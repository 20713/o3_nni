import argparse
import os
import numpy as np
import h5py
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matlab.engine
import pandas as pd

from .infer import load_model, predict
from .eval_plots import save_evaluation_plots


def _read_h5_scalar_attr(f, name):
    # 读取 HDF5 文件 attribute 中的标量（例如 OrbitNumber）
    # 兼容 attribute 可能是 python 标量、numpy 标量或 size=1 的 numpy 数组
    a = f.attrs.get(name)
    if a is None:
        return None
    if isinstance(a, np.ndarray) and a.size == 1:
        return int(a.reshape(-1)[0])
    try:
        return int(a)
    except Exception:
        return a


def _load_bremen_cache(bremen_l2_path):
    with Dataset(bremen_l2_path, "r") as bremen:
        print("load bremen data from", bremen_l2_path)
        orbitB = np.asarray(bremen.variables["orbit_number"][:]).reshape(-1)
        fovB = np.asarray(bremen.variables["FOV_number"][:]).reshape(-1)
        ozB = np.asarray(bremen.variables["ozone_concentration"][:]) * 6.0221e17
        tempB = np.asarray(bremen.variables["temperature_ecmwf"][:])
        presB = np.asarray(bremen.variables["pressure"][:])
        tghB = np.asarray(bremen.variables["altitude"][:])
    return {"orbitB": orbitB, "fovB": fovB, "ozB": ozB, "tempB": tempB, "presB": presB, "tghB": tghB}


def _gridfit_makima_engine(z0, wl0, rad_zwl, zq, lam, eng, smooth, project_root, mask_png_path=None):
    # 将单次观测的辐射场（高度×波长）映射到统一网格（zq×lam）上，并在内部使用对数辐射。
    #
    # 输入：
    # - z0: 1D，高度坐标（观测本身的切点高度；长度通常 101(0,...100)）
    # - wl0: 1D，波长坐标（观测本身的波长网格；长度通常 266(...)）
    # - rad_zwl: 2D，高度×波长的辐射（shape 通常为 (101, 266)）
    # - zq: 1D，目标高度网格（例如 0..60 km；长度 61）
    # - lam: 1D，目标波长通道（例如 7 个通道）
    # - eng: MATLAB engine 句柄（用于调用 gridfit/interp2）
    # - smooth: gridfit 的平滑参数（越大越平滑）
    # - project_root: MATLAB 函数所在目录（用于 addpath）
    # - mask_png_path: 可选，保存有效点 mask 热力图的路径（用于调试/质控）
    #
    # 输出：
    # - log_on_altius: 2D，shape=(len(zq), len(lam))，目标网格上的对数辐射
    if eng is None:
        raise RuntimeError("MATLAB engine is required but not available")
    if project_root:
        try:
            # 将项目目录加入 MATLAB 路径，确保可以调用 gridfit 等自定义函数
            eng.addpath(project_root, nargout=0)
        except Exception:
            pass
    # 将 1D 的高度轴 z0 与波长轴 wl0 扩展成 2D 坐标网格，与 rad_zwl(高度×波长)逐点对应
    # indexing="ij"：第 0 维对应 z0(行)，第 1 维对应 wl0(列)
    # Z0[i, j] = z0[i]：每一行是同一个高度值；WL0[i, j] = wl0[j] ：每一列是同一个波长值
    Z0, WL0 = np.meshgrid(z0, wl0, indexing="ij")
    # 有效辐射点：有限值且大于 0（后续要取对数，<=0 会无效）
    mask = np.isfinite(rad_zwl) & (rad_zwl > 0)
    if mask_png_path:
        # 可选：把 mask 保存为热力图（有效点=1，无效点=0），用于检查数据缺测/异常分布
        fig = plt.figure(figsize=(20, 6))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(mask.astype(np.uint8), aspect="auto", origin="lower", cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("wavelength index")
        ax.set_ylabel("altitude index")
        fig.tight_layout()
        fig.savefig(mask_png_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
    if not np.any(mask):
        raise ValueError("No valid radiance points for gridfit")
    # 将二维网格上的有效点抽取为散点集合 (Z1, WL1, RAD2)
    # Z1/WL1 是有效点的坐标，RAD2 是对应的 log(radiance)
    Z1 = Z0[mask].astype(float).ravel()   #.ravel() ：展平为 1D 向量
    WL1 = WL0[mask].astype(float).ravel()
    RAD2 = np.log(rad_zwl[mask].astype(float)).ravel()
    # 转为 MATLAB 数据类型，供 eng.gridfit / eng.interp2 使用
    Z1_ml = matlab.double(Z1.tolist())
    WL1_ml = matlab.double(WL1.tolist())
    RAD2_ml = matlab.double(RAD2.tolist())
    z0_ml = matlab.double(np.asarray(z0, dtype=float).tolist())
    wl0_ml = matlab.double(np.asarray(wl0, dtype=float).tolist())
    # 在 (z0, wl0) 原网格上对散点 log(radiance) 做平滑拟合，得到完整的规则网格场 logRAD3
    # gridfit 输入：
    # - Z1_ml：有效点的高度坐标（散点，1D）
    # - WL1_ml：有效点的波长坐标（散点，1D）
    # - RAD2_ml：有效点的观测值 log(radiance)（散点，1D）
    # - z0_ml：目标规则网格的高度轴（1D）
    # - wl0_ml：目标规则网格的波长轴（1D）
    out = eng.gridfit(Z1_ml, WL1_ml, RAD2_ml, z0_ml, wl0_ml,
                      'smooth', float(smooth),
                      'interp', 'bilinear',
                      'solver', 'normal',
                      'regularizer', 'gradient',
                      'extend', 'warning',
                      'tilesize', float('inf'),
                      nargout=1)
    logRAD3 = np.array(out, dtype=float)
    # MATLAB 的 interp2(X, Y, V, Xq, Yq) 这里需要的 V 维度与后续输入坐标对应，故做一次转置
    logRAD3_T = logRAD3.T #.T ：转置为 (波长,高度) 格式

    # 构建目标查询网格 (zq, lam)：将辐射场从 (z0, wl0) 映射到统一高度×通道
    Zq, LAMq = np.meshgrid(np.asarray(zq, dtype=float), np.asarray(lam, dtype=float), indexing="ij")
    LAM_ml = matlab.double(LAMq.tolist())
    Z_ml = matlab.double(Zq.tolist())
    WL0_ml = matlab.double(np.asarray(wl0, dtype=float).tolist())
    Z0_ml = matlab.double(np.asarray(z0, dtype=float).tolist())
    logRAD3_ml = matlab.double(logRAD3_T.tolist())
    # 二次插值：在拟合后的规则网格场上，用 makima 插值到目标 (zq, lam) 网格
    RAD4L_ml = eng.interp2(WL0_ml, Z0_ml, logRAD3_ml, LAM_ml, Z_ml, "makima", nargout=1)
    log_on_altius = np.array(RAD4L_ml, dtype=float)
    return log_on_altius

def validate(
    omps_l1_path,
    bremen_l2_path,
    model_path,
    smooth=10.0,
    output_dir=None,
    run_out_dir=None,
    save_pdf=True,
    iS=2,
    iT_start=1,
    iT_stop=162,
    iT_step=1,
    eng=None,
    ctx=None,
    bremen_cache=None,
):
    # 用 OMPS L1G 辐射观测驱动 ONNI 模型预测臭氧剖面，并与 Bremen L2 剖面对比。
    #
    # 主要数据流：
    # 1) 读取 Bremen L2（netCDF）：orbit/FOV 与臭氧、温度、气压剖面
    # 2) 读取 OMPS L1（HDF5）：经纬度、太阳几何、切点高度、辐射谱
    # 3) 对 OMPS 单观测辐射做 log + 高度/波长映射（gridfit + makima）
    # 4) 计算归一化辐射特征，做 PCA 投影，拼接 sza/saa，输入 NN 得到 O3 VMR(0..60 km)
    # 5) 与 Bremen L2 的 O3 VMR 剖面绘图对比，保存到 PDF
    base=os.path.dirname(__file__)

    # 严格校验文件是否存在
    for path in [omps_l1_path, bremen_l2_path, model_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"无法找到文件: {path}")

    if ctx is None:
        ctx = load_model(model_path=model_path)

    aux = ctx.get("pca_config")
    if aux is None:
        raise ValueError("model checkpoint does not contain pca_config; please re-train with updated model_train.py")
    inorm = int(aux["inorm"]) # 归一化参考高度索引（1-based；训练与推理必须一致，通常 inorm=41 对应 40 km）
    print(f"use inorm:{inorm}")
    # ONNI 使用的 7 个通道中心波长（单位与 wl0 统一：nm）
    lam = aux["wav"]#wav_chan=np.array([300, 315, 351, 525, 600, 675, 745], dtype=float)
    print(f"use wav:{lam}")
    npcChan = aux["npc_wav"].astype(int).tolist()
    print(f"use npc_wav:{npcChan}")
    Uoz = aux["Uoz"]
    YMoz = aux["YMoz"]
    zq = np.asarray(aux["radiance_grid"], dtype=float).reshape(-1)
    lz = np.asarray(aux["vmr_grid"], dtype=float).reshape(-1)
    if zq.size < 2:
        raise ValueError(f"radiance_grid is invalid: len={zq.size}")
    if lz.size < 2:
        raise ValueError(f"vmr_grid is invalid: len={lz.size}")
    if not (1 <= int(inorm) <= int(zq.size)):
        raise ValueError(f"inorm={inorm} out of range for radiance_grid (len={zq.size})")
    print(f"use radiance_grid:{zq}")
    print(f"use vmr_grid:{lz}")
    z_eval = np.arange(15.0, 46.0, 1.0, dtype=float)

    def _interp_profile(z_src, y_src, z_dst):
        z_src = np.asarray(z_src, dtype=float).reshape(-1)
        y_src = np.asarray(y_src, dtype=float).reshape(-1)
        m = np.isfinite(z_src) & np.isfinite(y_src)
        if np.count_nonzero(m) < 2:
            return np.full_like(z_dst, np.nan, dtype=float)
        z_use = z_src[m]
        y_use = y_src[m]
        order = np.argsort(z_use)
        z_use = z_use[order]
        y_use = y_use[order]
        z_unique, idx = np.unique(z_use, return_index=True)
        y_unique = y_use[idx]
        if z_unique.size < 2:
            return np.full_like(z_dst, np.nan, dtype=float)
        return np.interp(z_dst, z_unique, y_unique, left=np.nan, right=np.nan)

    project_root = os.path.dirname(os.path.dirname(__file__))
    eng_owned = False
    if eng is None:
        eng = matlab.engine.start_matlab()
        eng_owned = True
    try:
        try:
            # 把当前目录与项目根目录加入 MATLAB 路径，方便找 gridfit.m
            eng.addpath(base, nargout=0)
            eng.addpath(project_root, nargout=0)
        except Exception:
            pass

        if bremen_cache is None:
            bremen_cache = _load_bremen_cache(bremen_l2_path)
        orbitB = bremen_cache["orbitB"]
        fovB = bremen_cache["fovB"]
        ozB = bremen_cache["ozB"]
        tempB = bremen_cache["tempB"]
        presB = bremen_cache["presB"]
        tghB = bremen_cache["tghB"]

        with h5py.File(omps_l1_path, "r") as f:
            print("load omps data from",omps_l1_path)
            # OMPS L1：时间序列（bytes 的 ISO8601 字符串）
            timeUTC = f["/GRIDDED_DATA/DateTimeUTC"][()] #(DimAlongTrack=180)
            # 轨道号（从 HDF5 attribute 读取）
            orbit = _read_h5_scalar_attr(f, "OrbitNumber") #get orbit number from attribute of .h5
            # 地理与太阳几何（shape=(DimAlongTrack=180, DimAcrossTrack=3)）
            geoLat = f["/GEOLOCATION_DATA/Latitude_35km"][()] #(DimAlongTrack=180, DimAcrossTrack=3)
            geoLon = f["/GEOLOCATION_DATA/Longitude_35km"][()] #(DimAlongTrack=180, DimAcrossTrack=3)
            geoSAA = f["/GEOLOCATION_DATA/SolarAzimuth_35km"][()] #(DimAlongTrack=180, DimAcrossTrack=3)
            geoSZA = f["/GEOLOCATION_DATA/SolarZenithAngle_35km"][()] #(DimAlongTrack=180, DimAcrossTrack=3)
            # 原始波长网格（266 个），这里 *1e3 将单位统一到 nm
            wl0 = f["/GRIDDED_DATA/WavelengthGrid"][()].astype(float) * 1e3 #(DimWavelength=266)
            # 切点高度（shape=(180,3,101)）
            th = f["/GRIDDED_DATA/TangentHeight"][()] #（DimAlongTrack=180, DimAcrossTrack=3, DimVertical=101
            # 辐射（按本文件当前约定：shape=(180,3,101,266)；高度×波长）
            RAD0 = f["/GRIDDED_DATA/Radiance"] # (DimAlongTrack=180, DimAcrossTrack=3, DimVertical=101, DimWavelength=266)

            fig_compare = None
            ax1 = None
            ax2 = None
            pdf = None
            pdf_path = None
            output_root = output_dir if (output_dir is not None and str(output_dir).strip()) else os.path.join(base, "validate_omps_output")
            if run_out_dir is None or (isinstance(run_out_dir, str) and (not run_out_dir.strip())):
                run_ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                run_out_dir = os.path.join(output_root, run_ts)
            os.makedirs(run_out_dir, exist_ok=True)
            omps_tag = os.path.splitext(os.path.basename(omps_l1_path))[0]
            if bool(save_pdf):
                pdf_path = os.path.join(run_out_dir, f"ResultCompare_{omps_tag}.pdf")
                pdf = PdfPages(pdf_path)
                fig_compare = plt.figure(figsize=(11.69, 8.27))
                ax1 = fig_compare.add_subplot(1, 2, 1)
                ax2 = fig_compare.add_subplot(1, 2, 2)
            t_list = []
            y_list = []

            # iS：狭缝序号（1-based），转换到 0-based 作为数组索引
            iS0 = int(iS) - 1
            n_along = int(geoLat.shape[0])
            iT_start_eff = max(1, int(iT_start))
            iT_stop_eff = min(int(iT_stop), n_along)
            iT_step_eff = int(iT_step)
            if iT_start_eff > iT_stop_eff:
                raise ValueError(f"iT range [{iT_start_eff},{iT_stop_eff}] is empty for this OMPS file (n_along={n_along})")
            try:
                for iT in range(iT_start_eff, iT_stop_eff + 1, iT_step_eff):
                    # iT：沿轨观测序号（1-based），转换到 0-based
                    iT0 = int(iT) - 1
                    # 当前观测的地理/太阳几何（同一 iT0 下 3 个狭缝选择其一 iS0）
                    lat = float(geoLat[iT0, iS0])
                    lon = float(geoLon[iT0, iS0])
                    sza = float(geoSZA[iT0, iS0])
                    saa = float(np.abs(geoSAA[iT0, iS0]))
                    # 当前观测的切点高度（101 层）
                    z0 = np.asarray(th[iT0, iS0, :], dtype=float) # shape:(101,) value:[0.5,100.5,1]
                    # 当前观测的辐射二维场（高度×波长，shape=(101,266)）
                    rad_zwl = np.asarray(RAD0[iT0, iS0, :, :], dtype=float)
                    # 负值辐射视为无效点（后续 mask 会过滤；log 也要求 >0）
                    rad_zwl[rad_zwl < 0] = np.nan
                    mask_png = None
                    # 将辐射映射到统一网格（zq×lam），并输出对数辐射 log_on_altius
                    log_on_altius = _gridfit_makima_engine(z0, wl0, rad_zwl, zq, lam, eng, float(smooth), project_root, mask_png_path=mask_png)
                    radln = log_on_altius - log_on_altius[inorm - 1, :][None, :]

                    # 对每个通道做 PCA 投影并截断到该通道指定的 pc 数（npcChan）
                    Ait = []
                    # Yp：PCA 重构后的辐射（用于右图展示）；Yomps：原始 normalized log-radiance
                    Yp = np.zeros((len(lam), len(zq)))
                    Yomps = np.zeros((len(lam), len(zq)))
                    for ich in range(len(lam)):
                        npcs = int(npcChan[ich])
                        ydum = radln[:, ich]
                        Yomps[ich, :] = ydum
                        ym = YMoz[:, ich]
                        dy = ydum - ym
                        U = Uoz[:, :, ich]
                        a_full = dy @ U
                        Ait.append(a_full[:npcs])
                        Yp[ich, :] = ym + a_full[:npcs] @ U[:, :npcs].T

                    # 拼接所有通道的 PCA 系数，并追加 sza/saa 作为额外输入特征
                    Ait = np.concatenate(Ait, axis=0)
                    x = np.concatenate([Ait, np.array([sza, saa])], axis=0)[None, :]
                    # NN 推理输出：臭氧 VMR 剖面（长度 61，对应 0..60 km）
                    y2 = np.asarray(predict(x, ctx)[0], dtype=float).reshape(-1)
                    if not np.all(np.isfinite(y2)):
                        y2 = np.where(np.isfinite(y2), y2, np.nan)
                    # 用 orbit + FOV 在 Bremen L2 中找到对应剖面索引 idB（这里要求精确匹配）
                    idxs = np.where((orbitB == orbit) & (fovB == iT))[0]
                    if idxs.size == 0:
                        idxs = np.where((orbitB == orbit) & (fovB == iT0))[0]
                    if idxs.size == 0:
                        print(f"orbit={orbit} fov={iT} not found")
                        continue
                    idB = int(idxs[0])
                        #print(f"idB={idB}")

                    # Bremen 剖面：按时间索引 idB 直接取一行（shape=(61,)）
                    noz = ozB[idB, :].astype(float)
                    temp = tempB[idB, :].astype(float)
                    pres = presB[idB, :].astype(float)
                    nair = 100.0 * pres / temp / 1.306e-23
                    ozvmr = np.asarray(1e12 * noz / nair, dtype=float).reshape(-1)
                    tgh1 = np.asarray(tghB, dtype=float).reshape(-1)

                    t_eval = _interp_profile(tgh1, ozvmr, z_eval)
                    if y2.size != lz.size:
                        raise ValueError(f"vmr_grid length {lz.size} does not match y2 length {y2.size}")
                    y_eval = _interp_profile(lz, y2, z_eval)
                    if np.all(np.isfinite(t_eval)) and np.all(np.isfinite(y_eval)):
                        t_list.append(t_eval)
                        y_list.append(y_eval)

                    if pdf is not None:
                        ax1.clear()
                        ax1.plot(ozvmr, tgh1, "k-d", linewidth=2, markersize=3, label="BREMEN")
                        ax1.plot(y2, lz, "r-", linewidth=2, label="ONNI")
                        ax1.set_ylim([0, 60])
                        ax1.set_xlim([-1, 14])
                        ax1.grid(True, which="both", alpha=0.3)
                        ax1.set_xlabel("O$_3$ VMR [ppm]")
                        ax1.set_ylabel("z [km]")
                        ax1.legend(loc="best")

                        ax2.clear()
                        for k in range(Yp.shape[0]):
                            ax2.plot(Yp[k, :], zq, linewidth=1.5)
                        ax2.scatter(Yomps.reshape(-1), np.repeat(zq[None, :], Yomps.shape[0], axis=0).reshape(-1), c="k", s=8)
                        ax2.set_ylim([0, 60])
                        ax2.set_xlim([-3, 7])
                        ax2.grid(True, which="both", alpha=0.3)
                        ax2.set_xlabel("log(Normalized radiances (40 km))")
                        ax2.set_ylabel("z [km]")

                        t_raw = timeUTC[iT0]
                        if isinstance(t_raw, (bytes, np.bytes_)):
                            t_str = t_raw.decode("utf-8", errors="replace")
                        else:
                            t_str = str(t_raw)
                        fig_compare.suptitle( #iT0 是索引，从0开始，iT0+1 是FOV序号，即观测序号
                            f"Time={t_str} Orbit={orbit} Obs={iT0+1} Lat={lat:.4f} Lon={lon:.4f}  sza={sza:.2f} saa={saa:.2f}  smooth={smooth}",
                            fontsize=10,
                        )
                        fig_compare.tight_layout(rect=[0, 0, 1, 0.95])
                        pdf.savefig(fig_compare)
            finally:
                if pdf is not None:
                    pdf.close()
            if fig_compare is not None:
                plt.close(fig_compare)
            if pdf is not None:
                print(f"save {pdf_path}")

            if len(t_list) > 0:
                t_te = np.stack(t_list, axis=0)
                y_te = np.stack(y_list, axis=0)
                dy = y_te - t_te
                mse = float(np.mean(dy**2))
                reg = float(np.corrcoef(y_te.reshape(-1), t_te.reshape(-1))[0, 1])
                title = f"ONNI(red) vs Bremen(blue) 15-45 km; n_profile={t_te.shape[0]} mse={mse:.4g} corr={reg:.6f}"
                save_evaluation_plots(
                    t_te=t_te,
                    y_te=y_te,
                    out_dir=run_out_dir,
                    out_prefix="omps_eval_15_45km",
                    train_mse=mse,
                    test_mse=mse,
                    test_reg=reg,
                    z=z_eval,
                    title=title,
                )
    finally:
        if eng_owned:
            try:
                eng.quit()
            except Exception:
                pass
    if len(t_list) > 0:
        t_te = np.stack(t_list, axis=0)
        y_te = np.stack(y_list, axis=0)
        dy = y_te - t_te
        mse = float(np.mean(dy**2))
        reg = float(np.corrcoef(y_te.reshape(-1), t_te.reshape(-1))[0, 1])
        title = f"ONNI(red) vs Bremen(blue) 15-45 km; n_profile={t_te.shape[0]} mse={mse:.4g} corr={reg:.6f}"
        return {
            "t_te": t_te,
            "y_te": y_te,
            "mse": mse,
            "reg": reg,
            "n_profile": int(t_te.shape[0]),
            "z": z_eval,
            "title": title,
        }
    return {"t_te": None, "y_te": None, "n_profile": 0, "z": z_eval}


def main():
    # 命令行入口：指定输入文件/模型/参数，输出对比 PDF
    base = os.path.dirname(__file__)
    inputs_dir = os.path.join(base, "sample_files")
    default_out = os.path.join(base, "validate_omps_output")
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--omps", type=str, 
                    default=os.path.join(inputs_dir, "OMPS-NPP_LP-L1G-EV_v2.6_2016m0301t224735_o22510_2022m1005t174807.h5"), 
                    help="OMPS L1G 输入 HDF5 文件路径")
    ap.add_argument("--bremen", type=str, 
                    default=os.path.join(inputs_dir, "ESACCI-OZONE-L2-LP-OMPS_LP_SUOMI_NPP-IUP_UBR_V3_3NLC_UBR_HARMOZ_ALT-201603-fv0005.nc"), 
                    help="Bremen L2 输入 netCDF 文件路径")
    ap.add_argument("--model", type=str, 
                    default=os.path.join(inputs_dir, "model.pt"), 
                    help="训练好的模型文件路径")
    ap.add_argument("--smooth", type=float, default=10.0, help="MATLAB gridfit 的平滑参数（越大越平滑）")
    ap.add_argument("--out_dir", type=str, default=default_out, help="验证输出根目录（结果写入该目录下时间戳子目录）")
    ap.add_argument("--no_pdf", "--no-pdf", dest="no_pdf", action="store_true", help="不保存 PDF（仍会输出评估图 png）")
    ap.add_argument("--iS", type=int, default=2, help="狭缝序号（1..3），默认取中间狭缝 iS=2")
    ap.add_argument("--start", type=int, default=20, help="沿轨观测序号起始值 iT（1-based，默认 20）")
    ap.add_argument("--stop", type=int, default=160, help="沿轨观测序号结束值 iT（1-based，默认 160）")
    ap.add_argument("--step", type=int, default=1, help="沿轨观测序号步长（默认 1）")
    args = ap.parse_args()

    omps_input = str(args.omps)
    if not os.path.isdir(omps_input):
        validate(
            omps_l1_path=omps_input,
            bremen_l2_path=args.bremen,
            model_path=args.model,
            smooth=args.smooth,
            output_dir=args.out_dir,
            save_pdf=(not args.no_pdf),
            iS=args.iS,
            iT_start=args.start,
            iT_stop=args.stop,
            iT_step=args.step,
        )
        return

    omps_files = [
        os.path.join(omps_input, name)
        for name in sorted(os.listdir(omps_input))
        if name.lower().endswith(".h5")
    ]
    if len(omps_files) == 0:
        raise FileNotFoundError(f"no .h5 files found in directory: {omps_input}")

    batch_ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    batch_root = os.path.join(str(args.out_dir), batch_ts) #validate_omps_output\20260514_173743

    eng = matlab.engine.start_matlab()
    try:
        ctx = load_model(model_path=args.model)
        bremen_cache = _load_bremen_cache(args.bremen)
        all_t_list = []
        all_y_list = []
        for omps_path in omps_files:
            omps_tag = os.path.splitext(os.path.basename(omps_path))[0]
            run_out_dir = os.path.join(batch_root, omps_tag) #validate_omps_output\20260514_174223\OMPS-NPP_LP-L1G-EV_v2.6_2016m0301t004801_o22497_2022m1005t174727
            try:
                res = validate(
                    omps_l1_path=omps_path,
                    bremen_l2_path=args.bremen,
                    model_path=args.model,
                    smooth=args.smooth,
                    output_dir=args.out_dir,
                    run_out_dir=run_out_dir,
                    save_pdf=(not args.no_pdf),
                    iS=args.iS,
                    iT_start=args.start,
                    iT_stop=args.stop,
                    iT_step=args.step,
                    eng=eng,
                    ctx=ctx,
                    bremen_cache=bremen_cache,
                )
            except ValueError as e:
                print(f"skip {omps_path}: {e}")
                continue
            if res is not None and res.get("t_te") is not None and res.get("y_te") is not None:
                all_t_list.append(res["t_te"])
                all_y_list.append(res["y_te"])

        if len(all_t_list) > 0:
            t_all = np.concatenate(all_t_list, axis=0)
            y_all = np.concatenate(all_y_list, axis=0)
            dy = y_all - t_all
            mse = float(np.mean(dy**2))
            reg = float(np.corrcoef(y_all.reshape(-1), t_all.reshape(-1))[0, 1])
            title = f"ONNI(red) vs Bremen(blue) 15-45 km (ALL); n_profile={t_all.shape[0]} mse={mse:.4g} corr={reg:.6f}"
            save_evaluation_plots(
                t_te=t_all,
                y_te=y_all,
                out_dir=batch_root,
                out_prefix="omps_eval_15_45km_all",
                train_mse=mse,
                test_mse=mse,
                test_reg=reg,
                z=np.arange(15.0, 46.0, 1.0, dtype=float),
                title=title,
            )
    finally:
        try:
            eng.quit()
        except Exception:
            pass

if __name__ == "__main__":
    main()
