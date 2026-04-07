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


def _gridfit_makima_engine(z0, wl0, rad_zwl, zq, lam, eng, smooth, project_root, mask_png_path=None):
    # 将单次观测的辐射场（高度×波长）映射到统一网格（zq×lam）上，并在内部使用对数辐射。
    #
    # 输入：
    # - z0: 1D，高度坐标（观测本身的切点高度；长度通常 101）
    # - wl0: 1D，波长坐标（观测本身的波长网格；长度通常 266）
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
    omps_l1_path="",
    bremen_l2_path="",
    ozaux_path=None,
    model_path=None,
    smooth=10.0,
    save_pdf=None,
    show=True,
    iS=2,
    iT_start=1,
    iT_stop=162,
    iT_step=1,
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
    inputs_dir=os.path.join(base,"inputs")
    try:
        os.makedirs(inputs_dir,exist_ok=True)
    except Exception:
        pass
    out_dir=os.path.join(base,"outputs")
    try:
        os.makedirs(out_dir,exist_ok=True)
    except Exception:
        pass

    # 默认辅助数据/模型文件路径（若未显式指定）
    if ozaux_path is None or not ozaux_path:
        ozaux_path=os.path.join(out_dir,"ozAux3.npz")
    if model_path is None or not model_path:
        model_path=os.path.join(out_dir,"model.pt")

    # 若未给出输入文件或给出路径不存在，尝试在 inputs_dir 下找默认样例文件
    if not os.path.exists(omps_l1_path):
        omps_try=os.path.join(inputs_dir,os.path.basename(omps_l1_path)) if omps_l1_path else os.path.join(inputs_dir,"OMPS-NPP_LP-L1G-EV_v2.6_2016m0301t210605_o22509_2022m1005t174736.h5")
        if os.path.exists(omps_try):
            omps_l1_path=omps_try
        else:
            raise FileNotFoundError(omps_l1_path)
    if not os.path.exists(bremen_l2_path):
        bremen_try=os.path.join(inputs_dir,os.path.basename(bremen_l2_path)) if bremen_l2_path else os.path.join(inputs_dir,"ESACCI-OZONE-L2-LP-OMPS_LP_SUOMI_NPP-IUP_UBR_V3_3NLC_UBR_HARMOZ_ALT-201603-fv0005.nc")
        if os.path.exists(bremen_try):
            bremen_l2_path=bremen_try
        else:
            raise FileNotFoundError(bremen_l2_path)
    if not os.path.exists(ozaux_path):
        raise FileNotFoundError(ozaux_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    # ozaux 里保存了归一化高度索引、各通道 PCA 个数、PCA 基、均值谱等
    aux = np.load(ozaux_path)
    inorm = int(aux["inorm"]) # 归一化参考高度索引（1-based；训练与推理必须一致，通常 inorm=41 对应 40 km）
    print(f"inorm={inorm}")
    npcChan = aux["npcChan"].astype(int).tolist()
    Uoz = aux["Uoz"]
    YMoz = aux["YMoz"]

    # ONNI 使用的 7 个通道中心波长（单位与 wl0 统一：nm）
    lam = np.array([300, 315, 351, 525, 600, 675, 745], dtype=float)
    # 目标高度网格：0..60 km，共 61 层（与 NN 输出维度一致）
    lz = np.arange(61) # (61,）
    zq = lz.astype(float)

    # 加载训练好的网络 + scaler（推理需要一致的选列与缩放）
    ctx = load_model(model_path=model_path)
    project_root = os.path.dirname(os.path.dirname(__file__))
    # 启动 MATLAB 引擎，用于调用 gridfit.m 和 interp2
    eng = matlab.engine.start_matlab()
    try:
        try:
            # 把当前目录与项目根目录加入 MATLAB 路径，方便找 gridfit.m
            eng.addpath(base, nargout=0)
            eng.addpath(project_root, nargout=0)
        except Exception:
            pass

        with Dataset(bremen_l2_path, "r") as bremen:
            print("load bremen data from",bremen_l2_path)
            # Bremen L2：每条剖面都带 orbit_number 与 FOV_number，可用于和 OMPS 观测对齐
            orbitB = np.asarray(bremen.variables["orbit_number"][:]).reshape(-1) #(dim_time=52371)
            fovB = np.asarray(bremen.variables["FOV_number"][:]).reshape(-1)#(dim_time=52371)
            # Bremen 的臭氧浓度（这里乘以常数转为 number density 一类单位，后续再换算 VMR）
            ozB = np.asarray(bremen.variables["ozone_concentration"][:]) * 6.0221e17 #(dim_time=52371, dim_altitude=61)
            # 温度/气压剖面（用于把臭氧浓度换算成 VMR）
            tempB = np.asarray(bremen.variables["temperature_ecmwf"][:]) #(dim_time=52371, dim_altitude=61)
            presB = np.asarray(bremen.variables["pressure"][:]) #(dim_time=52371, dim_altitude=61)
            # Bremen 的高度轴（61 层）
            tghB = np.asarray(bremen.variables["altitude"][:]) #(dim_altitude=61) [5,1,66]

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

                # 单页对比图：左侧 O3 剖面，右侧辐射特征（OMPS vs 重构）
                fig_compare = plt.figure(figsize=(11.69, 8.27))
                ax1 = fig_compare.add_subplot(1, 2, 1)
                ax2 = fig_compare.add_subplot(1, 2, 2)
                # PDF 输出路径：若用户未指定，则使用 OMPS 文件名作为标签，避免覆盖
                if save_pdf:
                    pdf_path = save_pdf
                else:
                    omps_tag = os.path.splitext(os.path.basename(omps_l1_path))[0]
                    pdf_path = os.path.join(out_dir, f"ResultCompare_{omps_tag}.pdf")
                pdf = PdfPages(pdf_path)

                # iS：狭缝序号（1-based），转换到 0-based 作为数组索引
                iS0 = int(iS) - 1
                try:
                    for iT in range(iT_start, iT_stop + 1, iT_step):
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
                        #zq ：目标高度网格（你代码里是 0..60 ，长度 61）  lam ：目标波长通道（7个固定通道） rad_zwl ：原始辐射二维场（高度×波长）
                        mask_png = None#os.path.join(out_dir, f"mask_iS{iS0+1}_iT{iT0+1}.png")
                        # 将辐射映射到统一网格（zq×lam），并输出对数辐射 log_on_altius
                        log_on_altius = _gridfit_makima_engine(z0, wl0, rad_zwl, zq, lam, eng, float(smooth), project_root, mask_png_path=mask_png)
                        # 归一化中心点：以参考高度 inorm（1-based）处的 log(radiance) 作为基线，对整条廓线做差
                        # 这样可消除辐射绝对量级差异，得到 normalized log-radiance；该中心点与训练阶段保持一致（来自 ozAux3.npz）
                        radln = log_on_altius - log_on_altius[inorm - 1, :][None, :]

                        # 对每个通道做 PCA 投影并截断到该通道指定的 pc 数（npcChan）
                        Ait = []
                        # Yp：PCA 重构后的辐射（用于右图展示）；Yomps：原始 normalized log-radiance
                        Yp = np.zeros((len(lam), len(lz)))
                        Yomps = np.zeros((len(lam), len(lz)))
                        for ich in range(len(lam)):
                            npcs = int(npcChan[ich])
                            ydum = radln[:, ich]
                            Yomps[ich, :] = ydum
                            ym = YMoz[:, ich]
                            dy = ydum - ym
                            U = Uoz[:, :, ich]
                            # a_full：PCA 系数（全维）；只取前 npcs 个用于网络输入
                            a_full = dy @ U
                            Ait.append(a_full[:npcs])
                            # Yp：用前 npcs 个 PCA 分量重构，便于可视化对比
                            Yp[ich, :] = ym + a_full[:npcs] @ U[:, :npcs].T

                        # 拼接所有通道的 PCA 系数，并追加 sza/saa 作为额外输入特征
                        Ait = np.concatenate(Ait, axis=0)
                        x = np.concatenate([Ait, np.array([sza, saa])], axis=0)[None, :]
                        # NN 推理输出：臭氧 VMR 剖面（长度 61，对应 0..60 km）
                        y2 = np.asarray(predict(x, ctx)[0], dtype=float).reshape(-1)
                        if not np.all(np.isfinite(y2)):
                            y2 = np.where(np.isfinite(y2), y2, np.nan)
                        # 轨道号orbitB、观测号fovB：(dim_time=52371)
                        # 用 orbit + FOV 在 Bremen L2 中找到对应剖面索引 idB（这里要求精确匹配）
                        idxs = np.where((orbitB == orbit) & (fovB == iT))[0]
                        if idxs.size == 0:
                            print(f"orbit={orbit} fov={iT} not found")
                            continue
                        idB = int(idxs[0])
                        print(f"idB={idB}")

                        # Bremen 剖面：按时间索引 idB 直接取一行（shape=(61,)）
                        noz = ozB[idB, :].astype(float)
                        temp = tempB[idB, :].astype(float)
                        pres = presB[idB, :].astype(float)
                        # 由气压与温度计算空气数密度（此处常数来自原始实现）
                        nair = 100.0 * pres / temp / 1.306e-23
                        # 把臭氧数浓度换算为体积分数 VMR（ppm）
                        ozvmr = np.asarray(1e12 * noz / nair, dtype=float).reshape(-1)
                        # Bremen 高度轴（61）
                        tgh1 = np.asarray(tghB, dtype=float).reshape(-1)

                        # 左图：Bremen（黑） vs ONNI（红）臭氧剖面对比
                        ax1.clear()
                        ax1.plot(ozvmr, tgh1, "k-d", linewidth=2, markersize=3, label="BREMEN")
                        ax1.plot(y2, lz, "r-", linewidth=2, label="ONNI 7 channels.")
                        ax1.set_ylim([0, 60])
                        ax1.set_xlim([-1, 14])
                        ax1.grid(True, which="both", alpha=0.3)
                        ax1.set_xlabel("O$_3$ VMR [ppm]")
                        ax1.set_ylabel("z [km]")
                        ax1.legend(loc="best")

                        # 右图：各通道 normalized log-radiance 的重构曲线（Yp）与原始散点（Yomps）
                        ax2.clear()
                        for k in range(Yp.shape[0]):
                            ax2.plot(Yp[k, :], lz, linewidth=1.5)
                        ax2.scatter(Yomps.reshape(-1), np.repeat(lz[None, :], Yomps.shape[0], axis=0).reshape(-1), c="k", s=8)
                        ax2.set_ylim([0, 60])
                        ax2.set_xlim([-3, 7])
                        ax2.grid(True, which="both", alpha=0.3)
                        ax2.set_xlabel("log(Normalized radiances (40 km))")
                        ax2.set_ylabel("z [km]")

                        # timeUTC 是 bytes，显示前先 decode 为字符串
                        t_raw = timeUTC[iT0]
                        if isinstance(t_raw, (bytes, np.bytes_)):
                            t_str = t_raw.decode("utf-8", errors="replace")
                        else:
                            t_str = str(t_raw)
                        fig_compare.suptitle(
                            f"Time={t_str} Orbit={orbit} Obs={iT0} Lat={lat:.4f} Lon={lon:.4f}  sza={sza:.2f} saa={saa:.2f}  smooth={smooth}",
                            fontsize=10,
                        )
                        fig_compare.tight_layout(rect=[0, 0, 1, 0.95])

                        if pdf:
                            # 将当前页面写入 PDF
                            pdf.savefig(fig_compare)
                        if show:
                            plt.pause(0.1)
                finally:
                    if pdf:
                        # 关闭 PDF 文件句柄，确保内容落盘
                        pdf.close()
                if show:
                    plt.show()
                else:
                    plt.close(fig_compare)
    finally:
        try:
            # 关闭 MATLAB 引擎（释放资源）
            eng.quit()
        except Exception:
            pass


def main():
    # 命令行入口：指定输入文件/模型/参数，输出对比 PDF
    ap = argparse.ArgumentParser()
    ap.add_argument("--omps", type=str, default="", help="OMPS L1G 输入 HDF5 文件路径（不填则尝试使用 inputs/ 下的默认样例文件）")
    ap.add_argument("--bremen", type=str, default="", help="Bremen L2 输入 netCDF 文件路径（不填则尝试使用 inputs/ 下的默认样例文件）")
    ap.add_argument("--ozaux", type=str, default="", help="辅助数据 npz 路径（包含 PCA/归一化等；不填则使用 outputs/ozAux3.npz）")
    ap.add_argument("--model", type=str, default="", help="训练好的模型文件路径（不填则使用 outputs/model.pt）")
    ap.add_argument("--smooth", type=float, default=10.0, help="MATLAB gridfit 的平滑参数（越大越平滑）")
    ap.add_argument("--save-pdf", type=str, default="", help="对比图输出 PDF 路径（不填则写入 outputs/ResultCompare_<omps文件名>.pdf）")
    ap.add_argument("--no-show", action="store_true", help="不弹出交互式窗口，仅保存 PDF")
    ap.add_argument("--iS", type=int, default=2, help="狭缝序号（1..3），默认取中间狭缝 iS=2")
    ap.add_argument("--start", type=int, default=20, help="沿轨观测序号起始值 iT（1-based，默认 20）")
    ap.add_argument("--stop", type=int, default=162, help="沿轨观测序号结束值 iT（1-based，默认 162）")
    ap.add_argument("--step", type=int, default=1, help="沿轨观测序号步长（默认 1）")
    args = ap.parse_args()
    validate(
        omps_l1_path=args.omps,
        bremen_l2_path=args.bremen,
        ozaux_path=(args.ozaux if args.ozaux else None),
        model_path=(args.model if args.model else None),
        smooth=args.smooth,
        save_pdf=(args.save_pdf if args.save_pdf else None),
        show=(not args.no_show),
        iS=args.iS,
        iT_start=args.start,
        iT_stop=args.stop,
        iT_step=args.step,
    )


if __name__ == "__main__":
    main()
