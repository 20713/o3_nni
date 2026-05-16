import argparse
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 数据加载函数占位，实际运行时请确保 from .data_prepare import load_training_mat 可用
# ---------------------------------------------------------

def _parse_chan_list(chan_str):
    try:
        chan = tuple(int(x.strip()) for x in str(chan_str).split(",") if x.strip())
    except ValueError as e:
        raise ValueError(f"invalid --chan value: {chan_str}") from e
    return chan

def _compute_error_curves(log_radiance_data, radiance_orig, eps, m_max=None):
    """
    终极修正计算逻辑：
    1. 基于 log-radiance 计算 PCA。
    2. 计算重建残差 delta = log_orig - log_recon。
    3. 计算每个高度的标准差 sigma_z。
    4. 使用原始辐射强度作为权重 weight_z 对高度进行平均。
       - 权重保证了信号极弱的高度层（噪声区）不会错误地增加 npc 的数量。
    """
    n_valid, nz, n_chan = log_radiance_data.shape
    if m_max is None:
        m_max = nz
    m_max = int(m_max)

    e_curve = np.zeros((n_chan, m_max), dtype=float)
    npc_min = np.full((n_chan,), m_max, dtype=int)
    e_at_npc = np.full((n_chan,), np.nan, dtype=float)
    reached = np.zeros((n_chan,), dtype=bool)

    for c in range(n_chan):
        # 准备数据 (Samples, Altitudes)
        y = np.asarray(log_radiance_data[:, :, c], dtype=float)
        
        # 计算该波段在每个高度的平均辐射强度作为“信号权重”
        # 300nm 下，低层大气的权重会趋近于 0，有效滤除数值噪声
        weight_z = np.mean(radiance_orig[:, :, c], axis=0)
        weight_z = np.maximum(weight_z, 0) # 确保无负值
        if np.max(weight_z) > 0:
            weight_z /= np.max(weight_z) # 归一化至 [0, 1]
        
        # PCA 核心：中心化与 SVD
        ym = y.mean(axis=0)
        dy = y - ym[None, :]
        u, s, vh = np.linalg.svd(dy, full_matrices=False)
        scores = dy @ vh.T 

        for m in range(1, m_max + 1):
            if m >= nz:
                avg_sigma = 0.0
            else:
                # 重建对数空间残差
                delta = scores[:, m:] @ vh[m:, :]
                # 计算跨样本的标准差
                sigma_z = np.std(delta, axis=0)
                
                # 执行加权平均：RE = Σ(sigma_z * weight_z) / Σ(weight_z)
                # 这种方式能够真实反映“有信号层”的重建精度
                avg_sigma = np.sum(sigma_z * weight_z) / (np.sum(weight_z) + 1e-15)
            
            e_curve[c, m - 1] = float(avg_sigma)

        # 阈值筛选
        ok = np.where(e_curve[c, :] <= float(eps))[0]
        if ok.size > 0:
            idx = ok[0]
            npc_min[c] = idx + 1
            reached[c] = True
            e_at_npc[c] = e_curve[c, idx]
        else:
            e_at_npc[c] = e_curve[c, -1]

    return e_curve, npc_min, e_at_npc, reached

def plot_error_curves(e_curve, wav_chan, eps, out_path=None, show=False):
    n_chan, m_max = e_curve.shape
    x = np.arange(1, m_max + 1)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.turbo(np.linspace(0, 1, n_chan))
    
    y_percent = e_curve * 100.0
    for i in range(n_chan):
        ax.plot(x, y_percent[i, :], 'o-', ms=4, lw=1.2, alpha=0.8, 
                color=colors[i], mfc='none', label=f"{wav_chan[i]:g} nm")

    eps_pct = eps * 100.0
    ax.axhline(eps_pct, color='black', lw=1.5, label=f"Threshold {eps_pct:g}%")
    
    ax.set_yscale('log')
    ax.set_xlabel(r'$m_\lambda$ (Number of PCs)', fontsize=12)
    ax.set_ylabel('RE [%] (Weighted Avg Standard Deviation)', fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    ax.set_title("Average Relative Spectral Radiance Error (Weighted Logic)")
    
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=str, required=True, help="Path to training .mat")
    ap.add_argument("--chan", type=str, default="1,2,3,4,5,6,7,8", help="1-based channel ids")
    ap.add_argument("--nz", type=int, default=61, help="Number of altitude layers")
    ap.add_argument("--eps", type=float, default=0.003, help="Threshold (e.g. 0.003 for 0.3%)")
    ap.add_argument("--m_max", type=int, default=30)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # 导入数据加载模块
    from .data_prepare import load_training_mat
    wav, radiance, loki, vmr, sza, saa, zt, mat_used_path = load_training_mat(args.mat)

    wav_arr = np.asarray(wav).squeeze()
    chan_ids = _parse_chan_list(args.chan)
    idx_c = np.array(chan_ids) - 1
    wav_chan = wav_arr[idx_c]

    valid_idx = np.where(np.all(loki[:, idx_c], axis=1))[0]
    if valid_idx.size == 0:
        print("No valid samples found.")
        return

    nz = min(args.nz, radiance.shape[1])
    rad_orig = radiance[valid_idx][:, :nz, idx_c].copy()
    
    # 转换为对数空间。即使是加权平均，也需要底数保护以防数值异常。
    log_radiance = np.log(np.maximum(rad_orig, 1e-25))

    # 计算误差曲线（传入原始辐射作为权重参考）
    e_curve, npc_min, e_at_npc, reached = _compute_error_curves(
        log_radiance, rad_orig, eps=args.eps, m_max=args.m_max
    )

    print(f"\n[Weighted Results] eps={args.eps*100:g}%, samples={len(valid_idx)}")
    print(f"{'Chan':>5} {'Wav(nm)':>10} {'npc_min':>8} {'RE[%]':>12} {'Reached':>8}")
    for i, ch in enumerate(chan_ids):
        hit = "Yes" if reached[i] else "No"
        re_val = e_at_npc[i] * 100
        print(f"{ch:5d} {wav_chan[i]:10.2f} {npc_min[i]:8d} {re_val:12.4f} {hit:>8}")

    out_path = args.out if args.out else f"npc_weighted_{datetime.now().strftime('%H%M%S')}.png"
    plot_error_curves(e_curve, wav_chan, args.eps, out_path, args.show)
    print(f"\nDone. Plot saved to {out_path}")

if __name__ == "__main__":
    main()