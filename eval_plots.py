import os
import numpy as np
import matplotlib.pyplot as plt


def save_evaluation_plots(t_te, y_te, out_dir, out_prefix, train_mse, test_mse, test_reg, z=None, title=None):
    os.makedirs(out_dir,exist_ok=True)
    if z is None:
        z = np.arange(t_te.shape[1], dtype=float)
    else:
        z = np.asarray(z, dtype=float).reshape(-1)
        if t_te.shape[1] != z.size:
            raise ValueError(f"z length must match t_te.shape[1]; got len(z)={z.size}, t_te.shape[1]={t_te.shape[1]}")
    t_mean=t_te.mean(axis=0)
    t_std=t_te.std(axis=0)
    y_mean=y_te.mean(axis=0)
    y_std=y_te.std(axis=0)
    n_te=t_te.shape[0]
    plt.figure(1,figsize=(7,7))
    plt.clf()
    plt.plot(t_mean,z,"ob")
    plt.plot(y_mean,z,"r-",linewidth=2)
    plt.plot(t_mean+t_std,z,"ob",linewidth=1)
    plt.plot(t_mean-t_std,z,"ob",linewidth=1)
    plt.plot(y_mean+y_std,z,"r-",linewidth=1)
    plt.plot(y_mean-y_std,z,"r-",linewidth=1)
    plt.grid(True,which="both",alpha=0.3)
    plt.xlabel("O$_3$ vmr")
    plt.ylabel("z [km]")
    if title is None:
        plt.title(f"true(blue) vs pred(red); n_test={n_te} train_mse={train_mse:.4g} test_mse={test_mse:.4g} test_reg={test_reg:.6f}")
    else:
        plt.title(str(title))
    plt.ylim([float(z.min()), float(z.max())])
    plt.savefig(os.path.join(out_dir,f"{out_prefix}_mean_std.png"),dpi=150,bbox_inches="tight")
    print(f"save {os.path.join(out_dir,f'{out_prefix}_mean_std.png')}")
    dy=y_te-t_te
    p16=np.percentile(dy,16,axis=0)
    p50=np.percentile(dy,50,axis=0)
    p84=np.percentile(dy,84,axis=0)
    dsz=0.5*(p84-p16)
    plt.figure(2,figsize=(7,7))
    plt.clf()
    plt.plot(dsz,z,"k-",linewidth=2)
    plt.grid(True,which="both",alpha=0.3)
    plt.xlabel("ds [ppm]")
    plt.ylabel("z [km]")
    plt.title(f"ds=[p84-p16]/2  mean(ds)={dsz.mean():.4g}")
    plt.ylim([float(z.min()), float(z.max())])
    plt.savefig(os.path.join(out_dir,f"{out_prefix}_dsz.png"),dpi=150,bbox_inches="tight")
    print(f"save {os.path.join(out_dir,f'{out_prefix}_dsz.png')}")
    plt.figure(3,figsize=(7,7))
    plt.clf()
    plt.plot(p16,z,"b-",linewidth=2,label="p16")
    plt.plot(p50,z,"g-",linewidth=2,label="p50 (median)")
    plt.plot(p84,z,"r-",linewidth=2,label="p84")
    plt.grid(True,which="both",alpha=0.3)
    plt.xlabel("signed error [ppm]")
    plt.ylabel("z [km]")
    plt.title(f"signed error profile (test): p16/p50/p84; med(p50)={np.median(p50):.4g}")
    plt.legend()
    plt.ylim([float(z.min()), float(z.max())])
    plt.savefig(os.path.join(out_dir,f"{out_prefix}_signed_percentiles.png"),dpi=150,bbox_inches="tight")
    print(f"save {os.path.join(out_dir,f'{out_prefix}_signed_percentiles.png')}")
    plt.figure(4, figsize=(7, 7))
    plt.clf()
    bias_z = np.mean(dy, axis=0)
    mae_z  = np.mean(np.abs(dy), axis=0)
    rmse_z = np.sqrt(np.mean(dy**2, axis=0))
    plt.plot(bias_z, z, "g-", linewidth=2, label="Bias")
    plt.plot(mae_z, z, "b-", linewidth=2, label="MAE")
    plt.plot(rmse_z, z, "r-", linewidth=2, label="RMSE")
    plt.axvline(0, color="k", linewidth=1, alpha=0.5)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("error [ppm]")
    plt.ylabel("z [km]")
    plt.title(
        f"per-altitude errors: "
        f"mean|bias|={np.mean(np.abs(bias_z)):.4g}, "
        f"mean(MAE)={mae_z.mean():.4g}, "
        f"mean(RMSE)={rmse_z.mean():.4g}"
    )
    plt.legend()
    plt.ylim([float(z.min()), float(z.max())])
    plt.savefig(
        os.path.join(out_dir, f"{out_prefix}_bias_mae_rmse.png"),
        dpi=150,
        bbox_inches="tight"
    )
    print(f"save {os.path.join(out_dir, f'{out_prefix}_bias_mae_rmse.png')}")
