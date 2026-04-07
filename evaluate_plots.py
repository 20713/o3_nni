import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from .data_prepare import prepare_pca_features_and_io
from .infer import load_model, predict


def evaluate(mat_path,model_path=None,out_prefix="nni_py"):
    base=os.path.dirname(__file__)
    inputs=os.path.join(base,"inputs")
    try:
        os.makedirs(inputs,exist_ok=True)
    except Exception:
        pass
    if not os.path.exists(mat_path) or not mat_path:
        cand=os.path.join(inputs,"run_20260307_110701_SmartG_OutputXY_For_NNtrain.mat")
        mat_path=cand
    x,t,_,_=prepare_pca_features_and_io(mat_path)
    ctx=load_model(model_path=model_path)
    y=predict(x,ctx)

    n=x.shape[0]
    i1=int(0.70*n) #end of train set
    i2=int(0.85*n) #start of test set not val set
    y_tr=y[:i1]
    t_tr=t[:i1]
    y_te=y[i2:]
    t_te=t[i2:]
    n_te=t_te.shape[0]

    tr_mse=float(np.mean((y_tr-t_tr)**2))
    te_mse=float(np.mean((y_te-t_te)**2))
    reg=float(np.corrcoef(y_te.reshape(-1),t_te.reshape(-1))[0,1]) #reg 是 测试集 上的相关系数指标
    z=np.arange(t.shape[1],dtype=float)
    t_mean=t_te.mean(axis=0)
    t_std=t_te.std(axis=0)
    y_mean=y_te.mean(axis=0)
    y_std=y_te.std(axis=0)

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
    plt.title(f"true(blue) vs pred(red); n_test={n_te} train_mse={tr_mse:.4g} test_mse={te_mse:.4g} test_reg={reg:.6f}")
    plt.ylim([0,60])
    base=os.path.dirname(__file__)
    out_dir=os.path.join(base,"outputs")
    try:
        os.makedirs(out_dir,exist_ok=True)
    except Exception:
        pass
    plt.savefig(os.path.join(out_dir,f"{out_prefix}_mean_std.png"),dpi=150,bbox_inches="tight")

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
    plt.ylim([0,60])
    plt.savefig(os.path.join(out_dir,f"{out_prefix}_dsz.png"),dpi=150,bbox_inches="tight")

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
    plt.ylim([0,60])
    plt.savefig(os.path.join(out_dir,f"{out_prefix}_signed_percentiles.png"),dpi=150,bbox_inches="tight")

    return {"train_mse":tr_mse,"test_mse":te_mse,"test_reg":reg,"out_prefix":out_prefix}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mat",type=str,required=True)
    ap.add_argument("--model",type=str,default="")
    ap.add_argument("--out",type=str,default="nni_py")
    args=ap.parse_args()
    res=evaluate(args.mat,model_path=(args.model if args.model else None),out_prefix=args.out)
    print(res)


if __name__=="__main__":
    main()
