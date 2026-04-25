import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import argparse
import os
from .data_prepare import prepare_pca_features_and_io
from datetime import datetime
class MapMinMax:
    def __init__(self,remove_constant=True):
        self.remove_constant=remove_constant
        self.mask=None
        self.xmin=None
        self.scale=None
    def fit(self,X):
        xmin=X.min(axis=0)
        xmax=X.max(axis=0)
        if self.remove_constant:
            mask=(xmax>xmin)
        else:
            mask=np.ones_like(xmax,dtype=bool)
        self.mask=mask
        self.xmin=xmin[mask]
        rng=xmax[mask]-xmin[mask]
        rng_safe=np.where(rng==0,1.0,rng)
        self.scale=2.0/rng_safe
        return self
    def transform(self,X):
        X2=X[:,self.mask]
        Y=X2-self.xmin
        Y=Y*self.scale
        Y=Y-1.0
        return Y
class Net(nn.Module):
    def __init__(self,in_dim,out_dim=61,hid1=101,hid2=101):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,hid1),
            nn.Tanh(),
            nn.Linear(hid1,hid2),
            nn.Tanh(),
            nn.Linear(hid2,out_dim)
        )
    def forward(self,x):
        return self.net(x)
def load_prepared_npz(data_path):
    obj=np.load(data_path,allow_pickle=True)
    if "x" in obj and "t" in obj:
        x=obj["x"]
        t=obj["t"]
        return x,t
    raise ValueError(f"npz {data_path} must contain keys 'x' and 't'")


def save_evaluation_plots(t_te,y_te,out_dir,out_prefix,train_mse,test_mse,test_reg):
    os.makedirs(out_dir,exist_ok=True)
    z=np.arange(t_te.shape[1],dtype=float)
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
    plt.title(f"true(blue) vs pred(red); n_test={n_te} train_mse={train_mse:.4g} test_mse={test_mse:.4g} test_reg={test_reg:.6f}")
    plt.ylim([0,60])
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


def train_from_data(args):
    inp,out=load_prepared_npz(args.data_path)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device=args.device if (args.device is not None and str(args.device).strip()!="") else ("cuda" if torch.cuda.is_available() else "cpu")
    x_scaler=MapMinMax(remove_constant=True).fit(inp)
    t_scaler=MapMinMax(remove_constant=False).fit(out)
    x_scaled=x_scaler.transform(inp)
    t_scaled=t_scaler.transform(out)
    n=x_scaled.shape[0]
    i1=int(0.70*n)
    i2=int(0.85*n)
    x_tr=x_scaled[:i1]
    t_tr=t_scaled[:i1]
    x_va=x_scaled[i1:i2]
    t_va=t_scaled[i1:i2]
    x_te=x_scaled[i2:]
    t_te=t_scaled[i2:]
    ds_tr=TensorDataset(torch.from_numpy(x_tr).float(),torch.from_numpy(t_tr).float())
    ds_va=TensorDataset(torch.from_numpy(x_va).float(),torch.from_numpy(t_va).float())
    ds_te=TensorDataset(torch.from_numpy(x_te).float(),torch.from_numpy(t_te).float())
    dl_tr=DataLoader(ds_tr,batch_size=args.batch_size,shuffle=False)
    dl_va=DataLoader(ds_va,batch_size=args.batch_size,shuffle=False)
    net=Net(x_tr.shape[1]).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=args.lr)
    loss_fn=nn.MSELoss()
    best_va=np.inf
    best_state=None
    patience=args.epochs//10
    wait=0
    total_epochs=max(1,int(args.epochs))
    bar_width=30
    print(f"training on {device}; train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}",flush=True)
    for epoch in range(args.epochs):
        net.train()
        epoch_tr_loss=0.0
        batch_count=0
        for xb,yb in dl_tr:
            xb=xb.to(device)
            yb=yb.to(device)
            opt.zero_grad()
            pred=net(xb)
            loss=loss_fn(pred,yb)
            loss.backward()
            opt.step()
            epoch_tr_loss+=float(loss.item())
            batch_count+=1
        epoch_tr_loss/=max(1,batch_count)
        net.eval()
        with torch.no_grad():
            xv=torch.from_numpy(x_va).float().to(device)
            yv=torch.from_numpy(t_va).float().to(device)
            pv=net(xv)
            va_loss=loss_fn(pv,yv).item()
        improved=False
        if va_loss<best_va:
            best_va=va_loss
            best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            wait=0
            improved=True
        else:
            wait+=1
        progress=(epoch+1)/total_epochs
        filled=int(bar_width*progress)
        bar="="*filled+"."*(bar_width-filled)
        print(f"\r[{bar}] {epoch+1}/{total_epochs} train={epoch_tr_loss:.6g} val={va_loss:.6g} best={best_va:.6g} wait={wait}/{patience}",end="",flush=True)
        if improved or (epoch+1)==total_epochs or wait>=patience:
            print()
        if wait>=patience:
            print(f"early stop at epoch={epoch+1}, best_val={best_va:.6g}",flush=True)
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        xt=torch.from_numpy(x_tr).float().to(device)
        yt=torch.from_numpy(t_tr).float().to(device)
        pt=net(xt).cpu().numpy()
        xv=torch.from_numpy(x_va).float().to(device)
        yv=torch.from_numpy(t_va).float().to(device)
        pv=net(xv).cpu().numpy()
        xe=torch.from_numpy(x_te).float().to(device)
        ye=torch.from_numpy(t_te).float().to(device)
        pe=net(xe).cpu().numpy()
    def inv_scale(Ys,scaler):
        X=(Ys+1.0)/scaler.scale+scaler.xmin
        return X
    pt_i=inv_scale(pt,t_scaler)
    pv_i=inv_scale(pv,t_scaler)
    pe_i=inv_scale(pe,t_scaler)
    t_tr_i=inv_scale(t_tr,t_scaler)
    t_va_i=inv_scale(t_va,t_scaler)
    t_te_i=inv_scale(t_te,t_scaler)
    tr_mse=float(np.mean((pt_i-t_tr_i)**2))
    va_mse=float(np.mean((pv_i-t_va_i)**2))
    te_mse=float(np.mean((pe_i-t_te_i)**2))
    y_flat=pe_i.reshape(-1)
    t_flat=t_te_i.reshape(-1)
    reg=np.corrcoef(y_flat,t_flat)[0,1]
    # save model
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(str(args.out_dir),t)
    model_path=os.path.join(out_path,"model.pt")
    try:
        os.makedirs(out_path,exist_ok=True)
        print(f"create output dir {out_path}")
    except Exception:
        pass
    print(f"save model to {model_path}")
    torch.save({"state_dict":net.state_dict(),"x_scaler":{"mask":x_scaler.mask,"xmin":x_scaler.xmin,"scale":x_scaler.scale},"t_scaler":{"mask":t_scaler.mask,"xmin":t_scaler.xmin,"scale":t_scaler.scale}}, model_path)
    save_evaluation_plots(
        t_te=t_te_i,
        y_te=pe_i,
        out_dir=out_path,
        out_prefix="nni_py",
        train_mse=tr_mse,
        test_mse=te_mse,
        test_reg=float(reg),
    )
    return {"train_mse":tr_mse,"val_mse":va_mse,"test_mse":te_mse,"test_reg":float(reg),"model_path":model_path}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_path",type=str,default="./o3_nni/TRAIN_datasets/trainset_run_20260307_110701_SmartG_OutputXY_For_NNtrain_nz61_in41_nch7_ns25600_20260416_172511.npz")
    ap.add_argument("--epochs",type=int,default=5000)
    ap.add_argument("--out_dir",type=str,default="./o3_nni/TRAIN_outputs")
    ap.add_argument("--device",type=str,default="cuda:0")
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--batch_size",type=int,default=512)
    ap.add_argument("--seed",type=int,default=0)

    args=ap.parse_args()
    res=train_from_data(args)
    print(res)
if __name__=="__main__":
    main()
