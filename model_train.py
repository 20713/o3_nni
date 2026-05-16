import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import argparse
import os
from .data_prepare import prepare_pca_features_and_io
from .eval_plots import save_evaluation_plots
from .net import Net
from .scalers import MapMinMax
from datetime import datetime
def load_prepared_npz(data_path):
    with np.load(data_path, allow_pickle=True) as obj:
        if "x" in obj and "t" in obj:
            x = obj["x"]
            t = obj["t"]
            pca_keys = (
                "wav",
                "npc_wav",
                "inorm",
                "radiance_grid",
                "vmr_grid",
                "Uoz",
                "YMoz",
            )
            pca_config = {k: obj[k] for k in pca_keys if k in obj}
            print(pca_config["vmr_grid"])
            return x, t, pca_config
    raise ValueError(f"npz {data_path} must contain keys 'x' and 't'")

def train_from_data(args):
    inp,out,pca_config=load_prepared_npz(args.data_path)
    print("pca_config.keys()",pca_config.keys())
    print(f"loaded {args.data_path}")
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
    x_tr=x_scaled[:i1] #x表示输入特征，tr表示训练集
    t_tr=t_scaled[:i1] #t表示目标特征
    x_va=x_scaled[i1:i2]#va表示验证集
    t_va=t_scaled[i1:i2]
    x_te=x_scaled[i2:]#te表示测试集
    t_te=t_scaled[i2:]
    ds_tr=TensorDataset(torch.from_numpy(x_tr).float(),torch.from_numpy(t_tr).float())
    ds_va=TensorDataset(torch.from_numpy(x_va).float(),torch.from_numpy(t_va).float())
    ds_te=TensorDataset(torch.from_numpy(x_te).float(),torch.from_numpy(t_te).float())
    dl_tr=DataLoader(ds_tr,batch_size=args.batch_size,shuffle=True)#训练集打乱顺序
    dl_va=DataLoader(ds_va,batch_size=args.batch_size,shuffle=False)
    net=Net(x_tr.shape[1]).to(device)
    print("model architecture:")
    print(net)
    total_params=sum(p.numel() for p in net.parameters())
    trainable_params=sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"model params: total={total_params:,}, trainable={trainable_params:,}")
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=float(args.weight_decay))
    scheduler = None
    lr_scheduler = str(getattr(args, "lr_scheduler", "none")).strip().lower()
    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(args.lr_factor),
            patience=int(args.lr_patience),
            min_lr=float(args.min_lr),
            threshold=1e-4,
            threshold_mode="rel"
        )
    loss_fn=nn.MSELoss()
    best_va=np.inf
    best_state=None
    early_stop_patience = 100#args.epochs//10
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
        if scheduler is not None:
            scheduler.step(float(va_loss))
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
        cur_lr=float(opt.param_groups[0]["lr"])
        print(f"\r[{bar}] {epoch+1}/{total_epochs} train={epoch_tr_loss:.6g} val={va_loss:.6g} best={best_va:.6g} wait={wait}/{early_stop_patience} lr={cur_lr:.3g}",end="",flush=True)
        if improved or (epoch+1)==total_epochs or wait>=early_stop_patience:
            print()
        if wait>=early_stop_patience:
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
    model_path=os.path.join(out_path,"model.pth")
    try:
        os.makedirs(out_path,exist_ok=True)
        print(f"create output dir {out_path}")
    except Exception:
        pass
    print(f"save model to {model_path}")
    torch.save(
        {
            "state_dict": net.state_dict(),
            "x_scaler": {"mask": x_scaler.mask, "xmin": x_scaler.xmin, "scale": x_scaler.scale},
            "t_scaler": {"mask": t_scaler.mask, "xmin": t_scaler.xmin, "scale": t_scaler.scale},
            "pca_config": pca_config,
            #"train_data_path": str(args.data_path),
            #"train_args": vars(args),
            #"saved_at": str(t),
        },
        model_path,
    )
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
    ap.add_argument("--weight_decay",type=float,default=1e-5)
    ap.add_argument("--lr_scheduler",type=str,default="none",choices=["none","plateau"])
    ap.add_argument("--lr_factor",type=float,default=0.5)
    ap.add_argument("--lr_patience",type=int,default=50)
    ap.add_argument("--min_lr",type=float,default=1e-6)
    ap.add_argument("--batch_size",type=int,default=1024)
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args()
    res=train_from_data(args)
    print(res)
if __name__=="__main__":
    main()
