import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import argparse
import os
from .data_prepare import prepare_pca_features_and_io
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
def train_from_mat(mat_path,iter_max=200000,seed=0,device=None):
    inp,out,l,_=prepare_pca_features_and_io(mat_path)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device="cuda" if torch.cuda.is_available() else "cpu"
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
    dl_tr=DataLoader(ds_tr,batch_size=512,shuffle=False)
    dl_va=DataLoader(ds_va,batch_size=512,shuffle=False)
    net=Net(x_tr.shape[1]).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    loss_fn=nn.MSELoss()
    best_va=np.inf
    best_state=None
    patience=10000
    wait=0
    total_epochs=max(1,int(iter_max))
    bar_width=30
    print(f"training on {device}; train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}",flush=True)
    for epoch in range(iter_max):
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
    tr_mse=float(np.mean((pt_i-inv_scale(t_tr,t_scaler))**2))
    va_mse=float(np.mean((pv_i-inv_scale(t_va,t_scaler))**2))
    te_mse=float(np.mean((pe_i-inv_scale(t_te,t_scaler))**2))
    y_flat=pe_i.reshape(-1)
    t_flat=inv_scale(t_te,t_scaler).reshape(-1)
    reg=np.corrcoef(y_flat,t_flat)[0,1] #reg 是 测试集 上的相关系数指标
    base=os.path.dirname(__file__)
    out_dir=os.path.join(base,"outputs")
    try:
        os.makedirs(out_dir,exist_ok=True)
    except Exception:
        pass
    model_path=os.path.join(out_dir,"model.pt")
    torch.save({"state_dict":net.state_dict(),"x_scaler":{"mask":x_scaler.mask,"xmin":x_scaler.xmin,"scale":x_scaler.scale},"t_scaler":{"mask":t_scaler.mask,"xmin":t_scaler.xmin,"scale":t_scaler.scale}}, model_path)
    return {"train_mse":tr_mse,"val_mse":va_mse,"test_mse":te_mse,"test_reg":float(reg),"model_path":model_path}
def train_from_data(data_path,iter_max=200000,seed=0,device=None):
    inp,out=load_prepared_npz(data_path)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device="cuda" if torch.cuda.is_available() else "cpu"
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
    dl_tr=DataLoader(ds_tr,batch_size=512,shuffle=False)
    dl_va=DataLoader(ds_va,batch_size=512,shuffle=False)
    net=Net(x_tr.shape[1]).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    loss_fn=nn.MSELoss()
    best_va=np.inf
    best_state=None
    patience=10000
    wait=0
    total_epochs=max(1,int(iter_max))
    bar_width=30
    print(f"training on {device}; train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}",flush=True)
    for epoch in range(iter_max):
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
    tr_mse=float(np.mean((pt_i-inv_scale(t_tr,t_scaler))**2))
    va_mse=float(np.mean((pv_i-inv_scale(t_va,t_scaler))**2))
    te_mse=float(np.mean((pe_i-inv_scale(t_te,t_scaler))**2))
    y_flat=pe_i.reshape(-1)
    t_flat=inv_scale(t_te,t_scaler).reshape(-1)
    reg=np.corrcoef(y_flat,t_flat)[0,1]
    base=os.path.dirname(__file__)
    out_dir=os.path.join(base,"outputs")
    try:
        os.makedirs(out_dir,exist_ok=True)
    except Exception:
        pass
    model_path=os.path.join(out_dir,"model.pt")
    torch.save({"state_dict":net.state_dict(),"x_scaler":{"mask":x_scaler.mask,"xmin":x_scaler.xmin,"scale":x_scaler.scale},"t_scaler":{"mask":t_scaler.mask,"xmin":t_scaler.xmin,"scale":t_scaler.scale}}, model_path)
    return {"train_mse":tr_mse,"val_mse":va_mse,"test_mse":te_mse,"test_reg":float(reg),"model_path":model_path}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=str,default="")
    ap.add_argument("--mat",type=str,default="")
    ap.add_argument("--epochs",type=int,default=5000)
    args=ap.parse_args()
    base=os.path.dirname(__file__)
    inputs=os.path.join(base,"inputs")
    try:
        os.makedirs(inputs,exist_ok=True)
    except Exception:
        pass
    if args.data:
        res=train_from_data(args.data,iter_max=args.epochs)
    else:
        mat_path=args.mat
        if not mat_path or not os.path.exists(mat_path):
            cand=os.path.join(inputs,"run_20260307_110701_SmartG_OutputXY_For_NNtrain.mat")
            mat_path=cand
        res=train_from_mat(mat_path,iter_max=args.epochs)
    print(res)
if __name__=="__main__":
    main()
