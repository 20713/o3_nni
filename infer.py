import numpy as np
import torch
from torch import nn
import os
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
def load_model(model_path=None,device=None):
    if device is None:
        device="cuda" if torch.cuda.is_available() else "cpu"
    try:
        if model_path is None:
            base=os.path.dirname(__file__)
            model_path=os.path.join(base,"outputs","model.pt")
            print(f"loading model from {model_path}")
        obj=torch.load(model_path,map_location=device,weights_only=False)
    except TypeError:
        obj=torch.load(model_path,map_location=device)
    x_mask=obj["x_scaler"]["mask"]  # 输入特征保留掩码（训练阶段去除常量列后留下的列）
    x_xmin=obj["x_scaler"]["xmin"]
    x_scale=obj["x_scaler"]["scale"]
    t_mask=obj["t_scaler"]["mask"]
    t_xmin=obj["t_scaler"]["xmin"]
    t_scale=obj["t_scaler"]["scale"]
    in_dim=int(x_mask.sum())  # 网络输入维度=保留特征数
    net=Net(in_dim).to(device)
    net.load_state_dict(obj["state_dict"])
    net.eval()
    return {"net":net,"device":device,"x_mask":x_mask,"x_xmin":x_xmin,"x_scale":x_scale,"t_mask":t_mask,"t_xmin":t_xmin,"t_scale":t_scale}
def predict(x,ctx):
    x2=x[:,ctx["x_mask"]]  # 推理时仅选择训练阶段使用的特征列
    y=(x2-ctx["x_xmin"])*ctx["x_scale"]-1.0
    with torch.no_grad():
        yp=ctx["net"](torch.from_numpy(y).float().to(ctx["device"])).cpu().numpy()
    o=(yp+1.0)/ctx["t_scale"]+ctx["t_xmin"]
    return o
