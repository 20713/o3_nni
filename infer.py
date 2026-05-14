import numpy as np
import torch
import os
from .net import Net


def load_model(model_path=None,device=None):
    if model_path is None:
        raise ValueError("model_path is required; please provide a valid model path.")
    if device is None:
        device="cuda" if torch.cuda.is_available() else "cpu"
    try:
        obj=torch.load(model_path,map_location=device,weights_only=False)
        print(f"loaded model from {model_path}")
    except TypeError:
        obj=torch.load(model_path,map_location=device)
    x_mask=obj["x_scaler"]["mask"]  # 输入特征保留掩码（训练阶段去除常量列后留下的列）
    x_xmin=obj["x_scaler"]["xmin"]
    x_scale=obj["x_scaler"]["scale"]
    t_mask=obj["t_scaler"]["mask"]
    t_xmin=obj["t_scaler"]["xmin"]
    t_scale=obj["t_scaler"]["scale"]
    pca_config=obj.get("pca_config")
    in_dim=int(x_mask.sum())  # 网络输入维度=保留特征数
    net=Net(in_dim).to(device)
    net.load_state_dict(obj["state_dict"])
    net.eval()
    return {"net":net,"device":device,"x_mask":x_mask,"x_xmin":x_xmin,"x_scale":x_scale,"t_mask":t_mask,"t_xmin":t_xmin,"t_scale":t_scale,"pca_config":pca_config}
    
# def predict(x,ctx):
#     x2=x[:,ctx["x_mask"]]  # 推理时仅选择训练阶段使用的特征列
#     y=(x2-ctx["x_xmin"])*ctx["x_scale"]-1.0
#     print(np.nanmin(y), np.nanmax(y))
#     print(np.mean((y < -1) | (y > 1)))
#     with torch.no_grad():
#         yp=ctx["net"](torch.from_numpy(y).float().to(ctx["device"])).cpu().numpy()
#     o=(yp+1.0)/ctx["t_scale"]+ctx["t_xmin"]
#     return o

def predict(x, ctx):
    x2 = x[:, ctx["x_mask"]]
    y = (x2 - ctx["x_xmin"]) * ctx["x_scale"] - 1.0

    bad = (y < -1) | (y > 1)
    print("scaled min/max:", np.nanmin(y), np.nanmax(y))
    print("out-of-range ratio:", np.mean(bad))

    bad_idx = np.where(bad[0])[0]
    print("bad feature idx:", bad_idx)
    print("bad feature values:", y[0, bad_idx])

    with torch.no_grad():
        yp = ctx["net"](torch.from_numpy(y).float().to(ctx["device"])).cpu().numpy()

    o = (yp + 1.0) / ctx["t_scale"] + ctx["t_xmin"]
    return o
