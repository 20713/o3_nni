import argparse
import numpy as np
import os

from .data_prepare import prepare_pca_features_and_io
from .infer import load_model, predict


def run(mat_path,model_path=None):
    base=os.path.dirname(__file__)
    inputs=os.path.join(base,"inputs")
    try:
        os.makedirs(inputs,exist_ok=True)
    except Exception:
        pass
    if not os.path.exists(mat_path) or not mat_path:
        mat_path=os.path.join(inputs,"run_20260307_110701_SmartG_OutputXY_For_NNtrain.mat")
    x,t,l,aux=prepare_pca_features_and_io(mat_path)
    assert x.ndim==2 and t.ndim==2
    assert x.shape[0]==t.shape[0]
    assert x.shape[1]==99
    assert t.shape[1]==61
    assert l.ndim==1
    assert aux["Uoz"].shape[0]==61 and aux["Uoz"].shape[1]==61

    ctx=load_model(model_path=model_path)
    y=predict(x[:4],ctx)
    assert y.shape==(4,61)
    assert np.all(np.isfinite(y))
    return {"x_shape":x.shape,"t_shape":t.shape,"y_shape":y.shape}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mat",type=str,required=True)
    ap.add_argument("--model",type=str,default="")
    args=ap.parse_args()
    res=run(args.mat,model_path=(args.model if args.model else None))
    print(res)


if __name__=="__main__":
    main()
