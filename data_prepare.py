import numpy as np
from scipy.io import loadmat
import os
from datetime import datetime
import argparse
def load_training_mat(mat_path):
    """
    加载训练用 .mat 文件。
    需要包含变量：
    - radiance: (Nsample, Nz_raw, Nchan_raw) 或等价维度排布（本脚本按 radiance[:, nodes, il-1] 取通道）
    - LOKI: (Nsample, Nchan_raw) 逻辑有效性掩码（每通道是否有效）
    - vmr: (Nsample, Nz_raw) 目标臭氧剖面（至少包含前 nz 层）
    - SZA/SAA: (Nsample,) 太阳天顶角/方位角
    """
    if not mat_path:
        raise ValueError("mat_path is required; please pass a valid .mat path")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"mat_path {mat_path} not found")
    d=loadmat(mat_path,squeeze_me=True)
    print("load mat from",mat_path)
    radiance=d.get("radiance")
    LOKI=d.get("LOKI")
    vmr=d.get("vmr")
    SZA=d.get("SZA")
    SAA=d.get("SAA")
    return radiance,LOKI,vmr,SZA,SAA,mat_path
def build_big_xy(radiance,LOKI,il,nodes,n,iAll):
    """
    构建单个通道的 log(radiance) 训练矩阵。
    - il: 通道编号（1-based）
    - nodes: 选取的高度索引（0-based）
    - n: 取前 n 个样本（iAll=1），或在有效样本中取前 n 个（iAll=0）
    返回：
    - Y: (n, len(nodes)) 的 log(radiance)
    - LOKI: 原始有效性矩阵（用于后续筛选所有通道都有效的样本）
    """
    if iAll==0:
        lok1=np.where(LOKI[:,il-1])[0]
        if n>len(lok1):
            n=len(lok1)
        loks=lok1[:n]
    else:
        n=min(n,radiance.shape[0])
        loks=np.arange(n)
    Y=np.log(radiance[loks][:,nodes,il-1])
    return Y,LOKI
def prepare_pca_features_and_io(mat_path,inorm=41,chan=(1,2,3,4,5,6,7),npcChan=(8,9,9,14,18,19,20),nz=61,ns=25600,out_dir=None):
    """
    从 .mat 生成训练用特征 x 与标签 t，并同时保存 PCA 工件到 npz。
    参数含义：
    - inorm: 归一化参考高度索引（1-based）。以该高度处的 log(radiance) 为基线，对整条廓线做差。
    - chan: 使用的通道编号列表（1-based）。
    - npcChan: 每个通道保留的 PCA 主成分数，与 chan 一一对应（长度必须一致）。
    - nz: 使用的高度层数（取 radiance 的前 nz 层）。
    - ns: 样本数（取前 ns 个样本进入 PCA/训练特征构建）。
    - out_dir: 输出目录；为空时默认保存到脚本目录下 TRAIN_datasets。
    输出：
    - inp: 训练输入特征 x，shape=(Nvalid, sum(npcChan)+2)，其中 +2 对应 SZA/SAA
    - out: 训练标签 t，shape=(Nvalid, nz)
    - l: 有效样本索引（满足所有通道均有效）
    - meta: 关键工件与输出路径信息
    """
    radiance,LOKI,vmr,SZA,SAA,mat_used_path=load_training_mat(mat_path)
    nz=min(nz,radiance.shape[1])
    YY=np.zeros((ns,nz,len(chan)))
    YYY=np.zeros((ns,nz,len(chan)))
    YMoz=np.zeros((nz,len(chan)))
    dYMoz=np.zeros((nz,len(chan)))
    YMoz025=np.zeros((nz,len(chan)))
    YMoz975=np.zeros((nz,len(chan)))
    Uoz=np.zeros((nz,nz,len(chan)))
    nodes=np.arange(nz)
    for idx,ich in enumerate(chan):
        Y,LOK=build_big_xy(radiance,LOKI,ich,nodes,ns,1)
        YY[:,:,idx]=Y
        # 归一化：以 inorm（1-based）处的 log(radiance) 作为基线，对整条廓线做差
        # 例如 nz=61 且 z=0..60 km 时，inorm=41 对应 40 km
        YYY[:,:,idx]=Y-(Y[:,inorm-1][:,None])
    # 有效样本：要求所有通道都有效
    l=np.where(np.all(LOK,axis=1))[0]
    Ai=[]
    for idx,ich in enumerate(chan):
        npcs=npcChan[idx]
        Ydum=YYY[l,:,idx]
        # 每个通道独立 PCA：对 (Nvalid, nz) 的廓线做去均值并求协方差，再做 SVD 得到基 U
        Ym=Ydum.mean(axis=0)
        dY=Ydum-Ym[None,:]
        C=dY.T@dY
        U=np.linalg.svd(C,full_matrices=False)[2].T
        A=dY@U
        YMoz[:,idx]=Ym
        dYMoz[:,idx]=Ydum.std(axis=0)
        Yperc=np.percentile(Ydum,[2.5,97.5],axis=0)
        YMoz025[:,idx]=Yperc[0]
        YMoz975[:,idx]=Yperc[1]
        Uoz[:,:,idx]=U
        # 取该通道前 npcs 个 PCA 系数作为输入特征的一部分
        Ai.append(A[:,:npcs])
    # 拼接所有通道的 PCA 系数，并追加 SZA/SAA 两个几何特征
    Ai=np.concatenate(Ai,axis=1)
    inp=np.column_stack([Ai,SZA[l].reshape(-1,1),SAA[l].reshape(-1,1)])
    out=vmr[l,:nz]
    z=np.arange(nz)
    base=os.path.dirname(__file__)
    if out_dir is None or (isinstance(out_dir,str) and (not out_dir.strip())):
        out_dir=os.path.join(base,"TRAIN_datasets")
    try:
        os.makedirs(out_dir,exist_ok=True)
    except Exception:
        pass
    mat_tag=os.path.splitext(os.path.basename(mat_used_path))[0]
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    # 输出文件名编码关键配置：nz/inorm/通道数/样本数/时间戳，便于区分不同设置生成的数据集
    dataset_name=f"trainset_{mat_tag}_nz{nz}_in{inorm}_nch{len(chan)}_ns{ns}_{ts}.npz"
    dataset_path=os.path.join(out_dir,dataset_name)
    created_at=datetime.now().isoformat(timespec="seconds")
    payload=dict(
        # 训练集
        x=inp,
        t=out,
        l=l,
        # 工件（推理/复现需要）
        z=z,
        inorm=int(inorm),
        npcChan=np.array(npcChan),
        chan=np.array(chan),
        Uoz=Uoz,
        YMoz=YMoz,
        dYMoz=dYMoz,
        YMoz025=YMoz025,
        YMoz975=YMoz975,
        mat_path=str(mat_used_path),
        created_at=str(created_at),
    )
    np.savez_compressed(dataset_path,**payload)
    return inp,out,l,{"z":z,"inorm":inorm,"npcChan":np.array(npcChan),"Uoz":Uoz,"YMoz":YMoz,"out_traindataset_path":dataset_path}

def main():
    ap=argparse.ArgumentParser()
    # 默认路径以项目根目录为 cwd 时可直接使用；也可以传绝对路径
    ap.add_argument("--mat",type=str,default="./o3_nni/RTM_datasets/run_20260307_110701_SmartG_OutputXY_For_NNtrain.mat")
    ap.add_argument("--out_dir",type=str,default="./o3_nni/TRAIN_datasets",help="输出目录；不填则默认保存到 ./o3_nni/TRAIN_datasets")
    # --nc 是通道数（channel count），与 --ns（sample count）含义不同
    ap.add_argument("--nc",type=int,default=7,help="使用前 nc 个通道（默认全部通道=7）")
    ap.add_argument("--inorm",type=int,default=41)
    ap.add_argument("--nz",type=int,default=61)
    ap.add_argument("--ns",type=int,default=25600)
    args=ap.parse_args()
    # 通道与每通道 PCA 维数必须保持一一对应
    chan=(1,2,3,4,5,6,7,8)
    npcChan=(8,9,9,14,18,19,20,21)
    if args.nc<1 or args.nc>len(chan):
        raise ValueError(f"--nc must be in [1,{len(chan)}], got {args.nc}")
    chan=chan[:args.nc]
    print(f"use channels {chan}")
    print(f"use npcChan {npcChan}")
    npcChan=npcChan[:args.nc]
    x,t,l,meta=prepare_pca_features_and_io(args.mat,out_dir=args.out_dir,inorm=args.inorm,nz=args.nz,ns=args.ns,chan=chan,npcChan=npcChan)
    print({"x_shape":tuple(x.shape),"t_shape":tuple(t.shape),"n_valid":int(l.size),"dataset_path":meta.get("out_traindataset_path")})

if __name__=="__main__":
    main()
