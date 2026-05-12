import argparse
import os
from datetime import datetime
import numpy as np
from scipy.io import loadmat


def load_training_mat(mat_path):
    """
    Load a training .mat file.

    Required variables:
    - radiance: (Nsample, Nz_raw, Nchan_raw)
    - LOKI: (Nsample, Nchan_raw), per-channel validity mask
    - vmr: (Nsample, Nz_raw)
    - SZA/SAA: (Nsample,)
    """
    if not mat_path:
        raise ValueError("mat_path is required; please pass a valid .mat path")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"mat_path {mat_path} not found")
    d=loadmat(mat_path,squeeze_me=True)
    print("load mat from",mat_path)
    wav=d.get("WAV")
    radiance=d.get("radiance")
    LOKI=d.get("LOKI")
    vmr=d.get("vmr")
    SZA=d.get("SZA")
    SAA=d.get("SAA")
    return wav,radiance,LOKI,vmr,SZA,SAA,mat_path   


def prepare_pca_features_and_io(mat_path,inorm=41,chan=(1,2,3,4,5,6,7),npcChan=(8,9,9,14,18,19,20),nz=61,out_dir=None):

    wav,radiance,LOKI,vmr,SZA,SAA,mat_used_path=load_training_mat(mat_path)
    nz=min(nz,radiance.shape[1])
    nodes=np.arange(nz)
    wav_arr=np.asarray(wav).squeeze()
    if wav_arr.ndim!=1:
        raise ValueError(f"WAV must be 1D after squeeze, got shape={wav_arr.shape}")
    chan=tuple(int(c) for c in chan)
    if len(chan)==0:
        raise ValueError("chan is empty")
    if len(set(chan))!=len(chan):
        raise ValueError(f"chan has duplicate ids: {chan}")
    all_chan=tuple(range(1,int(wav_arr.shape[0])+1))
    bad=[c for c in chan if c not in all_chan]
    if bad:
        raise ValueError(f"chan contains unsupported ids {bad}; allowed: {all_chan}")
    npc_base=tuple(int(v) for v in npcChan)
    if len(npc_base)==len(chan):
        npcChan=npc_base
    else:
        if int(max(chan))>len(npc_base):
            raise ValueError(f"npcChan length={len(npc_base)} is not enough for selected chan={chan}")
        npcChan=tuple(npc_base[c-1] for c in chan)
    chan_idx=np.array(chan,dtype=int)-1
    if int(chan_idx.max())>=wav_arr.shape[0] or int(chan_idx.min())<0:
        raise ValueError(f"chan indices out of WAV range: chan={chan}, wav_len={wav_arr.shape[0]}")
    wav_chan=wav_arr[chan_idx] #排序？？？
    valid_sample_idx=np.where(np.all(LOKI[:,chan_idx],axis=1))[0]
    n_valid=valid_sample_idx.size
    log_radiance=np.log(radiance[valid_sample_idx][:,nodes][:,:,chan_idx])
    normalized_log_radiance=log_radiance-log_radiance[:,inorm-1,:][:,None,:]

    YMoz=np.zeros((nz,len(chan)))
    dYMoz=np.zeros((nz,len(chan)))
    YMoz025=np.zeros((nz,len(chan)))
    YMoz975=np.zeros((nz,len(chan)))
    Uoz=np.zeros((nz,nz,len(chan)))

    Ai=[]
    for idx,ich in enumerate(chan):
        npcs=npcChan[idx]
        Ydum=normalized_log_radiance[:,:,idx]
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
        Ai.append(A[:,:npcs])

    Ai=np.concatenate(Ai,axis=1)
    inp=np.column_stack([Ai,SZA[valid_sample_idx].reshape(-1,1),SAA[valid_sample_idx].reshape(-1,1)])
    out=vmr[valid_sample_idx,:nz]

    base=os.path.dirname(__file__)
    if out_dir is None or (isinstance(out_dir,str) and (not out_dir.strip())):
        out_dir=os.path.join(base,"TRAIN_datasets")
    try:
        os.makedirs(out_dir,exist_ok=True)
    except Exception:
        pass

    mat_tag=os.path.splitext(os.path.basename(mat_used_path))[0]
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_tokens=[f"{float(w):.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p") for w in wav_chan]
    wav_tag="-".join(wav_tokens)
    dataset_name=f"trainset_{mat_tag}_nz{nz}_in{inorm}_nch{len(chan)}_wav{wav_tag}_ns{n_valid}_{ts}.npz"
    dataset_path=os.path.join(out_dir,dataset_name)
    created_at=datetime.now().isoformat(timespec="seconds")
    payload=dict(
        x=inp,
        t=out,
        valid_sample_idx=valid_sample_idx,
        valid_samples_numbers=n_valid,
        z=np.array(nodes),
        inorm=inorm,
        npcChan=np.array(npcChan),
        chan=np.array(chan),
        wav_chan=np.array(wav_chan),
        Uoz=Uoz,
        YMoz=YMoz,
        dYMoz=dYMoz,
        YMoz025=YMoz025,
        YMoz975=YMoz975,
        mat_path=str(mat_used_path),
        created_at=str(created_at),
    )
    np.savez_compressed(dataset_path,**payload)
    return inp,out,n_valid,dataset_path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mat",type=str,default="./o3_nni/RTM_datasets/run_20260307_110701_SmartG_OutputXY_For_NNtrain.mat")
    ap.add_argument("--out_dir",type=str,default="./o3_nni/TRAIN_datasets",help="output directory")
    ap.add_argument("--chan",type=str,default="1,2,3,4,5,6,7",help="discrete channel ids, e.g. 1,2,4,7,8")
    ap.add_argument("--inorm",type=int,default=41)
    ap.add_argument("--nz",type=int,default=61,help="For prepare trainset,number of vertical levels(X,Y same) to use (keep first nz layers)")
    args=ap.parse_args()

    npcChan=(8,9,9,14,18,19,20,21) #接口
    try:
        chan=tuple(int(x.strip()) for x in str(args.chan).split(",") if x.strip())
    except ValueError as e:
        raise ValueError(f"invalid --chan value: {args.chan}") from e
    if len(chan)==0:
        raise ValueError("--chan is empty; provide at least one channel id")
    if len(set(chan))!=len(chan):
        raise ValueError(f"--chan has duplicate ids: {chan}")
    if min(chan)<1 or max(chan)>len(npcChan):
        raise ValueError(f"--chan must be in [1,{len(npcChan)}] for current npcChan base, got {chan}")
    selected_npcChan=tuple(npcChan[c-1] for c in chan)
    print(f"use selected channels {chan}")
    print(f"use selected npcChan {selected_npcChan}")
    x,t,n_valid,out_traindataset_path=prepare_pca_features_and_io(args.mat,out_dir=args.out_dir,inorm=args.inorm,nz=args.nz,chan=chan,npcChan=npcChan)
    print({
        "x_shape":tuple(x.shape),
        "t_shape":tuple(t.shape),
        "n_valid":int(n_valid),
        "dataset_path":out_traindataset_path
    })

if __name__=="__main__":
    main()
