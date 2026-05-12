import numpy as np

data = np.load(r"D:\binhou_iDATA\binhou_iDATA_codes\Mydata_NNItrain\Mydata_NNItrain\o3_nni\TRAIN_datasets\trainset_run_20260307_110701_SmartG_OutputXY_For_NNtrain_nz61_in41_nch7_wav745-675-600-525-351-315-300_ns21841_20260511_141516.npz")

print("Keys in npz:\n")

for key in data.files:
    arr = data[key]

    print(f"{key}:")
    print(f"  shape : {arr.shape}")
    print(f"  dtype : {arr.dtype}")
    print(f"  ndim  : {arr.ndim}")

    print()

print(data["wav_chan"])
print(type(data["wav_chan"]))
print(data["z"])
