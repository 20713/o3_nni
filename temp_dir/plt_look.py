import torch

path = r'D:\binhou_iDATA\binhou_iDATA_codes\Mydata_NNItrain\Mydata_NNItrain\o3_nni\TRAIN_outputs\20260515_120935\model.pth'

ckpt = torch.load(path, map_location='cpu')

print(type(ckpt))

if isinstance(ckpt, dict):
    print(ckpt.keys())
print(ckpt["pca_config"].keys())
print(ckpt["pca_config"]["radiance_grid"])
print(ckpt["pca_config"]["vmr_grid"])