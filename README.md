# Python 项目说明（NNI O3 + OMPS 验证）

本目录是原 MATLAB 项目的 Python 等价实现（训练 + 推断 + OMPS/Bremen 对照验证）。  
推荐流程：先用数据准备脚本生成“训练数据集 npz”（包含 x/t 与 PCA 工件），再用训练脚本加载该 npz 训练模型。

## 目录结构

- `o3_nni/`
  - `RTM_datasets/`：默认训练 `.mat` 路径所在目录（可自定义）
  - `TRAIN_datasets/`：数据准备脚本默认输出目录（可用 `--out_dir` 指定）
  - `TRAIN_outputs/`：训练输出目录（默认），按时间戳保存 `model.pt`
  - `inputs/`：保留的输入示例目录（部分脚本仍可使用）
  - `outputs/`：评估与验证输出目录（对比图、PDF 等）
  - `data_prepare.py`：从 `.mat` 生成训练数据集 npz（包含 x/t 与 PCA 工件）
  - `model_train.py`：两层 MLP 训练，保存 `model.pt`
  - `infer.py`：加载模型并推断
  - `evaluate_plots.py`：基于训练数据做整体评估出图
  - `validate_omps.py`：OMPS L1 + Bremen L2 对照验证（MATLAB Engine：`gridfit` + `interp2('makima')`）
  - `gridfit.m`：MATLAB `gridfit` 实现

## 数据准备（生成训练数据集 npz）

不传参数时，默认从 `./o3_nni/RTM_datasets/run_20260307_110701_SmartG_OutputXY_For_NNtrain.mat` 读取，并将输出写入 `./o3_nni/TRAIN_datasets`。

```bash
python -m o3_nni.data_prepare \
  --mat .\o3_nni\RTM_datasets\run_20260307_110701_SmartG_OutputXY_For_NNtrain.mat \
  --out_dir .\o3_nni\TRAIN_datasets \
  --nc 7 \
  --inorm 41 \
  --nz 61 \
  --ns 25600
```

关键信息：
- `--nc`：使用前 nc 个通道（与 `npcChan` 一一对应），默认 `7`
- `--ns`：使用的样本数（取前 ns 个样本），默认 `25600`
- `--inorm`：归一化参考层索引（1-based），默认 `41`（约 40 km）
- `--nz`：使用的高度层数（0..nz-1），默认 `61`
- 输出文件命名：`trainset_{mat_tag}_nz{nz}_in{inorm}_nch{nc}_ns{ns}_{timestamp}.npz`
- 输出 npz 同时包含训练集 `x/t` 与 PCA 工件（`Uoz/YMoz/npcChan/chan/inorm/z` 等）

## 训练

推荐使用数据集 npz 训练：

```bash
python -m o3_nni.model_train --data_path .\o3_nni\TRAIN_datasets\trainset_...npz --epochs 2000
```

输出：
- `o3_nni/TRAIN_outputs/{timestamp}/model.pt`

## 训练集评估出图

```bash
python -m o3_nni.evaluate_plots --mat .\o3_nni\RTM_datasets\run_20260307_110701_SmartG_OutputXY_For_NNtrain.mat --model .\o3_nni\TRAIN_outputs\{timestamp}\model.pt --out nni_py
```

输出：
- `o3_nni/outputs/nni_py_mean_std.png`
- `o3_nni/outputs/nni_py_dsz.png`
- `o3_nni/outputs/nni_py_signed_percentiles.png`

## OMPS 对照验证（导出 PDF）

该步骤在 Python 内启动 MATLAB Engine，调用 `gridfit.m` 与 `interp2(...,'makima')` 完成辐射场插值与补洞。

```bash
python -m o3_nni.validate_omps \
  --omps .\o3_nni\inputs\OMPS-NPP_LP-L1G-EV_v2.6_2016m0301t210605_o22509_2022m1005t174736.h5 \
  --bremen .\o3_nni\inputs\ESACCI-OZONE-L2-LP-OMPS_LP_SUOMI_NPP-IUP_UBR_V3_3NLC_UBR_HARMOZ_ALT-201603-fv0005.nc \
  --ozaux .\o3_nni\TRAIN_datasets\trainset_...npz \
  --model .\o3_nni\TRAIN_outputs\{timestamp}\model.pt \
  --no-show --smooth 10 --save-pdf .\o3_nni\outputs\compare.pdf
```

说明：
- `--ozaux` 可直接指定上一步生成的训练数据集 npz，因为其中已包含 `inorm/npcChan/Uoz/YMoz` 等工件
- 每个 iT 输出一页 PDF：左侧 O3 剖面（BREMEN vs ONNI），右侧为通道辐射的重构曲线与观测散点

## 运行位置与依赖

- 在包含 `o3_nni/` 的上一级目录运行模块命令，例如 `python -m o3_nni.data_prepare ...`
- 需要 Python 与相关依赖（`numpy/scipy/torch/matplotlib/h5py/netCDF4` 等）
- 运行 OMPS 验证需要 MATLAB Engine（用于 `gridfit` 与 `interp2('makima')`）
