# Python 项目说明（NNI O3 + OMPS 验证）

本目录是原 MATLAB 项目的 Python 等价实现（训练 + 推断 + OMPS/Bremen 对照验证）。  
推荐流程：先用数据准备脚本生成“训练数据集 npz”（包含 x/t 与 PCA 工件），再用训练脚本加载该 npz 训练模型。

## 目录结构

- `o3_nni/`
  - `RTM_datasets/`：默认训练 `.mat` 路径所在目录（可自定义）
  - `TRAIN_datasets/`：数据准备脚本默认输出目录（可用 `--out_dir` 指定）
  - `TRAIN_outputs/`：训练输出目录（默认），按时间戳保存 `model.pt` 与评估图片
  - `sample_files/`：保留的输入示例目录（部分脚本仍可使用）
  - `validate_omps_output/`：`validate_omps.py` 默认输出根目录（按时间戳建子目录保存 PDF）
  - `data_prepare.py`：从 `.mat` 生成训练数据集 npz（包含 x/t 与 PCA 工件）
  - `model_train.py`：两层 MLP 训练，保存 `model.pt`
  - `net.py`：统一的网络结构定义（训练/推断/验证共用）
  - `scalers.py`：归一化/反归一化相关（`MapMinMax`）
  - `eval_plots.py`：评估绘图公共函数（训练/验证可复用）
  - `infer.py`：加载模型并推断（`load_model` 需要显式提供 `model_path`）
  - `validate_omps.py`：OMPS L1 + Bremen L2 对照验证（MATLAB Engine：`gridfit` + `interp2('makima')`）
  - `gridfit.m`：MATLAB `gridfit` 实现

## 数据准备（生成训练数据集 npz）

不传参数时，默认从 `./o3_nni/RTM_datasets/run_20260307_110701_SmartG_OutputXY_For_NNtrain.mat` 读取，并将输出写入 `./o3_nni/TRAIN_datasets`。

```bash
python -m o3_nni.data_prepare \
  --mat ./o3_nni/RTM_datasets/run_20260307_110701_SmartG_OutputXY_For_NNtrain.mat \
  --out_dir ./o3_nni/TRAIN_datasets \
  --chan 1,2,3,4,5,6,7 \
  --inorm 41 \
  --nz 61
```

Note: data preparation uses all samples that are valid for the selected channels.

关键信息：
- `--chan`：离散通道选择（1-based），例如 `1,2,4,7,8`
- `--inorm`：归一化参考层索引（1-based），默认 `41`（约 40 km）
- `--nz`：使用的高度层数（0..nz-1），默认 `61`
- 输出文件命名：`trainset_{mat_tag}_nz{nz}_in{inorm}_nch{len(chan)}_wav{wav_tag}_ns{n_valid}_{timestamp}.npz`
- 输出 npz 同时包含训练集 `x/t` 与 PCA 工件（`Uoz/YMoz/npcChan/chan/wav_chan/inorm/z` 等）

## 训练

推荐使用数据集 npz 训练：

```bash
python -m o3_nni.model_train --data_path ./o3_nni/TRAIN_datasets/trainset_...npz --epochs 2000
```

输出（同一时间戳目录）：
- `o3_nni/TRAIN_outputs/{timestamp}/model.pt`
- `o3_nni/TRAIN_outputs/{timestamp}/nni_py_mean_std.png`
- `o3_nni/TRAIN_outputs/{timestamp}/nni_py_dsz.png`
- `o3_nni/TRAIN_outputs/{timestamp}/nni_py_signed_percentiles.png`
- `o3_nni/TRAIN_outputs/{timestamp}/nni_py_bias_mae_rmse.png`

说明：
- `model_train.py` 在训练结束后会自动生成上述评估图（前缀固定为 `nni_py`，无需额外参数）。

## OMPS 对照验证（导出 PDF）

该步骤在 Python 内启动 MATLAB Engine，调用 `gridfit.m` 与 `interp2(...,'makima')` 完成辐射场插值与补洞。

```bash
python -m o3_nni.validate_omps \
  --omps ./o3_nni/sample_files/OMPS-NPP_LP-L1G-EV_v2.6_2016m0301t224735_o22510_2022m1005t174807.h5 \
  --bremen ./o3_nni/sample_files/ESACCI-OZONE-L2-LP-OMPS_LP_SUOMI_NPP-IUP_UBR_V3_3NLC_UBR_HARMOZ_ALT-201603-fv0005.nc \
  --ozaux ./o3_nni/sample_files/ozAux3.npz \
  --model ./o3_nni/sample_files/model.pt \
  --no-show --smooth 10 --out_dir ./o3_nni/validate_omps_output
```

说明：
- `--ozaux` 可直接指定上一步生成的训练数据集 npz，因为其中已包含 `inorm/npcChan/Uoz/YMoz` 等工件
- `--ozaux` 不填时默认使用 `./o3_nni/sample_files/ozAux3.npz`
- `--model` 不填时默认使用 `./o3_nni/sample_files/model.pt`
- `--out_dir` 不填时默认写入 `./o3_nni/validate_omps_output/{timestamp}/`
- 每个 iT 输出一页 PDF：左侧 O3 剖面（BREMEN vs ONNI），右侧为通道辐射的重构曲线与观测散点

## 运行位置与依赖

- 在包含 `o3_nni/` 的上一级目录运行模块命令，例如 `python -m o3_nni.data_prepare ...`
- 需要 Python 与相关依赖（`numpy/scipy/torch/matplotlib/h5py/netCDF4/pandas` 等）
- 运行 OMPS 验证需要 MATLAB Engine（用于 `gridfit` 与 `interp2('makima')`）
