# Python 项目说明（NNI O3 + OMPS 验证）

本目录是原 MATLAB 项目的 Python 等价实现（训练 + 推断 + OMPS/Bremen 对照验证）。  
推荐流程：先用数据准备脚本生成“训练数据集 npz”（包含 x/t 与 PCA 工件），再用训练脚本加载该 npz 训练模型。

## 目录结构

- `o3_nni/`
  - `RTM_datasets/`：SmartG模拟生产的大规模 `.mat` 格式数据集所在目录
  - `TRAIN_datasets/`：数据准备脚本`data_prepare.py` 默认输出目录（可用 `--out_dir` 指定）
  - `TRAIN_outputs/`：模型训练脚本`model_train.py` 默认输出根目录，按时间戳保存 `model.pth` 与评估图片（可用 `--out_dir` 指定）
  - `sample_files/`：保留的输入示例目录（部分脚本仍可使用）
  - `validate_omps_output/`：omps卫星数据验证脚本`validate_omps.py` 默认输出根目录，按时间戳建子目录保存所反演的廓线对比PDF（可用 `--out_dir` 指定）
  - `data_prepare.py`：从 `.mat` 生成训练数据集 npz（包含 x/t 与 PCA 工件），默认输出目录为 `./o3_nni/TRAIN_datasets`
  - `model_train.py`：两层 MLP 训练，保存 `model.pth` 与评估图片，默认输出目录为 `./o3_nni/TRAIN_outputs`
  - `validate_omps.py`：OMPS L1 + Bremen L2 对照验证（MATLAB Engine：`gridfit` + `interp2('makima')`），默认输出目录为 `./o3_nni/validate_omps_output`
  - `net.py`：统一的网络结构定义（训练/推断/验证共用）
  - `scalers.py`：归一化/反归一化相关（`MapMinMax`）
  - `eval_plots.py`：评估绘图公共函数（训练/验证可复用）
  - `infer.py`：加载模型并推断（`load_model` 需要显式提供 `model_path`）
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
- 输出文件命名：`trainset_{mat_tag}_nz{nz}_in{inorm}_nch{len(chan)}_pc{pc_sum}_{pc_per_chan}_wav{wav_tag}_ns{n_valid}_{timestamp}.npz`
  - `pc_sum`：所选通道对应的 `npcChan` 总和
  - `pc_per_chan`：所选通道对应的 `npcChan` 列表（用 `-` 连接，例如 `5-6-6-10-10-10-10`）
- 输出 npz 同时包含训练集 `x/t` 与 PCA 工件（`Uoz/YMoz/npcChan/chan/wav_chan/inorm/z` 等）

## 训练

推荐使用数据集 npz 训练：

```bash
python -m o3_nni.model_train --data_path ./o3_nni/TRAIN_datasets/trainset_...npz --epochs 2000
```

可选：启用学习率调度（当验证集损失进入平台期时自动降学习率）：

```bash
python -m o3_nni.model_train \
  --data_path ./o3_nni/TRAIN_datasets/trainset_...npz \
  --epochs 2000 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --lr_scheduler plateau \
  --lr_factor 0.5 \
  --lr_patience 50 \
  --min_lr 1e-6
```

输出（同一时间戳目录）：
- `o3_nni/TRAIN_outputs/{timestamp}/model.pth`
- `o3_nni/TRAIN_outputs/{timestamp}/nni_py_mean_std.png`
- `o3_nni/TRAIN_outputs/{timestamp}/nni_py_dsz.png`
- `o3_nni/TRAIN_outputs/{timestamp}/nni_py_signed_percentiles.png`
- `o3_nni/TRAIN_outputs/{timestamp}/nni_py_bias_mae_rmse.png`

说明：
- `model_train.py` 在训练结束后会自动生成上述评估图（前缀固定为 `nni_py`，无需额外参数）。
- `--lr_scheduler plateau` 使用验证集 `va_loss` 触发降学习率；`lr_patience` 与训练早停（early stop）的 patience 相互独立。
- 训练保存的模型 checkpoint 会携带 `pca_config`（例如 `inorm/npcChan/chan/wav_chan/z/Uoz/YMoz` 等），供 OMPS 验证阶段直接使用。

## OMPS 对照验证（导出 PDF）

该步骤在 Python 内启动 MATLAB Engine，调用 `gridfit.m` 与 `interp2(...,'makima')` 完成辐射场插值与补洞。

```bash
python -m o3_nni.validate_omps \
  --omps ./o3_nni/sample_files/OMPS-NPP_LP-L1G-EV_v2.6_2016m0301t224735_o22510_2022m1005t174807.h5 \
  --bremen ./o3_nni/sample_files/ESACCI-OZONE-L2-LP-OMPS_LP_SUOMI_NPP-IUP_UBR_V3_3NLC_UBR_HARMOZ_ALT-201603-fv0005.nc \
  --model ./o3_nni/sample_files/model.pth \
  --no-show --smooth 10 --out_dir ./o3_nni/validate_omps_output
```

说明：
- `validate_omps.py` 默认从模型 checkpoint 中读取 `pca_config`（不再单独读取额外的 npz 配置文件）。
- `--model` 指向训练脚本生成的模型文件（例如 `o3_nni/TRAIN_outputs/{timestamp}/model.pth`）。
- `--out_dir` 不填时默认写入 `./o3_nni/validate_omps_output/{timestamp}/`
- 每个 iT 输出一页 PDF：左侧 O3 剖面（BREMEN vs ONNI），右侧为通道辐射的重构曲线与观测散点
- 额外输出一组汇总评估图（统一高度网格为 15–45 km）：
  - `omps_eval_15_45km_mean_std.png`
  - `omps_eval_15_45km_dsz.png`
  - `omps_eval_15_45km_signed_percentiles.png`
  - `omps_eval_15_45km_bias_mae_rmse.png`

## 运行位置与依赖

- 在包含 `o3_nni/` 的上一级目录运行模块命令，例如 `python -m o3_nni.data_prepare ...`
- 需要 Python 与相关依赖（`numpy/scipy/torch/matplotlib/h5py/netCDF4/pandas` 等）
- 运行 OMPS 验证需要 MATLAB Engine（用于 `gridfit` 与 `interp2('makima')`）

## 环境复现

本项目在 `o3_nni/` 下提供了当前可运行环境的快照文件，便于在新机器/新环境中复现：

- `o3_nni/environment.yml`：Conda 环境导出（包含 Python/torch/cuda 以及 pip 依赖段）
- `o3_nni/requirements_pip.txt`：`pip freeze` 结果（依赖快照备份）

推荐用 Conda 复现（会创建 `name:` 指定的环境名，例如 `ONNI`）：

```bash
conda env create -f ./o3_nni/environment.yml
conda activate ONNI
```

说明：
- `requirements_pip.txt` 通常不需要单独安装（因为 `environment.yml` 已包含 `pip:` 段）；它主要用于对比/排查依赖差异
- OMPS 验证还需要额外安装 MATLAB Engine（不包含在 `environment.yml`/`pip freeze` 中）
