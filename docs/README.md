# CT 肝脏肿瘤病灶分割 — 工作环境说明

本文档写给在本工作区执行任务的 Agent。目标任务是肝脏肿瘤病灶区 CT 图像分割。

## 数据与网络约束

数据受保密协议约束，**只存在于远程平台，不能下载到本地，也不能提交进 git**。`.gitignore` 中 `data/*`（仅保留 `data/.gitkeep` 占位）以及 `*.pt`、`*.pth`、`*.h5`、`*.nii` 类大产物同理：本地仓库里永远看不到真实数据，所有需要数据的操作都由用户在远程平台执行。

远程平台**无法从本地 SSH 连接，也没有 Jupyter**，只有一个 Linux 终端。本地与远程之间**只能通过 git 同步代码**。

因此 Agent 的工作方式是：在本地仓库改代码 → 提交并 push → 用户在远程 pull 后手动执行 → 把终端输出贴回本地。这意味着：

- 任何代码都必须能**非交互地一次性跑完**（`python train.py --fold 0` 这种），不要设计需要 Jupyter cell 或交互式调试的流程。
- 保证脚本可通过命令行参数与配置文件控制，不要在代码里写死本地绝对路径；路径一律相对仓库根目录或从参数读入。
- 远程执行前先做**小规模自检**（如 `--debug` 跑 1 个 case / 几个 iteration），并把关键信息（数据目录结构、case 数量、图像尺寸与 spacing、标签取值）打印成文本，方便用户直接贴回。
- 无法直接观测远程文件系统时，不要假设目录结构；先写探测脚本，再写训练代码。
- 每次需要远程执行，请**单独给出一段可直接粘贴到终端的命令**（优先 `python - <<'PY' ... PY` heredoc 或单行命令），不要要求用户在远程创建/编辑文件。

## 远程运行环境

conda 环境 `unet`（已激活），Python 3.11.16，解释器 `/home/phdauser004/.conda/envs/unet/bin/python`，系统为 Linux x86_64 (glibc 2.39)。

- GPU：1 × NVIDIA A100-PCIE-40GB（约 39.5 GiB 可用显存，sm_80，驱动 595.84）
- CPU：52 核
- torch 2.13.0+cu132，CUDA runtime 13.2，cuDNN 9.20.0，支持 bf16 与 TF32
- torchvision 0.28.0+cu132
- 医学影像与数据处理：monai 1.6.0、SimpleITK 2.5.6、nibabel 5.4.2、pydicom 3.0.2、scikit-image 0.26.0、scipy 1.17.1、scikit-learn 1.9.1、numpy 2.4.6、pandas 3.0.5
- 其他：matplotlib 3.11.2、pillow 12.3.0、tqdm 4.70.1、pyyaml 6.0.3、tensorboard 2.21.0、einops 0.8.2

**未安装**（不要依赖）：opencv / cv2、albumentations、torchaudio、h5py、cupy、seaborn、segmentation-models-pytorch。图像读写与几何变换请使用 SimpleITK / nibabel / PIL / MONAI 的 transform 体系。

安装新包需用户在远程手动执行，优先在现有依赖内解决问题。

## 命令速查

所有命令都在**仓库根目录**执行。**完整命令清单、每步的期望输出与报错处置统一在 `docs/baseline.md`**，
这里只列最常用的几条：

```bash
python -m src.selfcheck_data                                    # 数据回归自检（改过 dataset/配置后必跑）
python -m src.train --fold 0 --debug --set train.batch_size=16   # 冒烟自检（不落盘）
python -m src.train --fold 0 --out-dir runs/smoke_fold0 --set train.epochs=12   # 12 轮短跑
python -m src.train --fold 0                                    # 正式训练（--resume 续跑）
python scripts/probe_axis.py --case 31                          # 只在改动写盘逻辑后需要重跑
```

`src/selfcheck_data.py` 是**长期保留的回归自检**：核对「cache 与配置、代码三者是否自洽」，
任何一轮改了预处理 / dataset / 配置之后都应当重跑（秒级到十几秒，不需要 GPU、不写 runs/）。

## 模块一览

| 文件 | 作用 |
| --- | --- |
| `src/dataset.py` | 切片数据集（2.5D 三层窗 + 补边到 512×512）、`BalancedBatchSampler`（每批正负定比）、自实现 2D 增强、batch 契约 |
| `src/unet.py` | 手搓 2D U-Net（`DoubleConv2d` / `UNet2D`）；`in_channels` 与 `data.z_context` 两处自洽校验 |
| `src/losses.py` | 手搓 `DiceCELoss`（softmax + CE + soft Dice，`dice_positive_only` 只对含前景样本算 Dice）与 `build_loss(cfg)` |
| `src/infer.py` | 整卷推理：`predict_volume` / `seg_prob_to_label` / `load_label_volume`（拼卷逻辑与输入通道数无关） |
| `src/train.py` | 训练入口：前置校验 → 训练 → 每轮整卷验证（Dice/IoU/精确率/召回率/塌缩指标）→ 早停 → checkpoint / metrics.csv / TensorBoard；`--debug` / `--resume` |
| `src/selfcheck_data.py` | 数据侧回归自检（只读 cache）：补边 / 2.5D 三层窗 / 增强 / 平衡采样器 / 病人级隔离 |

## 其他约定

- 显存 40 GiB 单卡：优先 patch-based（如 96³–128³）训练，注意 `num_workers` 与 52 核的匹配。
- 代码风格与运行说明随改动一起更新，但**保持精简**（文档是给下一次开发看的，不是归档）。
- **病人级隔离是硬约束**：划分、采样、增强、2.5D 窗口都只能在本病例内部取数据，
  任何改动都要过 `src/selfcheck_data.py` 里那几条断言（口径见 `docs/preprocess_notes.md` 第四节）。
