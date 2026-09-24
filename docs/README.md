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

## 命令速查（哪些是正式流程、哪些只是自检）

所有命令都在**仓库根目录**执行，每步的期望输出与报错处置见 `docs/baseline.md`。

正式流程（数据 → 训练 → 评估，跑一遍就够）：

```bash
python scripts/preprocess.py        # 预处理 → cache/（第 1 轮，已跑过）
python scripts/check_cache.py       # 缓存体检（已跑过）
python scripts/make_splits.py       # 5 折划分 → data/splits.json（已跑过）
python -m src.train --fold 0 --debug   # 第 3 轮起：冒烟跑几个 iteration，看显存定 batch_size
for f in 0 1 2 3 4; do python -m src.train --fold $f; done   # 正式训练
python -m src.evaluate --all        # 第 4 轮：汇总 5 折指标
```

自检 / 核对类（改过对应代码后才需要重跑；只读 cache，不需要 GPU，不写 runs/）：

```bash
python -m src.selfcheck_data        # 第 2 轮：数据进模型的形态（shape/值域/标签/补边/增强/采样器）
python scripts/probe_axis.py --case 31   # 只在改动写盘逻辑后需要重跑
```

`src/selfcheck_data.py` 不是开发期的临时脚本，而是**长期保留的回归自检**：它核对的是
「cache 与配置、代码三者是否自洽」，任何一轮改了预处理、dataset、配置之后都应当重跑一遍
（秒级到十几秒）。它不产生训练产物，也不需要 GPU（日志里的显存行只是顺带报告）。

## 其他约定

- 显存 40 GiB 单卡：优先 patch-based（如 96³–128³）训练，注意 `num_workers` 与 52 核的匹配。
- 代码风格与运行说明随改动一起更新，但保持精简；实验配置、随机种子、指标口径要写清楚，便于跨轮次复现。
