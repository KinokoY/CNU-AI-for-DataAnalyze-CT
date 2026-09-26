# 运行手册（命令 + 判读）

只管"怎么跑、跑出什么算对"。数据事实见 `docs/data.md`，口径与坑见 `docs/preprocess_notes.md`，
背景与远程环境见 `docs/README.md`。**所有命令都在仓库根目录执行**。

## 1. 命令

```bash
# 环境确认（秒级）
python -c "import sys, torch, monai, SimpleITK, nibabel, numpy, scipy; print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0), monai.__version__, SimpleITK.Version_VersionString(), nibabel.__version__)"

# 正式流程
python scripts/preprocess.py           # → cache/（已跑过；默认写未压缩 .nii）
python scripts/inflate_cache.py --remove-gz   # 旧 .nii.gz 缓存就地转 .nii（已跑过，可省）
python scripts/check_cache.py          # 缓存体检（已跑过）
python scripts/make_splits.py          # → data/splits.json（已跑过）
python -m src.selfcheck_data           # 数据回归自检（2.5D / 平衡采样器 / 病人级隔离）
python -m src.train --fold 0 --debug --set train.batch_size=16   # 冒烟自检（不落盘）
python -m src.train --fold 0 --out-dir runs/smoke_fold0 --set train.epochs=12   # 12 轮短跑（判读用）
python -m src.train --fold 0            # 正式训练
python -m src.train --fold 0 --resume   # 续跑（读 runs/fold0/last.pt）
for f in 0 1 2 3 4; do python -m src.train --fold $f; done      # 5 折
python -m src.evaluate --fold 0 | --all # 待交付（第 5 轮）
```

常用开关：`--set k.v=...`（可多次）、`--device`、`--out-dir`、`--config`、
`--debug-cases / --debug-val-cases / --debug-iters`。
自检脚本可选：`--fold`、`--batches`、`--probe-case`、`--skip-sampler`、`--aug-samples`。
退出码：**0** 正常 / **2** 前置校验失败 / **3** NaN 损失 / **4** 续跑配置不一致 / **130** Ctrl-C。

## 2. 各步期望输出与判读

### 2.1 预处理（2–5 分钟）

尾部应为 `cache 中 image=25，label=25`、`完整性自检通过：25 例全部成功`。
判读：**含肿瘤必须 20 例**（0 例说明标签口径错了）、**spacing 全为 (1,1,1)**、image 是 `uint16` 且值域 ⊂ [0,1]、
每例 `tumor_slices <= n_slices`。出现 `与 docs/data.md 的 25 例预期不一致` 这类行 → 贴回。
清单过期但缓存有效时不必重跑：`python scripts/fetch_manifest.py` 只读重建（秒级）。

### 2.2 数据自检（十几秒，只读 cache、不需要 GPU）

必须**全绿**（打印 `自检通过`），否则不要进训练。重点看这四组：

| 输出 | 应有值 |
| --- | --- |
| `split=train fold=0` | 21 例 / 4470 层 / 含肿瘤 617（0.1380）；val 4 例 / 1513 层 / 149（0.0985） |
| `BalancedBatchSampler` | `bs=16 → 每批 8 正 + 8 阴`；一轮 `280` 个 batch；阳性层重复 ≈3.63；病人级隔离行的病例集合 = 本折 train |
| `batch 1：image (16, 3, 512, 512)` | 中心通道值域 `[0,1]`、label 取值 `{0,1}`、三层窗 `[[0,0,1],[0,1,2],…]`、含肿瘤 8/16 |
| `2.5D 窗口自检` | `中心通道与单层读取不一致 0 处`、`上下文通道与邻居层不一致 0 处`、`3/3 例` |

采样器自检里的 `每批阳性数 [8]`、`全阴性 batch=0`、`抽到的病例 …（越界 0 个）` 也是硬指标。
出现 `自检发现 N 个问题` → **把带 `-` 的行整段贴回**，别带着问题训练。

### 2.3 冒烟自检 `--debug`（约 1 分钟，**不落盘**）

只跑 3 个 iteration 打印形状/耗时/显存，再用当前（随机）权重对 1 例做整卷推理。核对：

- `image (16, 3, 512, 512)` / `label (16, 512, 512)` / `logits (16, 2, 512, 512)` —— 形状必须逐字对上；
- `2.5D 窗口` 连续三层、中心等于该样本的 `z`；`含肿瘤切片 8/16`；
- **GT 前景体素数 = 434721**（fold 0 的 case 33）—— 整条链路（切片轴 → 拼卷 → transpose → 裁回）正确的硬证据；
- 峰值显存与单步耗时（bs=16 实测 10969 MB / ≈0.19 s）；`--debug` 默认 `batch_size=2`，
  要按正式口径测就显式给 `--set train.batch_size=16`。

注意：`--debug` 只用该折 train 的**前 2 例**（fold 0 = 31/32，其中 32 无肿瘤），
采样器可能打印"阳性层少 → 退化为每批 M 个阳性"，**这是小子集的正常现象**，不是 bug。
随机权重下 `loss ≈1.1–1.3` 正常；第 1 个 iteration 明显慢是 cuDNN autotune，看第 2、3 个。

### 2.4 训练（每轮一行）

12 轮短跑实测（`batch_size=16`、fold 0，`runs/smoke_fold0_fixed`）：**280 个 batch / ≈54 s / 轮**、
验证 4 例整卷 ≈16 s、峰值显存 10969 MB、12 轮共 14.3 分钟、best 整卷 Dice **0.1983**。每轮日志形如：

```
epoch 12/12 | lr 1.70e-05 | 训练 loss 0.4613（dice 0.4426 + ce 0.0187）| 280 个 batch / 53.9 s［取数 1.1 s + 计算 52.7 s］| 验证整卷 Dice 0.1983［33:0.763 57:0.000 59:0.000 60:0.030；min 0.000 max 0.763］/ 15.8 s | IoU 0.1581 精确率 0.2207 召回率 0.1858 | 预测体素 431694（GT 453977）峰值概率 0.9999 | best 0.1983@ep12 | patience 0/20 | 峰值显存 10969 MB
```

判读（先看这五行）：

| 看什么 | 健康 | 出问题的样子与处置 |
| --- | --- | --- |
| `预测体素` vs `GT` | 同量级（0.5×–1.5× GT） | `0` → 塌缩（日志会额外告警）；`>3×GT` → 撒一片不定位，接着看下两行 |
| `峰值概率` | 0.9+，且**位置在肿瘤层** | 停在 0.0x → 模型没学会输出前景 |
| `dice` 项（= `1 − 肿瘤 soft Dice`） | 从 1.0 附近往下走 | **贴在 0.9 以上横盘 + `ce` < 0.01 = 塌缩**；`dice` 降而 `验证 Dice` 不涨 = 只拟合训练集 |
| `验证整卷 Dice` | 逐例看，别只看均值 | fold 0 实测：case 33（大病灶 434k 体素）能到 0.76，**57/59（4k/652 体素）长期 0** |
| `［取数 + 计算］` | 计算 ≫ 取数 | 取数 > 计算 → 先查缓存是不是 `.nii.gz`，再调大 `data.num_workers` |

`best` / `patience` 只在验证轮更新；`best.pt` 只在刷新时写，`last.pt` 每轮覆盖。
**训练不稳定**（同一配置两次短跑 best 0.0048 vs 0.1983）⇒ 单次跑不能当结论，
5 折必须报 mean±std 与逐例值。

### 2.5 5 折与续跑

```bash
for f in 0 1 2 3 4; do python -m src.train --fold $f; done
python -m src.train --fold 0 --resume        # Ctrl-C（退出码 130）后可直接续
```

`paths` 与 `train.epochs` 不参与 checkpoint 指纹（允许"先跑 5 轮试水再续到 200"）；
**其它配置项改了会拒绝续跑（退出码 4）并列出差异**。换了训练口径（采样/损失/通道数）后旧权重不能续跑，
直接重跑即可（全新启动会把上一轮的 `best.pt`/`last.pt`/`metrics.csv`/`tensorboard` 改名成 `*.prev`，
避免 `metrics.csv` 里同一个 epoch 出现两次）。

## 3. 报错 → 贴回什么

| 输出 | 含义 / 处置 |
| --- | --- |
| `cache_manifest.cfg_hash=… 与当前配置算出的 … 不一致` | 缓存来自别的预处理参数 → 重跑 `preprocess.py`（只改 `preprocess` 节才会变） |
| `读不到 cache 清单` / `cache/image/ 下缺 N 例` | 先 `fetch_manifest.py`，再 `preprocess.py` |
| `cache 清单与 data/splits.json 的病例集合不一致` | 缓存与划分不是同一批病人 → **贴回该行**，别训练 |
| `验证集里出现不含肿瘤的病例` | 划分被改坏（5 例仅肝脏只应进训练集）→ 检查 `data/splits.json` |
| `model.in_channels=… 与 data.z_context=… 不自洽` | 2.5D 配置改了一边忘另一边 → 配对（0↔1、1↔3、2↔5） |
| `预测卷 (H,W,Z) 与 GT 卷 (H,W,Z) 形状不一致` | 轴序或 `pad_offset` 裁回出错 → **贴整段 traceback**（最优先修的一类） |
| `本轮验证一个前景体素都没预测（pred_voxels=0…）` | 塌缩 → 贴回该行 + 本轮训练那行 |
| `epoch N 第 M 个 batch 的损失是 nan` | AMP 溢出/数据异常 → `--set train.amp=off` 复现，贴回该行 + 前后 20 行 |
| `TensorBoard 不可用（…）` | 只是告警，`metrics.csv` 照常写 |
| `TypeError: … unexpected keyword argument` | 库版本/参数名不匹配 → **贴整段 traceback** |
| 退出码 2 + 一串 `-` 行 | 前置校验失败 → **把带 `-` 的行整段贴回** |

## 4. 产物与 git 边界

- `runs/<run>/`：`best.pt`（评估用，含 `model_state`/`epoch`/`metric`/`cfg_hash`/病例/种子）、
  `last.pt`（每轮覆盖，额外含 optimizer/scheduler/best/patience，续跑用）、
  `metrics.csv`（列见 `src/train.py` 的 `CSV_COLUMNS`）、`run.json`（配置快照 + 环境 + 指纹）、
  `train.log`、`tensorboard/`。
- **不入库且只在远程**：`cache/`、`reports/`、`runs/`、所有 `*.pt` / `*.nii*`。
- **入库**：`data/splits.json`、`data/exclude_cases.json`、`configs/*.yaml`、`src/*.py`、`scripts/*.py`、`docs/*.md`。
- 随机种子固定 `train.seed=42`；预处理/划分/训练/评估四处的口径见各自脚本头部注释。
