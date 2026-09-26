# 运行手册（命令 + 判读）

只管"怎么跑、跑出什么算对"。数据事实见 `docs/data.md`，口径与坑见 `docs/preprocess_notes.md`，
背景与远程环境见 `docs/README.md`。**所有命令都在仓库根目录执行**。

## 1. 命令

```bash
# 环境确认（秒级）
python -c "import sys, torch, monai, SimpleITK, nibabel, numpy, scipy; print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0), monai.__version__, SimpleITK.Version_VersionString(), nibabel.__version__)"

# 正式训练（单折 / 5 折 / 续跑）
python -m src.train --fold 0 --debug --set train.batch_size=16   # 冒烟自检（不落盘，约 1 分钟）
python -m src.train --fold 0                                     # 单折正式训练
python -m src.train --fold 0 --resume                            # 中断后续跑（Ctrl-C 退出码 130）
for f in 0 1 2 3 4; do python -m src.train --fold $f; done        # 5 折

# 评估（读 runs/<run>/best.pt；--all 缺 checkpoint 的折只打印「跳过」）
python -m src.evaluate --fold 0                                  # 单折 + 5 例仅肝脏假阳性
python -m src.evaluate --all                                     # 5 折汇总
python -m src.evaluate --fold 0 --profile 57,59                  # 逐层剖面诊断（不写报告）
python -m src.evaluate --fold 0 --save-pred --save-raw           # 另存预测卷/概率图（保留 affine）
python -m src.evaluate --fold 0 --dry-run                        # 只打印计划，不推理不写报告

# 数据侧自检 / 探测（改过预处理或 dataset 之后才需要）
python -m src.selfcheck_data                                     # 只读 cache 的回归自检（十几秒）
python scripts/preprocess.py                                     # → cache/（已跑过）
python scripts/make_splits.py                                    # → data/splits.json（已跑过）
python scripts/fetch_manifest.py                                 # 只读重建 cache 清单（秒级）
python scripts/probe_axis.py --case 31                           # 只在改动写盘逻辑后需要重跑
```

常用开关：`--set k.v=...`（可多次）、`--device`、`--out-dir`、`--config`；
训练专属 `--debug / --debug-cases / --debug-val-cases / --debug-iters / --resume`；
评估专属 `--run-dir`（checkpoint 目录）、`--ckpt`（直接指文件）、`--folds 0,2`、`--fp-ckpt`、
`--skip-fp`、`--limit-cases N`、`--skip-config-check`、`--profile / --profile-all / --profile-bars`。

退出码：训练 **0** 正常 / **2** 前置校验失败 / **3** NaN 损失 / **4** 续跑配置不一致 / **130** Ctrl-C；
评估 **0** 正常 / **2** 前置校验或权重不匹配 / **1** 运行期异常（这类报告不要进结论）。

## 2. 训练

### 2.1 日志长什么样

一个 epoch 打 **6 行**：5 行进度（`train.log_every_epoch_lines=5`，间隔 = `ceil(一轮 batch 数 / 5)`）
加 1 行收尾摘要。**验证指标与 loss 在同一行**：

```
epoch 1 | iter 56 | loss 0.6321 (dice 0.6152 + ce 0.0169) | pos 0.501 | 10.8s
...
epoch 1 | iter 280 | loss 0.5988 (dice 0.5817 + ce 0.0171) | pos 0.500 | 53.6s
epoch 1/200 | lr 9.99e-04 | loss 0.5988 (dice 0.5817 + ce 0.0171) | pos 0.500 | 280 iters 53.9s [data 1.1s + compute 52.7s] | eval dice 0.0412 [33:0.150 57:0.000 59:0.000 60:0.000] iou 0.0301 prec 0.0512 rec 0.0489 | pred 210345 (gt 453977) peak 0.9987 | lesion 19/19 | cover>=0.2 1/19 | 16.2s | best 0.0412@1 | patience 0/20 | mem 10969MB
```

第 12 轮（fold 0 短跑实测，`runs/smoke_fold0_fixed`）的收尾行长这样：

```
epoch 12/12 | lr 1.70e-05 | loss 0.4613 (dice 0.4426 + ce 0.0187) | pos 0.500 | 280 iters 53.9s [data 1.1s + compute 52.7s] | eval dice 0.1983 [33:0.763 57:0.000 59:0.000 60:0.030] iou 0.1582 prec 0.2210 rec 0.1858 | pred 430364 (gt 453977) peak 0.9999 | lesion 19/19 | cover>=0.2 6/19 | 15.8s | best 0.1983@12 | patience 0/20 | mem 10969MB
```

`--resume` 时会先重放一行 `[上一次运行] epoch N/200 | ...`，让人一眼看到接的是哪一轮。

### 2.2 判读（先看这几项）

| 看什么 | 健康 | 出问题时 |
| --- | --- | --- |
| `loss` 的 `dice` 项（= `1 − 肿瘤 soft Dice`） | 从 1.0 附近往下走 | 贴在 0.9 以上横盘 + `ce` < 0.01 = **塌缩**；`dice` 降而 `eval dice` 不涨 = 只拟合训练集 |
| `eval dice` | 逐例看，别只看均值 | 只有大病灶涨、小病灶长期 0 是**预期**；`pred` 远大于 `gt` = 过预测 |
| `pred` vs `gt` | 同量级（0.5×–1.5×） | `0` → 塌缩（另打 WARNING）；`>3×gt` → 撒一片不定位 |
| `peak`（峰值概率） | 0.9+ 且位置在肿瘤层 | 停在 0.0x → 模型没学会输出前景 |
| `cover>=0.2 k/n` | 逐轮上升 | 长期 0 = 预测没盖住病灶（**这才是有区分度的病灶级口径**，见 2.4） |
| `[data + compute]` | compute ≫ data | data > compute → 先查缓存是不是 `.nii.gz`，再调大 `data.num_workers` |
| `mem` | 稳定不涨 | 逐轮上涨 = 显存泄漏，贴回整段日志 |

⚠️ **验证数值依赖轴序正确**：`predict_volume` 画布与 `load_label_volume` 必须是同一轴序。
改动读盘/拼卷/裁回后，先用 `--debug` 的「GT 434721 体素」（fold 0 的 case 33）+ 逐例 Dice 复核。

### 2.3 续跑（`--resume`）

- 每轮结束都覆盖 `runs/fold<k>/last.pt`（含模型 / 优化器 / 调度器 / AMP scaler / best / patience /
  该轮收尾日志行）。**Ctrl-C（130）、NaN 中断（3）、断电都只损失当前未完成的那一轮**。
- 续跑时校验：`fold` 必须一致、`cfg_hash` 必须一致（**`paths` 与 `train.epochs` 不参与指纹**，
  允许"先跑 50 轮再续到 200"）。不一致直接退出码 4 并列出差异。
- 继续的口径：epoch 从 `已完成 + 1` 接着数、lr 从调度器状态恢复（与一口气跑完严格一致）、
  采样顺序用 `sampler.set_epoch(已完成轮数)` 接上（**同一条序列，不会打乱**）。
- 想改训练口径（采样/损失/通道数/`target_hw`）⇒ 旧权重不能续跑，删掉 `runs/fold<k>/` 重跑即可
  （全新启动会把上一轮的 `best.pt`/`last.pt`/`metrics.csv`/`tensorboard` 改名成 `*.prev`，
  避免 `metrics.csv` 里同一个 epoch 出现两次）。
- 续跑前想只看状态不训练：

```bash
python - <<'PY'
import torch
ck = torch.load("runs/fold0/last.pt", map_location="cpu", weights_only=False)
print("epoch(已完成):", ck["epoch"], "| fold:", ck["fold"], "| cfg_hash:", ck["cfg_hash"])
print("metric:", ck.get("metric"), "| best:", ck.get("best"), "| patience:", ck.get("patience"))
print("有 optimizer/scheduler:", bool(ck.get("optimizer_state")), bool(ck.get("scheduler_state")))
print("上一轮收尾行:", ck.get("log_line") or "（旧版本 checkpoint，无此字段）")
PY
```

## 3. 评估

### 3.1 产物

| 文件 | 内容 |
| --- | --- |
| `reports/eval_fold<k>.json` | 该折的 checkpoint 元信息、口径、折级结论 `assessment`、逐例 `cases`（raw + clean 两套） |
| `reports/eval_summary.json` | 汇总：折表、macro/池化、分层检出、假阳性队列、全部逐例记录 |
| `reports/eval_summary.md` | 人读版：逐折表 / 汇总表 / 体积分档 / **覆盖比例分档** / 逐例表 / 仅肝脏 FP / 后处理前后 / 生成命令 |
| `runs/<run>/pred/*.nii.gz` | 仅 `--save-pred` 时：`<case>_gt` / `<case>_pred`（+ `--save-raw` 再落 `_pred_raw` / `_prob_u16`），**保留 cache 的 affine** |

流程：`best.pt` → 整卷推理（与训练期同一套 `predict_volume`）→ 后处理（删 <50 mm³ 孤立块）→ 指标 → 报告。
评估会重跑训练期那套前置校验，另加两条权重校验：`fold` 一致、`cfg_hash` 一致
（确要强行评估加 `--skip-config-check`，报告会记下这次跳过了校验）。

### 3.2 判读

| 看什么 | 健康 | 出问题时 |
| --- | --- | --- |
| `macro Dice`（主口径，逐例等权） | 与训练日志 best 的 `eval dice` 一致 | 对不上 → 权重/配置不是同一次训练（看 `checkpoint.epoch` 与 `cfg_hash`） |
| `池化 Dice` | 通常 ≫ macro（大病灶主导） | 池化高、macro 低 = 只有大病灶学会了，属预期 |
| **覆盖比例分档（2.2 节）** | 大病灶落在 ≥0.5 档 | 全挤在 `[0.00,0.10)` ⇒ 只是"撒的大片压到了一点"，等于没找到 |
| 逐例 `best_cover` + `dice` | 两者同向 | `best_cover` 高、`dice` 低 ⇒ 位置对但**撒太大**（看 `pred_mm3`/`fp_mm3`）；两者都低 ⇒ **没找到** |
| **仅肝脏 5 例的 FP** | 越低越好 | 命中例数高或 FP 体积到 10⁵ mm³ ⇒ 过预测 |
| 后处理前后 | Dice 略升、覆盖不掉 | 覆盖明显下降 = 删掉了真病灶 → 调小 `eval.min_lesion_mm3` |

⚠️ **`detection_rate` 这一列在本数据集上恒为 1.000**：判据「该 GT 病灶 ≥ `eval.detect_min_mm3`（10 mm³）
即算检出」被数据尺度撑满（20 例含肿瘤病人的肿瘤最小 652 mm³）。日志与 `eval_summary.md` 会显式打
`criterion_b_is_trivial` 警告 —— **看到它就别再读那个数**，改看覆盖比例分档与逐例 `best_cover`。

### 3.3 `--profile`：逐层剖面（没有查看器时的替代）

```
case 57：z 共 135 层｜GT 前景层 12 层（(58, 69)）｜预测前景层 91 层（(3, 93)）｜GT 4046 体素 / 预测 20261 体素｜重叠 0 体素
   z 区间   | GT 体素 | 预测体素 |   差   | GT                      预测
     56-60   |     180 |     1840 |  +1660 | ##                      ****
     ...
  读法：# = GT 在哪几层，* = 预测在哪几层。柱子位置对不上 = 撒错位置；预测柱子远高于 GT = 撒太大；
       预测的 z 范围盖不到 GT 的 z 范围 = 该病灶在轴向上被漏掉。
```

## 4. 报错 → 贴回什么

| 输出 | 含义 / 处置 |
| --- | --- |
| `cache_manifest.cfg_hash=… 与当前配置算出的 … 不一致` | 缓存来自别的预处理参数 → 重跑 `preprocess.py`（只改 `preprocess` 节才会变） |
| `读不到 cache 清单` / `cache/image/ 下缺 N 例` | 先 `fetch_manifest.py`，再 `preprocess.py` |
| `cache 清单与 data/splits.json 的病例集合不一致` | 缓存与划分不是同一批病人 → **贴回该行**，别训练 |
| `验证集里出现不含肿瘤的病例` | 划分被改坏（5 例仅肝脏只应进训练集）→ 检查 `data/splits.json` |
| `model.in_channels=… 与 data.z_context=… 不自洽` | 2.5D 配置改了一边忘另一边 → 配对（0↔1、1↔3、2↔5） |
| `预测卷 (H,W,Z) 与 GT 卷 (H,W,Z) 形状不一致` | 轴序或 `pad_offset` 裁回出错 → **贴整段 traceback**（最优先修的一类） |
| `本轮验证一个前景体素都没预测（pred_voxels=0…）` | 塌缩 → 贴回该行 + 本轮收尾摘要行 |
| `epoch N 第 M 个 batch 的损失是 nan` | AMP 溢出/数据异常 → `--set train.amp=off` 复现，贴回该行 + 前后 20 行 |
| `取数耗时 … 超过计算耗时 …` | 数据加载瓶颈 → 查缓存是否 `.nii.gz`，再调 `data.num_workers` |
| `TensorBoard 不可用（…）` | 只是告警，`metrics.csv` 照常写 |
| `找不到 checkpoint runs/fold<k>/best.pt` | 该折还没训练过（或不在默认 run 目录）→ 先训练，或 `--run-dir` 指向实际产物 |
| `checkpoint 与当前配置不一致（cfg_hash … vs …）` | 权重不是用当前配置训的 → 改回训练时的配置，或加 `--skip-config-check` |
| `属于 fold X，与本次要评估的 fold Y 不一致` | 权重指到了别的折 → 改 `--fold` 或换权重 |
| 退出码 2 + 一串 `-` 行 | 前置校验失败 → **把带 `-` 的行整段贴回** |
| `TypeError: … unexpected keyword argument` | 库版本/参数名不匹配 → **贴整段 traceback** |

## 5. 产物与 git 边界

- `runs/<run>/`：`best.pt`（评估用）、`last.pt`（续跑用，每轮覆盖）、`metrics.csv`（列见
  `src/train.py` 的 `CSV_COLUMNS`）、`run.json`（配置快照 + 环境 + 指纹 + 日志间隔）、
  `train.log`、`tensorboard/`、`pred/`（仅 `--save-pred`）。
- `reports/`：`eval_fold<k>.json`、`eval_summary.{json,md}`、`selfcheck_data.json`。
- **不入库且只在远程**：`cache/`、`reports/`、`runs/`、所有 `*.pt` / `*.nii*`。
- **入库**：`data/splits.json`、`data/exclude_cases.json`、`configs/*.yaml`、`src/*.py`、`scripts/*.py`、`docs/*.md`。
- 随机种子固定 `train.seed=42`；预处理/划分/训练/评估四处的口径见各自脚本头部注释。
