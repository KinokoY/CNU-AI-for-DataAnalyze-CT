# 管线口径与坑（速查，新会话先读这页）

只留**当前有效**的口径、关键数字与必须避开的坑。数字若与远程实测冲突，**以远程实测为准**。
数据本身见 `docs/data.md`，命令与判读见 `docs/baseline.md`。

## 一、缓存口径（已实测钉死，勿改）

- **文件**：`cache/image/<case>.nii`（未压缩，可 mmap）优先、兼容旧 `.nii.gz`；
  一律用 `src.utils.cache_file` / `cache_cases` 解析，**不要手写扩展名**。
- **为什么必须未压缩**：`.nii.gz` 无法 mmap，nibabel 每读一层要整卷解压（557 ms/层 ⇒ 取数 8.7 分钟/epoch、
  整卷推理 76 s/例）；未压缩 `.nii` 是 0.00 ms/层。已有压缩缓存用 `scripts/inflate_cache.py --remove-gz` 就地转换。
- **影像** `uint16`（HU 窗 `[-1000,1000]` → `[0,1]` → `×65535`）⇒ 输入 = `image/65535`，1 步长 ≈ 0.0305 HU。
- **掩膜** `uint8`、只含 `{0,1}` ⇒ 取前景用 `>0`（**不要写 `==2`**）。
- 全部 `spacing=(1,1,1)`；轴序与两库混用的禁忌见 `docs/data.md`。

## 二、关键数字（写 dataset / train 直接用）

- 面内尺寸：`512×512` 17 例，其余 342–436 共 8 例；**统一居中补边到 512×512**（不按尺寸分桶），
  补边占比按例 0%–55%（加权 ≈13%），补边值恒为 0。
- 切片数 `nz`：74–488；含肿瘤切片 **766/5982 = 12.81%**；fold 0 训练侧 **617/4470 = 13.80%**
  （fold 0：训练 21 例 / 验证 4 例 = 33/57/59/60）。
- fold 0 采样器实测（`batch_size=16`）：一轮 **280 个 batch**、每批 **8 正 + 8 阴**、
  阳性层重复 3.63 次、阴性层覆盖率 ≈58%、**全阴性 batch = 0**。
- 性能（2.5D、bs=16、A100 40GB）：单步 ≈0.19 s、一轮训练 ≈54 s（+ 整卷验证 ≈16 s）、
  **峰值显存 10969 MB**、取数 0.6–1.1 s/轮。

## 三、训练口径（当前有效）

| 项 | 口径 |
| --- | --- |
| 输入 | **2.5D 三层窗**：`data.z_context=1` → `model.in_channels=3`；样本 = `[z−1,z,z+1]`，**标签只监督中心层** |
| 窗口边界 | **端点复制**（`z=0` → `[0,0,1]`；`z=nz−1` → `[nz−2,nz−1,nz−1]`），**绝不跨病人取层** |
| 执行顺序 | 中心层补边 → 叠出 2.5D 窗口（中心层 + 未增强的真实邻居层）→ 整块送进增强：几何增强对所有通道同步、强度增强（gamma/噪声）只动中心通道；`z_context=0` 退化为单层 `(1,H,W)` |
| 采样 | `BalancedBatchSampler`：每批 `n_pos = clamp(round(bs×data.pos_ratio_train), 2, bs−1)` 阳性 + 其余阴性；阳性层按需重复（上限 `data.max_pos_repeat`）；一轮规模由 `data.epoch_samples`（null = 用切片数）决定；顺序只由 `(seed, epoch, 病例集合)` 派生 |
| 损失 | `DiceCELoss`：`softmax`、`batch=True`、`include_background=False`、**`dice_positive_only=True`**（Dice 只对含前景样本聚合，**CE 仍对整批**）；`ce_class_weights=None`、`λ_dice=λ_ce=1`、`smooth=1e-5` |
| 优化 | AdamW(lr=1e-3, wd=1e-4) + CosineAnnealingLR(T_max=epochs, eta_min=0)，每 epoch 一次；bf16 autocast **不启用 GradScaler**；`grad_clip_norm=0` |
| 验证 | 每轮**整卷推理**、macro 平均（先逐例再平均）；上报 Dice/IoU/精确率/召回率 + `pred_voxels` + `prob_peak` + 病灶检出/覆盖；**验证侧不做平衡采样**；`eval.threshold=0.5`（用 `>=`） |
| 网络 | 手搓 2D U-Net：4 级下采样 `32/64/128/256` + 瓶颈 512，`MaxPool2d` 下采样、`ConvTranspose2d` 上采样 + 跳跃拼接、`BatchNorm`+ReLU，1×1 head 输出 2 通道；`pad_to_multiple=16`（512 下不触发内部补边）；参数 9.243 M |
| 日志 | 一个 epoch **6 行**：`train.log_every_epoch_lines=5` 条进度（间隔 = `ceil(一轮 batch 数/5)`）+ 1 行收尾摘要，**验证指标与 loss 同一行**（见 `docs/baseline.md` 2.1） |
| 产物 | `runs/<run>/`：`best.pt`（评估用）/ `last.pt`（续跑，含 optimizer/scheduler/best/patience/上一轮日志行）/ `metrics.csv` / `run.json` / `train.log` / `tensorboard/` |
| 读盘策略 | `__init__` 只读 label 体素建索引；`__getitem__` 用每 worker 的 `lru_cache(8)` + `mmap` 只取需要的层（2.5D 取 3 层） |

## 四、评估口径（`src/metrics.py` 是唯一定义处）

| 项 | 口径 |
| --- | --- |
| 体素级 | `dice = (2TP+eps)/(2TP+FP+FN+eps)`、`iou = (TP+eps)/(TP+FP+FN+eps)`，`eps=1e-6`，**两边都空记 1.0**；与训练期选 best 用的是**同一个函数** |
| 精确率 / 召回率 | `TP/(TP+FP)`、`TP/(TP+FN)`；**预测为空时精确率记 0.0**、GT 为空时召回率记 0.0（5 例仅肝脏走这条） |
| 后处理 | 6 邻域 `scipy.ndimage.label`（与预处理统计同源，**别换 SimpleITK**）→ 删**严格小于** `eval.min_lesion_mm3`（默认 50 mm³）的孤立块 |
| **病灶覆盖质量（主口径）** | 每病灶 `overlap_frac = 交集体素 / 该 GT 病灶体素`（分母只取 GT，模型撒得再大也不会被撑到 1）；报 `mean/median_overlap_frac`、`n_covered`（≥0.20）、`n_covered_strict`（≥0.50）+ 5 档直方图 |
| 假阳性 | 两种口径：落在 GT 之外的**体素**总量 + **与 GT 零重叠的连通块**（个数/体积/最大块） |
| 汇总 | **macro 与池化两种都给**；`std` 用样本标准差（ddof=1）；另给逐例 `best_cover`，与 Dice 并排可区分「没找到」与「找歪了」 |
| 报告主表 | 用**后处理之后**的数字；后处理**之前**的同一组指标写进 JSON（`*_raw`）用于对照 |

**恒真问题（必须记住）**：原判据的第二条「该 GT 病灶 ≥ `eval.detect_min_mm3`（10 mm³）即算检出」在本
数据集上**恒真** —— 20 例含肿瘤病人的肿瘤体积下限是 652 mm³（全部 19 个 fold-0 val 病灶都 ≥ 652）。
于是 short-run 实测报出 **19/19 = 1.000**，而同一批病例里 57/59 的体素级 Dice 明明是 **0.0000**。
报告与日志会显式打 `criterion_b_is_trivial` 警告 ⇒ **看到它就别再读那个数**，改看覆盖比例分档与 `best_cover`。

## 五、必须记住的坑

1. **病人级隔离**：划分按病人；采样器只在本折 train 的切片索引内重排/重复；2.5D 窗口只在本病例内索引。
   `src/selfcheck_data.py` 有硬断言（抽到的病例必须 ⊆ 本折 train）。破坏它会让 Dice 虚高。
2. **补边偏移 `pad_offset`**：居中补边时内容不在原点（342→512 是 `(85,85)`），裁回原始面内尺寸必须用它；
   `src.dataset.pad_offset_of` 是唯一口径。
3. **`collate_samples` 是唯一 batch 契约**（别用 DataLoader 默认 collate：它会把 `orig_hw` 这类 tuple 列表转置）。
4. **不要拿 batch 内阳性比例当损失好坏的判据**：判据是训练 soft Dice 与收尾行里的 `eval dice`。
5. **轴序必须处处一致**：`predict_volume` 的画布是 `(H,W,Z)`，逐层写入 dataset 的 `[:, :, z]`，
   因此 `load_label_volume` **不能再转置前两维**。转错时方形病例形状不变 ⇒ 形状检查发现不了、Dice 静默接近 0
   （实测代价：0.0048 → 0.1983）。改动读盘/拼卷/裁回后，用 `--debug` 的「GT 434721 体素」+ 逐例 Dice 复核。
6. **小病灶系统性为 0**：fold 0 上 case 57（4 046 体素）/59（652）长期 0，case 33（434 721）能到 0.76
   ⇒ 报告要按病灶大小分层看；单次短跑不足以定论，5 折报 mean±std 与逐例值。
7. **`src/` 里"张量 → numpy"一律写 `np.asarray(张量)`**：本地 `_selftest.py` 的假 torch 比真库宽松
   （有 `.array` 等私有接口），而远程真 torch 上不存在。改完扫一次
   `Select-String -Path src/*.py -Pattern '\.array\b|\.numpy\(\)'`。
8. **本地自证是划算的**（几秒钟换一次远程来回）：
   - 纯判据抽成可单测的纯函数（如 `window_structure_problem`），第 4 轮有两条断言因未自证而远程误报；
   - 不能本地跑的纯函数用**合成数组**验数值口径，第 5 轮一次性抓到 4 个真 bug（删块时把 `keep[marks]`
     当成沿第 0 维的掩码、查表忘了把标记 0 置 0、`label_lesions` 与 `lesion_stats` 返回顺序不同导致解包错位、
     `int(0 维数组)` 在 numpy 2.x 抛异常）；
   - **跨模块导入按符号表核对**：`voxel_spacing` 定义在 `src.postprocess`，写成从 `src.metrics` 导入
     就是远程 `ImportError`（`src/metrics.py` 已用 `as` 形式显式再导出）。新增/改名模块级符号后，
     扫一遍所有 `from src.X import ...` 是否都能在 `src/X.py` 顶层找到。
