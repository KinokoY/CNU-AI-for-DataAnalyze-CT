# 预处理与数据管线备忘（速查，新会话先读这页）

只留**当前有效**的口径、关键数字与必须避开的坑。数字若与远程实测冲突，**以远程实测为准**。
数据本身见 `docs/data.md`，命令与判读见 `docs/baseline.md`。

## 一、缓存口径（已实测钉死，勿改）

- 25 例可用：20 含肿瘤 + 5 仅肝脏（32/34/38/41/47）；48–52 因几何错位剔除。
- **文件**：`cache/image/<case>.nii`（未压缩，可 mmap）优先、兼容旧 `.nii.gz`；
  一律用 `src.utils.cache_file` / `cache_cases` 解析，**不要手写扩展名**。
- **轴序**：nibabel `(nx, ny, nz)`，**切片轴在最后一维**（`a[:, :, k]` 是 `(ny,nx)`）；
  SimpleITK 读回是其转置 —— **两库不可混用**。
- **影像**：`uint16`，HU 窗 `[-1000,1000]` 线性映射到 `[0,1]` 后按 `1/65535` 量化 ⇒
  `image.float()/65535` 得输入；1 个量化步长 ≈ 0.0305 HU。
- **掩膜**：`uint8`，只含 `{0,1}`（label 2 = 肿瘤，label 1 肝脏已置 0）⇒ 取前景用 `>0`，**不要写 `==2`**。
- 全部 `spacing=(1,1,1)`。
- **为什么必须未压缩**：`.nii.gz` 无法 mmap，nibabel 每读一层要整卷解压（557 ms/层 ⇒ 取数 8.7 分钟/epoch、
  整卷推理 76 s/例）；未压缩 `.nii` 是 0.00 ms/层。已有压缩缓存用 `scripts/inflate_cache.py --remove-gz` 就地转换。

## 二、关键数字（写 dataset / train 直接用）

- 面内尺寸：`512×512` 17 例，其余 342–436 共 8 例；**统一居中补边到 512×512**（不再按尺寸分桶）。
  补边占比按例 0%–55%（加权 ≈13%），补边值恒为背景 0。
- 切片数 `nz`：74–488（各例不同，勿假设固定）。
- 含肿瘤切片：**766/5982 = 12.81%**；fold 0 训练侧 **617/4470 = 13.80%**（fold 0 训练 21 例 / 验证 4 例 = 33/57/59/60）。
- 划分：`data/splits.json`，5 折各 4 例验证 / 21 例训练，验证肿瘤体积 CV=0.317，
  **`split_level: "patient"`**（同病人的切片只在一侧）；5 例仅肝脏只进训练集。
- fold 0 采样器实测（`batch_size=16`）：一轮 **280 个 batch**、每批 **8 正 + 8 阴**、阳性层重复 3.63 次、
  阴性层覆盖率 ≈58%、**全阴性 batch = 0**。
- 性能（2.5D、bs=16、A100 40GB）：单步 ≈0.19 s、一轮训练 ≈54 s、验证 4 例整卷 ≈16 s、
  **峰值显存 10969 MB**；取数 0.6–1.1 s/轮（未压缩缓存）。

## 三、当前训练口径（第 4 轮起有效）

| 项 | 口径 |
| --- | --- |
| 输入 | **2.5D 三层窗**：`data.z_context=1` → `model.in_channels=3`；样本 = `[z−1,z,z+1]`，**标签只监督中心层** |
| 窗口边界 | **端点复制**（`z=0` → `[0,0,1]`；`z=nz−1` → `[nz−2,nz−1,nz−1]`），**绝不跨病人取层** |
| 执行顺序 | 中心层补边 → **叠出 2.5D 窗口**（中心层 + 未增强的真实邻居层）→ 整块送进增强流水线：几何增强对所有通道同步、强度增强（gamma/噪声）作用在全部通道上；`z_context=0` 时退化为单层 2D 的 `(1,H,W)` |
| 采样 | `BalancedBatchSampler`：每批 `n_pos = clamp(round(bs×data.pos_ratio_train), 2, bs−1)` 阳性 + 其余阴性；阳性层按需重复（上限 `data.max_pos_repeat`）；一轮规模由 `data.epoch_samples`（null = 用切片数）决定；顺序只由 `(seed, epoch, 病例集合)` 派生 |
| 损失 | `DiceCELoss`：`softmax`、`batch=True`、`include_background=False`、**`dice_positive_only=True`**（Dice 只对含前景样本聚合，**CE 仍对整批**）；`ce_class_weights=None`、`λ_dice=λ_ce=1`、`smooth=1e-5` |
| 优化 | AdamW(lr=1e-3, wd=1e-4) + CosineAnnealingLR(T_max=epochs, eta_min=0)，**每 epoch 一次**；bf16 autocast **不启用 GradScaler**；`grad_clip_norm=0` |
| 验证 | 每轮**整卷推理**、macro 平均（先逐例再平均）；上报 Dice/IoU/精确率/召回率 + `pred_voxels` + `prob_peak`；**验证侧不做平衡采样**（保持原始分布）；`eval.threshold=0.5`（用 `>=`） |
| 指标公式 | `dice = (2\|A∩B\|+eps)/(\|A\|+\|B\|+eps)`，`eps=1e-6`，两边都空记 1.0；预测为空时精确率记 0.0 且标 `precision_defined=False`（记 1.0 会骗人） |
| 网络 | 手搓 2D U-Net：4 级下采样 `32/64/128/256` + 瓶颈 512，`MaxPool2d` 下采样、`ConvTranspose2d` 上采样 + 跳跃拼接、`BatchNorm`+ReLU，1×1 head 输出 2 通道；前向按 `pad_to_multiple=16` 内部 replicate 补边（512 下不触发）；参数 9.243 M |
| 产物 | `runs/<run>/`：`best.pt`（评估用）/ `last.pt`（续跑）/ `metrics.csv`（每轮一行）/ `run.json` / `train.log` / `tensorboard/`；续跑用 `checkpoint["cfg_hash"]` 校验配置一致性 |
| 读盘策略 | `__init__` 只读 label 体素建索引；`__getitem__` 用每 worker 的 `lru_cache(8)` + `mmap` 只取需要的层（2.5D 取 3 层） |

## 三·补、评估口径（第 5 轮起有效，`src/metrics.py` 是唯一定义处）

| 项 | 口径 |
| --- | --- |
| 体重素口径 | 一律 `(H,W,Z)` 的 nibabel 轴序；预测与 GT 必须逐体素对齐（整卷推理按 `pad_offset` 裁回） |
| Dice / IoU | `dice = (2TP+eps)/(2TP+FP+FN+eps)`、`iou = (TP+eps)/(TP+FP+FN+eps)`，`eps=1e-6`，**两边都空记 1.0**；与训练期选 best 用的是**同一个函数**（`src.metrics`） |
| 精确率 / 召回率 | `TP/(TP+FP)`、`TP/(TP+FN)`；**预测为空时精确率记 0.0**（记 1.0 会骗人）、GT 为空时召回率记 0.0（5 例仅肝脏走这条） |
| 后处理 | 6 邻域 `scipy.ndimage.label`（与预处理统计同源，**别换 SimpleITK**）→ 删**严格小于** `eval.min_lesion_mm3`（默认 50 mm³）的孤立块；体积换算走 `to_mm3`（spacing 恒 (1,1,1)，但 1 体素=1 mm³ 这件事只在一处写） |
| 病灶级检出 | 逐 GT 病灶判：**与预测有任意重叠** 或 **该 GT 病灶 ≥ `eval.detect_min_mm3`（默认 10 mm³）** 即算检出；报告按体积分档（<100 / 100–1k / 1k–10k / 10k–100k / ≥100k）给分母与检出数 |
| 假阳性 | 两种口径都给：落在 GT 之外的**体素**总量，以及**与 GT 零重叠的连通块**（个数 / 体积 / 最大块）。5 例仅肝脏病人的 GT 全空，两者分别等于预测总体积与预测病灶总数 |
| 汇总 | **macro（逐例等权）与池化（所有体素合并后再算比率）两种都给**；`std` 用样本标准差（ddof=1，4 例/折时总体标准差会系统性偏小） |
| 报告主表 | 用**后处理之后**的数字；后处理**之前**的同一组指标写进 JSON（`voxel_raw` / `lesion_raw` / `fp_raw`），用于回答「删块值多少 Dice / 召回」 |

动机（`docs/todo.md` 第 5 轮）：肿瘤体积跨 3 个数量级，fold 0 上 case 33（434 721 体素）能到 0.76
而 57（4 046）/59（652）长期 0 ⇒ 任何「只看均值」的报告都会掩盖小病灶的系统性失败，
所以逐例值、分档检出、仅肝脏 FP 三条必须与均值一起报。

## 四、必须记住的坑

1. **病人级隔离**：划分按病人；采样器只在本折 train 的切片索引内重排/重复；2.5D 窗口只在本病例内索引。
   `src/selfcheck_data.py` 有硬断言（抽到的病例必须 ⊆ 本折 train）。破坏它会让 Dice 虚高。
2. **补边偏移 `pad_offset`**：居中补边时内容不在原点（342→512 是 `(85,85)`），裁回原始面内尺寸必须用它；
   `src.dataset.pad_offset_of` 是唯一口径。
3. **`collate_samples` 是唯一 batch 契约**（别用 DataLoader 默认 collate：它会把 `orig_hw` 这类 tuple 列表转置）。
4. **不要拿 batch 内阳性比例当损失好坏的判据**：真正的判据是训练 soft Dice 与 `val_dice_mean`。
5. **轴序必须处处一致（第 4 轮最大的坑）**：`predict_volume` 的画布是 `(H,W,Z)`，
   逐层写入的是 dataset 的 `[:, :, z]`（= nibabel 轴序），因此 `load_label_volume` **不能再转置前两维**。
   旧版多转置一次，方形病例（512×512）形状相同 ⇒ **形状检查发现不了，Dice 静默接近 0**。
   实测代价：修之前 fold 0 短跑 best 0.0048、修之后 0.1983（同一份损失/采样配置）。
   ⇒ 改动读盘/拼卷/裁回任一环节后，都要用 `--debug` 的「GT 434721 体素」+ 逐例 Dice 复核。
6. **小病灶系统性为 0**：fold 0 上 case 57（4 046 体素）/59（652）长期 0，case 33（434 721）能到 0.76
   ⇒ 报告要按病灶大小分层看，别只看均值；单次短跑也不足以定论，5 折报 mean±std 与逐例值。
7. **本地 `_selftest.py` 用的是假 torch**，比真库宽松（假 `Tensor` 有 `.array` / `astype()` 等私有接口）
   ⇒ `src/` 里"张量 → numpy"一律写 `np.asarray(张量)`；改完 `src/` 跑一次只读扫描
   `Select-String -Path src/*.py -Pattern '\.array\b|\.numpy\(\)'`。
8. **自检断言要先自证**：第 4 轮有两条断言分别误报了"端点复制"和"窗口边界"，
   纯判据要抽成可单测的纯函数（如 `window_structure_problem`），别只在远程试。
9. **不能本地跑的纯函数也要先用合成数组自证**（第 5 轮的 `postprocess` / `metrics` 就是这么验的）：
   一次性抓到 4 个真 bug —— 删块时把 `keep[marks]` 当成"沿第 0 维的掩码"用（IndexError）、
   查表忘了把标记 0（背景）置 0（干净卷求和会得到**全部**体素数）、`label_lesions` 与
   `lesion_stats` 的返回顺序不同（解包错位）、`int(0 维数组)` 在 numpy 2.x 上直接抛异常。
   共同点是**远程跑一次才发现、症状离现场很远**（报告数字全错却不报错），所以值得先在本地钉死。
10. **跨模块导入要按符号表核对，不要凭记忆**：第 5 轮 `voxel_spacing` 定义在 `src/postprocess`，
    却在 `src/evaluate` / `src/train` 里写成 `from src.metrics import voxel_spacing`
    —— 远程一跑就是 `ImportError: cannot import name`（本地 AST 检查抓不到"名字在别的模块里"）。
    现在 `src/metrics.py` 用 `from src.postprocess import voxel_spacing as voxel_spacing`
    **显式再导出**一次，让 `from src.metrics import ...` 成为唯一入口。
    新增/改名模块级符号后，用 AST 扫一遍所有 `from src.X import ...` 是否都能在 `src/X.py`
    顶层找到（这类检查几秒钟，比远程来回一轮便宜得多）。
