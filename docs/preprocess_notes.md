# 预处理 / 划分阶段备忘（新会话先读这页）

本页只记**已实测的事实**与**踩过的坑**，供新会话的 agent 直接沿用；细节见 `docs/baseline.md`、`docs/data.md`。

## 一、缓存口径（已由 `scripts/probe_axis.py` 在远程实测钉死，勿改）

- 25 例 = 20 含肿瘤 + 5 仅肝脏（32/34/38/41/47）；48–52 因几何错位剔除。
- 空间：全部 `spacing=(1,1,1)`；`nibabel` 读回形状 `(nx, ny, nz)`，**切片轴在最后一维** → `a[:, :, k]` 是 `(ny,nx)` 切片。
  同一文件 `SimpleITK` 读回是其转置 `(nz,ny,nx)`；**两库互为转置，不可混用**。
- 影像：`uint16`，HU 窗 `[-1000,1000]` 线性映射到 `[0,1]` 后按 `1/65535` 量化 → `/65535` 得 [0,1]（1 步长≈0.0305 HU）。
- 掩膜：`uint8`，只含 {0,1}（label 2 = 肿瘤；label 1 肝脏已置 0）。
- **文件：`cache/<kind>/<case>.nii`（未压缩，可 mmap）优先，兼容旧的 `.nii.gz`**；
  一律用 `src.utils.cache_file` / `cache_cases` 解析，不要手写扩展名。理由见 7.8。

## 二、关键数字（写 dataset / train 直接用）

- 面内尺寸：`512×512` 17 例，其余 342–436 一族 8 例；**最大面内 512**。
  现在**不再按面内尺寸分桶**（已废弃），而是把每个样本统一补边到 512×512 再用，
  详见第六节。
- 切片数 `nz`：74–488（各例不同，勿假设固定）。
- **含肿瘤切片 766 / 5982 = 12.81%**；采样子集 fold 0 是 617/4469 = 13.81%。
  采样器**不做重复过采样**：它保证「每个阳性层一轮恰好出现一次」，实际 batch 内比例≈整体比例
  （fold 0：bs=8 → 每批 1 个阳性 = 0.125）。要压缩全阴性 batch、提高每批阳性数，走大 batch_size（见 6.3）。
- 划分：`data/splits.json`，5 折各 4 例验证 / 21 例训练，验证体积 CV=0.317。

## 三、已踩的坑（每条都真实发生过）

1. **`np.asarray(sitk.GetArrayViewFromImage(...))` 会段错误**：视图指向临时图像内部缓冲区，临时对象回收后悬空。
   统一用 `GetArrayFromImage`（独立拷贝）。
2. **SITK 没有 `float16` 像素类型**：`sitkFloat16` 不存在；写盘只用 `uint16`/`float32`。
3. **per-slice 统计必须按切片轴求和**：nib 口径用 `lab.sum(axis=(0,1))`，SITK 口径用 `seg.sum(axis=(1,2))`；
   写错会得到"含肿瘤切片数 > 总切片数"这种不可能的结果（已有断言拦截）。
4. **SITK 崩溃不抛 Python 异常**：定位靠 `faulthandler` + `--verbose` 的步骤标记；另外
   `sitk.ConnectedComponent(fullyConnected=...)` 在 2.5.6 上参数名不兼容 → 连通域统一用 `scipy.ndimage.label`（6 邻域）。
5. **掩膜二值化口径**：`load_mask_binary` 已把掩膜变成 0/1，后续再按 `==2` 取前景会得到全 0 标签（曾真实发生）。
   现在原始标签只读一次，重采样后按 `>0` 取，并有断言。

## 四、远程独占产物（本地没有，且不能下载）

`cache/`、`reports/`（含 `preprocess_stats.md`、`cache_check.json`）、`cache/cache_manifest.json`、`runs/`
均在远程且受保密约束，**本地无法读取**。后续编码若需要其中的数字，必须让用户在远程执行脚本并把结果贴回。
需要局部数字时的替代做法：让用户在远程跑一行 `python - <<'PY' ... PY` 打印所需统计。

## 五、本地缺少、必须由远程 push 回来的文件

- ~~**`data/splits.json`**~~：**已入库**（远程跑 `make_splits.py` 后 push 回来了），本地已具备，
  可以直接写/查 `src/dataset.py`、`src/train.py`。

## 六、第 2 轮（Dataset + 补边 + 采样器 + 增强）已确认的事实

> 远程自检已通过（fold 0）：21 例 / 4469 层切片、含肿瘤 617 层；batch 形状恒为 (8,1,512,512)；
> 采样器 642 个 batch、阳性层 617 个一轮恰好各一次、覆盖切片 4469/4469、全阴性 25 个；
> 增强含肿瘤组 label 8/8 被改动、全背景组 8/8 保持全空；切片轴曲线（case 56）落在 z≈264-439。

### 6.1 统一补边（**取代了按面内尺寸分桶**）

- 样本单位：`index[i] = (case, z)`，`__getitem__` 返回
  `{"image": (1,512,512) float32∈[0,1], "label": (512,512) int64∈{0,1}, "case", "z", "orig_hw"}`。
- **读取顺序**：取一层 → `/65535` 还原 [0,1]（label 取 `>0`）→ **居中补边到 512×512**（`data.target_hw`
  + `data.pad_align=center`，image 与 label 同一偏移，补 0）→ 仅训练侧再施加增强。
  镜像做法（早期版本）：按精确面内尺寸分桶；`torch.stack` 要求同 batch 形状逐元素一致，
  于是采样器要先选桶再在桶内配额，小桶被摊薄、还产生大量全阴性 batch —— **已整条删除**
  （`bucket_key` / `format_bucket` / `BucketBatchSampler` / `pad_multiple` 都不存在了）。
- 512 是 16 的倍数 → U-Net 4 级下采样不需要内部 pad；`model.pad_to_multiple` 保留，
  只用来给 `data.target_hw` 做对齐（见 `src/dataset.resolve_target_hw`）。
- 面内尺寸实测（25 例，8 种）：512×512 17 例；其余 436/424/411/409/360/351×2/342 各 1 例。
  补边占比（远程实测，按例）：512 为 0%、436 27.5%、424 31.4%、411 35.6%、409 36.2%、
  360 50.6%、351 53.0%、342 55.4%。且增强在补边之后施加，非 512 病例的补边区在
  gamma/噪声/仿射后**不再严格为 0**。第 3 轮 loss 可考虑用 `orig_hw` + `pad_offset` 生成 ignore mask。
- **裁回原始尺寸必须用 `pad_offset`**：居中补边时内容不在原点（342→512 偏移 (85,85)），
  第 4 轮写死 `[:H, :W]` 会整体错位。`src.dataset.pad_offset_of(orig_hw, target_hw)` 是唯一口径。

### 6.2 batch 契约（`collate_samples`，第 3 轮训练直接照此取值）

`image (B,1,512,512) float32`、`label (B,512,512) int64`、`case list[str]`、`z list[int]`、
`orig_hw list[tuple]`（**补边前**的面内尺寸）、`pad_offset list[tuple]`（内容在画布里的左上角）。
**不要用 DataLoader 默认 collate**：它会把每样本一个 tuple 的 `orig_hw` 转置成两行列表，
形似 `(H,W)` 但 `for h, w in ...` 解包即炸（远程已炸过一次）。
比较 image 与 label 的空间维要**错开一个通道维**（`image (B,1,H,W)` vs `label (B,H,W)`）——
这条历史的坑在 `check_batch` 里改坏过一次。

### 6.3 采样器（`ProportionalBatchSampler`，单一池）

- 目标阳性数/批：`n_pos = min(batch_size, max(1, round(batch_size × pos_ratio_target)))`。
  **它是每批的上限，不是每批的实际比例**：一轮的阳性层总数固定（`P`），摊到 `B` 个 batch 上
  每批就是 `P/B`。fold 0 实测 `bs=8, pos_ratio_target=0.30 → n_pos=2`，但 `P=617, B=642`
  → **每批实际 1 个**（`0.125`），全阴性 batch 25 个（见下一条）。想提高每批阳性数只能加大
  `batch_size`：按 fold 0 的数字，`bs=16`（n_pos=5）是一轮 351 个 batch、每批 1~2 个阳性、全阴性 0 个。
- **一轮的 batch 数 = max(切片覆盖 `ceil(N/bs)`、阳性槽位 `ceil(P/n_pos)`、阴性槽位 `ceil(N_neg/(bs-n_pos))`）**，
  **可能大于** `ceil(N/bs)`。fold 0：`max(559, 309, 642) = 642`（阴性槽位才是上界），
  所以日志里 `切片数/batch_size ≈ 558.6` 只是参照值。
- 阳性层在整轮**均摊**：``第 q 批 = ceil(P*q/B) - ceil(P*(q-1)/B)``，每批再按 `n_pos` 封顶。
  不要用 ``max(q, ceil(P*q/B))``：那会把阳性前置到前几批、后面整段全 0。
  均摊后每个阳性层一轮**恰好出现一次**。
- **全阴性 batch 的个数 = max(0, B - P)**（当每批上限 ≥ `ceil(P/B)` 时），这是「全覆盖 + 均摊」
  下的下界，不是 bug：fold 0 必然有 `642-617 = 25` 个（本地穷举 B 的可行区间确认过下界就是 25）。
  比旧分桶实现的 117 个（566 个 batch 里）少约 4.7 倍；调 `batch_size` 会让它变（bs=16 时为 0）。
- 阴性层用轮转池：优先吐「本轮还没抽过」的，抽完才重排重复 ⇒ 保证**一轮覆盖每一层切片**
  （总槽位数 ≥ 切片数）。采样顺序只由 `(train.seed, epoch, 病例集合)` 用 sha256 派生，与 `num_workers` 无关。
- **可复现性的正确口径**：同 epoch + 同 seed 两次采样必须完全一致；**相邻 epoch 必须不同**
  （不同 epoch 种子不同是有意设计，别误判成"不可复现"）。
- `data` 节里的 `bucket_balance` / `min_pos_per_batch` 已删除；`pos_ratio_tolerance` 保留（自检阈值）。

### 6.4 自检判阈值的三条口径（旧版在这里误报过）

1. 训练侧每批阳性数按**采样器自己的计划区间**（`batch_targets()` 的 `floor`~`observed_max`）判，
   不是拿 `pos_ratio_target` 直接当阈值；
2. **验证侧不判阳性数**：val loader 顺序读、不做过采样，某个 batch 恰好落在「病例前 8 层无肿瘤」
   上（如 case 33）是正常的 —— 旧版把训练侧的桶预算套到 val batch 上，误报
   「阳性切片 0 超出该桶预算 [1,2]」；
3. 几何自检的期望值口径见 `selfcheck_data.check_affine_geometry` 的 docstring（踩过两次）。

### 6.5 增强（8 步，全部自实现）

`FlipSlice2D`（翻行、翻列各一次）、`Rotate90Slice2D`（整 90°）、`RandAffineSlice2D`
（旋转 ±15°/缩放 0.9–1.1/平移 ±10%，image 双线性、label 最近邻）、`GammaSlice2D`（随机 gamma
校正，单调保序的强度重排）、`GaussianNoiseSlice2D`（sigma 从 `U(0, noise_std)` 抽）、
夹回 [0,1]、label 二值化。串接用 `ComposeSteps` 替代 MONAI 的 `Compose`。
**为什么放弃 MONAI 做增强**：它的增强有两套签名（array 版吃数组、字典版 `*d` 才吃 dict 且
`keys` 是必需参数），参数名还跨版本改过（`axis` → `spatial_axis`、`shift_range` 在 1.6 已不存在），
远程在这一层连炸两次；而这几个增强本身只是几十行 numpy，自实现后可离线逐项断言。
不做弹性形变：逐层形变会破坏 z 方向一致性，对 2D 基线也没有收益。

增强的随机源：`random.Random(stable_seed(train.seed, fold, split) + 104729)`，与采样器种子错开；
同一折每次构建得到同一串增强参数（可复现），各样本之间仍互不相同。
**增强在补边之后**施加（见 6.1 的补边代价）。

自检脚本：`python -m src.selfcheck_data`（只读 cache，不需要 GPU；**不再需要 MONAI**）。
想核对某个不在当前折里的病例（如 case 56 在 fold 2 的 val）：`--probe-case 56`。

## 七、第 3 轮（模型 + 损失 + 训练）的编码口径

> 本节是**编码时定下的口径**（本地写代码时就钉死），远程跑过 `--debug` 与首折后回填实测数字。
> 运行手册（命令、期望输出、报错处置）在 `docs/baseline.md` 第 4 节。

### 7.1 网络（`src/unet.py`，从 `torch.nn` 手搓）

- `DoubleConv2d` = `Conv3x3(无 bias) → Norm → ReLU` × 2；`Down2d` = `MaxPool2d(2)` + `DoubleConv2d`；
  `Up2d` = `ConvTranspose2d(2,2)` + 拼接跳跃连接 + `DoubleConv2d`。
- `UNet2D`：编码 `encoder_channels=(32,64,128,256)`，瓶颈 `512`（最后一级 ×2），
  解码逐级回到 `32`，`1x1` 卷积输出 `out_channels=2`（0=背景，1=肿瘤）。
  **4 级下采样 = 16**，与 `model.pad_to_multiple=16` 对齐。
- 前向内部 `_pad_input`：H/W 不是 16 的整数倍时**补在右下**（`replicate`，不是补 0——补 0 等于贴一圈
  “空气”，会给第一层卷积造出假边界），算完再裁回原 H/W。`data.target_hw=512` 已是 16 的倍数，
  正式流程不触发这一步。
- 权重显式初始化：卷积 `kaiming_normal_(fan_in, relu)`、偏置 0、归一化 weight/bias = 1/0。
  写出来是为了「换 torch 版本默认初始化变了也不影响复现」。
- `norm` 支持 `batch`（默认）/ `instance` / `group` / `none`，第 6 轮想换只改配置。
- `load_encoder_pretrained(model, name, path)`：基础版 `name=None` → 记一行日志直接返回（随机初始化）；
  给 `resnet18/34` **直接报错并说明原因**（本版编码器是 DoubleConv，通道 32/64/128/256，
  与 resnet 的 64/128/256/512 残差块参数形状对不上），进阶版才做移植（只读本地权重文件、不走网络）。

### 7.2 损失（`src/losses.py`，不依赖 `monai.losses`）

- `DiceCELoss = λ_dice×(1 - Dice) + λ_ce×CE`，默认 `λ=1.0/1.0`；`softmax=True`、`batch=True`、
  **`include_background=false`**、`smooth=1e-5`、`to_onehot_y=False`（内部一律 one-hot）。
- **`include_background` 必须为 false（第 3 轮首次正式训练踩出来的坑，教科书式的坍缩）**：
  `true` 时损失存在一个「**全预测背景**」的平凡最优解 —— 背景 Dice→1、肿瘤 Dice→0，
  于是 dice 项 = `1-(1+0)/2 = 0.5`，加上 CE 可以压到 ≈0，合计 ≈0.5；
  而"找到肿瘤"要难得多。实测轨迹：初始 1.59 → epoch 2 `dice 0.4996 + ce 0.0078 = 0.507`、
  epoch 3 `0.5046`，每轮只降 0.001，**验证整卷 Dice 4 例恒为 0.0000**（一个肿瘤体素都没预测）。
  改成 `false` 后 dice 项 = `1 - 肿瘤 soft Dice`：全背景解的损失是 1.0，不找到肿瘤就降不下去，
  而且与上报指标（肿瘤整卷 Dice）同向。
  看日志判坍缩只需两件事：**`dice` 项贴在 0.5 附近不动 + `ce` 掉到 0.01 以下** → 已坍缩，别继续跑。
  若改成 false 后仍然欠检出（val Dice 长期为 0），下一批杠杆依次是
  `loss.ce_class_weights=[0.2, 1.0]`、`loss.lambda_ce=0.5`、`train.lr`。
- 为什么小病灶特别容易坍缩：一个 batch（16 层 × 512² ≈ 420 万像素）里肿瘤只占 **0.2%~0.4%**，
  背景 Dice 一项就占了 Dice 损失的 1/2，CE 也被背景像素主导 —— 这也正是 bs=16 有用的原因
  （每批必有 1~2 个阳性切片，肿瘤信号不会整批缺席）。
- **`batch=True` 的含义**：交集/并集先在 `(batch, H, W)` 上求和，每个类得到一个 Dice，再对类平均。
  batch 内常只有 1~2 个含肿瘤切片，逐样本口径会让「整批无肿瘤」的样本把梯度带偏。
- 数值口径：`softmax` 与 `CE` 都在 **float32** 上做（autocast 下也不降精度）；Dice 用 float32 概率。
- 随机权重时的量级参考：背景 Dice≈0.97、肿瘤 Dice≈0 → dice 项 ≈0.5；CE≈ln2≈0.69 → 合计 ≈1.2。
- **本轮不做补边区域 ignore mask**（用户拍板）：非 512 的 8 例补边占比 27%~55%，但补边值就是背景 0。
  第 6 轮若要做，口径是 `orig_hw` + `pad_offset` 生成 mask——注意仿射增强会把真实前景挪出内容框、
  同时把补边噪声挪进框内，mask 本身有轻微错配，得在文档里写清。

### 7.3 验证口径与整卷推理（`src/infer.py`，本轮提前落地）

- **指标**：验证集每例做整卷推理，逐例算整卷肿瘤 Dice，再对 4 例取平均（**macro**，每例等权）。
  不用「逐层平均」——空切片会把指标稀释；不用「全局聚合」——肿瘤体积跨 3 个数量级，会被大病灶主导。
- `predict_volume(model, case, cache_dir, cfg, device, pad_to_multiple=16, batch_slices=8,
  threshold=…, amp=…, tumor_channel=1) -> (prob[H,W,Z] float32, pred uint8, meta)`：
  逐层经 `CTSliceDataset` 读取（与训练同一套 `/65535` + 居中补边口径）→ 概率写进
  `(target_h, target_w, nz)` 画布 → **按 `pad_offset` 裁回原始面内尺寸**（写死 `[:H,:W]` 会整体错位）。
- **轴序**：cache 是 nibabel `(nx,ny,nz)`，逐层拼出来的卷是 **`(ny,nx,nz) = (H,W,Z)`**，
  与直接读出的 label 数组差一次**前两维转置**。`load_label_volume` 统一 `transpose(1,0,2)`，
  训练期验证与第 4 轮评估都用它，别在别处再写一遍。
- 阈值：`eval.threshold=0.5`（用 `>=`）；推理精度默认与训练一致（`eval.amp` → `train.amp`）。
- 显存/内存：整卷画布 float32，`512×512×488 ≈ 512 MB` 主机内存；前向是 `torch.no_grad`，
  `eval.infer_batch_slices` 控制一次几层。
- Dice 公式（训练期 `src/train.volume_dice` 与第 4 轮 `src/metrics.dice` 必须一致）：
  `(2|A∩B| + eps) / (|A| + |B| + eps)`，`eps=1e-6`，两边都空记 1.0。

### 7.4 AMP / 优化器 / 早停

- `train.amp: bf16` → 只用 autocast，**不启用 GradScaler**（bf16 与 fp32 同指数范围，不需要 loss scaling）；
  `fp16` 才启用 GradScaler；`off` = 纯 fp32。跨版本入口是 `src/utils.autocast_context` / `make_grad_scaler`
  （`torch.amp.*` 优先，回退 `torch.cuda.amp.*`；非 CUDA 设备给直通替身，训练循环只写一条梯度路径）。
- 优化器 `AdamW(lr=1e-3, wd=1e-4)` + `CosineAnnealingLR(T_max=epochs, eta_min=lr×min_lr_ratio)`；
  每 **epoch** 走一次 scheduler（不是每 batch）。
- 早停：按验证 macro Dice，`train.early_stop_patience=20`；`best.pt` 只在刷新时写，`last.pt` 每轮覆盖。
- 训练期数值保护：损失出现 NaN/Inf 立刻抛 `NonFiniteLoss` 并以退出码 3 结束（不把 NaN 权重写进 checkpoint）。

### 7.5 产物与续跑（`runs/fold<k>/`，不入库）

- `best.pt`（评估用）/ `last.pt`（续跑用）/ `metrics.csv`（每轮一行，列固定）/
  `run.json`（配置快照 + `cfg_hash` + `preprocess_cfg_hash` + 病例 + 环境）/ `train.log` / `tensorboard/`。
- `--resume`：从 `last.pt` 恢复 model/optimizer/scheduler/已完成轮数/`best`/`patience`，
  并用 `ProportionalBatchSampler.set_epoch(已完成轮数)` 把采样序列接上（`src/dataset.py` 本轮新增的方法）。
- checkpoint 指纹 = 配置去掉 `paths` 与 `train.epochs` 后的 hash：允许「先跑 5 轮试水再续到 200」，
  其余配置改动一律拒绝续跑（退出码 4）并打印差异行。

### 7.6 本轮修掉的一个坑：缓存指纹口径

manifest 里的 `cfg_hash` 原来由两处**不同**的写法产生：`scripts/preprocess.py` / `fetch_manifest.py`
用「排除 paths/train/model/eval」，`src/selfcheck_data.py` 用「排除 paths/train/model/eval/**data**」。
旧配置下两者恰好都只剩 `preprocess` 节，所以一直没暴露；但第 3 轮往配置里新增了与预处理无关的
`loss` 节，排除式写法会把 `loss` 算进指纹 → **缓存没变、训练启动校验却报「缓存来自别的预处理参数」**。

现在统一用 `src.utils.cache_fingerprint(cfg)`（**只取 `preprocess` 节**），四处（preprocess / fetch_manifest /
selfcheck_data / train）共用。对当前配置，它算出的值与 manifest 里已记录的
`7b4c48b4dc7ef880` 保持一致（`preprocess` 节本轮没动），所以远程不需要重跑预处理。

### 7.7 远程实测（第 3 轮 `--debug` 之后，fold 0）

结构类核对**全部通过**（`selfcheck_data` 也仍然通过，指纹 `7b4c48b4dc7ef880` 一致）：

- 形状：`image (B,1,512,512)` / `label (B,512,512)` / `logits (B,2,512,512)`，值域 [0,1]、label ⊂ {0,1}；
  补边偏移与 `orig_hw` 自洽（342→512 是 `(85,85)`、351→512 是 `(80,80)`、436→512 是 `(38,38)`）。
- 采样器：bs=2 时 `n_pos=n_neg=1`，一轮 batch 数 = 阴性层数（230 层 → 193 批，公式吻合）；
  bs=8 时一轮 33 批（2 例子集）、每批 1~2 个阳性。
- 整卷推理：`prob/pred (512,512,135)` 与 `GT` 同形状，`pad_offset=(0,0)`；
  **GT 前景 434721 体素 = `data/splits.json` 里 case 33 的 434721 mm³**（1mm³/体素）——
  这是「切片轴 → 拼卷 → transpose(1,0,2) → 裁回」整条链路正确的最硬证据。
- 显存（`max_memory_allocated`）：bs=2 → 5416 MB，bs=8 → 7671 MB；
  两点线性拟合 ≈ **376 MB/样本 + 4.7 GB 静态开销**（静态里主要是 cuDNN benchmark 的 autotune 工作区）。
  推算 bs=16 ≈ 10.7 GB、bs=32 ≈ 16.7 GB、bs=48 ≈ 22.7 GB（40 GB 卡）。
- **batch_size 定为 16**（第 3 轮按上面的实测定稿，不再是估算）：fold 0 上一轮 351 个 batch、
  每批 1~2 个阳性、**全阴性 batch 归零**，每 epoch 仍有 351 次参数更新；
  bs=32 虽然也能消除全阴性 batch，但更新次数掉到 176 —— 样本量这么小时不划算。
- 单步耗时（bs=8，稳态，排除第 1 个 iteration 的 autotune）：前向+损失 0.040 s / 反向 0.071 s /
  优化器 0.001 s ≈ **0.112 s/step**；第 1 个 iteration 的 forward 2.05 s + backward 10.4 s 是
  `cudnn.benchmark=True` 的一次性 autotune，属正常。
- 模型参数量实测 **9.243 M**（含解码器与 head；早期文档里写的 7.76 M 是估算，已按实测改正）。
- 随机初始化下的损失量级：dice 项 ≈0.68 + CE ≈0.91 ≈ **1.6**（不是 1.2；见 baseline 第 4.1 节的判读）。

**已修并远程验证（第 3 轮）**：`predict_volume` 单例 76 s 的根因就是压缩缓存——`.nii.gz` 无法 mmap，
nibabel **每读一层都把整卷解压一遍**。probe 实测（远程）：

| 读法 | 单层耗时 | 说明 |
| --- | --- | --- |
| `.nii.gz` + `np.asanyarray(proxy)[:, :, z]`（原路径） | **557 ms/层** | 135 层外推 75 s，与 76 s 实测吻合 |
| 整卷读一次再切层 | 0.45 s 读 + **0.00 ms/层** | 内存里切 |
| **未压缩 `.nii` + mmap** | **0.00 ms/层** | 打开 1 ms；就是现在采用的方案 |

训练侧同病：`make_train_loader` 纯取数 812 ms/batch（bs=8，num_workers=8）≈ **8.7 分钟/epoch**，
而 GPU 稳态只要 0.112 s/step ≈ 1.2 分钟/epoch。eval 前向 bf16 0.019 s / fp32 0.031 s，
说明 GPU 侧本来就没有问题（排查时先量这三件事，别先怀疑模型）。

转换后的实测（`inflate_cache.py --remove-gz` + `selfcheck_data` + `--debug --set train.batch_size=16`）：

- 50/50 个文件转换并逐体素校验通过，压缩原件已删；磁盘 1.27 GB → 3.24 GB；
  性能抽检 `mmap 逐层读取 0.00 ms/层`。
- `selfcheck_data` 照旧通过、指纹仍是 `7b4c48b4dc7ef880`（**不需要重跑预处理/重建清单**）；
  首个 train batch 冷读 6.803 s → **0.609 s**；采样器 bs=16：351 个 batch、每批 1~2 个阳性、
  **全阴性 batch = 0**、阳性层 617 次/去重 617、覆盖切片 4469/4469。
- `--debug`：**整卷推理 77 s → 2.6 s**；单步 0.183 s（前向 0.065 + 反向 0.117）；
  **峰值显存 bs=16 = 10920 MB / reserved 12152 MB**（bs=2 5416、bs=8 7671，三点拟合
  376 MB/样本 + 4.7 GB 静态）—— 与两点外推的 10.7 GB 只差 2%，`batch_size=16` 就此定稿。

### 7.8 缓存格式：未压缩 `.nii`（第 3 轮改的默认值）

- `scripts/preprocess.py` 现在默认写 **`.nii`**（未压缩）；`--compressed` 才写 `.nii.gz`。
- 读取端统一走 `src.utils.cache_file`（**`.nii` 优先、`.nii.gz` 兼容**）与 `cache_cases`（按目录清点病例，
  同一 case 两种扩展名只算一次）；遇到压缩缓存会打一次告警。涉及 dataset / infer / train / selfcheck /
  check_cache / fetch_manifest / probe_axis 七处，都不要再手写 `f"{case}.nii.gz"`。
- 已有压缩缓存不用重跑预处理：`python scripts/inflate_cache.py --remove-gz` 在同目录就地转成 `.nii`，
  逐体素校验（shape/dtype/affine/数组全等）后才删原件；**指纹与清单都不变**，不需要 fetch_manifest。
- 代价：磁盘约 3 GB → 4 GB（影像 uint16 ≈ 68 MB/例 + 掩膜 uint8 ≈ 34 MB/例）。
- `src/train.py` 每轮日志新增 `［取数 x s + 计算 y s］`，第 1 轮若出现
  `数据加载是瓶颈` 的告警，先查缓存格式再调 `num_workers`。

其余待回填：首折前几轮的 train loss 与 val macro Dice、单 epoch 耗时、早停轮数、5 折汇总。
