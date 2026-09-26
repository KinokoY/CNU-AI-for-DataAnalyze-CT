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

## 八、第 4 轮（平衡采样 + Dice 只算前景 + 2.5D 三层输入）：修「塌缩到全预测背景」

> 本节是**第 3 轮首折正式训练失败后的复盘与本轮口径**。运行手册见 `docs/baseline.md` 第 4 节，
> 远程实测数字待用户跑完 `selfcheck_data` / `--debug` / 短跑后回填。

### 8.1 第 3 轮首折的真实曲线：不是「训得不好」，是**塌缩**

用户贴回的 12 轮短跑（`runs/smoke_fold0`，`--set train.epochs=12`，bs=16，351 batch/轮）关键行：

| 现象 | 数值 | 含义 |
| --- | --- | --- |
| 训练 `dice` 项 | epoch 10→12 是 0.9287 → 0.8993，12 轮只降 0.03 | dice 项 = `1 − 肿瘤 soft Dice`，也就是肿瘤 soft 概率只有 ~0.10 |
| 训练 `ce` 项 | 0.0102 → 0.0087 | CE 从 0.0078 涨回 0.0102 又降回去，信噪比极低 |
| 验证整卷 Dice | `33:0.001 57:0.000 59:0.000 60:0.000`，best 0.0013 | **基本上一个前景体素都没预测** |
| `--debug` 的 `prob 峰值` | 0.005（第 3 轮） | 整卷最大肿瘤概率 0.5%，离 0.5 阈值差两个数量级 |

**判据**（写进 `docs/baseline.md` 4.2 的判读）：dice 项贴在 0.9 以上横盘而 ce 掉到 0.01 量级
= 已经在「背景盆地」里；`pred_voxels == 0`（第 4 轮新增的上报量）是它的直接证据。

**根因（三层，按重要性）**：

1. **每批的正向信号被稀释到接近 0（主因）**。老采样器保证「每个阳性层一轮恰好出现一次」，
   于是一轮 351 个 batch 分 617 个阳性层 ⇒ **每批只有 1~2 个含肿瘤切片**（12%），
   而这 1~2 层里肿瘤像素只占 0.2%~0.4%。CE 被背景像素彻底支配，dice 项虽然
   `include_background=false`，但 `batch=True` 是在整批 16 层的并集上聚合的，被 14~15 个
   全背景切片摊薄。老口径的「过采样倍数」实测只有 **0.87 倍**（`docs/preprocess_notes.md` 6.3
   里那句"几乎没有重复采样"其实已经预告了今天的结果）。
2. **阴性采样不给任何"少看背景"的余地**。老口径要求阴性层一轮尽量不重复 ⇒ 一轮必须把
   3852 个阴性层全部过一遍，batch 数被它顶到 351，比例自然回到自然分布。
3. **数值带宽**。2 通道 softmax 的判定等价于 `logit_1 − logit_0 ≥ 0`，塌缩时这个差是 −1.9
   （对应概率 0.13），离判定边界很远，所以验证侧只会看到 0.000 与 0.001 这种数字。

**结论**：管道、划分、轴序都没错（GT 434721 体素那条硬证据早已钉死），错的是**训练分布**
与**损失对稀有类的有效权重**。

### 8.2 本轮的四个改动

#### (1) `BalancedBatchSampler`：固定预算、改比例（取代 `ProportionalBatchSampler`）

- 每批**恒定** `n_pos` 个含肿瘤层 + `n_neg` 个不含肿瘤层，`n_pos = clamp(round(bs × pos_ratio_train),
  min_pos_per_batch, bs−1)`；默认 `pos_ratio_train=0.5` ⇒ bs=16 时**每批 8 正 + 8 阴**（旧口径 1~2 正）。
- 一轮的规模由**槽位预算**决定：`data.epoch_samples` 为 null 时预算 = 数据集切片数
  ⇒ **epoch 墙钟与旧口径基本不变**，只是比例改了（这是用户拍板的"固定预算、改比例"）。
- 阳性层用轮转池**重复采样**，重复次数上限 `data.max_pos_repeat`（默认 8）；阴性层不再要求
  一轮覆盖（覆盖率会掉到 30%~40%，日志显式打印）。**阴性池也会被重复抽**是预期行为。
- 比例被重复上限卡住、或阳性层太少撑不起下限时，实际值可能与目标不同 —— 这两种情况都会
  在 `describe()` 与 `--debug`/训练日志里显式说明（`repeat_capped` / `pool_limited`）。
- **不变的两条**：① 每个 batch 都含阳性（不再有"全阴性 batch 数为下界"这类讨论）；
  ② 采样顺序仍只由 `(train.seed, epoch, 病例集合)` 派生，同 epoch 可复现、相邻 epoch 不同，
  `--resume` 仍用 `set_epoch(已完成轮数)` 接上。

#### (2) 损失：`loss.dice_positive_only: true`

Dice 项**只在 `Y.sum()>0` 的样本上聚合**（`batch=true` 时先按样本掩码筛出含前景的切片再算
交集/并集），全背景样本不参与 Dice；**CE 仍然对整批所有样本计算**（背景那一半负责压假阳性，
不能一起丢掉）。一个 batch 里一个前景样本都没有时 dice 项记 0 而不是 NaN。
这与用户方案的「BCE 全体 + Dice 只对有前景样本」是同一件事，只是我们的 CE 是 softmax 口径。

#### (3) 2.5D 三层输入：`data.z_context=1` → `model.in_channels=3`

- 一个样本 = `[z−1, z, z+1]` 三层叠成通道维，标签恒取**中心层 z**；默认 `in_channels=3`。
- **端点复制**（`window_index`）：`z=0` 的窗口是 `[0,0,1]`、`z=nz−1` 是 `[nz−2,nz−1,nz−1]`，
  **绝不跨病人取层**（相邻病人之间没有空间连续性）。
- **执行顺序（关键，别调换）**：中心层先补边 → **只对中心层做增强** → 再把增强后的中心层与
  **未增强的**邻居层叠成 `(3,H,W)`。所以：
  * 几何增强天然对所有通道同步（它们共享同一个增强后的中心层 + 真实邻居层）；
  * gamma / 高斯噪声从定义上只动**被监督的那一层**，上下文保持真实灰度（自检会逐像素
    比对上下文通道与「相邻层单独读取」的结果）；
  * `z_context=0` 时叠层退化成"加一个长度为 1 的通道维"，与旧口径逐位一致 ——
    切换开关不需要两套增强代码。
- 整卷推理 `predict_volume` **一行都没改**：它逐层调 `dataset[i]`，dataset 给几个通道就喂几个
  通道（"读盘口径只有一份"的好处）。`meta` 里新增 `in_channels` / `z_context` 便于核对。
- 配置自洽在**两处**校验：`unet.UNet2D.__init__`（`in_channels == 2×z_context+1`）与
  `train.check_prerequisites`；`collate_samples(expect_channels=…)` 再兜一层。

#### (4) 验证侧补齐 IoU / 精确率 / 召回率 / 塌缩指标

- **验证侧完全不动分布**：val loader 仍顺序读、保持原始 ~10% 阳性（不做任何平衡采样）。
- 每轮整卷验证新增：`IoU`、`Precision`、`Recall`（同一 `eval.threshold`）、
  `pred_voxels_total`、`prob_peak_max`；`metrics.csv` **只在末尾追加列**
  （`val_iou_mean / val_precision_mean / val_recall_mean / val_pred_voxels / val_prob_peak`），
  第 4 轮的 `evaluate.py` 读旧 csv 也不会炸。
- 预测为空时 `precision` 记 **0.0 并标 `precision_defined=False`**：数学上未定义时记 1.0 会让
  日志看起来"完美"，而 recall 是 0 —— 第 3 轮首折就是这么骗过眼睛的。
- 日志新增一句塌缩告警：`本轮验证**一个前景体素都没预测**`。

#### (5) 顺带修掉的一处日志错位

`train.py` 原来**先 `scheduler.step()` 再打印 lr**，日志里的 `lr` 是"下一轮将要用的值"而不是
本轮实际用的值（`docs/baseline.md` 4.2 的示例日志一直有这个偏差）。现在改成**先取本轮 lr、
再 step**，日志与 TensorBoard 记录的 `train/lr` 就是真正作用于本轮权重更新的那个值。

### 8.3 病人级隔离（用户特别强调，本轮的所有改动都没有放松它）

- 样本单位仍是 `(case, z)`，`case` 只能来自本折 `train` / `val` 列表（`splits.json` 的
  `split_level: "patient"`）；平衡采样器只对切片索引做**重排与重复**，抽不到本折 train 以外的病例。
- `src/selfcheck_data.py` 的采样器自检新增一条硬断言：**一轮抽到的病例集合必须 ⊆ 本折 train**。
- 2.5D 窗口自检（`check_slice_window`）另外钉三件事：窗口层号全在 `[0, nz−1]` 内、
  中心通道逐像素等于「该 (case,z) 单独读一层」、上下文通道逐像素等于「同一病人相邻层」。

### 8.4 本轮**不做**的事（避免误读）

- 不改 `lr` / 调度器：第 3 轮的塌缩不是 lr 造成的，先换采样与损失口径；`train.lr` 与
  `train.min_lr_ratio` 留在配置里，等新口径的曲线出来再定（这是用户拍板的"暂不改值"）。
- 不启用 `loss.ce_class_weights`（默认仍 `null`）：它是**下一级杠杆**，只在"平衡采样 +
  `dice_positive_only` 之后 dice 项仍长期横盘"时才启用；临时试：`--set loss.ce_class_weights=[0.2,1.0]`。
- 仍不做补边区域 ignore mask（`orig_hw` + `pad_offset` 那套留到需要时再说）。
- `train.pos_ratio_target` 自本轮起**不再生效**（保留键只为兼容旧命令与旧 checkpoint 指纹；
  采样器构造时会打印一行"不再生效"的提示）。

### 8.5 本地离线自检（`_selftest.py`，168 项断言）

第 4 轮新增/改写的断言（本地已全部通过，运行 `python _selftest.py`）：

- 平衡采样器：每批阳性数 = 计划定值、batch 大小恒定、阳性重复 ≤ `max_pos_repeat`、
  同 epoch 可复现 / 相邻 epoch 不同、抽到的病例 ⊆ 数据集病例；
- `plan_balanced_slots` 纯函数：真实量级（P=617/bs=16 → 280 batch、8 正 8 阴、重复 3.63 次）、
  重复上限卡住比例、阳性极少时 batch 数被压回、全阴性数据集退化、`pool_limited`；
- 2.5D：`window_index` 的端点复制与越界报错、`z_context=1/2` 的通道数与窗口、
  中心通道 = 单层读取、上下文通道 = 相邻层单独读取、增强后形状与"上下文未被强度增强改动"；
- `collate_samples(expect_channels=…)` 的通道数校验；旧 `ProportionalBatchSampler` 已删除。

**注意**：本地自检用的是假 torch / 假 nibabel / 自造的小假数据，只验证**逻辑与形状口径**；
真实数据上的行为一律以远程 `python -m src.selfcheck_data` → `--debug` → 短跑 为准。

**【第 4 轮真实踩到的坑：本地假 torch 的私有属性掩盖了远程报错】**
第一次远程跑 `python -m src.selfcheck_data` 时在增强自检处 `AttributeError: 'Tensor' object has no
attribute 'array'` —— `_stubs/torch.py` 的假 `Tensor` 有个私有字段 ``.array``（内部就是 ndarray），
而**真 torch 的 `Tensor` 没有这个属性**（只有 `.numpy()` 与 `__array__`）。这类"本地能跑、远程报
AttributeError/TypeError"的问题，根因是本地 stub 比真库更宽松。已做的处理与约定：

- `src/selfcheck_data.py` 里所有"张量 → numpy"一律写 ``np.asarray(张量)``（等价于真 torch 的
  ``__array__``，假 torch 也实现了），**不再出现 ``.array``**；需要切下标时先转 numpy 再切。
- 本地自检里不要用**只有 stub 才有**的 API。当前 stub 提供的私有/宽松接口有：
  ``Tensor.array``、``Tensor.astype()``、``Tensor.ascontiguousarray()``、``Tensor.sum()/mean()/min()/max()``
  （真 torch 这些要带 ``dim=`` 或用 ``torch.*`` 函数）。写代码时按**真 torch** 的接口写。
- 每次改完 `src/` 后，本地除了 `_selftest.py`，还应跑一遍"只读静态扫描"：
  `Select-String -Path src/*.py -Pattern '\.array\b|\.numpy\(\)'`，确认没有依赖 stub 私有属性。

**【同一次远程自检里的第二个坑：自检断言自己误报了端点复制】**

`AttributeError` 修好之后，远程 `selfcheck_data` 又以退出码 1 停下：

```
- batch 1：样本 z=0 的窗口 [0, 0, 1] 层号不连续（越界应当用**端点复制**，不是跳过该层）
```

这条**是断言写错，不是数据错**：`z=0` 的窗口本来就该是 `[0,0,1]`（第 3 轮文档里写的就是这个形态）。
根因是我在 `check_batch` 里加了「相邻层号必须逐个 +1」的判据，把**正确的端点复制**当成了缺陷。
修的过程中又写坏过一版：试图从窗口反推 `nz` 来判断"重复是否只发生在端点"——**在端点处不可能**：
`[0,0,1]` 在「nz=2 的 z=0」与「nz≥3 的 z=0」下都是合法窗口，反推出来的 `nz` 会把合法窗口判死。
最终定下的分工（**别再合并这两层**）：

| 检查 | 位置 | 判据 | 能拿到什么 |
| --- | --- | --- | --- |
| 窗口**结构** | `selfcheck_data.window_structure_problem`（纯函数，本地可单测） | 长度 = 2r+1、`window[r] == z`、层号**非降且步长 ≤ 1** | 只有 batch 里的 `window` 与 `z`，**不需要 nz** ⇒ 不会误报 |
| 窗口**内容/边界** | `selfcheck_data.check_slice_window`（数据集级） | `window == window_index(z, nz, r)`、所有层号 ∈ `[0, nz-1]`、中心通道逐像素等于「单独读该层」 | 能拿到每例的 `nz` 与真实体素 ⇒ 这才是查「跨病人取层」的地方 |

配套改动：`window_index` 明确为「对**下标**做夹取 = edge padding」（本数据集 `nz` 是 74–488，
远大于 `r=1`，实际只会在首末各产生 1 个重复层号）；`_selftest.py` 里给
`window_structure_problem` 加了 11 条用例（含 `[0,0,1]` / `[1,1,2]` / `[8,9,9]` / `[0,0,0,1,2]`
这些端点形态，以及跳层/乱序/中心错位这些真缺陷），本地共 179 项断言。

其余待回填（第 4 轮远程跑完后）：平衡采样后的每批阳性数实测、2.5D 的显存/单步耗时、
新口径下首折前几轮的 dice 项与 val macro Dice、`pred_voxels` / `prob_peak` 是否脱离 0、
早停轮数、5 折汇总。

