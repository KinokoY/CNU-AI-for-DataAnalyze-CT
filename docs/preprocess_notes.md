# 预处理 / 划分阶段备忘（新会话先读这页）

本页只记**已实测的事实**与**踩过的坑**，供新会话的 agent 直接沿用；细节见 `docs/baseline.md`、`docs/data.md`。

## 一、缓存口径（已由 `scripts/probe_axis.py` 在远程实测钉死，勿改）

- 25 例 = 20 含肿瘤 + 5 仅肝脏（32/34/38/41/47）；48–52 因几何错位剔除。
- 空间：全部 `spacing=(1,1,1)`；`nibabel` 读回形状 `(nx, ny, nz)`，**切片轴在最后一维** → `a[:, :, k]` 是 `(ny,nx)` 切片。
  同一文件 `SimpleITK` 读回是其转置 `(nz,ny,nx)`；**两库互为转置，不可混用**。
- 影像：`uint16`，HU 窗 `[-1000,1000]` 线性映射到 `[0,1]` 后按 `1/65535` 量化 → `/65535` 得 [0,1]（1 步长≈0.0305 HU）。
- 掩膜：`uint8`，只含 {0,1}（label 2 = 肿瘤；label 1 肝脏已置 0）。

## 二、关键数字（写 dataset / train 直接用）

- 面内尺寸：`512×512` 17 例，其余 342–436 一族 8 例；按 16 对齐分 **6 个桶**（512/448/432/416×2/368/352×3）。
  **不同面内尺寸不能同 batch，必须分桶**；最大面内 512。
- 切片数 `nz`：74–488（各例不同，勿假设固定）。
- **含肿瘤切片 766 / 5982 = 12.81%** → 采样器把正样本提到 batch 内 25–35% 的基线。
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

## 六、第 2 轮（Dataset + 分桶采样器 + 增强）已确认的事实

- 样本单位：`index[i] = (case, z)`，`__getitem__` 返回
  `{"image": (1,H,W) float32∈[0,1], "label": (H,W) int64∈{0,1}, "case", "z", "orig_hw"}`。
- **batch 契约**（`collate_samples`，第 3 轮训练直接照此取值；**不要用 DataLoader 默认 collate**）：
  `image (B,1,H,W) float32`、`label (B,H,W) int64`、`case list[str]`、`z list[int]`、
  `orig_hw list[tuple[int,int]]`。默认 collate 会把每样本一个 tuple 的 `orig_hw` 转置成两行列表，
  形似 `(H,W)` 但 `for h, w in ...` 解包即炸（远程已炸过一次）；`collate_samples(verify=True)`
  顺带校验形状一致、`label⊂{0,1}`、`image⊂[0,1]`、无 NaN。
- 分桶键 = **精确面内尺寸 `(H, W)`**（如 `(342,342)` 与 `(351,351)` 是**两个桶**）。
  理由：`torch.stack` 要求同 batch 内形状逐元素一致，所以能同 batch 的充要条件就是精确尺寸相同。
  `pad_multiple=16` **只决定模型内部 pad 到多少**（`ceil(H/16)*16`），不参与分桶——早期版本拿
  「对齐 16 之后的值」当桶键，把 342 与 351 放进了同一个 `352x352` 桶，collate 时直接炸。
  `ds.bucket_of_case(case)` 返回桶键；`ds.bucket_stats()[key]["padded_hw"]` 给出该桶要 pad 到的尺寸。
- **batch 内阳性比例的真实口径**：`n_pos = min(batch_size, max(1, round(batch_size × pos_ratio_target)))`。
  `batch_size=8, pos_ratio_target=0.30` → `n_pos=2` → 实际比例 **0.25**（≈ 原始 12.81% 的 1.95 倍）。
  这是取整的必然结果，不是 bug；想贴近 0.30 就把 batch_size 提到 10/20。
- **小桶会被摊薄**：`352/432/448` 三个桶各只有 1–2 例病人、20–49 个阳性层，而一轮要出几十上百个
  batch → 每批阳性数只能是 `floor(P/B)`~`ceil(P/B)`（0 或 1），且必然有少量**全阴性 batch**。
  分配用「均摊」公式 ``第 q 批 = ceil(P*q/B) - ceil(P*(q-1)/B)``（**不要**用
  ``max(q, ceil(P*q/B))``：那会把阳性前置到前 P 批、后面整段全 0，远程实测 566 个 batch 里
  有 117 个全阴性，均摊后降到 16 个）。`BucketBatchSampler.bucket_budget()` 给出每桶的
  `ideal/expect/floor/empty_batches`，自检按它判阈值。
- 采样器保证：一轮 epoch 内每层切片至少出现一次、**阳性层恰好各一次**；阳性充足的桶里每批阳性数
  恒为 `n_pos`；采样顺序只由 `(train.seed, epoch, 病例集合)` 决定（用 sha256 派生，不用内置
  `hash()`），与 `num_workers` 无关，可复现。**可复现性的正确口径**：同 epoch+同 seed 两次采样
  必须完全一致；**相邻 epoch 必须不同**（不同 epoch 种子不同是有意设计，别误判成"不可复现"）。
- `num_workers` 等 DataLoader 参数从 `train` 节挪到了 `data` 节（`configs/default.yaml` 已补 `data` 节）。
- 增强 8 步（image/label 同步，仅训练；**全部自实现，不依赖 MONAI**）：
  `FlipSlice2D`（翻行、翻列各一次）、`Rotate90Slice2D`（整 90°）、`RandAffineSlice2D`
  （旋转 ±15°/缩放 0.9–1.1/平移 ±10%，image 双线性、label 最近邻）、`GammaSlice2D`（随机 gamma
  校正，单调保序的强度重排）、`GaussianNoiseSlice2D`（sigma 从 `U(0, noise_std)` 抽）、
  夹回 [0,1]、label 二值化。串接用 `ComposeSteps` 替代 MONAI 的 `Compose`。
  **为什么放弃 MONAI 做增强**：它的增强有两套签名（array 版吃数组、字典版 `*d` 才吃 dict 且
  `keys` 是必需参数），参数名还跨版本改过（`axis` → `spatial_axis`、`shift_range` 在 1.6 已不存在），
  远程在这一层连炸两次；而这几个增强本身只是几十行 numpy，自实现后可离线逐项断言。
  不做弹性形变：逐层形变会破坏 z 方向一致性，对 2D 基线也没有收益。
- 增强的随机源：`random.Random(stable_seed(train.seed, fold, split) + 104729)`，与采样器种子错开；
  同一折每次构建得到同一串增强参数（可复现），各样本之间仍互不相同。
- 自检脚本：`python -m src.selfcheck_data`（只读 cache，不需要 GPU；**不再需要 MONAI**）。
