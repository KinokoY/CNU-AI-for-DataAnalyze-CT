# 基础版（2D 肿瘤分割闭环）运行手册

本文档给出**逐条可粘贴**的远程执行命令与每步的期望输出。任务定义、数据事实与约束见
`docs/README.md` 与 `docs/data.md`；本文件只讲"怎么跑、跑出什么算对"。

约定：所有命令都在**仓库根目录**执行（远程 Linux 终端的当前目录就是项目根）。
跑完每一步请把终端输出贴回本地：第 1 步（预处理）与第 3 步（数据自检）是几何口径的依据，
第 4.1 步（`--debug`）是 batch_size 与训练链路的实测依据——这三处的输出尤其要贴回来。

> **第 4 轮（当前）改了什么、要重跑什么**：训练口径换成**平衡采样**（每批正负定比，默认各半）
> + **Dice 只对含前景的样本算** + **2.5D 三层输入**（`data.z_context=1` → `model.in_channels=3`）
> + 验证侧新增 IoU/精确率/召回率与塌缩指标。**缓存与划分不用动**（预处理指纹不变，不需要重跑
> `preprocess.py` / `fetch_manifest.py`）。远程要按顺序跑：
> `python -m src.selfcheck_data` → `python -m src.train --fold 0 --debug` →
> `python -m src.train --fold 0 --out-dir runs/smoke_fold0 --set train.epochs=12`（短跑判读）。
> 改动理由与第 3 轮首折的塌缩复盘见 `docs/preprocess_notes.md` 第八节。

---

## 0. 环境确认（秒级）

```bash
python -c "import sys, torch, monai, SimpleITK, nibabel, numpy, scipy; print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0), monai.__version__, SimpleITK.Version_VersionString(), nibabel.__version__, numpy.__version__, scipy.__version__)"
```

期望：`3.11.x 2.x.x NVIDIA A100-PCIE-40GB 1.6.0 2.5.6 5.4.2 2.4.x ...`

---

## 1. 预处理（预计 2–5 分钟）

```bash
python scripts/preprocess.py
```

做什么：剔除 48-52 → 地板值 padding 夹到 -1000 → 统一 RAS → 重采样到 1×1×1mm（影像线性 / 掩膜最近邻）
→ `clip(-1000,1000)` → `cache/image/<case>.nii`（**uint16，已归一化，未压缩**）+ `cache/label/<case>.nii`（uint8，仅 label 2）。

> **为什么缓存不压缩**（第 3 轮实测后改的默认值，别改回去）：`.nii.gz` 没法 mmap，
> nibabel 读一层 `a[:, :, z]` 会把**整卷解压一遍** = **557 ms/层**；未压缩 `.nii` 走 mmap = **0.00 ms/层**。
> 后果对比：训练取数 8.7 分钟/epoch → 秒级；整卷推理 76 s/例 → ~2 s/例。
> 需要省磁盘时可以用 `python scripts/preprocess.py --compressed` 写回 `.nii.gz`
> （读取端两种都认、未压缩优先，见 1.3）。

**缓存约定（下游 dataset.py 依赖，经 `scripts/probe_axis.py` 在远程实测确认，不要改动）**：

- **轴序**：cache 文件是标准的 **nibabel `(x, y, z)`** 布局，**切片轴在最后一维**：
  `nib.load(p).dataobj` 形状为 `(nx, ny, nz)`，`a[:, :, k]` 是一层 `(ny, nx)` 切片。
  SimpleITK 读同一个文件得到的是它的转置 `(nz, ny, nx)`——两库互为转置，**不要混用**。
  下游按「数组索引 + spacing」使用缓存，不解释 affine。
- **影像数值**：`uint16`，把 HU 窗 `[-1000, 1000]` 线性映射到 `[0, 1]` 后按 `1/65535` 量化存储。
  **dataset.py 里 `image.float() / 65535.0` 即得到 [0,1] 输入**，1 个量化步长 ≈ 0.0305 HU。
  （SimpleITK 没有 float16 像素类型；如需存原始 HU 可把 `preprocess.image_out_dtype` 设为 `float32`。）
- **扩展名**：`<case>.nii` 优先、`<case>.nii.gz` 兼容；一律通过 `src.utils.cache_file` 解析，
  不要在脚本里手写 `f"{case}.nii.gz"`。

产出：

| 文件 | 说明 |
| --- | --- |
| `cache/image/*.nii`、`cache/label/*.nii` | 每例两个文件，共 25 例（未压缩，可 mmap；旧的 `*.nii.gz` 也认） |
| `cache/cache_manifest.json` | 训练启动时校验 cache 与配置是否匹配；**只存在于远程**（不入库） |
| `reports/preprocess_stats.json` / `.md` | 逐 case 统计 + 尺寸分布汇总（**贴回这个 .md**） |

> 清单过期但缓存有效时，不必重跑预处理：`python scripts/fetch_manifest.py` 会从磁盘重建清单（只读、秒级）。

期望输出（日志尾部）：

```
cache 落盘格式：.nii（未压缩：nibabel 可 mmap，逐层读取是页缓存读）
cache 中 image=25，label=25（本次期望 25 例；按 .nii 优先、兼容 .nii.gz 计数）
完整性自检通过：25 例全部成功，image/label 文件集合与预期一致。
下一步：python scripts/check_cache.py（再跑 python scripts/make_splits.py）
```

自检要点：
- 出现 `可用 case 数为 N，与 docs/data.md 的 25 例预期不一致` → 把该行与失败明细贴回。
- 出现 `仅肝脏病例实测 [...]，与 docs 预期 [32,34,38,41,47] 不一致` → 贴回，这不中断流程但会影响划分。
- 出现 `label 2 体素数在重采样前后变化异常` → 贴回，说明该例 spacing 异常。
- 完整跑（不带 `--debug`/`--limit`）时会做完整性gate：任何一例失败、或 image/label 文件集合与
  预期 case 不一致，都会打印 `cache 不完整` 并以退出码 3 结束——此时**不要**进入训练。
- 想先验证脚本能跑通（只处理 1 例、秒级返回）：

```bash
python scripts/preprocess.py --debug
```

> 注意：`--debug`/`--limit` 属于部分运行，会明确提示"未做完整性判定"。
> 正式训练前必须用不带这两个开关的命令完整跑一遍，让 25 例的 cache 与清单一致。

### 1.1 缓存体检（强烈建议，秒级）

```bash
python scripts/check_cache.py
```

逐例核对：image/label 是否成对、shape 是否一致、spacing 是否为约定的 1mm、
label 是否只含 {0,1}、有无肿瘤、影像数值是否落在约定范围，并把实测与 `cache_manifest.json`
逐例比对；有任何不一致会打印 `发现 N 个问题` 并以退出码 1 结束。

期望输出：

```
case 数：25；含肿瘤 20 例；仅肝脏 5 例
含肿瘤的 case：[...20 个...]
仅肝脏的 case：[32, 34, 38, 41, 47]
体检报告：reports/cache_check.json
体检通过：几何、标签、值域与清单全部一致。
```

判读：**含肿瘤必须是 20 例**（若为 0 说明标签口径又错了），**spacing 必须全为 (1,1,1)**，
**image dtype 必须是 uint16**、归一化值域落在 `[0,1]`（乘 65535 后不超过 65535），
**每例 tumor_slices <= n_slices**。

### 1.2 轴序确认（只在改动写盘逻辑后才需要重跑）

```bash
python scripts/probe_axis.py --case 31
```

用途：确认 cache 文件被 nibabel / SimpleITK 读回时的真实形状与切片轴。
它同时打印沿每个轴求和得到的"非零层数"，用清单里的 `tumor_slices` 交叉验证哪一维是切片轴。

### 1.3 缓存格式：已有 `.nii.gz` 缓存怎么办（第 3 轮新增，一次性）

远程已有的是**压缩缓存**（`cache/image/<case>.nii.gz`）。`preprocess.py` 现在默认写未压缩 `.nii`，
读取端（`src/dataset.py` / `src/infer.py`）按「`.nii` 优先、`.nii.gz` 兼容」解析，
所以**不用重跑预处理、也不用改配置或指纹**，跑一次转换脚本即可：

```bash
python scripts/inflate_cache.py --dry-run      # 先看计划（不写文件）
python scripts/inflate_cache.py --remove-gz    # 转换成 .nii，校验通过后删掉 .nii.gz
```

做什么：对 `image/`、`label/` 两个子目录里的每个 `.nii.gz`，**在同目录**写出同名 `.nii`
（目录结构、文件名、体素、dtype、affine 全都不变，只差扩展名），逐体素比对通过后才删原件；
任何一例校验失败就报错退出且不删任何东西。

判读：
- 输出里应有 `image：25 例，其中压缩的 25 例` / `label：25 例，其中压缩的 25 例`；
- 每行 `xxx.nii.gz → xxx.nii：12.3 MB → 68.1 MB（uint16 (512,512,135)，0.45 s）；校验通过`；
- 结束打印 `转换完成：成功 50 个，跳过 0 个，失败 0 个（另有 0 个本来就是未压缩 .nii）`，
  以及一行性能抽检 `性能抽检（33.nii）：mmap 逐层读取 0.00 ms/层（压缩缓存实测 557 ms/层）`；
- 转换后 `python -m src.selfcheck_data` 应照旧通过（指纹 `7b4c48b4dc7ef880` 不变，清单不用重建）。

> 判断当前用的是哪种格式：`python -m src.selfcheck_data` 的 `cache：...（image=25，label=25；.nii 优先、兼容 .nii.gz）`
> 一行，若下面出现 `cache/image 里有 N 例仍是压缩的 .nii.gz` 的告警，就说明还没转换。

---

## 2. 划分（秒级）

```bash
python scripts/make_splits.py
```

做什么：20 例含肿瘤按肿瘤体积降序、每 5 例一档，档内最重的先配给"每例平均体积预算最小"的折；
5 例仅肝脏固定进每折 train。写出 `data/splits.json`（**纳入 git**）。

期望输出：

```
自检通过：5 折各 4 例验证，验证集并集覆盖全部 20 例含肿瘤病人，无重叠，仅肝病例只在 train。
验证集体积均衡：mean=... std=... CV=...
  fold 0: train=21 val=4 [...]（验证体积合计 ... mm3）
  ...
划分已写入：data/splits.json（请提交入库）
```

自检要点：每折 `val` 必须是 4 例，且 5 折 `val` 的并集恰好是 20 例含肿瘤病人；`train` 每折 21 例（16 含肿瘤 + 5 仅肝脏）。

---

## 3. 数据自检（第 2 轮交付、第 4 轮扩充，秒级到十几秒）

```bash
python -m src.selfcheck_data
```

做什么（只读 cache，不建模型、不训练）：核对 splits / 清单 / 预处理指纹 → 构建 train 与 val
两侧的 `CTSliceDataset`（都统一补边到 512×512、**2.5D 三层窗**）→ 实测 3 个 batch 的形态 →
打印一个病例的逐层肿瘤体素数曲线 → 自检增强、**2.5D 三层窗**与**平衡采样器** →
把结果写到 `reports/selfcheck_data.json`。

**第 4 轮新增/变化的输出行**（数字是 fold 0 的口径举例，实跑以贴回的为准）：

```
第 4 轮数据自检：Dataset（2.5D 3 通道 + 统一补边到 512x512）+ 平衡采样器 + 增强（fold=0）
整体阳性率：train 侧 0.1381（617/4469），val 侧 0.0985（149/1513）；**训练侧平衡采样目标 0.50**（≈ 3.6 倍过采样），验证侧保持原始分布不做平衡
BalancedBatchSampler：batch_size=16，data.pos_ratio_train=0.50 → 每批 8 正 + 8 阴（实际阳性占比 0.500）；seed=42
  数据集 4469 层切片（阳性 617 / 阴性 3852）；槽位预算 4469（data.epoch_samples=null→用切片数） → 一轮 280 个 batch / 4480 个槽位
  阳性层一轮重复 3.63 次（上限 data.max_pos_repeat=8，各层尽量均摊）；阴性层覆盖率 36.2%（不再要求一轮全过一遍）
  病人级隔离：样本索引只在本折 train 病例内重排与重复（不跨病人、不引入 val 病例）；病例集合 [...]
训练侧每批阳性数（计划值，定值）：8 正 + 8 阴 = 16；一轮 280 个 batch / 4480 个槽位；...
2.5D 窗口自检（z_context=1 → 3 通道）：抽 3 例查首/中/末层 —— 首层窗口 [0, 0, 1]、末层窗口 [nz-2, nz-1, nz-1]；中心通道与单层读取不一致 0 处；上下文通道与邻居层不一致 0 处；端点复制+中心一致+上下文干净 3/3 例；增强确实改了中心层 N/M 例
采样器自检：每轮 280 个 batch（计划 280）× 16 = 4480 个槽位；每批阳性数 [8]（计划 8 正 + 8 阴）；同 epoch 可复现=True；相邻 epoch 不同=True；batch 大小越界=0 种；阳性层出现 2240 次/去重 617 个（重复 3~4 次，上限 8）；阴性层出现 ... 次/去重 ... 个（覆盖率 36.2%）；全阴性 batch=0；抽到的病例 [...]（越界 0 个）
```

第 4 轮的判读要点：

- **`每批阳性数 [8]` 是定值**：每批恰好 `n_pos = clamp(round(bs × pos_ratio_train), min_pos_per_batch, bs−1)`
  个含肿瘤切片。旧的 `train.pos_ratio_target` 是"上限"、实际由均摊决定（每批 1~2 个），那个口径已废弃。
- **阳性层重复 3~4 次是设计**：阳性层只有 617 个，凑出 50% 的比例必须重复；上限 `data.max_pos_repeat`。
- **阴性层覆盖率掉到 ~36% 也是设计**：只有阳性层还保证"一轮至少出现一次"，阴性层改为轮转池重复抽取。
- **`（越界 0 个）`必须是 0**：病人级隔离断言，非 0 立即停手并贴回该行。
- **2.5D 行的两个"不一致 0 处"**：中心通道必须等于"该层单独读一次"、上下文通道必须等于"相邻层
  单独读一次"。非 0 说明窗口叠层/补边/通道顺序写错了，贴回该行与上面的 batch 行。

期望输出（关键几行；下面的数字是 fold 0 的**第 3 轮实测值**，第 4 轮改口径后会有变化，以实跑为准）：

```
划分：data/splits.json，5 折，每折 train=[21, 21, 21, 21, 21] / val=[4, 4, 4, 4, 4]
清单：25 例，预处理指纹 7b4c48b4dc7ef880（当前配置 7b4c48b4dc7ef880）一致
清单汇总：原始面内尺寸 {'512x512': 17, '436x436': 1, ..., '342x342': 1}（8 种）；切片数 74-488；含肿瘤切片 766/5982 = 0.1281
split=train fold=0：21 例、4469 层切片、含肿瘤 617 层（0.1381）、原始面内尺寸 6 种 → 统一补边到 512x512（center）
  原始   342x342：切片   480（含肿瘤   15 =  3.12%）  病例  1 例 [55]  补边占比 55.38%
  原始   512x512：切片  2187（含肿瘤  376 = 17.19%）  病例 16 例 [...]  补边占比 0.00%
split=val fold=0：4 例、1513 层切片、含肿瘤 149 层（0.0985）、原始面内尺寸 4 种 → 统一补边到 512x512（center）
batch 1：image (16, 3, 512, 512) / label (16, 512, 512)（2.5D 3 通道 = 层 [z-1, z, z+1] 为中心）；病例 [...]；z=[...]
        中心通道值域 [0.0000, 1.0000]；label 取值 [0, 1]；**含肿瘤切片 8/16 = 0.500**（平衡采样目标 0.50）；补边样本 .../16
        orig_hw [...]；补边偏移 [...]；三层窗 [[0,0,1], [0,1,2], ...]
        取该 batch 耗时 ... s（含首次冷读），其后 ... s
3 个 train batch 的空间维只有 1 种：[(512, 512)]（统一补边后应恒为 1 种）
逐层肿瘤体素数（case 56）：nibabel shape=(nx,ny,nz)=(411, 411, 478)，切片轴=最后一维，含肿瘤层 132/478，峰值在第 389 层
  首 10% 层（z<48）阳性占比 0.000，尾 10% 层阳性占比 0.104
增强自检（16 个样本 = 8 含肿瘤 + 8 全背景）：image 被改动 15 个（0.94）；含肿瘤组的 label 被改动 8/8（1.00）；全背景组 label 仍全空 8/8
自检通过：21 例 / 4469 层切片的形态、值域、标签、2.5D 三层窗、补边、增强与平衡采样比例均符合约定。
下一步：python -m src.train --fold 0 --debug（冒烟自检，不落盘）
```

判读要点：

- **batch 形状恒为 `(8, 1, 512, 512)` / `(8, 512, 512)`**：所有样本都补边到 `data.target_hw`，
  所以 `torch.stack` 永远合法。**不再有分桶/桶键/size_buckets 相关日志**——如果还看到
  `桶 512x512`、`各桶可达阳性数` 这类输出，说明远程拉到的还是旧代码。
- **`含肿瘤切片 1/8 = 0.125` 也是对的**：`pos_ratio_target=0.30` 经 `round(8×0.30)=2` 取整得
  `n_pos=2`（每批**上限**），但一轮只有 617 个阳性层要摊到 642 个 batch（见下一条），
  于是 617 个 batch 各 1 个阳性、25 个 batch 没有阳性，批次平均比例 617/5136 ≈ 0.12，
  与整体 0.138 基本持平——**过采样在这里体现为「阳性层一个不漏」，而不是每批硬凑 2 个**。
  想提高每批阳性数就把 `train.batch_size` 提到 16（`n_pos=5`）：按 fold 0 的数字算是
  一轮 351 个 batch、每批 1~2 个阳性、全阴性 0 个（本地推算，第 3 轮 `--debug` 会实测）。
- **一轮 batch 数 = max(切片覆盖、阳性槽位、阴性槽位)，可能大于 `ceil(切片数/batch_size)`**：
  fold 0 上 `ceil(4469/8)=559`，但要让 3852 个阴性层一轮不重复需要 `ceil(3852/6)=642`，
  所以是 **642**。自检会打印 `预期下限` 并按它等号比对，日志里的 `切片数/batch_size ≈ 558.6`
  只是参照值，别拿它心算。
- **全阴性 batch = 25 是当前参数下的理论最小值，不是 bug**：要做满「每个阳性层恰好一次 +
  每层切片至少一次」，就必须 642 个 batch，而一轮只有 617 个阳性层，于是必然有 `642-617=25`
  个 batch 全是阴性（本地扫过 B=300..900 的所有可行解，25 就是下界）。
  对比旧的分桶实现（fold 0 曾有 117 个全阴性 batch）已经降了约 4.7 倍。
  第 6 轮调 `batch_size` / `pos_ratio_target` 时这一项会一起变（bs=16 时就是 0 个）。
- **验证侧不判阳性数**：val loader 顺序读、不做过采样，batch 1/2 正好落在 case 33 的前 16 层
  （那 16 层本来就没有肿瘤，label 取值 `[0]`）是正常现象，日志会打印
  `不做过采样也不判阳性数——出现全阴性 batch 是正常的`。
- **补边偏移必须一起带下去**：`pad_offset` 是内容在 512 画布里的左上角（342→512 是 `(85,85)`），
  第 4 轮裁回原始面内尺寸要用它，写死 `[:H, :W]` 会整体错位。
- **逐层曲线应集中在中段**：若阳性层全挤在 `z≈0` 或 `z≈nz-1`，说明切片轴搞反了，把该曲线贴回。
  想核对不在当前折里的病例（如 case 56 在 fold 2 的 val）：`--probe-case 56`。
- **采样器可复现性的两个口径**（别搞混）：① 同一个 epoch、同一个 seed 两次采样必须**完全一致**
  （自检用两个新建采样器比较）；② **相邻 epoch 必须不同**（否则每轮都固定同一顺序，过采样退化）。
- **增强自检刻意分两组抽样**：含肿瘤的切片只占 13.8%，随机抽 8 个通常有 ~7 个 label 全空，
  而全空 label 怎么翻转都还是全空——所以自检会**专门抽含肿瘤的切片**来看「几何增强有没有改到 label」，
  另抽全背景切片确认「不会凭空造出前景」。看 `含肿瘤组的 label 被改动 N/M` 这一项。
- `值域` 必须落在 `[0,1]`、`label 取值` 必须只含 `{0,1}`、`image dtype` 必须是 float32。

常见报错与贴回内容：

| 输出 | 含义 / 该贴回什么 |
| --- | --- |
| `case 31 缺 image/label` | cache 不完整 → 先跑 `python scripts/preprocess.py` |
| `cfg_hash=... 与当前配置算出的 ... 不一致` | 缓存来自别的预处理参数 → 重跑 `preprocess.py` |
| `含肿瘤切片数 ... 与清单 ... 不符` | 清单过期 → `python scripts/fetch_manifest.py` 刷新 |
| `面内尺寸 ... 超过 data.target_hw ...` | 目标尺寸配小了（补边只能放大）→ 调大 `data.target_hw` |
| `batch 内 image ... 与 label ... 空间维不匹配` | 比较时忘了错开通道维（image 是 (B,1,H,W)）→ 贴回 |
| `阳性层覆盖不对` / `一轮没有覆盖全部切片` | 采样器的均摊或轮转池坏了 → 把该行与上面的采样器自检一起贴回 |
| `TypeError: ... got an unexpected keyword argument` | 多半是库里参数名/版本不匹配 → **贴整段 traceback** |
| 退出码 1 + `自检发现 N 个问题` | **把带 `-` 的行整段贴回**，先不要进入第 3 轮 |

只想快速过一遍、不跑完整轮采样：`python -m src.selfcheck_data --batches 1 --skip-sampler`。

**增强实现方式（重要变更）**：第 2 轮的增强**已改成完全自实现、不依赖 MONAI**
（`src/dataset.py` 里的 `FlipSlice2D` / `Rotate90Slice2D` / `RandAffineSlice2D` / `GammaSlice2D` /
`GaussianNoiseSlice2D`）。原因是 MONAI 的增强有两套签名（array 版吃数组、字典版 `*d` 才吃 dict
且 `keys` 是必需参数），参数名还跨版本改过（`axis`→`spatial_axis`、`shift_range` 在 1.6 已不存在），
我们在这一层连炸两次。现在这四个增强只有几十行 numpy，几何口径、随机性、值域都能在本地离线断言，
`Compose` 也由 `ComposeSteps` 替代。**MONAI 仍可用于其它用途，但这条链路不再需要它。**

| 参数（configs/default.yaml 的 data.augment） | 含义 |
| --- | --- |
| `flip_prob` | 翻行 / 翻列各一次，各自独立（image+label 同步） |
| `rotate90_prob` | 整 90° 旋转（k=1），label 无插值伪影 |
| `affine_prob` / `rotation_deg` / `scale_range` / `shift_frac` | 仿射：旋转 ±15°、缩放 0.9–1.1、平移 ±10%（image 双线性 / label 最近邻） |
| `gamma_prob` / `gamma_range` | 随机 gamma 校正 `img**gamma`（单调保序的强度重排；<1 提亮、>1 压暗） |
| `noise_prob` / `noise_std` | 高斯噪声；sigma 每次从 `U(0, noise_std)` 抽，故 `noise_std` 是强度上界 |

---

## 4. 训练（冒烟 → 标定 → 正式 → 续跑）：第 3 轮落地、第 4 轮换口径

交付物：`src/unet.py`（手搓 2D U-Net）、`src/losses.py`（手搓 Dice+CE）、`src/train.py`（训练入口），
以及 `src/infer.py`（整卷推理：`predict_volume` / `seg_prob_to_label` / `load_label_volume`）——
训练期每轮验证用它，`evaluate` 只在它上面补 `postprocess` / `metrics` / `evaluate`。

代码要点（细节见 `docs/preprocess_notes.md` 第七、八节）：

- 输入：**2.5D 三层窗**（`data.z_context=1` → `model.in_channels=3`，`[z−1, z, z+1]` 叠成通道，
  标签只监督中心层）。标量切片仍是 512×512、统一居中补边。
- 网络：4 级下采样 U-Net，编码 `32/64/128/256`、瓶颈 `512`，`MaxPool2d` 下采样、
  `ConvTranspose2d` 上采样 + 跳跃拼接，`BatchNorm` + ReLU，输出 2 通道 logits（0=背景 1=肿瘤）；
  前向内部按 `pad_to_multiple=16` 对齐（512 已是 16 的倍数，正式流程不会触发）。
- 采样：`BalancedBatchSampler` 每批固定 `n_pos` 正 + `n_neg` 阴（`data.pos_ratio_train=0.5`
  → bs=16 时 8 正 8 阴）；阳性层重复次数上限 `data.max_pos_repeat`；**验证侧不做平衡采样**。
- 损失：自实现 `DiceCELoss = 1×(1 - Dice) + 1×CE`，`softmax=True`、`batch=True`、`include_background=false`、
  **`dice_positive_only=true`（Dice 只对含前景的样本聚合，CE 仍对整批算）**、`smooth=1e-5`；
  **不做补边区域 ignore mask**（需要时再说）。
- 验证：每轮对验证集 4 例做**整卷推理**，逐例算整卷 Dice / IoU / 精确率 / 召回率再取平均（macro，
  每例等权），外加 `pred_voxels_total` 与 `prob_peak_max` 两个塌缩指标；
  用 Dice 选 `best.pt` 与早停；不逐层平均（空切片会把指标稀释掉）。
- AMP：`bf16` 只用 autocast，**不启用 GradScaler**（bf16 与 fp32 同指数范围，不需要 loss scaling）；
  `amp: fp16` 才启用 GradScaler；`amp: off` 走纯 fp32（排查 NaN 用）。
- 随机性：`train.seed` 固定 random/numpy/torch/cuda；采样顺序由 `(seed, epoch, 病例集合)` 派生，
  与 `num_workers` 无关；`--resume` 会把采样序列接到「已完成轮数」之后。

### 4.1 冒烟自检（秒级到 1 分钟，**不写盘**）

```bash
python -m src.train --fold 0 --debug
```

做什么：校验 `data/splits.json` 与 `cache/cache_manifest.json`（病例集合 + 预处理指纹 +
**2.5D 通道数和 `model.in_channels` 是否自洽**）→ 只取该折 **前 2 例**建数据集（`batch_size=2`）→
跑 3 个 iteration，打印每个 batch 的 shape / 中心通道值域 / label 取值 / **实测阳性比例** /
**三层窗层号** / 分段耗时（取 batch、搬 GPU、前向+损失、反向、优化器）/ 显存 allocated 与峰值 →
再用**当前（随机）权重**对 1 例验证病人做一次整卷推理，确认 `prob (H,W,Z)` 形状、`pad_offset` 裁回与
GT 卷形状一致 → 退出，**不创建 `runs/`**。

期望输出（**下面是 fold 0 的形态示例；`--set train.batch_size=16` 时的第 3 轮实测数字仍可对照**）：

```
前置校验通过：fold 0 的 train 21 例 [31, 32, ...] / val 4 例 [33, 57, 59, 60]；cache 清单 25 例，预处理指纹与配置一致；splits 指纹 c96d9c1092ad27a0
设备：cuda（NVIDIA A100-PCIE-40GB）；随机种子：42；目标面内尺寸：512x512（pad_align=center）；输入：2.5D 三层窗 z±1（3 通道）
--debug 冒烟自检：训练侧 2 例 [31, 32] / 230 层切片 / 一轮 18 个 batch；batch_size=16；输入 2.5D z±1（3 通道）；跑 3 个 iteration 后退出（**不写 runs/**）
BalancedBatchSampler：batch_size=16，data.pos_ratio_train=0.50 → 每批 8 正 + 8 阴（实际阳性占比 0.500）；seed=42
iteration 1：image (16, 3, 512, 512) / label (16, 512, 512)（logits (16, 2, 512, 512)）；病例 ['31', '32']；z=[0, 1, 2, ...]；2.5D 窗口 [[0, 0, 1], [0, 1, 2], [1, 2, 3], ...]
  值域 [0.0000, 1.0000]；label 取值 [0, 1]；**含肿瘤切片 8/16 = 0.500**（配置目标 0.50）；orig_hw [[512, 512], ...]；补边偏移 [[0, 0], ...]
  loss 1.20xx（dice 0.60xx + ce 0.6xx）；各 batch 分段耗时：取 batch ... s / 搬到 GPU ... s / 前向+损失 ... s / 反向 ... s / 优化器 ... s；显存 allocated ... MB / 峰值 ... MB
峰值显存（batch_size=16，512×512×3 通道输入，amp=bf16）：allocated ... MB / reserved ... MB
整卷推理自检：case 33（用当前（随机）权重，……**指标数值本身没有意义**）
  prob (512, 512, 135)（峰值 ...）/ pred (512, 512, 135)（前景 ... 体素）/ GT (512, 512, 135)（前景 434721 体素）；整卷 Dice = ...（随机权重）
  （随机权重下的指标同样没有意义，只看链路通不通）IoU ... / 精确率 ...（定义=...）/ 召回率 ...
  原始面内 512x512 → 补边画布 512x512，pad_offset=(0, 0)，nz=135；推理耗时 ~2.6 s（infer_batch_slices=8，阈值 0.50）
--debug 结束：**没有写盘**（runs/ 下不会出现本折产物）。
```

> 说明 0（**第 4 轮先看这两条**）：`image` 的通道维必须是 **3**（`2×data.z_context+1`），
> 且每个样本打印的 `2.5D 窗口` 必须是**连续三层**、中心那个等于该样本的 `z`；
> `含肿瘤切片` 必须是 `n_pos`（bs=16 → 8），若还是 1~2 说明拉到的仍是旧采样器。
>
> 说明 1：debug 用的 2 例是**该折 train 排序后的前 2 例**（fold 0 → [31, 32]，val 前 1 例 → 33），
> 所以每次 `--debug` 的病例都固定、日志可逐次对比。这两例都是 512×512，所以 `pad_offset` 是 `(0,0)`；
> 抽到非 512 病例时会出现 `(85,85)` 这类非零偏移（342→512 的偏移），这是正常的。
>
> 说明 2：**整卷推理自检的耗时是最能反映缓存格式的数**——未压缩 `.nii` 下 case 33（135 层）是 2.6 s；
> 还是 `.nii.gz` 时同一例要 **77 s**（见 1.3）。
>
> 说明 3：2.5D 之后每层要读 3 个切片（首末层是端点复制），**取数耗时会涨一点**，但未压缩缓存下
> 仍是毫秒级；若 `--debug` 里"取 batch"明显变慢，先按 1.3 确认缓存格式，再考虑调大 `data.num_workers`。

判读要点：

- **`image (B,3,512,512)` / `label (B,512,512)` / `logits (B,2,512,512)`**：形状必须逐字对得上，
  这是「dataset 叠层 → 模型 → 损失」三处口径对齐的唯一证据。
- **首个 iteration 的 loss 在 1.4~1.7 之间是正常的**：随机初始化时 logits 近似为 0，
  背景类拿不到「几乎全对」的 Dice（实测 dice 项 ≈0.68，对应背景 Dice≈0.65 而不是 0.97），
  CE 也略高于 ln2（实测 ≈0.91，说明随机权重下模型是"自信地乱猜"）。
  这是**两项各 1.0 权重**的合计值，不是异常；判读要看它随训练下降的趋势。
- **`--debug` 的「一轮 batch 数」看着偏大是正常的**：`batch_size=2` 时采样器算出的
  `n_pos=1, n_neg=1`，于是「阴性槽位」这条下界变成 ≈ 阴性层数（远大于 `切片数/2`）。
  当前默认 `batch_size=16` 时是全折 351 个 batch（口径见 `docs/preprocess_notes.md` 6.3）；
  想看真实量级就按默认配置再跑一次 `--debug`（不加 `--set`）。
- **实测阳性比例 = `data.pos_ratio_train`**（不是"上限"）：平衡采样器每批凑的阳性数就是
  `n_pos`（bs=16 → 8），所以 `--debug` 只跑 3 个 batch 也应该每批都是 8 个阳性切片。
  若只有 1~2 个，说明远程拉到的还是旧采样器（见 `docs/preprocess_notes.md` 8.2）。
- **`pad_offset` 与 `nz`**：`pad_offset` 只对非 512 病例非零（342→512 是 `(85,85)`）；若这里打印的
  面内尺寸与 `docs/preprocess_notes.md` 的面内尺寸表不符，说明 cache 或划分对不上，先别继续。
- **显存与单步耗时用来定 `batch_size`**。**第 3 轮的单通道实测**（`max_memory_allocated`，
  作为对照基线；第 4 轮 2.5D 之后要重新看 `--debug` 的实测值）：

  | batch_size | 峰值 allocated（1 通道） | 备注 |
  | --- | --- | --- |
  | 2 | 5416 MB | |
  | 8 | 7671 MB | |
  | **16（当前默认）** | **10920 MB**（reserved 12152 MB） | 单步 0.183 s |

  单通道三点拟合 ≈ **376 MB/样本 + 4.7 GB 静态开销**（静态里主要是 cuDNN autotune 的工作区）。
  2.5D（3 通道）之后第一级特征图变大，峰值会高于同 batch 的单通道值，**以 `--debug` 实测为准**，
  不做外推。40 GB 卡余量充足，`batch_size=16` 先照旧用，需要时再复测上调。
- **第 1 个 iteration 的 forward/backward 特别慢是正常的**（实测 2.05 s / 10.4 s）：
  `cudnn.benchmark=True` 的一次性 autotune，稳态是 0.040 s / 0.071 s。
- **整卷推理那一步的耗时**（实测 77 s）在把缓存换成未压缩 `.nii`（见 1.3）后会掉到 ~2 s；
  若仍是几十秒，先确认缓存格式，再怀疑模型。

标定某个 `batch_size`（**不给 `--set train.batch_size=` 时 `--debug` 固定用 2**，与正式训练无关）：

```bash
python -m src.train --fold 0 --debug                            # bs=2：最省事的链路自检
python -m src.train --fold 0 --debug --set train.batch_size=16  # 按正式训练的 batch_size 复测显存/耗时（推荐）
python -m src.train --fold 0 --debug --set train.batch_size=32  # 想试更大 batch 时用
```

想跳过整卷推理那一步：`--debug-val-cases 0`；想多跑几个 iteration：`--debug-iters 10`。

### 4.2 正式训练（单折）

```bash
python -m src.train --fold 0
```

做什么：建 train loader（**平衡采样**：每批 `n_pos` 正 + `n_neg` 阴 + 2D 增强）→ 训练一个 epoch →
每 `train.val_every` 轮做一次**整卷验证**（4 例）→ 按验证集整卷 Dice 的 macro 均值更新 `best.pt` /
累计 `patience` → 写 `last.pt`、追加 `metrics.csv`、写 TensorBoard →
`patience >= train.early_stop_patience`（默认 20）时早停。`epochs` 默认 200。

期望输出（形态示例）：开头是设备/配置/前置校验/模型/损失/优化器/采样器几段，然后是每轮一行：

```
设备：cuda（NVIDIA A100-PCIE-40GB）；随机种子：42；目标面内尺寸：512x512（pad_align=center）；输入：2.5D 三层窗 z±1（3 通道）
训练配置：epochs=200，batch_size=16，lr=0.001，val_every=1，早停 patience=20，grad_clip_norm=0，num_workers=8
模型：UNet2D(in=3 [2.5D 三层窗 z±1（3 通道）], out=2, encoder=[32, 64, 128, 256], bottleneck=512, 下采样 4 次（对齐 16）, norm=batch, 可训练参数 9.25 M)
DiceCELoss（自实现）：λ_dice=1 λ_ce=1；softmax=True batch=True include_background=False smooth=1e-05；to_onehot_y=False（内部一律 one-hot）；dice_positive_only=True（Dice 只在含前景的样本上算（CE 仍对整批算））；ce_class_weights=None
优化器：AdamW(lr=0.001, weight_decay=0.0001)；调度器：CosineAnnealingLR(T_max=200, eta_min=0)
AMP：bf16 autocast（**不启用 GradScaler**：bf16 与 fp32 同指数范围，不需要 loss scaling）
前置校验通过：fold 0 的 train 21 例 [...] / val 4 例 [...]
训练采样器（平衡采样）：batch_size=16 → 每批 8 正 + 8 阴（阳性占比 0.500，整体阳性率 0.1381 ⇒ 过采样 3.62 倍）；一轮 280 个 batch；阳性层重复 3.63 次（上限 8）、阴性层覆盖率 36.2%
训练 loader：21 例 / 4469 层切片 → 一轮 280 个 batch（batch_size=16）；验证 4 例（整卷推理）
首个 batch 形态：image (16, 3, 512, 512) / label (16, 512, 512)；病例 [...]；z=[...]；2.5D 窗口 [[0,0,1], [0,1,2], ...]；值域 [0.0000, 1.0000]；orig_hw [...]；补边偏移 [...]
epoch 1/200 | lr 1.00e-03 | 训练 loss …（dice … + ce …）| 280 个 batch / … s［取数 … s + 计算 … s］| 验证整卷 Dice …［33:… 57:… 59:… 60:…；min … max …］/ … s | IoU … 精确率 … 召回率 … | 预测体素 …（GT …）峰值概率 … | best …@ep1 | patience 0/20 | 峰值显存 … MB
...
训练结束：best 整卷 Dice … @ epoch …；本次跑完 … 轮（epoch 1 → …），用时 … 分钟
产物：runs/fold0（best.pt / last.pt / metrics.csv / run.json / train.log / tensorboard/）
```

> 上面的数字是**形态示例**：`batch_size=16` 时 fold 0 一轮 **280 个 batch**（平衡采样后由
> 「正负各半」的槽位预算算出，不再是 351），单步约 0.183 s（第 3 轮单通道实测）⇒ 计算 ~50 s；
> 2.5D 之后单步会更慢一些，以实测为准。验证 4 例整卷 ≈ 11 s。
> 若日志里 `取数` 涨到分钟级，先按 1.3 确认缓存格式，再考虑调大 `data.num_workers`。

判读要点：

- **训练 loss 前几轮应明显下降**，且 **`dice` 项必须从 1.0 附近往下走**（`dice_positive_only=true`
  之后它仍然是 `1 - 肿瘤 soft Dice`，但只在含前景的样本上聚合）。若 3~5 轮后 loss 完全不动或
  变成 `nan`，先按下面的报错表处理，别让它跑满 200 轮。
- **验证整卷 Dice 从 0.0x 起步是正常的**（随机权重常把整卷判成背景或乱判），看的是**趋势**。
- **第 4 轮新增的三行判读（比 Dice 更早暴露塌缩）**：
  * `预测体素 N（GT M）`：**N=0 就是"一个前景体素都没预测"**，日志会额外打一条警告；
  * `峰值概率`：整卷肿瘤概率的最大值。第 3 轮首折全程 0.005，健康训练应当稳步抬到 0.9+；
  * `IoU / 精确率 / 召回率`：预测为空时精确率记 0.0（`精确率 0.0000` + `预测体素 0` 同时出现即塌缩）；
    召回率长期为 0 而精确率很高，说明模型只敢圈很小的一块（欠检出）。
- **`验证整卷 Dice` 是 macro 均值**（先逐例算、再平均），括号里是逐例值（fold 0 的 val 是
  33/57/59/60，见 `data/splits.json`）：某例长期为 0 通常说明该例病灶最小/最难
  （`docs/data.md`：肿瘤体积跨 3 个数量级），不是 bug。
- **每轮那行的 `dice` / `ce` 两项要一起看**：`dice` 项就是 `1 - 肿瘤 soft Dice`，
  **`dice` 项贴在 0.9 以上横盘 + `ce` 掉到 0.01 以下是「塌缩到全预测背景」的特征**
  （第 3 轮首折的真实曲线：12 轮里 dice 0.93→0.90、ce 0.010→0.009、验证 Dice 恒 ≈0.000）。
  第 4 轮的平衡采样 + `dice_positive_only` 就是针对它；判读与修法见
  `docs/preprocess_notes.md` 8.1 / 8.2。
- **`best` 与 `patience`**：只有验证轮才更新；`patience` 达到 `early_stop_patience` 就停在那一轮，
  `best.pt` 仍指向历史最优。每轮都会覆盖 `last.pt`（续跑用），`best.pt` 只在刷新时写。
- **`峰值显存`** 是本次运行的历史峰值；把它与 `--debug` 的实测值对照，可以判断还能不能再加 batch。
- **每轮那行的 `［取数 x s + 计算 y s］` 是判断瓶颈的唯一依据**：正常应是「计算远大于取数」
  （未压缩缓存下取数几秒、计算几十秒到一两分钟）。若 `取数 > 计算`，说明数据加载拖住了 GPU：
  先看 `python -m src.selfcheck_data` 有没有 `仍是压缩的 .nii.gz` 告警（有就按 1.3 转换），
  已经是 `.nii` 再考虑调大 `data.num_workers`（52 核，可到 16）。注意 **2.5D 每层要读 3 个切片**，
  取数耗时会比单通道略高，这是预期内的。第 1 轮 epoch 若出现
  `取数耗时 ... 超过计算耗时 ...：**数据加载是瓶颈**` 的告警，也是同一件事。

### 4.2.1 短跑烟测：先拿一份「能预测出东西」的权重给下游开发用

本次 12 轮日志显示，运行时显式覆盖了损失为 `include_background=true`、
`dice_positive_only=false`，与本配置默认的 `false/true` 相反；当时整卷 GT 还被额外
转置，所报最佳 Dice 0.0066 不能作为真实模型表现。现已修正 GT 轴序与 2.5D 几何增强。
拉取本次代码后先运行 `python -m src.selfcheck_data --fold 0`，确认
`整卷 GT/训练标签对齐 ... 逐像素一致=True`。随后用下面命令从头短跑，
不要加旧的 `--set loss.*` 参数，也不要从旧 `last.pt` 续跑。

```bash
python -m src.train --fold 0 --out-dir runs/smoke_fold0 --set train.epochs=12
```

用途：整卷推理 → 3D 后处理 → 指标 → 报告这一段的开发需要一份**真实**的 `best.pt`；
12 个 epoch 约十几分钟，足够让模型从「全预测背景」变成「至少圈得出肿瘤」，于是后处理、
病灶级检出、FP 统计这些代码路径才真的被走到（空预测会把所有分支都走成退化路径，掩盖 bug）。

**第 4 轮它就是「口径修好没有」的第一道验收**（比正式 200 轮便宜得多）：

- `dice` 项应从 1.0 附近**明显下降**（跌破 0.7 才算真的在找肿瘤）；
- 每轮那行的 `预测体素` 必须 > 0，且 `峰值概率` 明显抬起来（不再停在 0.00x）；
- `验证整卷 Dice` 至少有一例 > 0.05，`metrics.csv` 里 `best_dice` 不为 0；
- **仍然不要求指标好看**，它是下游代码的输入 + 口径是否修好的信号。
- 若 12 轮后 `dice` 项仍贴在 0.9 横盘、`预测体素` 仍是 0：**停下来把整段日志贴回**，
  下一级杠杆是 `--set loss.ce_class_weights=[0.2,1.0]` 或回调 `train.pos_ratio_train`，
  别直接开 200 轮。

注意：
- **写进独立目录** `--out-dir runs/smoke_fold0`，别和正式训练的 `runs/fold0` 混在一起；
- 全新开始（非 `--resume`）时 `train.py` 会把同目录里上一轮的
  `best.pt` / `last.pt` / `metrics.csv` / `tensorboard` 改名成 `*.prev` 只留一代 ——
  否则 `metrics.csv` 是追加写的，新一轮的 epoch 1..N 会接在旧内容后面，同一个 epoch 号出现两次；
- 12 epoch 的权重**不是结果**，评估报告里要标注它的来源（epoch 数与 run 目录）。
- 下游要评这份权重时，`python -m src.evaluate --fold 0 --run-dir runs/smoke_fold0`（待交付该开关）。

### 4.2.2 5 折依次跑

```bash
for f in 0 1 2 3 4; do python -m src.train --fold $f; done
```

跑完 5 折后进入第 4 轮的评估（`python -m src.evaluate --all`）。

### 4.3 产物（`runs/fold<k>/`，不入库）

| 文件 | 内容 / 用途 |
| --- | --- |
| `best.pt` | 验证整卷 Dice 最优的权重；含 `model_state` / `epoch` / `metric` / `cfg` / `cfg_hash` / 病例列表 / 种子。**第 4 轮评估读它** |
| `last.pt` | 每轮覆盖，额外含 `optimizer_state` / `scheduler_state` / `best` / `patience`；`--resume` 读它 |
| `metrics.csv` | 每轮一行：`epoch, lr, train_loss, train_dice, train_ce, train_batches, train_pos_ratio, train_seconds, val_dice_mean, val_dice_min, val_dice_max, val_seconds, is_best, best_dice, patience, epoch_seconds, gpu_peak_mb` |
| `run.json` | 本次运行的元信息（配置快照、`cfg_hash`、`preprocess_cfg_hash`、病例、环境版本、最终 best） |
| `train.log` | 与终端同样的日志（DEBUG 级） |
| `tensorboard/` | `train/loss`、`train/lr`、`val/dice_mean`、`val/dice_case_<case>` 等标量（`train.tensorboard=false` 可关） |

### 4.4 中断与续跑

```bash
python -m src.train --fold 0 --resume
```

- 从 `last.pt` 恢复 `model/optimizer/scheduler`、已完成的轮数、`best` 与 `patience`，
  并把采样顺序接到 `epoch = 已完成轮数 + 1`（`BalancedBatchSampler.set_epoch`），
  与「一口气跑完」的采样序列一致；
- `paths` 与 `train.epochs` 不参与 checkpoint 指纹，所以「先 `--set train.epochs=5` 试水、再 `--resume`
  跑满 200」是允许的；**其它任何配置项改了都会拒绝续跑（退出码 4）并列出差异**；
- **第 3 轮的 `last.pt`/`best.pt` 不能用来续跑**：本轮换了采样口径、损失口径与输入通道数，
  配置指纹必然不同（会以退出码 4 明确拒绝）。要重新开始就在 `runs/fold<k>/` 里删掉旧权重，
  或者直接跑（全新开始时旧的 `best.pt`/`last.pt`/`metrics.csv`/`tensorboard` 会自动改名成 `*.prev`）；
- Ctrl-C 中断时 `last.pt` 是上一轮结束时的状态（退出码 130），可直接 `--resume`。

### 4.5 退出码

| 码 | 含义 | 处置 |
| --- | --- | --- |
| 0 | 正常结束（含 `--debug` 跑完、`--resume` 发现已跑满） | 看日志最后的 `best` 与产物 |
| 2 | 前置校验失败（splits / 清单 / cache 文件 / 模型与损失搭配 / **2.5D 通道数不自洽**） | 按带 `-` 的行修，见 4.6 |
| 3 | 训练出现 NaN/Inf 损失（或 `--debug` 的整卷推理形状不符） | `--set train.amp=off` 复现；或调小 `train.lr` |
| 4 | `--resume` 的 checkpoint 缺失 / 属于别的折 / 与当前配置不一致 | 按日志里的差异行处理，或删掉 `last.pt` 重跑 |
| 130 | 被 Ctrl-C 中断 | `--resume` 续跑 |

### 4.6 常见报错与贴回内容

| 输出 | 含义 / 该贴回什么 |
| --- | --- |
| `cache_manifest.cfg_hash=... 与当前配置算出的 ... 不一致` | 缓存来自别的预处理参数 → 重跑 `python scripts/preprocess.py`（只改 `preprocess` 节才会变；改 `loss`/`train`/`data.z_context` 等不会） |
| `读不到 cache 清单 ... 先跑 python scripts/fetch_manifest.py` | 远程没有清单 → 先跑 `fetch_manifest.py`（只读 cache、秒级） |
| `cache 清单与 data/splits.json 的病例集合不一致：只在清单里 [...]，只在划分里 [...]` | 缓存与划分不是同一批病人 → **贴回这一行**，先别训练 |
| `验证集里出现不含肿瘤的病例 [...]` | 划分被改坏（5 例仅肝脏只应进训练集）→ 贴回，并检查 `data/splits.json` |
| `cache/image/ 下缺 N 例：[...]` | cache 不完整 → 重跑 `preprocess.py` |
| `model.out_channels=1 时 loss.softmax 必须为 false` | 配置自相矛盾（单通道是 sigmoid 口径）→ 改回 `out_channels: 2` |
| `model.in_channels=... 与 data.z_context=... 不自洽：2.5D 窗口的通道数必须是 2×z_context+1 = ...` | 2.5D 配置改了一边忘了另一边 → 把 `data.z_context` 与 `model.in_channels` 配对（1↔3、0↔1、2↔5） |
| `batch 内 image 通道数是 N，与 data.z_context 推出的 M 不一致` | dataset 给的通道数与配的对不上（多半是 `z_context` 被 `--set` 改过）→ 贴回该行 |
| `image 里出现非有限值` / `image 值域 [...] 超出 [0,1]` | 增强或叠层把值推出值域 → **贴回该行 + 前后 20 行日志** |
| `--debug 出现非有限损失` / 训练 `epoch N 第 M 个 batch 的损失是 nan` | AMP 溢出或数据异常 → 先 `--set train.amp=off` 复现；贴回该行 + 前后 20 行日志 |
| `预测卷 (H,W,Z) 与 GT 卷 (H,W,Z) 形状不一致` | 轴序或 `pad_offset` 裁回出错 → **贴整段 traceback 与该行**（这是最需要立刻修的一类） |
| `tumor_channel=... 超出输出通道数 ...` / `target 形状 ... 与 logits ... 不匹配` | 模型输出通道与标签口径不一致 → 贴回该行 |
| `本轮验证**一个前景体素都没预测**（pred_voxels=0，峰值概率 ...）` | 就是第 3 轮首折的塌缩形态 → 贴回该行 + 本轮训练那行（看 `dice`/`ce`），见 `docs/preprocess_notes.md` 8.1 |
| `自检发现 N 个问题` + 带 `-` 的行（selfcheck_data） | 数据侧口径不对 → 把带 `-` 的行整段贴回 |
| `TensorBoard 不可用（...）` | 只是告警：`metrics.csv` 照常写；把告警贴回即可 |
| `TypeError: ... got an unexpected keyword argument` | 多半是库里参数名/版本不匹配 → **贴整段 traceback** |
| 退出码 2 + 一串 `-` 行 | 前置校验失败：**把带 `-` 的行整段贴回** |

只想快速验证训练链路（不写产物、1 例验证、3 个 iteration）：`python -m src.train --fold 0 --debug`。

### 4.7 第 4 轮起可调的口径（先别在首折乱调）

| 参数 | 作用 | 备注 |
| --- | --- | --- |
| `train.batch_size` | 每批样本数 | 第 3 轮定为 **16**；2.5D 后峰值显存以 `--debug` 实测为准 |
| `data.pos_ratio_train` | **训练侧每批阳性切片占比目标**（0.5 = 正负各半） | 每批阳性数 = `clamp(round(bs×该值), min_pos_per_batch, bs−1)`；配 `--set data.pos_ratio_train=0.3` 可回退到 nnU-Net 式的偏保守口径 |
| `data.min_pos_per_batch` | 每批阳性数下限 | 防止 batch 很小时比例取整成 0 |
| `data.max_pos_repeat` | 单个阳性层一轮最多重复几次 | 太小会让实际比例达不到 `pos_ratio_train`（日志会打印实际值）；太大有过拟合到少数层的风险 |
| `data.epoch_samples` | 一轮总槽位数预算 | `null` = 用数据集切片数（epoch 墙钟与旧口径一致） |
| `data.z_context` / `model.in_channels` | 2.5D 半径 / 输入通道数 | 必须配对：`0↔1`、`1↔3`、`2↔5`；两处都会校验 |
| `train.pos_ratio_target` | **已废弃**（旧口径的每批阳性上限） | 保留键只为兼容旧命令/旧 checkpoint 指纹；采样器构造时会提示"不再生效" |
| `train.lr` / `train.min_lr_ratio` | 初始学习率 / 余弦退火下界（`eta_min = lr × 该值`） | 第 4 轮**先不改**（塌缩不是 lr 的锅）；要看曲线再定 |
| `train.early_stop_patience` | 连续多少轮没有提升就停 | 默认 20 |
| `train.val_every` | 每多少轮验证一次 | 验证要跑 4 例整卷，`val_every=2` 可省一半时间 |
| `loss.include_background` | Dice 是否含背景类 | **必须 false**（true 存在全预测背景的平凡最优解，证据见 `docs/preprocess_notes.md` 7.2） |
| `loss.dice_positive_only` | Dice 是否只在含前景的样本上算 | 默认 **true**（第 4 轮新增）；`false` 可回退到"整批聚合" |
| `loss.ce_class_weights` | CE 的类别权重（如 `[0.2, 1.0]`） | **下一级杠杆**，默认 `null`；只在平衡采样后仍欠检出时启用 |
| `loss.lambda_dice` / `loss.lambda_ce` | 两项权重 | 默认 1.0 / 1.0 |
| `eval.threshold` / `eval.infer_batch_slices` | 概率→标签阈值 / 整卷推理批大小 | 训练期验证与评估共用同一口径 |

---

## 5. 后续步骤（评估，命令占位）

```bash
# 交付 src/postprocess.py + src/metrics.py + src/evaluate.py 后：
python -m src.evaluate --fold 0
python -m src.evaluate --all             # 汇总 5 折均值 ± 标准差
```

调试用的配置覆盖（不改文件）：

```bash
python -m src.train --fold 0 --debug --set train.batch_size=4 --set train.epochs=5
```

---

## 6. 运行产物与 git 边界

- **不入库且只存在于远程**：`cache/`（含 `cache_manifest.json`）、`reports/`、`runs/`，以及所有 `*.pt` / `*.nii*`。
  数据受保密协议约束不能下载，所以本地仓库看不到这些文件；后续编码所需的关键数字都记在
  `docs/preprocess_notes.md` 里。
- **入库**：`data/splits.json`、`data/exclude_cases.json`、`configs/*.yaml`、`src/*.py`、`scripts/*.py`、`docs/*.md`。
  注意：`data/splits.json` 由远程运行 `make_splits.py` 生成，**必须在远程提交并 push 回来**，本地才会有。
- 随机种子固定为 `train.seed`（默认 42），预处理/划分/训练/评估四处的口径见各自脚本头部注释。
