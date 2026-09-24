# 基础版（2D 肿瘤分割闭环）运行手册

本文档给出**逐条可粘贴**的远程执行命令与每步的期望输出。任务定义、数据事实与约束见
`docs/README.md` 与 `docs/data.md`；本文件只讲"怎么跑、跑出什么算对"。

约定：所有命令都在**仓库根目录**执行（远程 Linux 终端的当前目录就是项目根）。
跑完每一步请把终端输出贴回本地，尤其是第 1、2 步——后面 Dataset 的补边尺寸与 batch 形态要按实测尺寸定稿。

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
→ `clip(-1000,1000)` → `cache/image/<case>.nii.gz`（**uint16，已归一化**）+ `cache/label/<case>.nii.gz`（uint8，仅 label 2）。

**缓存约定（下游 dataset.py 依赖，经 `scripts/probe_axis.py` 在远程实测确认，不要改动）**：

- **轴序**：cache 文件是标准的 **nibabel `(x, y, z)`** 布局，**切片轴在最后一维**：
  `nib.load(p).dataobj` 形状为 `(nx, ny, nz)`，`a[:, :, k]` 是一层 `(ny, nx)` 切片。
  SimpleITK 读同一个文件得到的是它的转置 `(nz, ny, nx)`——两库互为转置，**不要混用**。
  下游按「数组索引 + spacing」使用缓存，不解释 affine。
- **影像数值**：`uint16`，把 HU 窗 `[-1000, 1000]` 线性映射到 `[0, 1]` 后按 `1/65535` 量化存储。
  **dataset.py 里 `image.float() / 65535.0` 即得到 [0,1] 输入**，1 个量化步长 ≈ 0.0305 HU。
  （SimpleITK 没有 float16 像素类型；如需存原始 HU 可把 `preprocess.image_out_dtype` 设为 `float32`。）

产出：

| 文件 | 说明 |
| --- | --- |
| `cache/image/*.nii.gz`、`cache/label/*.nii.gz` | 每例两个文件，共 25 例 |
| `cache/cache_manifest.json` | 训练启动时校验 cache 与配置是否匹配；**只存在于远程**（不入库） |
| `reports/preprocess_stats.json` / `.md` | 逐 case 统计 + 尺寸分布汇总（**贴回这个 .md**） |

> 清单过期但缓存有效时，不必重跑预处理：`python scripts/fetch_manifest.py` 会从磁盘重建清单（只读、秒级）。

期望输出（日志尾部）：

```
cache 中 image=25，label=25
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

## 3. 数据自检（第 2 轮交付，秒级到十几秒）

```bash
python -m src.selfcheck_data
```

做什么（只读 cache，不建模型、不训练）：核对 splits / 清单 / 预处理指纹 → 构建 train 与 val
两侧的 `CTSliceDataset`（都统一补边到 512×512）→ 实测 3 个 batch 的形态 → 打印一个病例的逐层
肿瘤体素数曲线 → 自检增强与采样器 → 把结果写到 `reports/selfcheck_data.json`。

期望输出（关键几行；train 侧 617/4469、val 侧 149/1513 是 fold 0 的实测值，
`559 个 batch`、`全阴性 batch=0`、逐种补边占比是按这些数字与当前配置**推算**的期望值，用来对照）：

```
划分：data/splits.json，5 折，每折 train=[21, 21, 21, 21, 21] / val=[4, 4, 4, 4, 4]
清单：25 例，预处理指纹 xxxxxxxx（当前配置 xxxxxxxx）一致
原始面内尺寸与清单逐例一致（21 例）：{'512x512': 17, '352x352': 3, ...}
补边口径：统一补到 512x512（center）；逐种原始尺寸的补边像素占比 {'342x342': '55.5%', ...}
整体阳性率：train 侧 0.1381（617/4469），val 侧 0.0985（149/1513）
训练侧可达阳性数/批：理想 2 / 本轮计划 2~2（617 个阳性层摊到 559 个 batch）；计划前 12 批 = [2, 2, ...]
batch 1：image (8, 1, 512, 512) / label (8, 512, 512)；病例 [...]
        值域 [0.0000, 1.0000]；label 取值 [0, 1]；**含肿瘤切片 2/8 = 0.250**（配置目标 0.30）；
        补边样本 0/8（补边像素均值 0.0%）
        取该 batch 耗时 0.9 s（含首次冷读）
8 个 train batch 的空间维只有 1 种：[(512, 512)]（统一补边后应恒为 1 种）
采样器自检：每轮 559 个 batch（预期下限 559；切片数/batch_size ≈ 558.6）；同 epoch 可复现=True；
        相邻 epoch 不同=True；batch 大小越界=0 种；阳性层抽到 617 次/去重 617 个（共 617）；
        覆盖切片 4469/4469；全阴性 batch=0
自检通过：21 例 / 4469 层切片的形态、值域、标签、补边、增强与采样比例均符合约定。
下一步：第 3 轮 python -m src.train --fold 0 --debug
```

判读要点：

- **batch 形状恒为 `(8, 1, 512, 512)` / `(8, 512, 512)`**：所有样本都补边到 `data.target_hw`，
  所以 `torch.stack` 永远合法。**不再有分桶/桶键/size_buckets 相关日志**——如果还看到
  `桶 512x512`、`各桶可达阳性数` 这类输出，说明远程拉到的还是旧代码。
- **`含肿瘤切片 2/8 = 0.250`（而不是 0.30）是对的**：`pos_ratio_target=0.30` 经
  `round(8×0.30)=2` 取整后，每个 batch 恒为 2 个肿瘤切片，批次平均比例就是 0.25 =
  原始 12.81% 的约 1.95 倍过采样。想更接近 0.30 就把 `train.batch_size` 提到 10/20，或把
  `train.pos_ratio_target` 改成 0.25（两者等价，改一个即可）。
- **每批阳性数按「采样器自己的计划区间」判**（日志里的 `本轮计划 lo~hi`），不是拿 0.30 当阈值：
  阳性层稀少时每批放不满是「一轮覆盖每一层 + 阳性均摊」的取舍。
- **默认配置下不再出现全阴性 batch**（旧的分桶实现在 fold 0 上有 117 个，现在按 fold 0 的
  4469 层 / 617 个阳性层算：559 个 batch × 2 阳 = 1118 个槽位装得下 617，日志里应为
  `全阴性 batch=0`）。出现非 0 说明阳性均摊或补位逻辑坏了。
- **一轮 batch 数 = max(切片覆盖、阳性槽位、阴性槽位)**，可能大于 `ceil(切片数/batch_size)`；
  自检会打印 `预期下限` 并等号比对，不要按「切片数/8」心算。
- **验证侧不判阳性数**：val loader 顺序读、不做过采样，某个 batch 落在「病例前几层无肿瘤」
  （如 case 33）是正常现象。自检若报「阳性切片 0 超出预算」这类错，先确认远程代码是不是旧的
  （旧版把训练侧预算套到 val batch 上，就是那条误报）。
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

## 4. 后续步骤（代码分轮交付，命令占位）

```bash
# 第 3 轮交付 src/unet.py + src/train.py 后：
python -m src.train --fold 0 --debug     # 只跑数个 iteration，打印输入 shape 与显存占用
python -m src.train --fold 0             # 正式训练

# 第 4 轮交付 src/evaluate.py 后：
python -m src.evaluate --fold 0
python -m src.evaluate --all             # 汇总 5 折均值 ± 标准差
```

调试用的配置覆盖（不改文件）：

```bash
python -m src.train --fold 0 --debug --set train.batch_size=4 --set train.epochs=5
```

---

## 5. 运行产物与 git 边界

- **不入库且只存在于远程**：`cache/`（含 `cache_manifest.json`）、`reports/`、`runs/`，以及所有 `*.pt` / `*.nii.gz`。
  数据受保密协议约束不能下载，所以本地仓库看不到这些文件；后续编码所需的关键数字都记在
  `docs/preprocess_notes.md` 里。
- **入库**：`data/splits.json`、`data/exclude_cases.json`、`configs/*.yaml`、`src/*.py`、`scripts/*.py`、`docs/*.md`。
  注意：`data/splits.json` 由远程运行 `make_splits.py` 生成，**必须在远程提交并 push 回来**，本地才会有。
- 随机种子固定为 `train.seed`（默认 42），预处理/划分/训练/评估四处的口径见各自脚本头部注释。
