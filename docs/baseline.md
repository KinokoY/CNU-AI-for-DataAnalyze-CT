# 基础版（2D 肿瘤分割闭环）运行手册

本文档给出**逐条可粘贴**的远程执行命令与每步的期望输出。任务定义、数据事实与约束见
`docs/README.md` 与 `docs/data.md`；本文件只讲"怎么跑、跑出什么算对"。

约定：所有命令都在**仓库根目录**执行（远程 Linux 终端的当前目录就是项目根）。
跑完每一步请把终端输出贴回本地，尤其是第 1、2 步——后面 Dataset 的 batch 与分桶要按实测尺寸定稿。

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

## 3. 后续步骤（代码分轮交付，命令占位）

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

## 4. 运行产物与 git 边界

- **不入库且只存在于远程**：`cache/`（含 `cache_manifest.json`）、`reports/`、`runs/`，以及所有 `*.pt` / `*.nii.gz`。
  数据受保密协议约束不能下载，所以本地仓库看不到这些文件；后续编码所需的关键数字都记在
  `docs/preprocess_notes.md` 里。
- **入库**：`data/splits.json`、`data/exclude_cases.json`、`configs/*.yaml`、`src/*.py`、`scripts/*.py`、`docs/*.md`。
  注意：`data/splits.json` 由远程运行 `make_splits.py` 生成，**必须在远程提交并 push 回来**，本地才会有。
- 随机种子固定为 `train.seed`（默认 42），预处理/划分/训练/评估四处的口径见各自脚本头部注释。
