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
→ `clip(-1000,1000)` → `cache/image/<case>.nii.gz`（float16 HU）+ `cache/label/<case>.nii.gz`（uint8，仅 label 2）。

产出：

| 文件 | 说明 |
| --- | --- |
| `cache/image/*.nii.gz`、`cache/label/*.nii.gz` | 每例两个文件，共 25 例 |
| `cache/cache_manifest.json` | 训练启动时校验 cache 与配置是否匹配 |
| `reports/preprocess_stats.json` / `.md` | 逐 case 统计 + 尺寸分布汇总（**贴回这个 .md**） |

期望输出（日志尾部）：

```
cache 中 image=25，label=25
下一步：python scripts/make_splits.py
```

自检要点：
- 出现 `可用 case 数为 N，与 docs/data.md 的 25 例预期不一致` → 把该行与失败明细贴回。
- 出现 `仅肝脏病例实测 [...]，与 docs 预期 [32,34,38,41,47] 不一致` → 贴回，这不中断流程但会影响划分。
- 想先验证脚本能跑通（只处理 1 例、秒级返回）：

```bash
python scripts/preprocess.py --debug
```

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

- **不入库**：`cache/`、`reports/`、`runs/`（已加入 `.gitignore`），以及所有 `*.pt` / `*.nii.gz`。
- **入库**：`data/splits.json`、`data/exclude_cases.json`、`configs/*.yaml`、`src/*.py`、`scripts/*.py`、`docs/*.md`。
- 随机种子固定为 `train.seed`（默认 42），预处理/划分/训练/评估四处的口径见各自脚本头部注释。
