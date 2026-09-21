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

- **`data/splits.json`**：由远程 `python scripts/make_splits.py` 生成，本地仓库当前**没有**这个文件。
  写 `src/dataset.py` / `src/train.py` 之前必须确认它已在本地（远程 `git add data/splits.json && git push`），
  否则只能在本地按上面的 fold 归属硬编码兜底——不推荐。
