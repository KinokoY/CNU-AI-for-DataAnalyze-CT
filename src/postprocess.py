"""3D 后处理：连通域标记、按体积删小块、病灶体积分档（第 5 轮新增）。

整体功能（全部只吃 ``(H, W, Z)`` 的 numpy 数组，不读盘、不依赖 torch）：
    1. ``to_mm3`` —— 体素数 ↔ mm³ 的唯一换算口径（``体素数 × spacing 三向乘积``）；
    2. ``label_lesions`` —— 6 邻域连通域标记，返回逐病灶体积（mm³）与包围盒；
    3. ``lesion_stats`` —— 只统计、不重算标记（``label_lesions`` 的轻量版，供假阳性统计复用）；
    4. ``remove_small_lesions`` —— 删掉小于 ``min_lesion_mm3`` 的孤立连通块（默认 50 mm³）；
    5. ``lesion_size_bin`` —— 把病灶体积分档（报告要按病灶大小分层看，别只看均值）。

为什么用 ``scipy.ndimage.label`` 而不是 ``SimpleITK.ConnectedComponent``：
    *  **轴序口径唯一**：本项目的数组是 nibabel 轴序 ``(nx, ny, nz)``，切片轴在**最后一维**
        （见 `docs/preprocess_notes.md` 第一节）；SimpleITK 读回是它的转置，两库混用会静默错位。
        ``scipy.ndimage`` 直接作用在 numpy 数组上，不引入第二套坐标系。
    *  **与预处理统计同源**：第 1 轮数肿瘤体积用的就是 ``scipy.ndimage``（6 邻域），
        评估侧沿用同一口径，体素数才对得上（``docs/data.md`` 的肿瘤体积与 ``data/splits.json``
        的 ``tumor_volume_mm3`` 就是那套口径）。

每个病例的 spacing 在预处理后恒为 ``(1,1,1)``（``docs/preprocess_notes.md`` 第一节），
此时 1 体素 = 1 mm³；但换算一律走 ``to_mm3`` 而不写死数字，避免将来改重采样口径时
「体素数被当成 mm³」这类静默错误（``docs/data.md`` 第 1 节：原始 spacing 有 13 种，
体素数 ≠ mm³，只有预处理之后才恰好相等）。

依赖：numpy / scipy.ndimage。
前后接口：上游是 ``src.infer.predict_volume`` 的二值预测卷与 ``load_label_volume`` 的 GT 卷；
    下游是 ``src.metrics``（病灶级检出 / 假阳性统计）与 ``src.evaluate``（报告）。
用法：``cleaned, n_removed, removed_mm3 = remove_small_lesions(pred_bin, 50.0, spacing)``。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage

#: 6 邻域结构元（与第 1 轮预处理的连通域统计口径一致，**不要改成 18/26 邻域**）
STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)

#: 病灶体积分档边界（mm³，下界含、上界不含）。口径来源：本数据集肿瘤体积跨 3 个数量级
#: （0.65–435 cm³，见 `docs/data.md` 第 5 节），fold 0 的 val 里 59 = 652 mm³、57 = 4046 mm³
#: 长期为 0 ⇒ 报告必须按体积分层给检出率，只看均值会掩盖小病灶的系统性失败。
DEFAULT_SIZE_BINS: tuple = (
    (0.0, 100.0, "微型 <100"),
    (100.0, 1000.0, "小型 100–1k"),
    (1000.0, 10000.0, "中型 1k–10k"),
    (10000.0, 100000.0, "大型 10k–100k"),
    (100000.0, float("inf"), "巨块 ≥100k"),
)

#: 二值输入的判定阈值（``> 0.5``）：预测卷是 uint8 {0,1}，概率卷也能直接用同一口径
BINARY_THRESHOLD = 0.5


# --------------------------------------------------------------------------------------
# 体积换算与分档
# --------------------------------------------------------------------------------------

def to_mm3(voxel_count, spacing, unit: bool = False):
    """体素数 → mm³：``体素数 × spacing[0] × spacing[1] × spacing[2]``。

    ``spacing`` 是**单个体素的物理尺寸**（mm），顺序与数组轴序一致（nibabel 的 ``(nx, ny, nz)``）。
    预处理后全是 ``(1,1,1)``，此时返回值等于体素数；写成函数是为了让「单位」这件事只有一处定义
    （``docs/data.md`` 第 1 节：原始数据 spacing 有 13 种，体素数 ≠ mm³，必须按各自 spacing 换算）。

    ``unit=True`` 时只返回「1 体素 = 多少 mm³」这个系数（``tissue_volume`` 那种批量换算用）。
    """
    values = [float(v) for v in (spacing if spacing is not None else (1.0, 1.0, 1.0))]
    if len(values) != 3:
        raise ValueError(f"spacing 必须是 3 个值（x,y,z 各向），收到 {spacing!r}")
    if any(v <= 0 for v in values):
        raise ValueError(f"spacing 必须全为正数，收到 {spacing!r}")
    factor = values[0] * values[1] * values[2]
    if unit:
        return factor
    return float(np.asarray(voxel_count, dtype=np.float64) * factor)


def voxel_spacing(cfg: dict | None) -> tuple:
    """从配置里取体素物理尺寸（mm）。

    本项目只有一个事实来源：``preprocess.target_spacing``（默认 ``[1,1,1]``，预处理已把全部病例
    重采样到该 spacing）。缓存 ``cache_manifest.json`` 里逐例的 ``spacing`` 也是同一个值；
    这里不去读清单，是为了让 ``postprocess`` / ``metrics`` 保持「纯函数 + 显式入参」，
    不依赖任何文件。
    """
    raw = ((cfg or {}).get("preprocess") or {}).get("target_spacing") or [1.0, 1.0, 1.0]
    values = tuple(float(v) for v in raw)
    if len(values) != 3:
        raise ValueError(f"preprocess.target_spacing 必须是 3 个值，收到 {raw!r}")
    return values


def lesion_size_bin(volume_mm3: float, bins: Sequence = DEFAULT_SIZE_BINS) -> str:
    """把病灶体积映射到分档名（``bins`` 是 ``(下界, 上界, 名称)`` 序列，下界含、上界不含）。

    超出最后一档上界（默认 ``inf``）时返回最后一档的名字；小于第一档下界时返回第一档的名字，
    保证任何正体积都有归属（``0`` 与负值返回 ``"无效"``，便于在报告里一眼看出异常输入）。
    """
    value = float(volume_mm3)
    if not np.isfinite(value) or value <= 0:
        return "无效"
    for low, high, name in bins:
        if float(low) <= value < float(high):
            return str(name)
    return str(bins[-1][2])


# --------------------------------------------------------------------------------------
# 连通域标记
# --------------------------------------------------------------------------------------

def _as_binary3d(binary, name: str = "binary3d") -> np.ndarray:
    """把输入规整成 ``(H,W,Z)`` 的 bool 数组，并对明显不对的形状直接报错。"""
    array = np.asarray(binary)
    if array.ndim != 3:
        raise ValueError(f"{name} 必须是 3D 数组（本项目是 (H,W,Z) 的 nibabel 轴序），"
                         f"收到 shape={array.shape}")
    if array.size and not np.isfinite(np.asarray(array, dtype=np.float32)).all():
        raise ValueError(f"{name} 里出现 NaN/Inf，先排查上游（概率图异常？）再谈连通域")
    return array > BINARY_THRESHOLD


def lesion_stats(labels: np.ndarray, spacing: Sequence[float]) -> tuple:
    """统计已经标好的连通域，返回 ``(n_lesions, per_lesion_mm3, slices)``。

    ``labels`` 必须是 ``scipy.ndimage.label`` 输出的整型标记图（0 = 背景）。
    ``per_lesion_mm3`` 按**标记值顺序**给出（下标 i 对应标记 i+1，即第 i 个病灶）。
    ``slices`` 是逐病灶的 ``{"label", "voxels", "volume_mm3", "bbox"}``，``bbox`` 是
    ``(z0, z1, y0, y1, x0, x1)`` 的**半开区间**（用于快速定位病灶，报告里只报 z 范围）。

    ``spacing`` 是单个体素的物理尺寸；换算走 ``to_mm3``，不写「1 体素 = 1 mm³」。
    """
    marks = np.asarray(labels)
    if marks.ndim != 3:
        raise ValueError(f"labels 必须是 3D 标记图，收到 shape={marks.shape}")
    factor = to_mm3(1, spacing, unit=True)
    n_lesions = int(marks.max()) if marks.size else 0
    if n_lesions <= 0:
        return 0, [], []
    # 一次 bincount 拿到全部病灶的体素数（比逐病灶 sum 快一个量级，整卷 512×512×488 上很明显）
    counts = np.bincount(marks.ravel(), minlength=n_lesions + 1)
    voxels = [int(v) for v in counts[1:]]
    volumes = [round(float(v) * factor, 3) for v in voxels]
    objects = ndimage.find_objects(marks)
    slices: list = []
    for index, (count, volume) in enumerate(zip(voxels, volumes)):
        bbox = None
        if index < len(objects) and objects[index] is not None:
            z_slice, y_slice, x_slice = objects[index]
            bbox = [int(z_slice.start), int(z_slice.stop),
                    int(y_slice.start), int(y_slice.stop),
                    int(x_slice.start), int(x_slice.stop)]
        slices.append({"label": int(index + 1), "voxels": int(count),
                       "volume_mm3": float(volume), "bbox": bbox})
    return int(n_lesions), volumes, slices


def label_lesions(binary3d, spacing, return_slices: bool = False) -> tuple:
    """6 邻域连通域标记，返回 ``(int_labels, n_lesions, per_lesion_mm3)``。

    ``int_labels`` 是与入参同形状的 ``int32`` 标记图（0 = 背景，1..n 为病灶编号，编号顺序
    按数组扫描顺序，只保证确定性、不保证与体积大小相关）。``per_lesion_mm3`` 下标 i 对应标记 i+1。

    ``return_slices=True`` 时返回四元组，第 4 项是逐病灶的包围盒明细（见 ``lesion_stats``）。
    ``spacing`` 必填：本数据集预处理后恒为 ``(1,1,1)``，但**体素数≠mm³ 这件事必须由调用方显式表态**
    （原始数据 spacing 有 13 种，见 ``docs/data.md`` 第 1 节）。
    """
    binary = _as_binary3d(binary3d)
    marks, n_lesions = ndimage.label(binary, structure=STRUCTURE_6)
    marks = marks.astype(np.int32, copy=False)
    n_lesions, volumes, slices = lesion_stats(marks, spacing)
    if return_slices:
        return marks, int(n_lesions), volumes, slices
    return marks, int(n_lesions), volumes


def remove_small_lesions(binary3d, min_voxels: float, spacing, return_slices: bool = False) -> tuple:
    """删掉体积 **小于** ``min_voxels`` mm³ 的孤立连通块，返回 ``(cleaned, n_removed, removed_mm3)``。

    参数口径（``docs/baseline.md`` 与 ``configs/default.yaml`` 的 ``eval.min_lesion_mm3``）：
        * ``min_voxels`` 的单位是 **mm³**（尽管参数名叫 voxels，保留它是为了与第 4 轮定的接口签名
          ``remove_small_lesions(binary3d, min_voxels, spacing)`` 一致）；判据是严格小于 ⇒
          体积恰好等于阈值的病灶**保留**；
        * ``spacing`` 必填（理由同 ``label_lesions``）；
        * 返回的 ``removed_mm3`` 是被删掉病灶的体积合计，报告里用它标「后处理删掉了多少体素」。

    ``return_slices=True`` 时返回四元组，第 4 项是**被删掉的那些病灶**的明细（逐病灶体积 + 包围盒），
    便于在报告里列出「最小的几个被删块」。
    """
    binary = _as_binary3d(binary3d)
    threshold = float(min_voxels)
    if threshold < 0:
        raise ValueError(f"min_voxels（mm³）不能为负，收到 {min_voxels!r}")
    if threshold == 0:
        # 阈值为 0 时没有任何病灶会被删（严格小于 0 不可能成立）——直接短路，省一次标记
        if return_slices:
            return binary.astype(np.uint8), 0, 0.0, []
        return binary.astype(np.uint8), 0, 0.0

    marks, n_lesions, volumes, slices = label_lesions(binary, spacing, return_slices=True)
    if n_lesions == 0:
        if return_slices:
            return marks.astype(np.uint8), 0, 0.0, []
        return marks.astype(np.uint8), 0, 0.0

    # 标记号 0 是背景，**必须一起置 0**：查表写错方向时背景会变成 1，
    # 干净的卷求和会得到「全部体素数」而不是病灶体素数（本地逻辑自证抓到过一次）。
    keep = np.ones(n_lesions + 1, dtype=bool)
    keep[0] = False
    removed_records: list = []
    removed_mm3 = 0.0
    for index, (volume, record) in enumerate(zip(volumes, slices), start=1):
        if float(volume) < threshold:
            keep[index] = False
            removed_records.append(record)
            removed_mm3 += float(volume)
    removed_mm3 = round(removed_mm3, 3)
    # **用 LUT + np.where，不要写 ``keep[marks]``**：标记图是 int32、取值范围 0..n_lesions，
    # 直接 `keep[marks]` 会把它当"沿第 0 维的布尔掩码"用（长度不匹配 → IndexError）。
    # 正确写法是先按标记号查表得到与体素同形状的布尔掩码，再取 0/1。本地逻辑自证抓到过一次。
    cleaned = np.ascontiguousarray(np.where(keep[marks], 1, 0).astype(np.uint8))
    n_removed = int(len(removed_records))
    if return_slices:
        return cleaned, n_removed, removed_mm3, removed_records
    return cleaned, n_removed, removed_mm3


# --------------------------------------------------------------------------------------
# 假阳性统计（仅供 src.metrics 复用，单独放在这里是因为它只跟连通域有关）
# --------------------------------------------------------------------------------------

def unmatched_lesion_stats(pred_bin, gt_bin, spacing) -> dict:
    """统计**预测里与 GT 完全没有重叠**的病灶（= 假阳性病灶）。

    返回 ``{"n_lesions", "volume_mm3", "largest_mm3", "largest_bbox", "volumes_mm3"}``。
    GT 全空（5 例仅肝脏病人）时每个预测病灶都算假阳性 —— 这正是那 5 例要单独报告的东西。
    """
    pred = _as_binary3d(pred_bin, "pred_bin")
    gt = _as_binary3d(gt_bin, "gt_bin")
    marks, n_lesions, volumes, slices = label_lesions(pred, spacing, return_slices=True)
    if n_lesions == 0:
        return {"n_lesions": 0, "volume_mm3": 0.0, "largest_mm3": 0.0,
                "largest_bbox": None, "volumes_mm3": []}
    # 逐病灶的「是否碰到 GT」：对 (marks, gt) 做一次联合 bincount，比逐病灶切片快得多
    hit = np.bincount(marks[gt].ravel(), minlength=n_lesions + 1) > 0
    keep = [index for index in range(1, n_lesions + 1) if not bool(hit[index])]
    if not keep:
        return {"n_lesions": 0, "volume_mm3": 0.0, "largest_mm3": 0.0,
                "largest_bbox": None, "volumes_mm3": []}
    kept_volumes = [float(volumes[index - 1]) for index in keep]
    largest_index = int(np.argmax(kept_volumes))
    return {
        "n_lesions": int(len(keep)),
        "volume_mm3": round(float(sum(kept_volumes)), 3),
        "largest_mm3": round(float(kept_volumes[largest_index]), 3),
        "largest_bbox": slices[keep[largest_index] - 1]["bbox"],
        "volumes_mm3": [round(v, 3) for v in kept_volumes],
    }


if __name__ == "__main__":  # pragma: no cover - 直接运行本文件时给出正确入口
    raise SystemExit("本模块是库，不是脚本；整卷评估由 python -m src.evaluate 调用"
                     "（后处理口径见 docs/preprocess_notes.md 第三节）。")
