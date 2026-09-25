"""预处理：把 data/ 下的原始 volume/segmentation 转成 1mm RAS 的 cache，并统计逐 case 事实。

整体功能：剔除 data/exclude_cases.json 中的病例（48-52 几何错位）→ 把地板值 padding 夹到 -1000 →
        统一 RAS → 重采样到 1x1x1mm（影像线性 / 掩膜最近邻）→ 按全局窗 clip(-1000,1000) →
        影像按缓存口径转成归一化 uint16、掩膜只保留 label 2 存 uint8 到 cache/，同时产出可粘贴的统计报告。
前后接口：上游是 data/volume-<id>.nii + segmentation-<id>.nii 与 configs/default.yaml；
        下游给 scripts/make_splits.py 提供 reports/preprocess_stats.json、给 src/dataset.py 提供 cache/
        （cache 里影像是 uint16 的 [0,1] 归一化值，除以 65535 即为 [0,1]）。
用法：在仓库根目录执行 ``python scripts/preprocess.py``；快速自检 ``python scripts/preprocess.py --debug``。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:  # 允许从仓库根直接 python scripts/preprocess.py 运行
    from src.utils import (
        cache_fingerprint,
        format_kv_table,
        load_config,
        load_json,
        rel_to_root,
        resolve_path,
        save_json,
        save_report,
        setup_logger,
    )
except ModuleNotFoundError:  # pragma: no cover - 兜底：把仓库根塞进 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import (  # type: ignore
        cache_fingerprint,
        format_kv_table,
        load_config,
        load_json,
        rel_to_root,
        resolve_path,
        save_json,
        save_report,
        setup_logger,
    )

LOGGER = setup_logger("preprocess")

VOLUME_RE = re.compile(r"^volume[-_](\d+)$")
SEG_RE = re.compile(r"^segmentation[-_](\d+)$")
NII_SUFFIXES = (".nii.gz", ".nii")

# 与 config 中 preprocess 节同名的默认值；config 里的值会覆盖它，因此这里只是兜底
DEFAULTS = {
    "target_spacing": [1.0, 1.0, 1.0],
    "orientation": "RAS",
    "floor_margin": 1.0,
    "hu_clip": [-1000.0, 1000.0],
    "floor_clamp_value": -1000.0,
    "image_out_dtype": "uint16",
}


# --------------------------------------------------------------------------------------
# 文件发现
# --------------------------------------------------------------------------------------

def strip_nii_suffix(name: str) -> str:
    """去掉 .nii / .nii.gz 后缀，返回文件名主干。"""
    low = name.lower()
    for suffix in NII_SUFFIXES:
        if low.endswith(suffix):
            return name[: -len(suffix)]
    return name


def discover_cases(data_dir: Path) -> tuple[dict, dict]:
    """扫描数据目录，返回 (volumes, segmentations)：case_id(int) -> 绝对路径。"""
    volumes: dict = {}
    segs: dict = {}
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}（用 --data-dir 指定）")

    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        stem = strip_nii_suffix(path.name)
        if not stem:
            continue
        mv = VOLUME_RE.match(stem)
        if mv:
            volumes.setdefault(int(mv.group(1)), path)
            continue
        ms = SEG_RE.match(stem)
        if ms:
            segs.setdefault(int(ms.group(1)), path)
    return volumes, segs


def select_cases(volumes: dict, segs: dict, excluded: set) -> list:
    """取 volume/segmentation 成对且未被剔除的 case id，按数值升序返回。"""
    paired = sorted(set(volumes) & set(segs))
    missing_seg = sorted(set(volumes) - set(segs))
    missing_vol = sorted(set(segs) - set(volumes))
    if missing_seg:
        LOGGER.warning("有 volume 但缺 segmentation 的 case（已跳过）：%s", missing_seg)
    if missing_vol:
        LOGGER.warning("有 segmentation 但缺 volume 的 case（已跳过）：%s", missing_vol)
    selected = [c for c in paired if c not in excluded]
    dropped = [c for c in paired if c in excluded]
    if dropped:
        LOGGER.info("按排除清单剔除 %d 例：%s", len(dropped), dropped)
    return selected


# --------------------------------------------------------------------------------------
# 单个 case 的几何处理
# --------------------------------------------------------------------------------------

def load_as_float(volume_path: Path):
    """读取影像为 float32 的 SimpleITK 图像（保持原 spacing/origin/direction）。"""
    import SimpleITK as sitk

    img = sitk.ReadImage(str(volume_path))
    if img.GetPixelID() != sitk.sitkFloat32:
        img = sitk.Cast(img, sitk.sitkFloat32)
    return img


def load_seg_labels(seg_path: Path):
    """读取原始掩膜为 uint8 标签图（保留 0/1/2 原值），用于统计原始标签分布。"""
    import SimpleITK as sitk

    seg = sitk.ReadImage(str(seg_path))
    if seg.GetPixelID() != sitk.sitkUInt8:
        seg = sitk.Cast(seg, sitk.sitkUInt8)
    return seg


def mask_from_label(seg_labels, label: int = 2):
    """把标签图按 ``== label`` 二值化成 uint8 的 0/1 掩膜图。"""
    import SimpleITK as sitk

    return sitk.Cast(sitk.BinaryThreshold(seg_labels, lowerThreshold=label, upperThreshold=label,
                                          insideValue=1, outsideValue=0),
                     sitk.sitkUInt8)


def floor_stats(arr: np.ndarray, margin: float) -> tuple[float, int]:
    """返回 (地板值, 地板体素数)：与最小值相差不超过 margin 的体素视为卷外 padding。"""
    vmin = float(arr.min())
    n_floor = int(np.count_nonzero(arr <= vmin + float(margin)))
    return vmin, n_floor


def apply_floor_clamp(img, floor_value: float, margin: float, clamp_to: float):
    """把地板体素（<= floor_value + margin）夹到 clamp_to，其余保持不变。"""
    import SimpleITK as sitk

    mask = sitk.BinaryThreshold(img, lowerThreshold=float(floor_value) + float(margin),
                                upperThreshold=float(np.finfo(np.float32).max),
                                insideValue=1, outsideValue=0)
    return sitk.Cast(sitk.Mask(img, mask, outsideValue=float(clamp_to)), sitk.sitkFloat32)


def resample_ras_1mm(img, target_spacing, interpolator: int, orientation: str = "RAS"):
    """先把方位统一到 ``orientation``（默认 RAS），再重采样到目标 spacing（影像线性、掩膜最近邻）。

    输出网格按「输入物理范围 / target_spacing」计算；因为已经轴对齐，direction 是置换矩阵，
    该窗口就是紧致的，不会出现倾斜包围盒导致的尺寸爆炸。这里显式给定 size 与 origin，
    不让 SITK 自行推导输出网格。
    """
    import SimpleITK as sitk

    oriented = sitk.DICOMOrient(img, str(orientation))
    original_spacing = oriented.GetSpacing()
    original_size = oriented.GetSize()
    new_size = [int(np.ceil(original_size[i] * abs(original_spacing[i]) / abs(float(target_spacing[i]))))
                for i in range(3)]
    new_size = [max(1, n) for n in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([float(s) for s in target_spacing])
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(oriented.GetDirection())
    resampler.SetOutputOrigin(oriented.GetOrigin())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(oriented), oriented, new_size


def crop_origin_to_nonzero(img, arr: np.ndarray):
    """把输出 origin 前移到前景体素最小角的物理偏移处，影像与掩膜用同一次偏移。

    动机：本流程下游只按「数组索引 + spacing」使用缓存，不解释 affine；把两卷（image 与 label）
    的 origin 统一挪到同一处，是为了让它们的世界坐标差恒为常数偏移，避免以后有人误以为两卷
    origin 不同、需要额外对齐。前景为空（仅肝脏病例的 label）时不做偏移。
    """
    import SimpleITK as sitk

    idx = np.nonzero(arr > 0)
    if idx[0].size == 0:
        return img, (0, 0, 0)
    ijk_min = [int(idx[0].min()), int(idx[1].min()), int(idx[2].min())]
    direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    offset = direction @ (direction.T @ (spacing * np.asarray(ijk_min, dtype=np.float64)))
    new_origin = np.asarray(img.GetOrigin(), dtype=np.float64) + offset
    out = sitk.Image(img)
    out.SetOrigin([float(x) for x in new_origin])
    return out, tuple(ijk_min)


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def process_case(case_id: int, vol_path: Path, seg_path: Path, cache_dir: Path, pre: dict,
                 verbose: bool = False) -> dict:
    """处理一个 case：地板夹取 → RAS → 1mm 重采样 → 全局窗 → 落盘 cache，返回该 case 的统计。

    verbose=True 时在每一步几何操作前打印标记：SITK 是 C++ 实现，异常会以 SIGSEGV（段错误）
    直接杀掉进程而不抛 Python 异常，留下"最后一条标记"是定位崩溃点的唯一手段。
    """
    import SimpleITK as sitk

    def step(msg: str) -> None:
        if verbose:
            LOGGER.info("    [case %d] %s", case_id, msg)

    target_spacing = [float(s) for s in pre["target_spacing"]]
    hu_lo, hu_hi = (float(x) for x in pre["hu_clip"])
    margin = float(pre["floor_margin"])
    clamp_to = float(pre["floor_clamp_value"])
    orientation = str(pre["orientation"])
    out_dtype = str(pre["image_out_dtype"]).lower()
    if out_dtype not in ("float32", "uint16"):
        # SimpleITK 没有 float16 像素类型；这里显式拦掉错误配置而不是留到 WriteImage 才炸
        raise ValueError(f"preprocess.image_out_dtype 只支持 'float32' 或 'uint16'，收到 {out_dtype!r}"
                         f"（注意 SimpleITK 没有 float16 像素类型）")

    rec: dict = {"case": case_id, "status": "OK", "error": ""}
    step(f"读取影像 {vol_path.name}")
    vol_img = load_as_float(vol_path)
    step(f"读取掩膜 {seg_path.name}，size={vol_img.GetSize()}")
    seg_labels = load_seg_labels(seg_path)          # 原始标签图（0/1/2），用于统计与取 label 2
    seg_img = mask_from_label(seg_labels, label=2)  # 只用 label 2 的二值掩膜

    rec["original_shape"] = tuple(int(s) for s in vol_img.GetSize())
    rec["original_spacing"] = tuple(round(float(s), 4) for s in vol_img.GetSpacing())
    rec["original_orientation"] = "".join(sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(vol_img.GetDirection()))

    # 1) 地板值 padding：在原始整数域上判定，避免插值把地板值摊开。
    #    注意：GetArrayViewFromImage 返回的是指向 SITK 图像内部缓冲区的视图，
    #    若把 ReadImage 的结果写成临时对象（sitk.GetArrayViewFromImage(sitk.ReadImage(...))），
    #    临时图像会被立刻回收，视图随即悬空 —— 读它的内容就是段错误。
    #    这里统一用 GetArrayFromImage（返回独立拷贝），统计完立刻 del，内存占用可控。
    step("统计地板值")
    raw_arr = sitk.GetArrayFromImage(vol_img)  # vol_img 仍在作用域内，且此处为拷贝
    floor_value, n_floor = floor_stats(raw_arr, margin)
    rec["floor_value"] = floor_value
    rec["floor_voxels"] = n_floor
    rec["floor_fraction"] = float(n_floor / max(1, raw_arr.size))
    del raw_arr

    step("统计掩膜原始标签分布")
    seg_arr_raw = sitk.GetArrayFromImage(seg_labels)
    labels, counts = np.unique(seg_arr_raw, return_counts=True)
    rec["mask_label_counts_raw"] = {int(v): int(c) for v, c in zip(labels.tolist(), counts.tolist())}
    n_label2_raw = int(rec["mask_label_counts_raw"].get(2, 0))
    del seg_arr_raw

    step(f"地板夹取（floor={floor_value}，比例 {rec['floor_fraction']:.4f}）")
    img_clamped = apply_floor_clamp(vol_img, floor_value, margin, clamp_to)

    # 2) 统一方位 + 3) 重采样到 target_spacing（影像线性、掩膜最近邻）
    step(f"影像 {orientation} + 重采样到 {target_spacing}")
    img_rs, _, _ = resample_ras_1mm(img_clamped, target_spacing, sitk.sitkLinear, orientation)
    step(f"影像重采样完成，输出 size={img_rs.GetSize()}")
    step(f"掩膜 {orientation} + 重采样（最近邻）")
    seg_rs, _, _ = resample_ras_1mm(seg_img, target_spacing, sitk.sitkNearestNeighbor, orientation)
    step("掩膜重采样完成")

    # 4) 全局窗：先记录窗内饱和比例，再 clip 到 [hu_lo, hu_hi]
    step("全局窗 clip")
    img_arr = sitk.GetArrayFromImage(img_rs).astype(np.float32, copy=False)
    rec["clip_low_fraction"] = float(np.count_nonzero(img_arr < hu_lo) / max(1, img_arr.size))
    rec["clip_high_fraction"] = float(np.count_nonzero(img_arr > hu_hi) / max(1, img_arr.size))
    np.clip(img_arr, hu_lo, hu_hi, out=img_arr)

    step("掩膜二值化（seg_img 已是 0/1，取 >0）并转 uint8")
    seg_arr = np.asarray(sitk.GetArrayFromImage(seg_rs))
    seg_arr = (seg_arr > 0).astype(np.uint8)
    if seg_arr.size and int(seg_arr.max()) > 1:
        raise RuntimeError(f"case {case_id} 的二值掩膜出现 >1 的取值 {np.unique(seg_arr).tolist()}，"
                           f"说明 label 2 的筛选环节口径不一致")
    n_label2_rs = int(np.count_nonzero(seg_arr))
    # 结构自检：原始有 label 2 的病例，重采样后不应该是空的（空说明朝向/网格处理有 bug）
    if n_label2_raw > 0 and n_label2_rs == 0:
        raise RuntimeError(f"case {case_id} 原始 label 2 有 {n_label2_raw} 个体素，"
                           f"但 1mm 重采样后前景为 0 —— 朝向/RAS 重定向处理有误")
    if n_label2_raw > 0:
        ratio = n_label2_rs / max(1, n_label2_raw)
        rec["label2_voxels_resampled"] = n_label2_rs
        rec["label2_voxel_ratio"] = round(float(ratio), 6)
        if not (0.2 <= ratio <= 5.0):
            LOGGER.warning("case %d 的 label 2 体素数在重采样前后变化异常：%d -> %d（比例 %.3f），"
                           "请检查该例的 spacing", case_id, n_label2_raw, n_label2_rs, ratio)

    # 5) 落盘：数组保持 SimpleITK 的 (nz, ny, nx) 内存顺序，经 GetImageFromArray 后
    #    几何信息与 img_rs/seg_rs 完全匹配，因此直接把重采样后的几何照抄过来即可。
    #    不要在这里对数组做 transpose：那会让图像尺寸与待拷贝的几何不一致，
    #    CopyInformation 会直接抛错；而下游只需要「索引 + spacing」，转置没有任何收益。
    mask_sitk = sitk.GetImageFromArray(seg_arr.astype(np.uint8))
    mask_sitk.CopyInformation(seg_rs)
    seg_rs = None  # 掩膜几何已拷进 mask_sitk，其余大对象尽早释放以降低峰值内存
    seg_img = None
    seg_labels = None

    if out_dtype == "float32":
        # 直接存 clip 后的 HU，归一化留给 dataset.py
        image_to_write = sitk.GetImageFromArray(img_arr.astype(np.float32))
    else:
        # 存「已归一化」的 uint16：窗内 HU 线性映射到 [0,1] 后按 1/65535 量化。
        # 这样 cache 与训练口径完全一致（dataset.py 只需 /65535 还原到 [0,1]），
        # 且 uint16 是 SITK/NIfTI 原生支持的类型（SITK 没有 float16 像素类型）。
        norm_arr = (img_arr.astype(np.float32) - hu_lo) / max(1e-6, (hu_hi - hu_lo))
        np.clip(norm_arr, 0.0, 1.0, out=norm_arr)
        image_to_write = sitk.GetImageFromArray(np.rint(norm_arr * 65535.0).astype(np.uint16))
        rec["image_quant_step_hu"] = round(float((hu_hi - hu_lo) / 65535.0), 6)
    image_to_write.CopyInformation(img_rs)

    # 影像与掩膜使用同一套 origin（都前移到前景最小角），下游重建整卷时按索引对齐即可，
    # 不必再关心 affine：两卷的世界坐标差是常数偏移。
    image_with_origin, shift_ijk = crop_origin_to_nonzero(image_to_write, seg_arr)
    mask_with_origin, _ = crop_origin_to_nonzero(mask_sitk, seg_arr)
    rec["origin_shift_ijk"] = list(shift_ijk)

    # 6) 统计：以「像素数 + spacing」为准换算 mm3，不依赖 affine。
    #    轴序实测结论（scripts/probe_axis.py 在远程确认，下游 dataset.py 依赖它）：
    #      * SimpleITK 内存数组是 (nz, ny, nx)（本函数内 seg_arr / img_arr 就是这个顺序）；
    #      * 写成 .nii.gz 后，用 nibabel 读回得到的是 (nx, ny, nz) —— 两个库互为转置；
    #      * 因此「切片轴在 cache 文件的最后一维」，nib 读回时 a[:, :, k] 才是一层 (ny, nx) 切片。
    #    统计放在写盘之前：统计若失败就不该留下看似成功的缓存文件。
    step("统计切片与连通域")
    voxel_mm3 = float(np.prod([float(s) for s in img_rs.GetSpacing()]))
    n_tumor_voxels = int(np.count_nonzero(seg_arr))
    # 每层前景体素数：对 (y, x) 两个轴求和，长度才是 nz
    per_slice = seg_arr.sum(axis=(1, 2))
    tumor_slices = int(np.count_nonzero(per_slice >= 1))
    tiny_slices = int(np.count_nonzero((per_slice >= 1) & (per_slice < 10)))
    if tumor_slices > int(img_arr.shape[0]):
        raise RuntimeError(f"case {case_id} 的含肿瘤切片数 {tumor_slices} 超过总切片数 "
                           f"{int(img_arr.shape[0])}，per-slice 统计的轴用错了")
    if int(per_slice.size) != int(img_arr.shape[0]):
        raise RuntimeError(f"case {case_id} 的 per-slice 数组长度 {int(per_slice.size)} "
                           f"与切片数 {int(img_arr.shape[0])} 不一致")

    # 三种形状全部记录下来，避免下游再猜轴序：
    #   sitk_shape_zyx：本函数用的 SimpleITK 内存布局
    #   nib_shape_xyz ：cache 文件（.nii.gz）用 nibabel 读回时的布局，切片轴在最后
    #   inplane_hw    ：面内尺寸 (ny, nx)；nz 为切片数
    rec["new_shape_zyx"] = tuple(int(s) for s in img_arr.shape)
    rec["new_shape_xyz"] = tuple(int(s) for s in img_arr.shape[::-1])
    rec["inplane_hw"] = (int(img_arr.shape[1]), int(img_arr.shape[2]))
    rec["new_spacing"] = tuple(round(float(s), 4) for s in img_rs.GetSpacing())
    rec["voxel_mm3"] = round(voxel_mm3, 6)
    rec["n_slices"] = int(img_arr.shape[0])
    rec["tumor_slices"] = tumor_slices
    rec["tumor_slice_ratio"] = round(float(tumor_slices / max(1, rec["n_slices"])), 6)
    rec["tiny_tumor_slices"] = tiny_slices
    rec["tumor_voxels"] = n_tumor_voxels
    rec["tumor_volume_mm3"] = round(float(n_tumor_voxels * voxel_mm3), 4)
    rec["has_tumor"] = bool(n_tumor_voxels > 0)

    if n_tumor_voxels > 0:
        # 6 邻域连通域。用 scipy.ndimage.label 而不是 sitk.ConnectedComponent：
        # 后者的 fullyConnected 关键字在 SimpleITK 各版本间命名不一致（2.5.6 上直接 TypeError），
        # scipy 是环境既有依赖、行为稳定，且 6 邻域口径与评估阶段的后处理一致。
        from scipy import ndimage

        structure = ndimage.generate_binary_structure(seg_arr.ndim, 1)  # 1 = 6 邻域
        labeled, n_components = ndimage.label(seg_arr, structure=structure)
        component_sizes = np.bincount(labeled.ravel())[1:]  # 跳过 label 0（背景）
        sizes = [int(x) for x in component_sizes.tolist() if x > 0]
        rec["n_components"] = int(n_components)
        rec["min_component_voxels"] = int(min(sizes)) if sizes else 0
        rec["min_component_mm3"] = round(float(min(sizes) * voxel_mm3), 4) if sizes else 0.0
        rec["component_sizes_voxels"] = sorted(sizes, reverse=True)[:10]
    else:
        rec["n_components"] = 0
        rec["min_component_voxels"] = 0
        rec["min_component_mm3"] = 0.0
        rec["component_sizes_voxels"] = []

    step(f"写 cache（image dtype={image_with_origin.GetPixelIDTypeAsString()}，"
         f"数组轴序 (nz,ny,nx)，nibabel 读回为 (nx,ny,nz)）")
    image_dir = cache_dir / "image"
    label_dir = cache_dir / "label"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_writer = sitk.ImageFileWriter()
    image_writer.SetFileName(str(image_dir / f"{case_id}.nii.gz"))
    image_writer.SetUseCompression(True)
    image_writer.Execute(image_with_origin)

    label_writer = sitk.ImageFileWriter()
    label_writer.SetFileName(str(label_dir / f"{case_id}.nii.gz"))
    label_writer.SetUseCompression(True)
    label_writer.Execute(sitk.Cast(mask_with_origin, sitk.sitkUInt8))

    # 7) 回读自校验：从刚写的文件独立复算一遍关键量，确认落盘内容与内存统计一致。
    #    目的是让"体积/切片数这类统计量"不再依赖人眼判断是否合理 —— 写错轴、写错量都会当场暴露。
    written = sitk.ReadImage(str(label_dir / f"{case_id}.nii.gz"))
    written_arr = np.asarray(sitk.GetArrayFromImage(written))
    written_spacing = [float(s) for s in written.GetSpacing()]
    written_voxel_mm3 = float(np.prod(written_spacing))
    written_tumor_voxels = int(np.count_nonzero(written_arr))
    if tuple(written_arr.shape) != rec["new_shape_zyx"]:
        raise RuntimeError(f"case {case_id}：回读 label 的 shape {tuple(written_arr.shape)} "
                           f"与记录的 {rec['new_shape_zyx']} 不一致")
    if abs(written_voxel_mm3 - rec["voxel_mm3"]) > 1e-6:
        raise RuntimeError(f"case {case_id}：回读 voxel_mm3={written_voxel_mm3} "
                           f"与记录的 {rec['voxel_mm3']} 不一致")
    if written_tumor_voxels != rec["tumor_voxels"]:
        raise RuntimeError(f"case {case_id}：回读肿瘤体素数 {written_tumor_voxels} "
                           f"与记录的 {rec['tumor_voxels']} 不一致")
    written_volume = round(float(written_tumor_voxels * written_voxel_mm3), 4)
    if abs(written_volume - rec["tumor_volume_mm3"]) > 1e-3:
        raise RuntimeError(f"case {case_id}：回读肿瘤体积 {written_volume} "
                           f"与记录的 {rec['tumor_volume_mm3']} 不一致")
    written_slices = int(np.count_nonzero(written_arr.sum(axis=(1, 2)) >= 1))
    if written_slices != rec["tumor_slices"]:
        raise RuntimeError(f"case {case_id}：回读含肿瘤切片数 {written_slices} "
                           f"与记录的 {rec['tumor_slices']} 不一致")

    # 8) 再用 nibabel 读一次，确认「下游 dataset.py 实际会看到什么」：
    #    nib 的数组是 SITK 的转置，切片轴在最后一维；这一步把该约定钉进每例的运行时校验，
    #    避免以后改动写盘逻辑时又悄悄换掉轴序。
    import nibabel as nib

    nib_arr = np.asanyarray(nib.load(str(label_dir / f"{case_id}.nii.gz")).dataobj)
    if tuple(nib_arr.shape) != rec["new_shape_xyz"]:
        raise RuntimeError(f"case {case_id}：nibabel 读回 shape {tuple(nib_arr.shape)} "
                           f"与记录的 {rec['new_shape_xyz']} 不一致（轴序约定被破坏）")
    nib_slices = int(np.count_nonzero(nib_arr.sum(axis=(0, 1)) >= 1))
    if nib_slices != rec["tumor_slices"]:
        raise RuntimeError(f"case {case_id}：nibabel 口径下含肿瘤切片数 {nib_slices} "
                           f"与记录的 {rec['tumor_slices']} 不一致")
    if int(nib_arr.sum()) != rec["tumor_voxels"]:
        raise RuntimeError(f"case {case_id}：nibabel 读回肿瘤体素数 {int(nib_arr.sum())} "
                           f"与记录的 {rec['tumor_voxels']} 不一致")

    return rec


def build_aggregate(records: list, pre: dict, pad_to_multiple: int = 16) -> dict:
    """汇总所有 case：面内尺寸分布、分桶情况、肿瘤切片占比、体积量级。

    尺寸一律用「面内 (ny, nx)」口径：SimpleITK 数组是 (nz, ny, nx)，切片数 nz 各例不同、
    面内尺寸才是决定分桶与显存的量。
    """
    ok = [r for r in records if r["status"] == "OK"]
    failed = [r for r in records if r["status"] != "OK"]
    inplane = Counter((int(r["new_shape_zyx"][1]), int(r["new_shape_zyx"][2])) for r in ok)
    n_slices_values = Counter(int(r["n_slices"]) for r in ok)
    spacings = Counter(tuple(r["new_spacing"]) for r in ok)
    orig_spacings = Counter(tuple(r["original_spacing"]) for r in ok)
    orig_orientations = Counter(r["original_orientation"] for r in ok)

    max_hw = [0, 0]
    for shape in inplane:
        max_hw[0] = max(max_hw[0], int(shape[0]))
        max_hw[1] = max(max_hw[1], int(shape[1]))

    mult = int(pad_to_multiple)
    buckets = Counter((int(np.ceil(shape[0] / mult) * mult), int(np.ceil(shape[1] / mult) * mult))
                      for shape in inplane for _ in range(inplane[shape]))

    tumor_slices = sum(int(r["tumor_slices"]) for r in ok)
    total_slices = sum(int(r["n_slices"]) for r in ok)
    inconsistent = [r["case"] for r in ok if int(r["tumor_slices"]) > int(r["n_slices"])]
    if inconsistent:
        # 兜底：per-slice 统计若用错轴，含肿瘤切片数会超过总切片数，必须显式暴露而不是写进报告
        raise RuntimeError(f"以下 case 的含肿瘤切片数超过总切片数：{inconsistent}，"
                           f"说明 per-slice 统计的轴用错了")
    with_tumor = [r["case"] for r in ok if r["has_tumor"]]
    liver_only = [r["case"] for r in ok if not r["has_tumor"]]
    volumes = [float(r["tumor_volume_mm3"]) for r in ok if r["has_tumor"]]
    floor_values = Counter(r["floor_value"] for r in ok)

    return {
        "axis_convention": ("cache 文件用 nibabel 读回时形状为 (nx, ny, nz)（等价 (W, H, Z)），"
                            "切片轴在最后一维：a[:, :, k] 是一层 (ny, nx) 切片；"
                            "SimpleITK 读回时是它的转置 (nz, ny, nx)。两库互为转置（probe_axis.py 实测）。"),
        "n_cases_ok": len(ok),
        "n_cases_failed": len(failed),
        "failed_cases": [r["case"] for r in failed],
        "cases": [r["case"] for r in ok],
        "cases_with_tumor": with_tumor,
        "cases_liver_only": liver_only,
        "n_cases_with_tumor": len(with_tumor),
        "n_cases_liver_only": len(liver_only),
        "inplane_shapes": {"x".join(str(v) for v in k): v for k, v in sorted(inplane.items())},
        "unique_inplane_shapes": len(inplane),
        "n_slices_distribution": {str(k): v for k, v in sorted(n_slices_values.items())},
        "n_slices_min": min(n_slices_values) if n_slices_values else 0,
        "n_slices_max": max(n_slices_values) if n_slices_values else 0,
        "max_hw": max_hw,
        "pad_to_multiple": mult,
        "size_buckets": {"x".join(str(v) for v in k): v for k, v in sorted(buckets.items())},
        "n_size_buckets": len(buckets),
        "new_spacing": {"x".join(str(v) for v in k): v for k, v in sorted(spacings.items())},
        "original_spacings": {"x".join(str(v) for v in k): v for k, v in sorted(orig_spacings.items())},
        "original_orientations": dict(orig_orientations),
        "floor_values": {str(k): v for k, v in sorted(floor_values.items())},
        "total_slices": total_slices,
        "tumor_slices": tumor_slices,
        "tumor_slice_ratio_overall": round(float(tumor_slices / max(1, total_slices)), 6),
        "tumor_volume_mm3_min": round(min(volumes), 4) if volumes else None,
        "tumor_volume_mm3_median": round(float(np.median(volumes)), 4) if volumes else None,
        "tumor_volume_mm3_max": round(max(volumes), 4) if volumes else None,
        "target_spacing": [float(s) for s in pre["target_spacing"]],
        "hu_clip": [float(x) for x in pre["hu_clip"]],
    }


def build_markdown(report: dict) -> str:
    """把统计报告渲染成可直接粘贴的 markdown。"""
    agg = report["aggregate"]
    columns = ["case", "original_shape", "original_spacing", "original_orientation", "floor_value",
               "floor_fraction", "new_shape_zyx", "new_shape_xyz", "n_slices", "tumor_slices",
               "tumor_slice_ratio", "label2_voxel_ratio", "n_components", "min_component_voxels",
               "min_component_mm3", "tumor_volume_mm3", "clip_low_fraction"]
    lines: list = []
    lines.append("# 预处理统计报告（scripts/preprocess.py）")
    lines.append("")
    lines.append(f"- 配置指纹 cfg_hash：`{report.get('cfg_hash', 'NA')}`")
    lines.append(f"- 排除病例：{report.get('excluded_cases')}")
    lines.append(f"- 成功 {agg['n_cases_ok']} 例，失败 {agg['n_cases_failed']} 例 {agg['failed_cases']}")
    lines.append(f"- 含肿瘤 {agg['n_cases_with_tumor']} 例：{agg['cases_with_tumor']}")
    lines.append(f"- 仅肝脏 {agg['n_cases_liver_only']} 例：{agg['cases_liver_only']}")
    lines.append(f"- 目标 spacing：{agg['target_spacing']}；实际达到：{agg['new_spacing']}")
    lines.append(f"- 全局窗 hu_clip：{agg['hu_clip']}（地板值分布：{agg['floor_values']}）")
    lines.append("")
    lines.append(f"- 轴序约定：{agg['axis_convention']}")
    lines.append("- 逐 case 的 `new_shape_zyx` 是 SimpleITK 内存布局，"
                 "`new_shape_xyz` 是 nibabel 读回 cache 文件时的布局（下游 dataset.py 用后者）。")
    lines.append("")
    lines.append("## 1. 面内尺寸分布（决定 batch、分桶与显存）")
    lines.append("")
    lines.append(f"- 唯一面内尺寸数：{agg['unique_inplane_shapes']}；面内 (ny,nx) -> case 数：{agg['inplane_shapes']}")
    lines.append(f"- 最大面内 (ny,nx)：{agg['max_hw']}；按 pad_to_multiple={agg['pad_to_multiple']} 分桶："
                 f"{agg['size_buckets']}（共 {agg['n_size_buckets']} 个桶）")
    lines.append(f"- 切片数 nz：min={agg['n_slices_min']} max={agg['n_slices_max']}；"
                 f"分布 -> case 数：{agg['n_slices_distribution']}")
    lines.append("")
    lines.append("## 2. 切片级前景占比（采样器依据）")
    lines.append("")
    lines.append(f"- 总切片 {agg['total_slices']}，含肿瘤切片 {agg['tumor_slices']}，"
                 f"占比 {agg['tumor_slice_ratio_overall']:.4f}")
    lines.append(f"- 肿瘤体积 mm3：min={agg['tumor_volume_mm3_min']} "
                 f"median={agg['tumor_volume_mm3_median']} max={agg['tumor_volume_mm3_max']}")
    lines.append("")
    lines.append("## 3. 逐 case 明细")
    lines.append("")
    lines.append(format_kv_table(report["cases"], columns))
    lines.append("")
    lines.append("## 4. 判读提示")
    lines.append("")
    lines.append("- 面内尺寸不唯一是正常的：面内 spacing 0.666-0.9766 重采样到 1mm 后会得到不同面内尺寸；"
                 "下游不做任何 resize，只按桶 padding 到 16 的整数倍。")
    lines.append("- `n_components` > 1 或 `min_component_voxels` 很小，说明存在孤立小病灶，"
                 "评估阶段的 3D 后处理阈值（eval.min_lesion_mm3）要据此判断。")
    lines.append("- `tumor_slices` 为 0 的 case 就是仅肝脏病例，只进训练集、不进验证集。")
    lines.append("- `label2_voxel_ratio` = 重采样前后 label 2 体素数之比，正常应在 0.2-5.0 之间；"
                 "越界说明该例 spacing 异常，已在运行日志里告警。")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CT 预处理：统一 spacing/朝向、剪裁地板值、落盘 cache 并统计")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件，默认 configs/default.yaml")
    parser.add_argument("--data-dir", default=None, help="覆盖 data/ 位置")
    parser.add_argument("--cache-dir", default=None, help="覆盖 cache/ 位置")
    parser.add_argument("--debug", action="store_true", help="只处理第一个 case，打印详细中间量后退出")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个 case（0 = 全部），用于崩溃定位")
    parser.add_argument("--verbose", action="store_true",
                        help="每个 case 打印逐步标记（SITK 段错误时靠最后一条标记定位）")
    parser.add_argument("--set", dest="overrides", action="append", default=None,
                        help="覆盖配置项，如 --set preprocess.target_spacing=[1,1,1]（可多次）")
    args = parser.parse_args(argv)

    # SITK 崩溃（SIGSEGV）不会抛 Python 异常，开 faulthandler 至少能看到 C 层栈
    import faulthandler

    faulthandler.enable()

    cfg = load_config(args.config, args.overrides)
    pre = dict(DEFAULTS)
    pre.update(cfg.get("preprocess", {}) or {})
    model_cfg = cfg.get("model", {}) or {}
    pad_to_multiple = int(model_cfg.get("pad_to_multiple", 16))
    # 早失败：配置里写错类型时不要等到第 1 例处理到最后一步才炸
    if str(pre["image_out_dtype"]).lower() not in ("float32", "uint16"):
        LOGGER.error("preprocess.image_out_dtype 只支持 'float32' 或 'uint16'，收到 %r"
                     "（注意 SimpleITK 没有 float16 像素类型）", pre["image_out_dtype"])
        return 2
    paths = cfg.get("paths", {}) or {}

    data_dir = resolve_path(args.data_dir or paths.get("data", "data"))
    cache_dir = resolve_path(args.cache_dir or paths.get("cache", "cache"))
    exclude_path = resolve_path(paths.get("exclude", "data/exclude_cases.json"))

    exclude_doc = load_json(exclude_path, default={}) or {}
    excluded = {int(item["case"]) for item in exclude_doc.get("exclude", [])}
    liver_only_expected = set(int(x) for x in exclude_doc.get("liver_only_expected", []))

    volumes, segs = discover_cases(data_dir)
    LOGGER.info("数据目录 %s：volume=%d，segmentation=%d", rel_to_root(data_dir), len(volumes), len(segs))

    cases = select_cases(volumes, segs, excluded)
    if args.debug:
        cases = cases[:1]
        LOGGER.info("[debug] 只处理 1 个 case：%s", cases)
    elif args.limit > 0:
        cases = cases[: args.limit]
        LOGGER.info("[limit] 只处理前 %d 个 case：%s", args.limit, cases)
    if not cases:
        LOGGER.error("没有任何可处理的 case，请检查 --data-dir 与排除清单。")
        return 2

    # 缓存指纹只由 preprocess 节决定（src.utils.cache_fingerprint）：
    # 第 3 轮新增的 loss 节与缓存无关，不应让清单失效
    cfg_hash = cache_fingerprint(cfg)
    records: list = []
    for i, case_id in enumerate(cases, 1):
        LOGGER.info("[%d/%d] case %d ...", i, len(cases), case_id)
        try:
            rec = process_case(case_id, volumes[case_id], segs[case_id], cache_dir, pre,
                               verbose=bool(args.verbose or args.debug))
        except Exception as exc:  # noqa: BLE001 - 单 case 失败不中断整体
            LOGGER.exception("case %d 处理失败：%s", case_id, exc)
            rec = {"case": case_id, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        records.append(rec)
        if rec["status"] == "OK":
            shape = tuple(int(s) for s in rec["new_shape_zyx"])
            LOGGER.info("    shape(z,y,x) %s -> %s，含肿瘤切片 %d/%d，肿瘤体积 %.1f mm3",
                        tuple(rec["original_shape"]), shape,
                        rec["tumor_slices"], rec["n_slices"], rec["tumor_volume_mm3"])

    import SimpleITK as sitk  # 版本号写进报告供复现
    import nibabel as nib

    aggregate = build_aggregate(records, pre, pad_to_multiple=pad_to_multiple)
    report = {
        "cfg_hash": cfg_hash,
        "created_from_config": rel_to_root(resolve_path(args.config)),
        "data_dir": rel_to_root(data_dir),
        "cache_dir": rel_to_root(cache_dir),
        "excluded_cases": sorted(excluded),
        "preprocess": {k: pre[k] for k in DEFAULTS},
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "SimpleITK": sitk.Version_VersionString(), "nibabel": nib.__version__},
        "aggregate": aggregate,
        "cases": records,
    }

    stats_path = resolve_path(paths.get("preprocess_stats", "reports/preprocess_stats.json"))
    json_path, md_path = save_report(report, stats_path, md_builder=build_markdown)

    manifest = {
        "cfg_hash": cfg_hash,
        "cache_dir": rel_to_root(cache_dir),
        "preprocess": {k: pre[k] for k in DEFAULTS},
        "axis_convention": aggregate["axis_convention"],
        "cases": [
            {
                "case": int(r["case"]),
                "sitk_shape_zyx": [int(s) for s in r["new_shape_zyx"]],
                "nib_shape_xyz": [int(s) for s in r["new_shape_xyz"]],
                "inplane_hw": [int(s) for s in r["inplane_hw"]],
                "spacing": list(r["new_spacing"]),
                "n_slices": int(r["n_slices"]),
                "tumor_slices": int(r["tumor_slices"]),
                "tumor_voxels": int(r["tumor_voxels"]),
                "tumor_volume_mm3": float(r["tumor_volume_mm3"]),
                "has_tumor": bool(r["has_tumor"]),
                "n_components": int(r["n_components"]),
                "min_component_mm3": float(r["min_component_mm3"]),
            }
            for r in records if r["status"] == "OK"
        ],
        "aggregate": {k: aggregate[k] for k in
                      ("n_cases_ok", "n_cases_with_tumor", "n_cases_liver_only", "cases_with_tumor",
                       "cases_liver_only", "inplane_shapes", "max_hw", "size_buckets", "n_size_buckets",
                       "n_slices_min", "n_slices_max", "total_slices", "tumor_slices",
                       "tumor_slice_ratio_overall", "new_spacing")},
    }
    manifest_path = save_json(manifest, paths.get("cache_manifest", "cache/cache_manifest.json"))

    # ---- 自检 ----
    expected_ids = {str(int(c)) for c in cases}
    present_img = {p.name.split(".")[0] for p in (cache_dir / "image").glob("*.nii.gz")} \
        if (cache_dir / "image").is_dir() else set()
    present_lab = {p.name.split(".")[0] for p in (cache_dir / "label").glob("*.nii.gz")} \
        if (cache_dir / "label").is_dir() else set()
    n_written, n_label = len(present_img), len(present_lab)
    LOGGER.info("cache 中 image=%d，label=%d（本次期望 %d 例）", n_written, n_label, len(expected_ids))

    ran_without_failure = aggregate["n_cases_failed"] == 0
    next_step = "python scripts/check_cache.py"
    if args.debug or args.limit > 0:
        LOGGER.info("[部分运行] 未做完整性判定；请用不带 --debug/--limit 的命令跑完整 25 例后再依赖 cache。")
    elif not ran_without_failure:
        LOGGER.error("有 %d 例处理失败（%s），cache 不完整，请修复后重跑；"
                     "不要用当前 cache 进入训练。", aggregate["n_cases_failed"], aggregate["failed_cases"])
        return 3
    else:
        if present_img != expected_ids or present_lab != expected_ids:
            LOGGER.error("cache 文件集合与预期 case 不一致：缺 %s，多 %s",
                         sorted(expected_ids - present_img), sorted(present_img - expected_ids))
            return 3
        if aggregate["n_cases_ok"] != 25:
            LOGGER.warning("可用 case 数为 %d，与 docs/data.md 的 25 例预期不一致，请核对排除清单与数据目录。",
                           aggregate["n_cases_ok"])
        got_liver_only = set(aggregate["cases_liver_only"])
        if liver_only_expected and got_liver_only != liver_only_expected:
            LOGGER.warning("仅肝脏病例实测 %s，与 docs 预期 %s 不一致（不中断，请在报告中确认）。",
                           sorted(got_liver_only), sorted(liver_only_expected))
        LOGGER.info("完整性自检通过：%d 例全部成功，image/label 文件集合与预期一致。", len(expected_ids))
        next_step = "python scripts/check_cache.py（再跑 python scripts/make_splits.py）"

    if args.debug:
        LOGGER.info("[debug] 单个 case 的完整统计：\n%s", json.dumps(records[0], ensure_ascii=False, indent=2))
        LOGGER.info("[debug] 未做 25 例与仅肝病例的预期核对。")

    LOGGER.info("统计报告：%s%s", rel_to_root(json_path), f" 与 {rel_to_root(md_path)}" if md_path else "")
    LOGGER.info("缓存清单：%s", rel_to_root(manifest_path))
    LOGGER.info("下一步：%s", next_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
