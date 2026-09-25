"""从已有 cache 本地生成 cache_manifest.json（不重跑预处理）。

整体功能：直接扫描 cache/image 与 cache/label，统计每例的形状、spacing、含肿瘤切片数、肿瘤体积与
        连通域个数，生成与 scripts/preprocess.py 同构的清单；用于"缓存数据本身正确、只是清单口径
        过期（例如轴序修正后）"时刷新清单，省掉几分钟的重新预处理。
前后接口：上游是 cache/ 下已有的 nii.gz；下游与 preprocess.py 产出的清单同格式，
        供 scripts/check_cache.py 与 src/train.py 校验；**不修改任何 nii.gz**。
用法：仓库根目录执行 ``python scripts/fetch_manifest.py``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from src.utils import cache_fingerprint, load_config, rel_to_root, resolve_path, save_json, setup_logger
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import (  # type: ignore
        cache_fingerprint,
        load_config,
        rel_to_root,
        resolve_path,
        save_json,
        setup_logger,
    )

LOGGER = setup_logger("fetch_manifest")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="从已有 cache 本地生成 cache_manifest.json")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件")
    parser.add_argument("--cache-dir", default=None, help="覆盖 cache/ 位置")
    parser.add_argument("--set", dest="overrides", action="append", default=None, help="覆盖配置项")
    args = parser.parse_args(argv)

    import nibabel as nib
    from scipy import ndimage

    cfg = load_config(args.config, args.overrides)
    paths = cfg.get("paths", {}) or {}
    pre = cfg.get("preprocess", {}) or {}
    model = cfg.get("model", {}) or {}
    cache_dir = resolve_path(args.cache_dir or paths.get("cache", "cache"))
    image_dir, label_dir = cache_dir / "image", cache_dir / "label"
    if not label_dir.is_dir() or not image_dir.is_dir():
        LOGGER.error("cache 目录不完整：%s", rel_to_root(cache_dir))
        return 2

    # 与 preprocess.py / selfcheck_data.py / train.py 用同一套口径指纹（只依赖 preprocess 节）
    cfg_hash = cache_fingerprint(cfg)

    cases: list = []
    for path in sorted(label_dir.glob("*.nii.gz"), key=lambda p: int(p.name.split(".")[0])):
        case = int(path.name.split(".")[0])
        lab = np.asanyarray(nib.load(str(path)).dataobj)
        img_img = nib.load(str(image_dir / f"{case}.nii.gz"))
        spacing = [round(float(z), 4) for z in img_img.header.get_zooms()[:3]]
        voxel_mm3 = float(np.prod(spacing))
        # 轴序实测（probe_axis.py）：nib 布局 (nx, ny, nz)，切片轴在最后一维
        per_slice = lab.sum(axis=(0, 1))
        tumor_voxels = int((lab > 0).sum())
        tumor_slices = int((per_slice > 0).sum())
        if tumor_slices > int(lab.shape[2]):
            raise RuntimeError(f"case {case}：含肿瘤切片数 {tumor_slices} 超过总切片数 {lab.shape[2]}")

        if tumor_voxels > 0:
            structure = ndimage.generate_binary_structure(3, 1)
            labeled, n_components = ndimage.label(lab, structure=structure)
            sizes = [int(x) for x in np.bincount(labeled.ravel())[1:].tolist() if x > 0]
        else:
            n_components, sizes = 0, []

        cases.append({
            "case": case,
            "sitk_shape_zyx": [int(s) for s in lab.shape[::-1]],
            "nib_shape_xyz": [int(s) for s in lab.shape],
            "inplane_hw": [int(lab.shape[1]), int(lab.shape[0])],
            "spacing": spacing,
            "n_slices": int(lab.shape[2]),
            "tumor_slices": tumor_slices,
            "tumor_voxels": tumor_voxels,
            "tumor_volume_mm3": round(float(tumor_voxels * voxel_mm3), 4),
            "has_tumor": bool(tumor_voxels > 0),
            "n_components": int(n_components),
            "min_component_mm3": round(float(min(sizes) * voxel_mm3), 4) if sizes else 0.0,
            "source": "fetch_manifest.py",
        })

    with_tumor = [c["case"] for c in cases if c["has_tumor"]]
    liver_only = [c["case"] for c in cases if not c["has_tumor"]]
    inplane: dict = {}
    for c in cases:
        key = f"{c['inplane_hw'][0]}x{c['inplane_hw'][1]}"
        inplane[key] = inplane.get(key, 0) + 1
    mult = int(model.get("pad_to_multiple", 16))
    buckets: dict = {}
    for c in cases:
        h = int(np.ceil(c["inplane_hw"][0] / mult) * mult)
        w = int(np.ceil(c["inplane_hw"][1] / mult) * mult)
        key = f"{h}x{w}"
        buckets[key] = buckets.get(key, 0) + 1
    total_slices = sum(c["n_slices"] for c in cases)
    tumor_slices = sum(c["tumor_slices"] for c in cases)

    manifest = {
        "cfg_hash": cfg_hash,
        "cache_dir": rel_to_root(cache_dir),
        "preprocess": {k: pre.get(k) for k in
                       ("target_spacing", "orientation", "hu_clip", "floor_clamp_value", "image_out_dtype")},
        "axis_convention": ("cache 文件用 nibabel 读回时形状为 (nx, ny, nz)（等价 (W, H, Z)），"
                            "切片轴在最后一维：a[:, :, k] 是一层 (ny, nx) 切片；"
                            "SimpleITK 读回时为它的转置 (nz, ny, nx)。两库互为转置。"),
        "cases": cases,
        "aggregate": {
            "n_cases_ok": len(cases),
            "n_cases_with_tumor": len(with_tumor),
            "n_cases_liver_only": len(liver_only),
            "cases_with_tumor": with_tumor,
            "cases_liver_only": liver_only,
            "inplane_shapes": inplane,
            "max_hw": [max(c["inplane_hw"][0] for c in cases), max(c["inplane_hw"][1] for c in cases)],
            "size_buckets": buckets,
            "n_size_buckets": len(buckets),
            "n_slices_min": min(c["n_slices"] for c in cases),
            "n_slices_max": max(c["n_slices"] for c in cases),
            "total_slices": total_slices,
            "tumor_slices": tumor_slices,
            "tumor_slice_ratio_overall": round(float(tumor_slices / max(1, total_slices)), 6),
        },
        "notes": ["本清单由 scripts/fetch_manifest.py 从已有 cache 生成，未重新处理影像；"
                  "若 cache 内容或预处理配置变更，请重跑 scripts/preprocess.py"],
    }

    out = save_json(manifest, paths.get("cache_manifest", "cache/cache_manifest.json"))
    LOGGER.info("扫描 %d 例：含肿瘤 %d，仅肝脏 %d", len(cases), len(with_tumor), len(liver_only))
    LOGGER.info("面内尺寸分布：%s", inplane)
    LOGGER.info("按 pad_to_multiple=%d 分桶：%s（共 %d 个桶）", mult, buckets, len(buckets))
    LOGGER.info("切片数范围：%d - %d；含肿瘤切片 %d/%d = %.4f",
                manifest["aggregate"]["n_slices_min"], manifest["aggregate"]["n_slices_max"],
                tumor_slices, total_slices, manifest["aggregate"]["tumor_slice_ratio_overall"])
    LOGGER.info("清单已写入：%s", rel_to_root(out))
    LOGGER.info("下一步：python scripts/check_cache.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
