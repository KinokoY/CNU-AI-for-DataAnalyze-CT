"""缓存体检：读回 cache/ 下的全部产物，逐项核对几何、标签与数值口径是否符合约定。

整体功能：对每一例检查 image/label 是否成对、shape 是否一致、spacing 是否为约定的 1mm 三项相同、
        label 取值是否只含 {0,1}、有无肿瘤、影像数值是否落在约定范围（uint16 时为 [0,65535]），
        并把实测结果与 cache_manifest.json 逐例比对，不一致就报错并以非零码退出。
前后接口：上游是 scripts/preprocess.py（或 fetch_manifest.py）产出的 cache/ 与 cache_manifest.json
        （cache 文件为 ``<case>.nii``，未压缩优先，兼容旧的 ``.nii.gz``）；
        下游给 src/dataset.py / src/train.py 一个"缓存可信"的前置保证。
用法：仓库根目录执行 ``python scripts/check_cache.py``；只要汇总不要逐例明细就加 ``--quiet``。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from src.utils import (
        cache_cases,
        cache_file,
        load_config,
        load_json,
        rel_to_root,
        resolve_path,
        save_json,
        setup_logger,
        warn_compressed_cache,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import (  # type: ignore
        cache_cases,
        cache_file,
        load_config,
        load_json,
        rel_to_root,
        resolve_path,
        save_json,
        setup_logger,
        warn_compressed_cache,
    )

LOGGER = setup_logger("check_cache")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="cache 体检：核对几何、标签与归一化幅度")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件")
    parser.add_argument("--cache-dir", default=None, help="覆盖 cache/ 位置")
    parser.add_argument("--quiet", action="store_true", help="只打印汇总，不打印逐例明细")
    parser.add_argument("--set", dest="overrides", action="append", default=None, help="覆盖配置项")
    args = parser.parse_args(argv)

    import nibabel as nib

    cfg = load_config(args.config, args.overrides)
    paths = cfg.get("paths", {}) or {}
    pre = cfg.get("preprocess", {}) or {}
    hu_lo, hu_hi = (float(x) for x in pre.get("hu_clip", [-1000.0, 1000.0]))
    target_spacing = [float(s) for s in pre.get("target_spacing", [1.0, 1.0, 1.0])]
    # cache 影像类型：uint16 = 已归一化到 [0,1] 后按 1/65535 量化（默认）；
    #                float32 = 未归一化的 HU（clip 到 hu_clip 之后）
    expect_out_dtype = str(pre.get("image_out_dtype", "uint16")).lower()
    if expect_out_dtype == "uint16":
        expect_nib_dtype, val_lo, val_hi, val_desc = "uint16", 0.0, 65535.0, "归一化值 [0,1]（×65535）"
    else:
        expect_nib_dtype, val_lo, val_hi, val_desc = "float32", hu_lo, hu_hi, f"HU [{hu_lo}, {hu_hi}]"

    cache_dir = resolve_path(args.cache_dir or paths.get("cache", "cache"))
    image_dir, label_dir = cache_dir / "image", cache_dir / "label"
    if not image_dir.is_dir() or not label_dir.is_dir():
        LOGGER.error("cache 目录不存在或缺子目录：%s", rel_to_root(cache_dir))
        return 2
    LOGGER.info("期望的缓存口径：image_out_dtype=%s，target_spacing=%s，值域 %s",
                expect_out_dtype, target_spacing, val_desc)

    case_ids = cache_cases(cache_dir, "image")
    label_ids = cache_cases(cache_dir, "label")
    images = [cache_file(cache_dir, "image", c) for c in case_ids]
    labels = [cache_file(cache_dir, "label", c) for c in label_ids]
    for path in images[:1] + labels[:1]:
        warn_compressed_cache(path, LOGGER)      # 压缩缓存只提醒一次（逐层读取会整卷解压）
    problems: list = []

    if len(images) != len(labels):
        problems.append(f"image 文件数 {len(images)} != label 文件数 {len(labels)}")
    img_ids = {str(c) for c in case_ids}
    lab_ids = {str(c) for c in label_ids}
    if img_ids != lab_ids:
        problems.append(f"image/label 文件名不成对，差集：{sorted(img_ids ^ lab_ids)}")

    manifest = load_json(paths.get("cache_manifest", "cache/cache_manifest.json"), default={}) or {}
    manifest_cases = {int(r["case"]): r for r in manifest.get("cases", [])}
    if not manifest_cases:
        problems.append("没有读到 cache_manifest.json —— 请先跑 python scripts/fetch_manifest.py，"
                        "或完整跑一遍 python scripts/preprocess.py")
    else:
        disk_ids = {int(c) for c in img_ids}
        if set(manifest_cases) != disk_ids:
            problems.append(f"清单里的 case 集合与磁盘不一致：磁盘 {sorted(disk_ids)} "
                            f"清单 {sorted(manifest_cases)}（可能是只跑了 --limit/--debug 的部分缓存）")
        if not manifest.get("axis_convention"):
            LOGGER.warning("清单里没有 axis_convention 字段：它来自更早版本的脚本，"
                           "请跑 python scripts/fetch_manifest.py 刷新清单")

    rows: list = []
    for case in case_ids:
        img = nib.load(str(cache_file(cache_dir, "image", case)))
        arr = np.asanyarray(img.dataobj)
        lab_img = nib.load(str(cache_file(cache_dir, "label", case)))
        lab = np.asanyarray(lab_img.dataobj)
        zooms = tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])
        lab_zooms = tuple(round(float(z), 4) for z in lab_img.header.get_zooms()[:3])

        # 轴序实测（scripts/probe_axis.py）：nibabel 读回的数组是 (nx, ny, nz)，
        # 切片轴在最后一维，所以每层前景体素数要对 (x, y) 求和。
        per_slice = lab.sum(axis=(0, 1))
        if int(per_slice.size) != int(lab.shape[2]):
            problems.append(f"case {case}：per-slice 长度 {int(per_slice.size)} "
                            f"与切片数 {int(lab.shape[2])} 不一致")
        uniq = np.unique(lab)
        img_dtype = str(img.get_data_dtype())
        if img_dtype == "uint16":
            value_range = (0.0, 1.0)
        elif img_dtype in ("float32", "float64"):
            value_range = (float(arr.min()), float(arr.max()))
        else:
            value_range = (float("nan"), float("nan"))

        row = {
            "case": case,
            "nib_shape_xyz": tuple(int(s) for s in arr.shape),
            "inplane_hw": (int(lab.shape[1]), int(lab.shape[0])),
            "n_slices": int(lab.shape[2]),
            "spacing": zooms,
            "dtype": img_dtype,
            "label_dtype": str(lab_img.get_data_dtype()),
            "label_values": uniq.tolist()[:5],
            "tumor_voxels": int((lab > 0).sum()),
            "tumor_slices": int((per_slice > 0).sum()),
            "img_min": round(float(arr.min()), 2),
            "img_max": round(float(arr.max()), 2),
            "value_min": round(value_range[0], 4),
            "value_max": round(value_range[1], 4),
            "nan": int(np.count_nonzero(~np.isfinite(np.asarray(arr, dtype=np.float32)))),
        }
        rows.append(row)

        if row["tumor_slices"] > int(lab.shape[2]):
            problems.append(f"case {case}：含肿瘤切片数 {row['tumor_slices']} 超过总切片数 {int(lab.shape[2])}")

        if tuple(arr.shape) != tuple(lab.shape):
            problems.append(f"case {case}：image shape {arr.shape} != label shape {lab.shape}")
        if zooms != tuple(round(s, 4) for s in target_spacing) or lab_zooms != zooms:
            problems.append(f"case {case}：spacing image={zooms} label={lab_zooms}，"
                            f"期望 {tuple(round(s, 4) for s in target_spacing)}")
        if img_dtype != expect_nib_dtype:
            problems.append(f"case {case}：image dtype={img_dtype}，"
                            f"与 config 的 image_out_dtype={expect_out_dtype} 不符")
        if len(uniq) and (int(uniq.max()) > 1 or int(uniq.min()) < 0):
            problems.append(f"case {case}：label 取值 {uniq.tolist()[:5]} 不是 0/1 二值")
        if row["nan"]:
            problems.append(f"case {case}：影像里有 {row['nan']} 个非有限值")
        if row["img_min"] < val_lo - 1e-3 or row["img_max"] > val_hi + 1e-3:
            problems.append(f"case {case}：影像取值 [{row['img_min']}, {row['img_max']}] "
                            f"超出期望范围 {val_desc}")
        if int(case) in manifest_cases:
            rec = manifest_cases[int(case)]
            if list(row["nib_shape_xyz"]) != [int(s) for s in rec.get("nib_shape_xyz", [])]:
                problems.append(f"case {case}：nibabel 读回的 shape 与清单不符 "
                                f"{row['nib_shape_xyz']} vs {rec.get('nib_shape_xyz')}")
            if int(row["tumor_slices"]) != int(rec["tumor_slices"]):
                problems.append(f"case {case}：含肿瘤切片数 {row['tumor_slices']} "
                                f"与清单 {rec['tumor_slices']} 不符")
            if int(row["tumor_voxels"]) != int(rec.get("tumor_voxels", row["tumor_voxels"])):
                problems.append(f"case {case}：肿瘤体素数 {row['tumor_voxels']} "
                                f"与清单 {rec.get('tumor_voxels')} 不符")

    n_tumor = sum(1 for r in rows if r["tumor_voxels"] > 0)
    dtypes = Counter(r["dtype"] for r in rows)
    summary = {
        "cache_dir": rel_to_root(cache_dir),
        "n_cases": len(rows),
        "n_cases_with_tumor": n_tumor,
        "n_cases_liver_only": len(rows) - n_tumor,
        "cases_with_tumor": [int(r["case"]) for r in rows if r["tumor_voxels"] > 0],
        "cases_liver_only": [int(r["case"]) for r in rows if r["tumor_voxels"] == 0],
        "image_dtypes": dict(dtypes),
        "spacings": {str(k): v for k, v in Counter(r["spacing"] for r in rows).items()},
        "inplane_shapes": {str(k): v for k, v in
                           Counter(tuple(r["inplane_hw"]) for r in rows).items()},
        "n_slices_min": min((r["n_slices"] for r in rows), default=0),
        "n_slices_max": max((r["n_slices"] for r in rows), default=0),
        "tumor_slices_total": sum(r["tumor_slices"] for r in rows),
        "voxels_total": sum(int(np.prod(r["nib_shape_xyz"])) for r in rows),
        "axis_convention": "nibabel 读回为 (nx, ny, nz)，切片轴在最后一维；SimpleITK 读回是其转置",
        "img_min_overall": min((r["img_min"] for r in rows), default=None),
        "img_max_overall": max((r["img_max"] for r in rows), default=None),
        "image_out_dtype": expect_out_dtype,
        "value_range_expected": val_desc,
        "quant_step_hu": round((hu_hi - hu_lo) / 65535.0, 6) if expect_out_dtype == "uint16" else None,
        "problems": problems,
        "rows": rows,
    }

    report_path = resolve_path(Path(paths.get("reports", "reports")) / "cache_check.json")
    json_path = save_json(summary, report_path)

    LOGGER.info("cache 目录：%s", rel_to_root(cache_dir))
    LOGGER.info("case 数：%d；含肿瘤 %d 例；仅肝脏 %d 例", len(rows), n_tumor, len(rows) - n_tumor)
    LOGGER.info("含肿瘤的 case：%s", summary["cases_with_tumor"])
    LOGGER.info("仅肝脏的 case：%s", summary["cases_liver_only"])
    LOGGER.info("影像 dtype 分布：%s；spacing 分布：%s", summary["image_dtypes"], summary["spacings"])
    LOGGER.info("面内尺寸分布：%s", summary["inplane_shapes"])
    LOGGER.info("切片数范围：%d - %d；含肿瘤切片合计 %d",
                summary["n_slices_min"], summary["n_slices_max"], summary["tumor_slices_total"])
    LOGGER.info("影像原始值域（全体）：[%s, %s]，期望 %s",
                summary["img_min_overall"], summary["img_max_overall"], val_desc)
    if summary["quant_step_hu"] is not None:
        LOGGER.info("uint16 量化步长：%.6f HU/级（越小越好，理论下限由 hu_clip 跨度决定）",
                    summary["quant_step_hu"])
        LOGGER.info("  即 dataset.py 里 image.float()/65535 得到 [0,1]，等价于 HU 窗 [-1000,1000]")

    if not args.quiet:
        LOGGER.info("")
        LOGGER.info("case | nib_shape_xyz | inplane_hw | n_slices | spacing | dtype | label 值 | "
                    "tumor_voxels | tumor_slices | 归一化后 [0,1] 值域")
        for r in rows:
            LOGGER.info("%s | %s | %s | %d | %s | %s | %s | %d | %d | [%s, %s]",
                        r["case"], r["nib_shape_xyz"], r["inplane_hw"], r["n_slices"], r["spacing"],
                        r["dtype"], r["label_values"], r["tumor_voxels"], r["tumor_slices"],
                        r["value_min"], r["value_max"])

    LOGGER.info("体检报告：%s", rel_to_root(json_path))
    if problems:
        LOGGER.error("发现 %d 个问题：", len(problems))
        for item in problems:
            LOGGER.error("  - %s", item)
        return 1
    LOGGER.info("体检通过：几何、标签、值域与清单全部一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
