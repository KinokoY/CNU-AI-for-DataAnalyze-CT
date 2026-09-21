"""缓存体检：读回 cache/ 下的全部产物，逐项核对几何、标签与归一化幅度是否符合约定。

整体功能：对每一例检查 image/label 是否成对、shape 是否一致、spacing 是否为 1mm 三轴相同、
        label 取值是否只含 {0,1}、有无肿瘤、以及影像 HU 是否落在全局窗内；再把实测结果与
        cache_manifest.json 的记录逐例比对，不一致就报错并以非零码退出。
前后接口：上游是 scripts/preprocess.py 产出的 cache/ 与 cache/cache_manifest.json；
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
    from src.utils import load_config, load_json, rel_to_root, resolve_path, save_json, setup_logger
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import (  # type: ignore
        load_config,
        load_json,
        rel_to_root,
        resolve_path,
        save_json,
        setup_logger,
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

    cache_dir = resolve_path(args.cache_dir or paths.get("cache", "cache"))
    image_dir, label_dir = cache_dir / "image", cache_dir / "label"
    if not image_dir.is_dir() or not label_dir.is_dir():
        LOGGER.error("cache 目录不存在或缺子目录：%s", rel_to_root(cache_dir))
        return 2

    images = sorted(image_dir.glob("*.nii.gz"), key=lambda p: int(p.name.split(".")[0]))
    labels = sorted(label_dir.glob("*.nii.gz"), key=lambda p: int(p.name.split(".")[0]))
    problems: list = []

    if len(images) != len(labels):
        problems.append(f"image 文件数 {len(images)} != label 文件数 {len(labels)}")
    img_ids = {p.name.split(".")[0] for p in images}
    lab_ids = {p.name.split(".")[0] for p in labels}
    if img_ids != lab_ids:
        problems.append(f"image/label 文件名不成对，差集：{sorted(img_ids ^ lab_ids)}")

    manifest = load_json(paths.get("cache_manifest", "cache/cache_manifest.json"), default={}) or {}
    manifest_cases = {int(r["case"]): r for r in manifest.get("cases", [])}
    if manifest_cases and set(manifest_cases) != {int(c) for c in img_ids}:
        problems.append(f"清单里的 case 集合与磁盘不一致：磁盘 {sorted(int(c) for c in img_ids)} "
                        f"清单 {sorted(manifest_cases)}")

    rows: list = []
    for path in images:
        case = path.name.split(".")[0]
        img = nib.load(str(path))
        arr = np.asanyarray(img.dataobj)
        lab_img = nib.load(str(label_dir / f"{case}.nii.gz"))
        lab = np.asanyarray(lab_img.dataobj)
        zooms = tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])
        lab_zooms = tuple(round(float(z), 4) for z in lab_img.header.get_zooms()[:3])

        per_slice = lab.reshape(-1, lab.shape[2]).sum(axis=0)
        uniq = np.unique(lab)
        row = {
            "case": case,
            "shape_zyx": tuple(int(s) for s in arr.shape),
            "spacing": zooms,
            "dtype": str(img.get_data_dtype()),
            "label_dtype": str(lab_img.get_data_dtype()),
            "label_values": uniq.tolist()[:5],
            "tumor_voxels": int((lab > 0).sum()),
            "tumor_slices": int((per_slice > 0).sum()),
            "img_min": round(float(arr.min()), 2),
            "img_max": round(float(arr.max()), 2),
            "nan": int(np.count_nonzero(~np.isfinite(np.asarray(arr, dtype=np.float32)))),
        }
        rows.append(row)

        if tuple(arr.shape) != tuple(lab.shape):
            problems.append(f"case {case}：image shape {arr.shape} != label shape {lab.shape}")
        if zooms != tuple(round(s, 4) for s in target_spacing) or lab_zooms != zooms:
            problems.append(f"case {case}：spacing image={zooms} label={lab_zooms}，"
                            f"期望 {tuple(round(s, 4) for s in target_spacing)}")
        if len(uniq) and (int(uniq.max()) > 1 or int(uniq.min()) < 0):
            problems.append(f"case {case}：label 取值 {uniq.tolist()[:5]} 不是 0/1 二值")
        if row["nan"]:
            problems.append(f"case {case}：影像里有 {row['nan']} 个非有限值")
        if row["img_min"] < hu_lo - 1e-3 or row["img_max"] > hu_hi + 1e-3:
            problems.append(f"case {case}：影像范围 [{row['img_min']}, {row['img_max']}] "
                            f"超出全局窗 [{hu_lo}, {hu_hi}]")
        if case in {str(c) for c in manifest_cases}:
            rec = manifest_cases[int(case)]
            if list(row["shape_zyx"]) != [int(s) for s in rec["shape_zyx"]]:
                problems.append(f"case {case}：shape 与清单不符 {row['shape_zyx']} vs {rec['shape_zyx']}")
            if int(row["tumor_slices"]) != int(rec["tumor_slices"]):
                problems.append(f"case {case}：含肿瘤切片数 {row['tumor_slices']} "
                                f"与清单 {rec['tumor_slices']} 不符")

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
                           Counter((r["shape_zyx"][1], r["shape_zyx"][2]) for r in rows).items()},
        "n_slices_min": min((r["shape_zyx"][0] for r in rows), default=0),
        "n_slices_max": max((r["shape_zyx"][0] for r in rows), default=0),
        "tumor_slices_total": sum(r["tumor_slices"] for r in rows),
        "voxels_total": sum(int(np.prod(r["shape_zyx"])) for r in rows),
        "img_min_overall": min((r["img_min"] for r in rows), default=None),
        "img_max_overall": max((r["img_max"] for r in rows), default=None),
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
    LOGGER.info("影像值域（全体）：[%s, %s]", summary["img_min_overall"], summary["img_max_overall"])

    if not args.quiet:
        LOGGER.info("")
        LOGGER.info("case | shape_zyx | spacing | dtype | label 值 | tumor_voxels | tumor_slices | img_min | img_max")
        for r in rows:
            LOGGER.info("%s | %s | %s | %s | %s | %d | %d | %s | %s",
                        r["case"], r["shape_zyx"], r["spacing"], r["dtype"], r["label_values"],
                        r["tumor_voxels"], r["tumor_slices"], r["img_min"], r["img_max"])

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
