"""轴序探针：确定缓存文件被 nibabel / SimpleITK 读回时的真实数组形状与切片轴。

整体功能：对同一个 .nii.gz 分别用 SimpleITK 与 nibabel 读取，打印各自的数组形状、spacing 与
        "沿各轴求和得到的前景层数"，从而唯一确定切片轴在数组的哪一维，并检查两个库是否互为转置。
前后接口：只读 data/ 下的原始 NIfTI 与 cache/ 下的缓存（若存在），不写任何文件。
用法：仓库根目录执行 ``python scripts/probe_axis.py --case 31``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def describe(tag: str, arr: np.ndarray, spacing) -> None:
    """打印一个数组的形状、spacing 与沿每个轴求和后的非零层数。"""
    print(f"[{tag}] shape={tuple(int(s) for s in arr.shape)} spacing={spacing}")
    for axis in range(arr.ndim):
        per = (arr > 0).sum(axis=tuple(i for i in range(arr.ndim) if i != axis))
        print(f"    axis={axis}: 非零层数={int((per > 0).sum())}/{int(per.size)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="确定缓存文件的真实轴序")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--case", type=int, default=31)
    args = ap.parse_args()

    import SimpleITK as sitk
    import nibabel as nib

    print(f"SimpleITK {sitk.Version_VersionString()} / nibabel {nib.__version__} / numpy {np.__version__}")

    # ---- 1) 原始文件 ----
    raw_path = Path(args.data_dir) / f"volume-{args.case}.nii"
    if raw_path.exists():
        sitk_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(raw_path)))
        nib_img = nib.load(str(raw_path))
        nib_arr = np.asanyarray(nib_img.dataobj)
        print("\n=== 原始 volume ===")
        describe("SimpleITK", sitk_arr, "n/a")
        describe("nibabel", nib_arr, "n/a")
        print(f"两者是否互为转置（nib 的 shape == sitk shape 反转）："
              f"{tuple(nib_arr.shape) == tuple(reversed(sitk_arr.shape))}")
        print(f"nibabel header zooms = {tuple(round(float(z), 4) for z in nib_img.header.get_zooms()[:3])}")

    # ---- 2) 缓存文件 ----
    lab_path = Path(args.cache_dir) / "label" / f"{args.case}.nii.gz"
    img_path = Path(args.cache_dir) / "image" / f"{args.case}.nii.gz"
    if lab_path.exists():
        print("\n=== 缓存 label ===")
        s_lab = sitk.GetArrayFromImage(sitk.ReadImage(str(lab_path)))
        n_lab = np.asanyarray(nib.load(str(lab_path)).dataobj)
        describe("SimpleITK", s_lab, "n/a")
        describe("nibabel", n_lab, "n/a")
        print(f"转置关系：{tuple(n_lab.shape) == tuple(reversed(s_lab.shape))}")
    else:
        print(f"\n[跳过] 缓存不存在：{lab_path}")

    if img_path.exists():
        print("\n=== 缓存 image ===")
        s_img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        n_img_file = nib.load(str(img_path))
        n_img = np.asanyarray(n_img_file.dataobj)
        print(f"SimpleITK shape={tuple(int(s) for s in s_img.shape)} dtype={s_img.dtype}")
        print(f"nibabel  shape={tuple(int(s) for s in n_img.shape)} dtype={n_img.dtype} "
              f"zooms={tuple(round(float(z), 4) for z in n_img_file.header.get_zooms()[:3])}")
        print(f"转置关系：{tuple(n_img.shape) == tuple(reversed(s_img.shape))}")
        print(f"取值：min={int(n_img.min())} max={int(n_img.max())}（uint16 归一化应为 0..65535）")

    # ---- 3) 与清单比对 ----
    import json

    mpath = Path(args.cache_dir) / "cache_manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        rec = next((r for r in manifest.get("cases", []) if int(r["case"]) == args.case), None)
        if rec:
            print(f"\n清单记录 shape_zyx={rec['shape_zyx']} tumor_slices={rec['tumor_slices']}")
            print(f"清单的 shape 是否等于 nibabel 读回的 shape："
                  f"{[int(x) for x in rec['shape_zyx']] == [int(x) for x in n_lab.shape]}")

    print("\n判读：哪个轴的『非零层数』与清单 tumor_slices 接近且不超过它，切片轴就是那一维；")
    print("      nz 为切片数，ny/nx 为面内尺寸。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
