"""预处理崩溃定位：对单个 case 逐步执行几何处理，打印每步前后，定位 segfault 在哪一步。

整体功能：只处理一个 case（默认 31），在每一步几何操作前后打印标记，并在 C 层崩溃时由 faulthandler
        打出栈；同时打印重采样前后的大小/间距/内存占用估算，用于确认是否存在尺寸爆炸。
前后接口：只读 data/ 下的原始 NIfTI，不写任何文件（除非 --out 指定）；与 scripts/preprocess.py 使用同一套 SITK 调用。
用法：仓库根目录执行 ``python scripts/probe_resample.py --case 31``。
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def log(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="逐步定位预处理 segfault")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--case", type=int, default=31)
    ap.add_argument("--target-spacing", default="1,1,1")
    ap.add_argument("--stop-after", default="all",
                    choices=["read", "mask", "floor", "orient", "size", "resample", "all"],
                    help="在这一步之后停住，用于二分定位")
    args = ap.parse_args()

    # 让 C 层信号（SIGSEGV）也打印 Python 栈
    faulthandler.enable()

    import SimpleITK as sitk

    log(f"SimpleITK {sitk.Version_VersionString()}，numpy {np.__version__}，python {sys.version.split()[0]}")
    target_spacing = [float(x) for x in args.target_spacing.split(",")]

    vol_path = Path(args.data_dir) / f"volume-{args.case}.nii"
    seg_path = Path(args.data_dir) / f"segmentation-{args.case}.nii"
    log(f"volume: {vol_path}（存在={vol_path.exists()}）")
    log(f"seg   : {seg_path}（存在={seg_path.exists()}）")

    log("step 1/7 ReadImage volume ...")
    raw = sitk.ReadImage(str(vol_path))
    log(f"  读入 pixelID={raw.GetPixelIDTypeAsString()} size={raw.GetSize()} "
        f"spacing={tuple(round(s, 4) for s in raw.GetSpacing())}")

    log("step 2/7 Cast volume -> float32 ...")
    vol_img = sitk.Cast(raw, sitk.sitkFloat32) if raw.GetPixelID() != sitk.sitkFloat32 else raw
    log(f"  ok，估算内存 {vol_img.GetSize()[0] * vol_img.GetSize()[1] * vol_img.GetSize()[2] * 4 / 1e6:.1f} MB")
    if args.stop_after == "read":
        return 0

    log("step 3/7 ReadImage + threshold mask ...")
    seg_u8 = sitk.ReadImage(str(seg_path))
    if seg_u8.GetPixelID() != sitk.sitkUInt8:
        seg_u8 = sitk.Cast(seg_u8, sitk.sitkUInt8)
    log(f"  seg pixelID={seg_u8.GetPixelIDTypeAsString()} size={seg_u8.GetSize()}")
    mask = sitk.Cast(sitk.BinaryThreshold(seg_u8, 2, 2, 1, 0), sitk.sitkUInt8)
    log(f"  ok，前景体素数={int(np.count_nonzero(sitk.GetArrayViewFromImage(mask)))}")
    if args.stop_after == "mask":
        return 0

    log("step 4/7 floor stats + clamp ...")
    arr = sitk.GetArrayViewFromImage(raw)
    vmin = float(arr.min())
    n_floor = int(np.count_nonzero(arr <= vmin + 1.0))
    log(f"  floor={vmin} n_floor={n_floor} frac={n_floor / arr.size:.4f}")
    del arr
    fmask = sitk.BinaryThreshold(vol_img, lowerThreshold=vmin + 1.0,
                                 upperThreshold=float(np.finfo(np.float32).max), insideValue=1, outsideValue=0)
    vol_clamped = sitk.Cast(sitk.Mask(vol_img, fmask, outsideValue=-1000.0), sitk.sitkFloat32)
    log(f"  ok，clamp 后 min={float(sitk.GetArrayViewFromImage(vol_clamped).min())}")
    if args.stop_after == "floor":
        return 0

    log("step 5/7 DICOMOrient -> RAS ...")
    oriented = sitk.DICOMOrient(vol_clamped, "RAS")
    log(f"  ok，size={oriented.GetSize()} spacing={tuple(round(s, 4) for s in oriented.GetSpacing())} "
        f"origin={tuple(round(o, 2) for o in oriented.GetOrigin())}")
    if args.stop_after == "orient":
        return 0

    log("step 6/7 计算输出网格 ...")
    in_spacing = oriented.GetSpacing()
    in_size = oriented.GetSize()
    new_size = [int(round(in_size[i] * (in_spacing[i] / target_spacing[i]))) for i in range(3)]
    avg = [abs(in_spacing[i]) for i in range(3)] * (np.abs(np.asarray(oriented.GetDirection()).reshape(3, 3)).sum(axis=0))
    proj_size = [int(np.ceil(in_size[i] * avg[i] / target_spacing[i])) for i in range(3)]
    log(f"  按 spacing 比例估算 new_size={new_size}")
    log(f"  按投影窗口估算     proj_size={proj_size}（d={tuple(round(x, 3) for x in avg)}）")
    log(f"  输出体素总数(投影口径)={int(np.prod(proj_size))}，float32 约 {int(np.prod(proj_size)) * 4 / 1e6:.1f} MB")
    if args.stop_after == "size":
        return 0

    log("step 7/7 ResampleImageFilter（影像，线性）...")
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing([float(s) for s in target_spacing])
    rs.SetSize([int(x) for x in proj_size])
    rs.SetOutputDirection(oriented.GetDirection())
    rs.SetOutputOrigin(oriented.GetOrigin())
    rs.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(0.0)
    out = rs.Execute(oriented)
    log(f"  ok，输出 size={out.GetSize()} spacing={tuple(round(s, 4) for s in out.GetSpacing())}")

    log("额外：对掩膜做同一套重采样（最近邻）...")
    rs2 = sitk.ResampleImageFilter()
    rs2.SetOutputSpacing([float(s) for s in target_spacing])
    rs2.SetSize([int(x) for x in proj_size])
    rs2.SetOutputDirection(oriented.GetDirection())
    rs2.SetOutputOrigin(oriented.GetOrigin())
    rs2.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    rs2.SetInterpolator(sitk.sitkNearestNeighbor)
    rs2.SetDefaultPixelValue(0.0)
    mask_out = rs2.Execute(sitk.DICOMOrient(mask, "RAS"))
    m = np.asarray(sitk.GetArrayFromImage(mask_out))
    log(f"  ok，掩膜前景体素数={int(np.count_nonzero(m))}（label 值={np.unique(m)[:5].tolist()}）")
    log("全部步骤完成，没有崩溃。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
