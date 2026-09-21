#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""CT 数据探针：只读地打印数据目录的全部结构与内容事实，用于在看不到数据的情况下推进分割任务。

数据受保密协议约束只存在于远程平台，本地无数据可用，因此需要由用户在本项目根目录执行本脚本，
并把完整 stdout 贴回，作为撰写 ``docs/data.md`` 与后续数据处理 / 训练的实测依据。

用法
----
在本项目根目录（默认当前 Linux 工作目录就是项目根）执行::

    # 主用法：扫描 ./data，打印全部 7 节报告
    python scripts/probe_data.py

    # 显式指定数据目录（默认即为 data）
    python scripts/probe_data.py --data-dir data

参数说明与默认值：

===================================  ==================================================
``--data-dir PATH``                  数据目录，默认 ``data``；相对路径相对当前工作目录
``--vol-pattern REGEX``              影像文件名正则（不含扩展名），默认 ``^volume[-_](\d+)$``
``--seg-pattern REGEX``              掩膜文件名正则（不含扩展名），默认 ``^segmentation[-_](\d+)$``
``--limit-cases N``                  只检查前 N 个 case，默认 ``0`` 表示全部（快速自检用）
``--skip-stats``                     只看文件系统与头信息，不读体素（秒级返回，但无强度/标签/规模）
``--write-doc PATH``                 额外把完整输出写入该文件；**不要写进仓库**，用完请删除
``-h`` / ``--help``                  打印参数帮助
===================================  ==================================================

其他说明：

* **脚本只读数据**，不写入、不修改任何数据文件；进度与警告走 stderr，报告正文走 stdout。
* 输出为逐行 ``key=value`` 纯文本（中文列名，数字保持原文），可直接整段复制粘贴。
* 退出码：``0`` 脚本跑完（不代表数据没问题）；``2`` 命令行参数或正则非法。
* 单个 case 读取失败不会中断整体，该 case 记为 ``status=READ_ERROR`` 并继续。
* 约 30 个 case 全量读取体素需数十秒；请把**完整 stdout 原样贴回**，用于生成 ``docs/data.md``。
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import nibabel as nib

# --------------------------------------------------------------------------------------
# 常量与工具
# --------------------------------------------------------------------------------------

NII_SUFFIXES = (".nii.gz", ".nii", ".hdr", ".img", ".nrrd", ".mha", ".mhd")

# 目录清单：这些子目录名通常是元数据而非图像数据，单独报告、不递归进去统计
LIKELY_META_DIRS = {"__macosx", ".git", ".ipynb_checkpoints", "derivatives", "labels", "metrics"}

# DICOM 判定用的文件头魔数（128 字节 preamble 之后的 "DICM"）
DICOM_MAGIC = b"DICM"

VOLUME_PATTERNS = (re.compile(r"^volume[-_](\d+)$"),)
SEG_PATTERNS = (re.compile(r"^segmentation[-_](\d+)$"),)

# CT 场景下"明显属于空气"的 HU 阈值，仅用于给出一个参考口径，不是硬编码假设
CT_AIR_HU = -900.0
# 掩膜包围盒是否包含在身体包围盒内的容差（体素）
BBOX_TOL = 2


def eprint(*args) -> None:
    print(*args, file=sys.stderr)


def fmt_num(v) -> str:
    """把数值格式化成紧凑字符串；缺失值统一输出 'NA'。"""
    if v is None:
        return "NA"
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not np.isfinite(f):
            return "NA"
        if f == 0:
            return "0"
        a = abs(f)
        if a >= 1e5 or a < 1e-3:
            return f"{f:.4g}"
        return f"{f:.4f}".rstrip("0").rstrip(".")
    return str(v)


def fmt_tuple(vals, nd: int = 4) -> str:
    if vals is None:
        return "NA"
    try:
        seq = [float(x) for x in np.ravel(np.asarray(vals))]
    except Exception:
        return "NA"
    return "(" + ",".join(fmt_num(round(s, nd)) for s in seq) + ")"


def global_percentiles(arr: np.ndarray, mask: np.ndarray | None, qs) -> list:
    """返回给定 qs 的分位数；mask 非空时只统计 mask 内体素。"""
    flat = np.asarray(arr).ravel()
    if mask is not None:
        m = np.asarray(mask).ravel()
        flat = flat[m]
    if flat.size == 0:
        return [None for _ in qs]
    return [float(np.percentile(flat, q)) for q in qs]


def voxel_volume_mm3(zooms) -> float:
    v = 1.0
    for z in zooms:
        v *= float(z)
    return v


def nonzero_bbox(mask: np.ndarray):
    """返回掩膜的紧凑包围盒，格式 ((x0,x1),(y0,y1),(z0,z1))（闭区间，体素索引）。"""
    idx = np.nonzero(mask)
    if len(idx) == 0 or idx[0].size == 0:
        return None
    return tuple((int(a.min()), int(a.max())) for a in idx)


def bbox_inside(inner, outer, tol: int = BBOX_TOL) -> bool:
    if inner is None or outer is None:
        return False
    return all(
        inner[d][0] >= outer[d][0] - tol and inner[d][1] <= outer[d][1] + tol
        for d in range(len(inner))
    )


def bbox_str(bb) -> str:
    if bb is None:
        return "NA"
    return "[" + " ".join(f"{a}:{b}" for a, b in bb) + "]"


# --------------------------------------------------------------------------------------
# 1. 文件系统枚举
# --------------------------------------------------------------------------------------

def scan_filesystem(data_dir: str) -> dict:
    """遍历 data_dir，产出文件清单与结构信息（不读图像内容）。"""
    info = {
        "dir": os.path.abspath(data_dir),
        "exists": os.path.isdir(data_dir),
        "top_entries": [],
        "subdirs": [],
        "n_files_total": 0,
        "n_nii": 0,
        "n_dicom": 0,
        "other_ext": collections.Counter(),
        "other_samples": [],
        "zero_byte": [],
        "total_bytes": 0,
    }
    if not info["exists"]:
        return info

    for name in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, name)
        if os.path.isdir(full):
            try:
                n_inside = len(os.listdir(full))
            except OSError:
                n_inside = -1
            info["subdirs"].append((name, n_inside))
        else:
            info["top_entries"].append(name)

    for root, dirs, files in os.walk(data_dir):
        rel = os.path.relpath(root, data_dir)
        # 不递归进明显的元数据目录
        if rel != "." and os.path.basename(root).lower() in LIKELY_META_DIRS:
            dirs[:] = []
            continue
        for fn in sorted(files):
            full = os.path.join(root, fn)
            info["n_files_total"] += 1
            try:
                size = os.path.getsize(full)
            except OSError:
                size = -1
            if size >= 0:
                info["total_bytes"] += size
                if size == 0:
                    info["zero_byte"].append(os.path.relpath(full, data_dir))
            low = fn.lower()
            if low.endswith((".nii", ".nii.gz")):
                info["n_nii"] += 1
            elif low.endswith((".dcm", ".dicom")) or re.fullmatch(r"i\d+\.\w+", low):
                info["n_dicom"] += 1
            else:
                # 用魔数再判一次 DICOM（无扩展名的情况）
                is_dcm = False
                if size > 132:
                    try:
                        with open(full, "rb") as fh:
                            fh.seek(128)
                            is_dcm = fh.read(4) == DICOM_MAGIC
                    except OSError:
                        pass
                if is_dcm:
                    info["n_dicom"] += 1
                else:
                    ext = os.path.splitext(low)[1] or "(no-ext)"
                    info["other_ext"][ext] += 1
                    if len(info["other_samples"]) < 20:
                        info["other_samples"].append(os.path.relpath(full, data_dir))
    return info


def strip_known_suffix(name: str) -> str:
    for suf in NII_SUFFIXES:
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return name


def classify_files(data_dir: str) -> dict:
    """把 NIfTI 文件名分成 volume / segmentation / 其他，并做 case 配对。"""
    res = {
        "volumes": {},      # case_id -> 相对路径
        "segs": {},         # case_id -> 相对路径
        "unmatched": [],    # 既不匹配 volume-* 也不匹配 segmentation-* 的 NIfTI
        "duplicates": [],   # 同一 case 同类型出现多个文件
    }
    if not os.path.isdir(data_dir):
        return res

    for root, dirs, files in os.walk(data_dir):
        rel_dir = os.path.relpath(root, data_dir)
        if rel_dir != "." and os.path.basename(root).lower() in LIKELY_META_DIRS:
            dirs[:] = []
            continue
        for fn in sorted(files):
            if fn.startswith("._"):  # macOS 资源叉文件
                continue
            if not fn.lower().endswith((".nii", ".nii.gz")):
                continue
            rel = os.path.relpath(os.path.join(root, fn), data_dir)
            stem = strip_known_suffix(fn)
            kind = None
            case = None
            for pat in VOLUME_PATTERNS:
                m = pat.match(stem)
                if m:
                    kind, case = "volume", m.group(1)
                    break
            if kind is None:
                for pat in SEG_PATTERNS:
                    m = pat.match(stem)
                    if m:
                        kind, case = "seg", m.group(1)
                        break
            if kind is None:
                res["unmatched"].append(rel)
                continue
            bucket = res["volumes"] if kind == "volume" else res["segs"]
            if case in bucket:
                res["duplicates"].append(f"{rel} (与 {bucket[case]} 同为 {kind} {case})")
                continue
            bucket[case] = rel
    return res


def sort_cases(ids) -> list:
    """按数值排序，非数值 id 排在后面按字符串排序。"""
    numeric = sorted((int(c) for c in ids if str(c).isdigit()))
    others = sorted(c for c in ids if not str(c).isdigit())
    return [str(c) for c in numeric] + others


# --------------------------------------------------------------------------------------
# 2. 图像与标签检查
# --------------------------------------------------------------------------------------

def load_array(path: str) -> np.ndarray:
    """读取 NIfTI 并返回 numpy 数组（保持原始 dtype）。"""
    img = nib.load(path)
    return np.asanyarray(img.dataobj)


def inspect_pair(case: str, vol_path: str, seg_path: str) -> dict:
    """检查一个 case 的 volume/seg 对，返回一个 dict（全部是纯 python 标量/字符串）。"""
    rec = {
        "case": case,
        "status": "OK",
        "error": "",
        "vol_path": vol_path,
        "seg_path": seg_path,
        "vol_bytes": os.path.getsize(vol_path) if os.path.exists(vol_path) else None,
        "seg_bytes": os.path.getsize(seg_path) if os.path.exists(seg_path) else None,
    }

    # ---- 头信息 ----
    img = nib.load(vol_path)
    seg = nib.load(seg_path)
    rec["vol_shape"] = tuple(int(s) for s in img.shape)
    rec["seg_shape"] = tuple(int(s) for s in seg.shape)
    rec["shape_match"] = rec["vol_shape"] == rec["seg_shape"]
    rec["vol_zooms"] = tuple(float(z) for z in img.header.get_zooms()[: len(img.shape)])
    rec["seg_zooms"] = tuple(float(z) for z in seg.header.get_zooms()[: len(seg.shape)])
    rec["spacing_match"] = np.allclose(rec["vol_zooms"], rec["seg_zooms"], atol=1e-4, rtol=1e-4)
    rec["vol_dtype"] = str(img.get_data_dtype())
    rec["seg_dtype"] = str(seg.get_data_dtype())
    rec["vol_axcodes"] = "".join(nib.aff2axcodes(img.affine))
    rec["seg_axcodes"] = "".join(nib.aff2axcodes(seg.affine))
    rec["affine_match"] = bool(np.allclose(img.affine, seg.affine, atol=1e-3))
    rec["max_affine_diff"] = float(np.max(np.abs(img.affine - seg.affine)))
    rec["vol_affine"] = np.array2string(img.affine, precision=4, suppress_small=True).replace("\n", " | ")
    total_vol_mm3 = float(np.prod(rec["vol_zooms"])) if rec["vol_zooms"] else None
    rec["voxel_mm3"] = total_vol_mm3
    rec["voxel_mm"] = float(round(total_vol_mm3 ** (1.0 / 3.0), 4)) if total_vol_mm3 else None

    # ---- 数据 ----
    vol = load_array(vol_path)
    seg_arr = load_array(seg_path)

    # ---- 标签取值 ----
    uniq, counts = np.unique(seg_arr, return_counts=True)
    rec["seg_values"] = [(fmt_num(u), int(c)) for u, c in zip(uniq.tolist(), counts.tolist())]
    rec["n_unique_labels"] = int(len(uniq))
    rec["has_background"] = bool(0 in uniq.tolist())
    rec["seg_is_integer"] = bool(np.all(np.isfinite(seg_arr)) and np.all(seg_arr == np.round(seg_arr)))
    rec["seg_min"] = float(np.min(seg_arr)) if seg_arr.size else None
    rec["seg_max"] = float(np.max(seg_arr)) if seg_arr.size else None
    rec["seg_negative"] = bool(rec["seg_min"] is not None and rec["seg_min"] < 0)
    rec["seg_fractional"] = bool(not rec["seg_is_integer"])

    # 前景（>0）与"类 1"（==1，若有）
    fg = seg_arr > 0
    rec["fg_voxels"] = int(np.count_nonzero(fg))
    rec["fg_fraction_total"] = float(rec["fg_voxels"] / max(1, seg_arr.size))

    # ---- 图像强度统计 ----
    v = np.asarray(vol)
    finite = np.isfinite(v)
    rec["vol_nonfinite"] = int(v.size - np.count_nonzero(finite))
    vf = v[finite]
    rec["vol_dtype_is_float"] = bool(np.issubdtype(v.dtype, np.floating))
    if vf.size:
        rec["vol_min"] = float(np.min(vf))
        rec["vol_max"] = float(np.max(vf))
        rec["vol_mean"] = float(np.mean(vf, dtype=np.float64))
        rec["vol_std"] = float(np.std(vf, dtype=np.float64))
    qs = (0.5, 1, 5, 25, 50, 75, 95, 99, 99.5)
    rec["vol_percentiles"] = global_percentiles(vf, None, qs)
    rec["pct_qs"] = qs

    # 身体掩膜：CT 场景优先用空气阈值（HU < -900 视为空气）；
    # 若影像本身非负（无符号数据/已是窗内值），则退化为 >0。
    vol_min = rec.get("vol_min", 0)
    if (not rec["vol_dtype_is_float"]) and vol_min is not None and vol_min >= 0:
        body = v > 0
    else:
        body = v > CT_AIR_HU
    body = body & finite
    rec["body_voxels"] = int(np.count_nonzero(body))
    rec["body_fraction_total"] = float(rec["body_voxels"] / max(1, v.size))
    rec["body_percentiles"] = global_percentiles(v, body, qs)
    rec["fg_percentiles"] = global_percentiles(v, fg, qs) if rec["fg_voxels"] > 0 else [None] * len(qs)

    # ---- 归一化参考：身体区域 z-score 参数 ----
    if rec["body_voxels"] > 0:
        bv = v[body].astype(np.float64)
        rec["body_mean"] = float(bv.mean())
        rec["body_std"] = float(bv.std())
        rec["zscore_clip_lo"] = float(np.percentile(bv, 0.5))
        rec["zscore_clip_hi"] = float(np.percentile(bv, 99.5))
    else:
        rec["body_mean"] = rec["body_std"] = None
        rec["zscore_clip_lo"] = rec["zscore_clip_hi"] = None

    # ---- 逐切片统计 ----
    # 约定：shape 与 zooms 均为 nibabel 轴的原始顺序，对标准 NIfTI 第 3 轴是 Z（头脚方向）。
    # 4D 两种情况要分开：shape=(X,Y,Z,1)（单通道，常见）与 shape=(X,Y,Z,C>1)（多通道/多时序）。
    if vol.ndim == 3:
        axis = 2
    elif vol.ndim == 4:
        axis = 2
        if int(vol.shape[3]) > 1:
            eprint(f"[警告] case {case} 是 4D 且第 4 维={int(vol.shape[3])}（多通道/多时序），"
                   f"本节的逐切片与体素统计口径可能不适用，请人工确认后再据此设计训练。")
    else:
        axis = max(0, vol.ndim - 1)
    rec["ndim"] = int(vol.ndim)
    rec["slice_axis"] = int(axis)
    n_slices = int(vol.shape[axis]) if vol.ndim >= 3 else int(vol.shape[0])

    body_slices = np.count_nonzero(np.sum(body, axis=tuple(i for i in range(body.ndim) if i != axis)))
    fg_per_slice = np.sum(fg, axis=tuple(i for i in range(fg.ndim) if i != axis))
    rec["n_slices"] = n_slices
    rec["body_slices"] = int(body_slices)
    rec["fg_slices"] = int(np.count_nonzero(fg_per_slice))

    # 首尾连续空切片（体积与掩膜）
    first_nz = last_nz = None
    if rec["fg_slices"] > 0:
        nz_idx = np.nonzero(fg_per_slice)[0]
        first_nz, last_nz = int(nz_idx[0]), int(nz_idx[-1])
    rec["fg_first_slice"] = first_nz
    rec["fg_last_slice"] = last_nz
    body_per_slice = np.sum(body, axis=tuple(i for i in range(body.ndim) if i != axis))
    bnz = np.nonzero(body_per_slice)[0]
    rec["body_first_slice"] = int(bnz[0]) if bnz.size else None
    rec["body_last_slice"] = int(bnz[-1]) if bnz.size else None

    # 每个前景切片上的连通区域数（4 邻域？—— 这里用一维近似：统计每行的变化段，成本低且能发现强碎片化）
    # 更直接且有意义的碎片指标：单个前景切片的面积分布
    if rec["fg_slices"] > 0:
        areas = fg_per_slice[fg_per_slice > 0]
        rec["fg_area_min"] = int(areas.min())
        rec["fg_area_median"] = float(np.median(areas))
        rec["fg_area_max"] = int(areas.max())
        rec["fg_area_mean"] = float(areas.mean())
        rec["max_lesion_frac_of_slice"] = float(
            areas.max() / max(1, int(np.prod([vol.shape[i] for i in range(vol.ndim) if i != axis])))
        )
    else:
        rec["fg_area_min"] = rec["fg_area_median"] = rec["fg_area_max"] = rec["fg_area_mean"] = None

    # 单体素/极小碎片占比（标注噪声的粗略信号）
    if rec["fg_voxels"] > 0 and vol.ndim == 3:
        # 逐切片：统计"面积 == 1"的切片数占比
        tiny_slices = int(np.count_nonzero(fg_per_slice == 1))
        rec["fragile_slices_area1"] = tiny_slices
        rec["fragile_slices_frac"] = float(tiny_slices / max(1, rec["fg_slices"]))
    else:
        rec["fragile_slices_area1"] = None
        rec["fragile_slices_frac"] = None

    # 病灶在体积中的位置（相对坐标，判断是否贴边）
    bb_fg = nonzero_bbox(fg)
    bb_body = nonzero_bbox(body)
    rec["fg_bbox"] = bb_fg
    rec["body_bbox"] = bb_body
    rec["fg_bbox_inside_body"] = bbox_inside(bb_fg, bb_body)
    if bb_fg is not None:
        cx = [(bb_fg[d][0] + bb_fg[d][1]) / 2.0 for d in range(len(bb_fg))]
        rec["fg_center_norm"] = tuple(
            round(cx[d] / max(1, vol.shape[d]), 3) for d in range(len(cx))
        )
    else:
        rec["fg_center_norm"] = None

    # 肿瘤体积 cm^3
    if rec["voxel_mm3"] and rec["fg_voxels"]:
        rec["tumor_volume_cm3"] = float(rec["fg_voxels"] * rec["voxel_mm3"] / 1000.0)
        rec["tumor_frac_of_body"] = float(rec["fg_voxels"] / max(1, rec["body_voxels"]))
    else:
        rec["tumor_volume_cm3"] = None
        rec["tumor_frac_of_body"] = None

    return rec


# --------------------------------------------------------------------------------------
# 3. 报告输出
# --------------------------------------------------------------------------------------

def hr(title: str, ch: str = "=") -> None:
    print()
    print(ch * 78)
    print(title)
    print(ch * 78)


def p(line: str = "") -> None:
    print(line)


def report_filesystem(fs: dict) -> None:
    hr("[1] 文件系统与目录结构")
    p(f"data_dir={fs['dir']}")
    p(f"exists={fmt_num(fs['exists'])}")
    if not fs["exists"]:
        p("结论：目录不存在，后续检查全部跳过。请确认 --data-dir 是否写对。")
        return
    p(f"top_level_files={len(fs['top_entries'])}")
    p(f"top_level_subdirs={len(fs['subdirs'])}")
    if fs["subdirs"]:
        for name, n in fs["subdirs"][:30]:
            p(f"  subdir={name} entries={n}")
        if len(fs["subdirs"]) > 30:
            p(f"  ... 其余 {len(fs['subdirs']) - 30} 个子目录省略")
    p(f"files_total_recursive={fs['n_files_total']}")
    p(f"nii_files={fs['n_nii']}")
    p(f"dicom_like_files={fs['n_dicom']}")
    p(f"total_size_mb={fmt_num(round(fs['total_bytes'] / 1024 / 1024, 2))}")
    if fs["zero_byte"]:
        p(f"zero_byte_files={len(fs['zero_byte'])} -> {fs['zero_byte'][:10]}")
    if fs["other_ext"]:
        p("other_ext_counts=" + ", ".join(f"{k}:{v}" for k, v in fs["other_ext"].most_common(15)))
    if fs["other_samples"]:
        p("other_samples=" + ", ".join(fs["other_samples"]))
    if fs["top_entries"]:
        show = fs["top_entries"][:80]
        p(f"top_level_listing(first {len(show)} of {len(fs['top_entries'])}):")
        for name in show:
            p(f"  {name}")


def report_pairing(cls: dict) -> None:
    hr("[2] 文件名解析与 case 配对")
    vols, segs = cls["volumes"], cls["segs"]
    p(f"volume_matched={len(vols)}")
    p(f"segmentation_matched={len(segs)}")
    p(f"unmatched_nii={len(cls['unmatched'])}")
    for u in cls["unmatched"][:20]:
        p(f"  unmatched={u}")
    if cls["duplicates"]:
        p(f"duplicate_case_files={len(cls['duplicates'])}")
        for d in cls["duplicates"][:20]:
            p(f"  duplicate={d}")
    only_vol = sort_cases(set(vols) - set(segs))
    only_seg = sort_cases(set(segs) - set(vols))
    both = sort_cases(set(vols) & set(segs))
    p(f"case_ids_both={len(both)}")
    p(f"case_ids_volume_only={len(only_vol)}" + (f" -> {only_vol[:30]}" if only_vol else ""))
    p(f"case_ids_seg_only={len(only_seg)}" + (f" -> {only_seg[:30]}" if only_seg else ""))
    if both:
        p(f"case_id_min={both[0]}")
        p(f"case_id_max={both[-1]}")
        ids = [int(c) for c in both if c.isdigit()]
        if ids:
            missing = sorted(set(range(min(ids), max(ids) + 1)) - set(ids))
            p(f"case_id_range={min(ids)}-{max(ids)}")
            p(f"gaps_in_range={len(missing)}" + (f" -> {missing[:30]}" if missing else ""))
    p("naming_convention=volume-<id>.nii 与 segmentation-<id>.nii 一一对应（脚本按此正则解析）")


def report_geometry(recs: list) -> None:
    hr("[3] 几何与头信息（每个 case 一行）")
    p("说明：case 用 intersection 口径（volume 与 segmentation 都存在才计算）。")
    p("")
    p("case | vol_shape | seg_shape | spacing_mm | dtype(v/s) | axcodes(v/s) | affine_match | max_diff | voxel_mm | size_MB(v/s)")
    if any(r.get("status") == "OK_SKIPPED" for r in recs):
        p("注：status=OK_SKIPPED 表示该 case 因 --skip-stats 只读了头信息，seg_* 与误差列显示 NA。")
    for r in recs:
        if r.get("status") != "OK":
            p(f"{r['case']} | status={r.get('status')} error={str(r.get('error', ''))[:120]}")
            continue
        p(
            f"{r['case']} | {r['vol_shape']} | {r.get('seg_shape', 'NA')} | {fmt_tuple(r['vol_zooms'], 4)} | "
            f"{r['vol_dtype']}/{r.get('seg_dtype', 'NA')} | {r.get('vol_axcodes', 'NA')}/{r.get('seg_axcodes', 'NA')} | "
            f"{fmt_num(r.get('affine_match'))} | {fmt_num(r.get('max_affine_diff'))} | {fmt_num(r.get('voxel_mm'))} | "
            f"{fmt_num(round((r.get('vol_bytes') or 0) / 1e6, 2))}/{fmt_num(round((r.get('seg_bytes') or 0) / 1e6, 2))}"
        )

    ok = [r for r in recs if r.get("status") == "OK"]
    if not ok:
        return

    shapes = collections.Counter(r["vol_shape"] for r in ok)
    spacings = collections.Counter(tuple(round(z, 4) for z in r["vol_zooms"]) for r in ok)
    axcodes = collections.Counter(r.get("vol_axcodes", "NA") for r in ok)
    dtypes = collections.Counter(f"{r['vol_dtype']}/{r.get('seg_dtype', 'NA')}" for r in ok)
    ndims = collections.Counter(r.get("ndim", 3) for r in ok)

    p("")
    p("--- 聚合 ---")
    p(f"n_dims_distribution={dict(ndims)}")
    p(f"vol_dtype/seg_dtype_distribution={dict(dtypes)}")
    p(f"unique_shapes={len(shapes)}")
    for s, c in shapes.most_common(10):
        p(f"  shape={s} count={c}")
    p(f"unique_spacings_mm={len(spacings)}")
    for s, c in spacings.most_common(10):
        p(f"  spacing={s} count={c}")
    p(f"unique_axcodes={len(axcodes)} -> {dict(axcodes)}")
    p(f"shape_match_all={fmt_num(all(r['shape_match'] for r in ok))}")
    p(f"spacing_match_all={fmt_num(all(r['spacing_match'] for r in ok))}")
    p(f"affine_match_all={fmt_num(all(r['affine_match'] for r in ok))}")
    if not all(r["affine_match"] for r in ok):
        bad = [r["case"] for r in ok if not r["affine_match"]]
        p(f"affine_mismatch_cases={bad[:20]}")
    p(f"max_affine_diff_overall={fmt_num(max(r['max_affine_diff'] for r in ok))}")

    dims = np.array([r["vol_shape"] for r in ok], dtype=np.int64)
    p(f"shape_min_per_axis={tuple(int(x) for x in dims.min(axis=0))}")
    p(f"shape_max_per_axis={tuple(int(x) for x in dims.max(axis=0))}")
    p(f"shape_mean_per_axis={tuple(round(float(x), 1) for x in dims.mean(axis=0))}")

    # 可行的 patch 尺寸建议
    min_dim = dims.min(axis=0)
    cand = [d for d in (64, 96, 128, 160, 192) if d <= int(min_dim.min())]
    p(f"patch_size_candidates_uniform={cand}（均能整除/覆盖所有 case 的最小边 {int(min_dim.min())}）")
    if set(ndims) - {3}:
        p(f"[!] 存在非 3D 数据 ndim={dict(ndims)}：多通道/多时序卷不能按单通道 CT 处理，需单独确认。")
    if len(shapes) > 1:
        p(f"[!] shape 不统一（{len(shapes)} 种）：训练时必须 resize/crop 到统一尺寸，或使用 patch 采样。")
    if len(spacings) > 1:
        p(f"[!] spacing 不统一（{len(spacings)} 种）：重采样到统一 spacing 会更稳，否则物理尺度不一致。")
    p("reference_affine(第一个 case)=" + ok[0]["vol_affine"])


def report_intensity(recs: list) -> None:
    hr("[4] 图像强度分布（归一化依据）")
    ok = [r for r in recs if r["status"] == "OK"]
    if not ok:
        p("无可用 case。")
        return
    qs = ok[0]["pct_qs"]
    qhdr = ",".join(f"p{fmt_num(q)}" for q in qs)
    p("--- 整幅图像 ---")
    p(f"case | min | max | mean | std | {qhdr} | nonfinite")
    for r in ok:
        p(
            f"{r['case']} | {fmt_num(r['vol_min'])} | {fmt_num(r['vol_max'])} | {fmt_num(r['vol_mean'])} | "
            f"{fmt_num(r['vol_std'])} | {','.join(fmt_num(x) for x in r['vol_percentiles'])} | {r['vol_nonfinite']}"
        )
    p("")
    p("--- 身体区域（CT: 体素 > -900HU；无符号数据: 体素 > 0）---")
    p(f"case | body_voxels | body_frac | mean | std | {qhdr} | body_slices")
    for r in ok:
        p(
            f"{r['case']} | {r['body_voxels']} | {fmt_num(r['body_fraction_total'])} | {fmt_num(r.get('body_mean'))} | "
            f"{fmt_num(r.get('body_std'))} | {','.join(fmt_num(x) for x in r['body_percentiles'])} | {r['body_slices']}"
        )
    p("")
    p("--- 前景/肿瘤区域（只看掩膜内体素的强度）---")
    p(f"case | fg_voxels | {qhdr}")
    for r in ok:
        p(
            f"{r['case']} | {r['fg_voxels']} | "
            f"{','.join(fmt_num(x) for x in r['fg_percentiles'])}"
        )
    p("")
    p("--- 聚合（中位数口径，用于选窗/归一化） ---")
    body_lo = [r["body_percentiles"][1] for r in ok]        # p1
    body_hi = [r["body_percentiles"][7] for r in ok]        # p99
    body_med = [r["body_percentiles"][4] for r in ok]       # p50
    p(f"body_p1_median={fmt_num(np.median(body_lo))} body_p50_median={fmt_num(np.median(body_med))} body_p99_median={fmt_num(np.median(body_hi))}")
    vmin = [r["vol_min"] for r in ok]
    vmax = [r["vol_max"] for r in ok]
    p(f"global_min={fmt_num(min(vmin))} global_max={fmt_num(max(vmax))}")
    p("提示：若 global_min < -1000 或 global_max > 4000，说明强度不是标准 HU，归一化前需确认。")
    lo = [r["zscore_clip_lo"] for r in ok if r.get("zscore_clip_lo") is not None]
    hi = [r["zscore_clip_hi"] for r in ok if r.get("zscore_clip_hi") is not None]
    if lo and hi:
        p("z-score 裁剪窗参考（body 区域 p0.5 / p99.5 的中位数）："
          f"lo={fmt_num(np.median(lo))} hi={fmt_num(np.median(hi))} -> "
          "MONAI: ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True)")
    p("z-score 参考（对 body 区域）：per-case body_mean/body_std 已在上面列出；"
      "也可用 ScaleIntensityRangePercentiles 并按 body p1/p99 裁剪离群值。")


def report_labels(recs: list) -> None:
    hr("[5] 标签语义与取值")
    ok = [r for r in recs if r["status"] == "OK"]
    if not ok:
        p("无可用 case。")
        return
    p("case | n_labels | values(value:count) | has_bg | integer | negative | fractional")
    value_sets = collections.Counter()
    for r in ok:
        vals = r["seg_values"]
        vs = ",".join(f"{v}:{c}" for v, c in vals)
        value_sets[tuple(v for v, _ in vals)] += 1
        p(
            f"{r['case']} | {r['n_unique_labels']} | {vs} | "
            f"{fmt_num(r['has_background'])} | {fmt_num(r['seg_is_integer'])} | "
            f"{fmt_num(r['seg_negative'])} | {fmt_num(r['seg_fractional'])}"
        )
    p("")
    p("--- 聚合 ---")
    p(f"unique_value_set_patterns={len(value_sets)}")
    for pat, c in value_sets.most_common(10):
        p(f"  labels={pat} count={c}")
    all_vals = sorted({v for r in ok for v, _ in r["seg_values"]})
    p(f"union_of_all_labels={all_vals}")
    p(f"all_integer={fmt_num(all(r['seg_is_integer'] for r in ok))}")
    p(f"any_negative={fmt_num(any(r['seg_negative'] for r in ok))}")
    p(f"any_empty_mask={fmt_num(any(r['fg_voxels'] == 0 for r in ok))}")
    if all_vals and set(all_vals) <= {"0", "1"}:
        p("结论：二分类掩膜（0=背景，1=肿瘤/病灶）。")
    elif all_vals and set(all_vals) <= {"0", "1", "2"}:
        p("结论：最多三类。请在报告里确认 1/2 的语义（例如 1=肝脏 2=肿瘤，或 1=肿瘤 2=其他）。")
    else:
        p("结论：取值超出 {0,1,2}，需人工确认标签定义。")


def report_scale(recs: list) -> None:
    hr("[6] 任务规模：肿瘤体积、切片覆盖、patch 可行性")
    ok = [r for r in recs if r["status"] == "OK"]
    if not ok:
        p("无可用 case。")
        return
    p("case | tumor_voxels | tumor_cm3 | frac_of_body | fg_slices/total | fg_slice_range | area_min/med/max | area1_frac | fg_in_body | center_norm")
    for r in ok:
        rng = "NA"
        if r["fg_first_slice"] is not None:
            rng = f"{r['fg_first_slice']}-{r['fg_last_slice']}"
        p(
            f"{r['case']} | {r['fg_voxels']} | {fmt_num(r.get('tumor_volume_cm3'))} | "
            f"{fmt_num(r.get('tumor_frac_of_body'))} | {r['fg_slices']}/{r['n_slices']} | {rng} | "
            f"{fmt_num(r.get('fg_area_min'))}/{fmt_num(r.get('fg_area_median'))}/{fmt_num(r.get('fg_area_max'))} | "
            f"{fmt_num(r.get('fragile_slices_frac'))} | {fmt_num(r['fg_bbox_inside_body'])} | {r.get('fg_center_norm')}"
        )
    p("")
    p("--- 聚合 ---")
    fgs = np.array([r["fg_voxels"] for r in ok], dtype=np.float64)
    cm3 = np.array([r["tumor_volume_cm3"] for r in ok if r.get("tumor_volume_cm3") is not None], dtype=np.float64)
    p(f"fg_voxels: min={int(fgs.min())} median={int(np.median(fgs))} mean={fmt_num(fgs.mean())} max={int(fgs.max())}")
    if cm3.size:
        p(f"tumor_cm3: min={fmt_num(cm3.min())} median={fmt_num(np.median(cm3))} mean={fmt_num(cm3.mean())} max={fmt_num(cm3.max())}")
        p(f"pct_cases_with_tumor_under_1cm3={fmt_num(round(float(np.mean(cm3 < 1.0)), 3))}")
    ratios = [r["fg_slices"] / max(1, r["n_slices"]) for r in ok]
    p(f"fg_slice_ratio: min={fmt_num(min(ratios))} median={fmt_num(np.median(ratios))} max={fmt_num(max(ratios))}")
    p(f"cases_with_bbox_outside_body={[r['case'] for r in ok if not r['fg_bbox_inside_body']]}")
    p(f"body_slice_ratio_median={fmt_num(np.median([r['body_slices'] / max(1, r['n_slices']) for r in ok]))}")
    p("")
    p("--- 掩膜语义合理性自检（防止把肝脏/器官掩膜误当病灶）---")
    cov = [r.get("tumor_frac_of_body") for r in ok if r.get("tumor_frac_of_body") is not None]
    if cov:
        p(f"fg_over_body_ratio: min={fmt_num(min(cov))} median={fmt_num(np.median(cov))} max={fmt_num(max(cov))}")
        big = [r["case"] for r in ok if (r.get("tumor_frac_of_body") or 0) > 0.35]
        p(f"cases_where_mask_covers_over_35pct_of_body={big}")
        p("判读：肝肿瘤通常只占身体的 0.1%~5%；若多数 case 远高于此，掩膜很可能是肝脏/器官而非病灶。")
    contrast = []
    for r in ok:
        if r.get("fg_percentiles") and r["fg_percentiles"][4] is not None and r.get("body_mean") is not None:
            std = r.get("body_std") or 0.0
            if std > 0:
                contrast.append((r["fg_percentiles"][4] - r["body_mean"]) / std)
    if contrast:
        p(f"fg_median_minus_body_mean_in_body_std: min={fmt_num(min(contrast))} median={fmt_num(np.median(contrast))} max={fmt_num(max(contrast))}")
        p("判读：该值为前景中位强度相对身体均值的 z 分数；明显偏离 0 说明病灶与肝实质有强度差，任务可学。")
    m = [r["case"] for r in ok if r["body_voxels"] == 0]
    if m:
        p(f"cases_with_empty_body_mask={m}（空气阈值口径可能不适配该数据，请核对强度范围）")
    p("")
    p("--- 数据划分提示 ---")
    ids = [int(r["case"]) for r in ok if str(r["case"]).isdigit()]
    if ids:
        p(f"n_cases={len(ids)}")
        p("若做 5 折交叉验证，每折约 " + str(round(len(ids) / 5, 1)) + f" 个 case（训练约 {round(len(ids) * 0.8, 1)}，验证约 {round(len(ids) * 0.2, 1)}）。")
        p("注意：小样本 + 肿瘤体积差异大时，建议按 tumor_cm3 分层划分（分层依据见上表）。")


def report_conclusion(recs: list, cls: dict) -> None:
    hr("[7] 机器可读的一句话总结（可直接贴进 docs/data.md）")
    ok = [r for r in recs if r["status"] == "OK"]
    if not ok:
        p("no_valid_cases=1")
        return
    shapes = {r["vol_shape"] for r in ok}
    spacings = {tuple(round(z, 4) for z in r["vol_zooms"]) for r in ok}
    ax = {r["vol_axcodes"] for r in ok}
    dtypes = {r["vol_dtype"] for r in ok}
    seg_dtypes = {r["seg_dtype"] for r in ok}
    labels = sorted({v for r in ok for v, _ in r["seg_values"]}, key=lambda s: float(s))
    cm3 = [r["tumor_volume_cm3"] for r in ok if r.get("tumor_volume_cm3") is not None]
    p(f"n_cases={len(ok)}")
    p(f"mask_semantics={'binary' if labels == ['0', '1'] else 'multiclass'}")
    p(f"label_values={labels}")
    p(f"volume_dtype={sorted(dtypes)} segmentation_dtype={sorted(seg_dtypes)}")
    p(f"shape_uniform={fmt_num(len(shapes) == 1)} unique_shapes={len(shapes)}")
    p(f"spacing_uniform={fmt_num(len(spacings) == 1)} unique_spacings={len(spacings)}")
    p(f"axcodes={sorted(ax)}")
    p(f"shape_example={ok[0]['vol_shape']}")
    p(f"spacing_example={tuple(round(z, 4) for z in ok[0]['vol_zooms'])}")
    p(f"tumor_cm3_median={fmt_num(np.median(cm3)) if cm3 else 'NA'}")
    p(f"foreground_slice_ratio_median={fmt_num(np.median([r['fg_slices'] / max(1, r['n_slices']) for r in ok]))}")
    p(f"hint=volume-<id>.nii 为影像，segmentation-<id>.nii 为同空间掩膜，配对 id 共 {len(cls['volumes'])} 个 volume / {len(cls['segs'])} 个 segmentation")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CT 数据探针：打印数据目录的全部结构与内容事实")
    ap.add_argument("--data-dir", default="data", help="数据目录（相对仓库根或绝对路径），默认 data")
    ap.add_argument("--vol-pattern", default=r"^volume[-_](\d+)$",
                    help="volume 文件名的正则（不含扩展名），默认 ^volume[-_](\\d+)$")
    ap.add_argument("--seg-pattern", default=r"^segmentation[-_](\d+)$",
                    help="segmentation 文件名的正则（不含扩展名），默认 ^segmentation[-_](\\d+)$")
    ap.add_argument("--limit-cases", type=int, default=0, help="只检查前 N 个 case（0 = 全部）")
    ap.add_argument("--write-doc", default="", help="额外把完整输出写入该路径（不影响 stdout）")
    ap.add_argument("--skip-stats", action="store_true", help="只做文件系统与头信息检查，不读体素数据")
    args = ap.parse_args(argv)

    # 允许通过命令行覆盖命名正则（正则写错时给出明确提示，而不是抛 traceback）
    global VOLUME_PATTERNS, SEG_PATTERNS
    try:
        VOLUME_PATTERNS = (re.compile(args.vol_pattern),)
        SEG_PATTERNS = (re.compile(args.seg_pattern),)
    except re.error as exc:
        eprint(f"[错误] 文件名正则无法编译：{exc}")
        eprint(r"        示例：--vol-pattern '^volume[-_](\d+)$' --seg-pattern '^segmentation[-_](\d+)$'")
        return 2

    buf = []

    class Tee:
        def write(self, s):
            buf.append(s)
            sys.__stdout__.write(s)

        def flush(self):
            sys.__stdout__.flush()

    if args.write_doc:
        sys.stdout = Tee()

    try:
        hr("CT 数据探针报告")
        p(f"生成时间(UTC)={datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
        p(f"python={sys.version.split()[0]}")
        p(f"numpy={np.__version__} nibabel={nib.__version__}")
        p(f"cwd={os.getcwd()}")
        p("脚本只读数据，不写入任何数据文件。")
        p("运行方式：在项目根目录执行 python scripts/probe_data.py（详见脚本文件头 docstring）。")

        fs = scan_filesystem(args.data_dir)
        report_filesystem(fs)

        cls = classify_files(args.data_dir)
        report_pairing(cls)

        cases = sort_cases(set(cls["volumes"]) & set(cls["segs"]))
        only_vol = sort_cases(set(cls["volumes"]) - set(cls["segs"]))
        only_seg = sort_cases(set(cls["segs"]) - set(cls["volumes"]))
        if args.limit_cases > 0:
            cases = cases[: args.limit_cases]
            p(f"\n[限制] --limit-cases={args.limit_cases}，只检查 {len(cases)} 个 case。")

        recs = []
        if not cls["volumes"] and not cls["segs"]:
            p("\n[!] 没有解析出任何 volume/segmentation 文件，跳过图像检查。")
            p("    可能原因：命名不同（用 --vol-pattern/--seg-pattern 覆盖），或数据在子目录中。")
        else:
            total = len(cases)
            for i, case in enumerate(cases, 1):
                vp = os.path.join(args.data_dir, cls["volumes"][case])
                sp = os.path.join(args.data_dir, cls["segs"][case])
                # 进度打到 stderr，不污染 stdout 的报告正文
                eprint(f"[{i}/{total}] case {case} ...")
                if args.skip_stats:
                    rec = {"case": case, "status": "OK_SKIPPED", "error": "", "vol_path": vp, "seg_path": sp,
                           "vol_bytes": os.path.getsize(vp) if os.path.exists(vp) else None,
                           "seg_bytes": os.path.getsize(sp) if os.path.exists(sp) else None}
                    img = nib.load(vp)
                    rec.update({
                        "vol_shape": tuple(int(s) for s in img.shape), "seg_shape": None, "ndim": img.ndim,
                        "vol_zooms": tuple(float(z) for z in img.header.get_zooms()[: img.ndim]),
                        "vol_dtype": str(img.get_data_dtype()),
                    })
                    recs.append(rec)
                    continue
                try:
                    recs.append(inspect_pair(case, vp, sp))
                except Exception as exc:  # noqa: BLE001 - 单 case 失败不应中断整体
                    eprint(f"[warn] case {case} 检查失败: {type(exc).__name__}: {exc}")
                    recs.append({"case": case, "status": "READ_ERROR",
                                 "error": f"{type(exc).__name__}: {exc}",
                                 "vol_path": vp, "seg_path": sp, "vol_bytes": None, "seg_bytes": None})
            if not args.skip_stats:
                report_geometry(recs)
                report_intensity(recs)
                report_labels(recs)
                report_scale(recs)
                report_conclusion(recs, cls)
            else:
                # --skip-stats 下 seg_* 字段不存在，report_geometry 用 .get() 兜底输出 NA
                report_geometry(recs)
                p()
                p("[说明] --skip-stats 只读头信息，未读取体素：seg_shape/affine 对比/强度/标签/规模各节均未计算。")

        if only_vol or only_seg:
            p()
            p("[!] 存在无法配对的 case：volume_only=" + str(only_vol[:20]) + " seg_only=" + str(only_seg[:20]))
            p("    这些 case 不能直接用于有监督训练，需人工确认。")

        p()
        p("[!] 本报告是撰写 docs/data.md 的实测依据：请把 stdout 整段原样贴回对话。")
        p("    不要用 --write-doc 把报告文件落在仓库里（probe_report.txt 已加入 .gitignore，写完请删除）。")
        hr("报告结束", "-")
        p("请把以上全部 stdout 原样贴回，用于撰写 docs/data.md。")
    finally:
        if args.write_doc:
            sys.stdout = sys.__stdout__
            with open(args.write_doc, "w", encoding="utf-8") as fh:
                fh.write("".join(buf))
            eprint(f"[info] 已写入 {args.write_doc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
