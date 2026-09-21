#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""CT 标签语义探针：判定 segmentation 中 label 1 / label 2 各自代表什么器官，并顺带核对强度定标与重复病例。

它输出 6 节定向补测报告（逐 label 统计、嵌套关系、肿瘤/肝脏体积比、整数域与 HU 域对照、仿射不一致病例的重采样核对、重复病例指纹），
用于回答"这是肝脏+肝内肿瘤多类分割还是单类病灶分割"这一决定训练目标的问题。

用法
----
在本项目根目录执行::

    # 主用法：扫描 ./data，打印全部 6 节补测报告
    python scripts/probe_labels.py

    # 只跑前 3 个 case，快速确认脚本本身能跑通
    python scripts/probe_labels.py --limit-cases 3

参数说明与默认值：

===================================  ==================================================
``--data-dir PATH``                  数据目录，默认 ``data``；相对路径相对当前工作目录
``--limit-cases N``                  只检查前 N 个 case，默认 ``0`` 表示全部
``--vol-pattern REGEX``              影像文件名正则，默认 ``^volume[-_](\d+)$``
``--seg-pattern REGEX``              掩膜文件名正则，默认 ``^segmentation[-_](\d+)$``
``--skip-affine-fix``                跳过仿射不一致病例的重采样核对（该步最慢）
``-h`` / ``--help``                  打印参数帮助
===================================  ==================================================

判读要点（输出里会给出明确结论）：

* 若某 label 完全包含另一 label 且体积远大于它，则前者是器官（肝脏）、后者是其中的病灶。
* 若两个 label 有大量体素相交，则属于互斥（mutually exclusive）标注，不能当作嵌套关系。
* 若某 label 大部分体素贴到身体外部或落在空气阈值以下，则该 label 可能是伪影或标注错误。
* 输出为逐行 ``key=value`` 纯文本；脚本只读数据，不写入任何文件；进度与警告走 stderr。
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


def eprint(*args) -> None:
    print(*args, file=sys.stderr)


def fmt_num(v) -> str:
    """把数值格式化成紧凑字符串；缺失值统一输出 NA。"""
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
        if abs(f) >= 1e5 or abs(f) < 1e-3:
            return f"{f:.4g}"
        return f"{f:.2f}"
    return str(v)


def pct(values, q):
    """安全分位数：空输入返回 None。"""
    if values is None or len(values) == 0:
        return None
    return float(np.percentile(values, q))


def summarize(arr: np.ndarray, mask: np.ndarray | None = None, qs=(0.5, 1, 5, 25, 50, 75, 95, 99, 99.5)) -> dict:
    """返回给定体素集合的强度摘要（分位数 + 均值 + 标准差）。"""
    flat = np.asarray(arr).ravel()
    if mask is not None:
        flat = flat[np.asarray(mask).ravel()]
    if flat.size == 0:
        return {"n": 0}
    flat = flat.astype(np.float64)
    out = {"n": int(flat.size), "mean": float(flat.mean()), "std": float(flat.std())}
    for q in qs:
        out[f"p{q}"] = float(np.percentile(flat, q))
    return out


def bbox_of(mask: np.ndarray):
    """紧凑包围盒 ((x0,x1),(y0,y1),(z0,z1))，空掩膜返回 None。"""
    idx = np.nonzero(mask)
    if idx[0].size == 0:
        return None
    return tuple((int(a.min()), int(a.max())) for a in idx)


def bbox_str(bb) -> str:
    return "NA" if bb is None else "[" + " ".join(f"{a}:{b}" for a, b in bb) + "]"


def bbox_contains(outer, inner) -> bool:
    if outer is None or inner is None:
        return False
    return all(outer[d][0] <= inner[d][0] and inner[d][1] <= outer[d][1] for d in range(len(outer)))


def erode6(mask: np.ndarray) -> np.ndarray:
    """6 邻域二值腐蚀；用 numpy 实现，避免引入 scipy 依赖。"""
    out = mask.copy()
    out[1:, :, :] &= mask[:-1, :, :]
    out[:-1, :, :] &= mask[1:, :, :]
    out[:, 1:, :] &= mask[:, :-1, :]
    out[:, :-1, :] &= mask[:, 1:, :]
    out[:, :, 1:] &= mask[:, :, :-1]
    out[:, :, :-1] &= mask[:, :, 1:]
    return out


def dilate6(mask: np.ndarray) -> np.ndarray:
    """6 邻域二值膨胀；边界向外不扩散。"""
    out = mask.copy()
    out[1:, :, :] |= mask[:-1, :, :]
    out[:-1, :, :] |= mask[1:, :, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


# --------------------------------------------------------------------------------------
# 文件枚举与配对（与 probe_data.py 同口径，保持两脚本一致）
# --------------------------------------------------------------------------------------

NII_SUFFIXES = (".nii.gz", ".nii")
VOLUME_PATTERNS = (re.compile(r"^volume[-_](\d+)$"),)
SEG_PATTERNS = (re.compile(r"^segmentation[-_](\d+)$"),)


def strip_suffix(name: str) -> str:
    for suf in NII_SUFFIXES:
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return name


def collect_pairs(data_dir: str) -> tuple:
    """返回 (volumes, segs, 未配对列表)，均为 case_id -> 文件名。"""
    vols, segs, unmatched = {}, {}, []
    if not os.path.isdir(data_dir):
        return vols, segs, unmatched

    def first_match(patterns, stem):
        """按顺序尝试每个正则，返回第一个匹配到的 case id；都不匹配则返回 None。"""
        for pat in patterns:
            m = pat.match(stem)
            if m:
                return m.group(1)
        return None

    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith((".nii", ".nii.gz")):
            continue
        stem = strip_suffix(fn)
        for patterns, bucket, kind in ((VOLUME_PATTERNS, vols, "volume"),
                                       (SEG_PATTERNS, segs, "segmentation")):
            cid = first_match(patterns, stem)
            if cid is None:
                continue
            if cid in bucket:
                unmatched.append(f"{fn} (重复的 {kind} {cid}，已有 {bucket[cid]})")
            else:
                bucket[cid] = fn
            break
        else:
            unmatched.append(fn)
    return vols, segs, unmatched


def sort_cases(ids) -> list:
    numeric = sorted((int(c) for c in ids if str(c).isdigit()))
    others = sorted(c for c in ids if not str(c).isdigit())
    return [str(c) for c in numeric] + others


# --------------------------------------------------------------------------------------
# 单个 case 的定向检查
# --------------------------------------------------------------------------------------

def inspect_case(case: str, vol_path: str, seg_path: str, skip_affine_fix: bool) -> dict:
    rec = {"case": case, "status": "OK", "error": ""}

    vimg = nib.load(vol_path)
    simg = nib.load(seg_path)
    vol = np.asanyarray(vimg.dataobj)
    seg_raw = np.asanyarray(simg.dataobj)

    rec["vol_shape"] = tuple(int(s) for s in vimg.shape)
    rec["seg_shape"] = tuple(int(s) for s in simg.shape)
    zooms = tuple(float(z) for z in vimg.header.get_zooms()[:3])
    rec["zooms"] = zooms
    rec["voxel_mm3"] = float(np.prod(zooms))
    rec["vol_dtype"] = str(vimg.get_data_dtype())
    rec["seg_dtype"] = str(simg.get_data_dtype())
    rec["vol_axcodes"] = "".join(nib.aff2axcodes(vimg.affine))
    rec["seg_axcodes"] = "".join(nib.aff2axcodes(simg.affine))
    rec["affine_match"] = bool(np.allclose(vimg.affine, simg.affine, atol=1e-3))
    rec["affine_maxdiff"] = float(np.max(np.abs(vimg.affine - simg.affine)))
    rec["resampled"] = False

    # 仿射不一致时，把掩膜重采样到影像网格，保证后续所有体素级比较是在同一坐标系里做的
    if not rec["affine_match"] and not skip_affine_fix and rec["vol_shape"] == rec["seg_shape"]:
        seg = resample_from_to(simg, vimg, order=0)
        seg = np.asanyarray(seg.dataobj)
        rec["resampled"] = True
    else:
        seg = seg_raw

    uniq, counts = np.unique(seg, return_counts=True)
    rec["labels"] = [(fmt_num(u), int(c)) for u, c in zip(uniq.tolist(), counts.tolist())]
    rec["label_set"] = [int(u) for u in uniq.tolist()]

    # ---- 身体掩膜：与 probe_data.py 完全相同的口径（整数域 > -900），保证可比 ----
    body = vol > -900
    rec["body_voxels"] = int(np.count_nonzero(body))
    vmin = float(np.min(vol)) if vol.size else None
    vmax = float(np.max(vol)) if vol.size else None
    rec["vol_min"], rec["vol_max"] = vmin, vmax
    # 分位数用固定步长采样估计：精度足够，且不会为 3 亿体素的大卷分配整份副本
    rec["vol_p0.5"] = pct(vol.ravel()[:: 16], 0.5)
    rec["vol_p99.5"] = pct(vol.ravel()[:: 16], 99.5)

    # ---- 逐 label 的强度与形状摘要 ----
    label_info = {}
    for lab in rec["label_set"]:
        m = seg == lab
        n = int(np.count_nonzero(m))
        info = {"n": n, "vol_cm3": n * rec["voxel_mm3"] / 1000.0 if rec["voxel_mm3"] else None}
        if lab > 0 and n > 0:
            s = summarize(vol, m)
            info.update({k: s.get(k) for k in ("mean", "std", "p0.5", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99.5")})
            info["bbox"] = bbox_of(m)
        label_info[lab] = info
    rec["label_info"] = label_info

    # 身体内强度分位数，用于判断“整数域 vs HU 域”
    # 用固定步长在身体掩膜上采样，避免 np.nonzero 为大卷生成上亿条索引
    bvals = vol.ravel()[::16][body.ravel()[::16]].astype(np.float64)
    if bvals.size > 2_000_000:
        bvals = bvals[:: max(1, bvals.size // 2_000_000)]
    if bvals.size == 0:
        rec["body_p0.5"] = rec["body_p99.5"] = rec["body_p1"] = rec["body_p99"] = None
    else:
        rec["body_p0.5"] = float(np.percentile(bvals, 0.5))
        rec["body_p99.5"] = float(np.percentile(bvals, 99.5))
        rec["body_p1"] = float(np.percentile(bvals, 1))
        rec["body_p99"] = float(np.percentile(bvals, 99))

    # ---- label 之间的嵌套 / 相交关系（只对两个前景 label 有意义） ----
    fg_labels = [l for l in rec["label_set"] if l > 0]
    rel = {}
    if len(fg_labels) == 2:
        a, b = fg_labels
        ma, mb = seg == a, seg == b
        na, nb = int(ma.sum()), int(mb.sum())
        inter = int(np.count_nonzero(ma & mb))
        small, large = (a, b) if na <= nb else (b, a)
        msmall, mlarge = (ma, mb) if na <= nb else (mb, ma)
        nsmall, nlarge = (na, nb) if na <= nb else (nb, na)
        rel = {
            "a": a, "b": b, "n_a": na, "n_b": nb, "intersection": inter,
            "small": small, "large": large, "n_small": nsmall, "n_large": nlarge,
            "frac_small_in_large": float(np.count_nonzero(msmall & mlarge) / max(1, nsmall)),
            "frac_large_in_small": float(np.count_nonzero(msmall & mlarge) / max(1, nlarge)),
            "bbox_contains": bbox_contains(bbox_of(mlarge), bbox_of(msmall)),
            "vol_ratio_small_over_large": float(nsmall / max(1, nlarge)),
        }
        # 小 label 是否贴到身体外（用膨胀一圈后仍不在 body 里的比例）
        ring = dilate6(msmall) & ~msmall
        if np.count_nonzero(ring) > 0:
            rel["frac_small_ring_outside_body"] = float(np.count_nonzero(ring & ~body) / np.count_nonzero(ring))
        # 小 label 的实心度：腐蚀一圈后留存比例（薄环状结构会接近 0）
        rel["small_erode1_survival"] = float(np.count_nonzero(erode6(msmall)) / max(1, nsmall))
    rec["relation"] = rel

    # ---- 大 label 的“实心度”：判断它是不是肝脏那种实心器官 ----
    if len(fg_labels) >= 1:
        big = max(fg_labels, key=lambda l: label_info[l]["n"])
        mbig = seg == big
        bb = bbox_of(mbig)
        if bb is not None:
            cube = int(np.prod([bb[d][1] - bb[d][0] + 1 for d in range(3)]))
            rec["big_label"] = big
            rec["big_label_bbox_fill"] = float(label_info[big]["n"] / max(1, cube))
        # 实心度：腐蚀一圈后仍留存的体素比例。实心团块应远大于 0（成片留存），
        # 薄壁/环状/血管样结构腐蚀一圈后几乎全部消失。
        core = erode6(mbig)
        rec["big_label_erode1_survival"] = float(np.count_nonzero(core) / max(1, int(np.count_nonzero(mbig))))

    # ---- HU 定标线索 ----
    rec["air_peak"] = None
    if vol.size:
        hist, edges = np.histogram(vol.ravel()[::16], bins=60,
                                   range=(float(np.percentile(vol, 0.1)), float(np.percentile(vol, 99.9))))
        rec["air_peak"] = float(edges[int(np.argmax(hist))])
    rec["p1_eq_p25"] = bool(rec.get("vol_p0.5") is not None and rec["body_p1"] == rec["body_p0.5"])

    # 供“整数域 vs HU 域”判断：空气地板值相对 -1000 / -1024 的位置
    rec["floor_offset_vs_1000"] = None if vmin is None else float(vmin + 1000.0)
    rec["floor_offset_vs_1024"] = None if vmin is None else float(vmin + 1024.0)

    # ---- 重复病例指纹（对原始体素字节做 sha1）----
    rec["vol_sha1"] = hashlib.sha1(np.ascontiguousarray(vol).tobytes()).hexdigest()
    rec["seg_sha1"] = hashlib.sha1(np.ascontiguousarray(seg_raw).tobytes()).hexdigest()
    return rec


# --------------------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------------------

def hr(title: str, ch: str = "=") -> None:
    print()
    print(ch * 78)
    print(title)
    print(ch * 78)


def report_labels(recs: list) -> None:
    hr("[1] 逐 label 摘要（体积与强度）：判定每个 label 是什么组织")
    print("case | label | voxels | vol_cm3 | mean | std | p1 | p50 | p99 | bbox")
    for r in recs:
        if r["status"] != "OK":
            continue
        for lab, info in sorted(r["label_info"].items()):
            print(
                f"{r['case']} | {lab} | {info['n']} | {fmt_num(info.get('vol_cm3'))} | "
                f"{fmt_num(info.get('mean'))} | {fmt_num(info.get('std'))} | "
                f"{fmt_num(info.get('p1'))} | {fmt_num(info.get('p50'))} | {fmt_num(info.get('p99'))} | "
                f"{bbox_str(info.get('bbox'))}"
            )
    print()
    print("--- 按 label 聚合（中位数）---")
    for lab in (1, 2):
        vols_ = [r["label_info"][lab]["vol_cm3"] for r in recs
                 if r["status"] == "OK" and lab in r["label_info"] and r["label_info"][lab].get("vol_cm3") is not None]
        meds = [r["label_info"][lab].get("p50") for r in recs
                if r["status"] == "OK" and lab in r["label_info"] and r["label_info"][lab].get("p50") is not None]
        n_cases = sum(1 for r in recs if r["status"] == "OK" and lab in r["label_info"])
        if vols_:
            print(f"label {lab}: 出现 {n_cases}/{len(recs)} 个 case | vol_cm3 中位={fmt_num(np.median(vols_))} "
                  f"范围={fmt_num(min(vols_))}-{fmt_num(max(vols_))} | 强度中位={fmt_num(np.median(meds)) if meds else 'NA'}")
    print("判读：体积大且强度落在软组织范围(~40-70HU 增强肝)的是器官（肝脏）；体积小得多的是其中的病灶。")


def report_nesting(recs: list) -> None:
    hr("[2] label 嵌套关系：小 label 是否完全落在大 label 内部（决定性证据）")
    print("case | labels | small | large | n_small | n_large | frac_small_in_large | bbox_contains | vol_ratio | ring_outside_body | small_erode1_survival")
    any_two = False
    for r in recs:
        if r["status"] != "OK" or not r.get("relation"):
            continue
        any_two = True
        rel = r["relation"]
        print(
            f"{r['case']} | {r['label_set']} | label {rel['small']} | label {rel['large']} | "
            f"{rel['n_small']} | {rel['n_large']} | {fmt_num(rel['frac_small_in_large'])} | "
            f"{fmt_num(rel['bbox_contains'])} | {fmt_num(rel['vol_ratio_small_over_large'])} | "
            f"{fmt_num(rel.get('frac_small_ring_outside_body'))} | "
            f"{fmt_num(rel.get('small_erode1_survival'))}"
        )
    if not any_two:
        print("（没有同时含两个前景 label 的 case）")
    print()
    print("--- 聚合 ---")
    fracs = [r["relation"]["frac_small_in_large"] for r in recs if r.get("relation")]
    ratios = [r["relation"]["vol_ratio_small_over_large"] for r in recs if r.get("relation")]
    inter = [r["relation"]["intersection"] for r in recs if r.get("relation")]
    if fracs:
        print(f"n_two_label_cases={len(fracs)}")
        print(f"frac_small_in_large: min={fmt_num(min(fracs))} median={fmt_num(np.median(fracs))} max={fmt_num(max(fracs))}")
        print(f"vol_ratio_small/large: min={fmt_num(min(ratios))} median={fmt_num(np.median(ratios))} max={fmt_num(max(ratios))}")
        print(f"label 相交体素: max={max(inter)}（若全为 0 说明两类互斥，属于嵌套标注而非多标签重叠）")
        print("判读：frac_small_in_large 接近 1 且相交为 0 → 小 label 是大器官内部的独立病灶（肝脏+肿瘤）。")
    print()
    print("--- 大 label 是否为实心器官 ---")
    print("case | big_label | bbox_fill | erode1_survival")
    for r in recs:
        if r["status"] != "OK" or "big_label" not in r:
            continue
        print(f"{r['case']} | label {r['big_label']} | bbox_fill={fmt_num(r.get('big_label_bbox_fill'))} | "
              f"erode1_survival={fmt_num(r.get('big_label_erode1_survival'))}")
    print("判读：bbox_fill 0.3-0.7 且 erode1_survival 明显大于 0（成片留存）是实心器官（肝脏）特征；")
    print("      erode1_survival 接近 0 说明是薄壁/环状/血管样结构，不是实心器官。")


def report_scale_task(recs: list) -> None:
    hr("[3] 任务语义：肝脏体积、肿瘤体积、肿瘤占肝脏比例")
    print("case | liver_cm3 | tumor_cm3 | tumor/liver | lesion_slices | tumor_present")
    for r in recs:
        if r["status"] != "OK":
            continue
        info = r["label_info"]
        big = r.get("big_label")
        small = None
        if r.get("relation"):
            small = r["relation"]["small"]
        liver = info.get(big, {}).get("vol_cm3") if big is not None else None
        tumor = info.get(small, {}).get("vol_cm3") if small is not None else None
        ratio = (tumor / liver) if (tumor and liver) else None
        print(
            f"{r['case']} | {fmt_num(liver)} | {fmt_num(tumor)} | {fmt_num(ratio)} | "
            f"{'-' if small is None else info[small]['n']} | {fmt_num(small is not None)}"
        )
    print()
    ratios = []
    for r in recs:
        if r["status"] == "OK" and r.get("relation"):
            big, small = r["relation"]["large"], r["relation"]["small"]
            ln = r["label_info"][big]["n"]
            sn = r["label_info"][small]["n"]
            ratios.append(sn / max(1, ln))
    n_tumor = sum(1 for r in recs if r["status"] == "OK" and r.get("relation"))
    print(f"有肿瘤 label 的 case = {n_tumor}/{len(recs)}；只有肝脏的 case = {len(recs) - n_tumor}/{len(recs)}")
    if ratios:
        print(f"tumor/liver 比例: min={fmt_num(min(ratios))} median={fmt_num(np.median(ratios))} max={fmt_num(max(ratios))}")
        print("判读：若该比例中位数在 1%-20% 量级，符合“肝脏+肝内肿瘤”多类标注。")


def report_intensity_scaling(recs: list) -> None:
    hr("[4] 强度定标：整数域与 HU 域的对照（决定归一化与裁剪）")
    print("case | dtype | vol_min | vol_max | body_p0.5 | body_p1 | body_p99 | body_p99.5 | floor+1000 | floor+1024")
    for r in recs:
        if r["status"] != "OK":
            continue
        print(
            f"{r['case']} | {r['vol_dtype']} | {fmt_num(r['vol_min'])} | {fmt_num(r['vol_max'])} | "
            f"{fmt_num(r['body_p0.5'])} | {fmt_num(r['body_p1'])} | {fmt_num(r['body_p99'])} | "
            f"{fmt_num(r['body_p99.5'])} | {fmt_num(r['floor_offset_vs_1000'])} | {fmt_num(r['floor_offset_vs_1024'])}"
        )
    print()
    print("--- 聚合：空气地板值分布（-1000 附近即为标准 HU 口径）---")
    floors = collections.Counter(r["vol_min"] for r in recs if r["status"] == "OK")
    print(f"vol_min 取值分布 = {dict(floors)}")
    offs = [abs(r["floor_offset_vs_1000"]) for r in recs if r["status"] == "OK"]
    print(f"|vol_min + 1000| 中位 = {fmt_num(np.median(offs))}")
    print("判读：")
    print("  若多数 case 的 vol_min 落在 -1024/-1000 附近 → 基本是 HU 口径（可能整体偏了 24）。")
    print("  若出现 -2048/-3024 等 2 倍/3 倍关系 → 该 case 是另一套缩放（如窗外重建值或含 padding），需单独裁剪。")
    print()
    print("--- 判读说明 ---")
    print("如需看某 case 的强度直方图形状（空气峰/软组织峰位置），可对单个 case 跑：")
    print("  python scripts/probe_labels.py --limit-cases 1")
    print("其中 body_p0.5 / body_p99.5 已给出身体内的极值区间，供选归一化窗口使用。")


def report_affine(recs: list) -> None:
    hr("[5] 仿射不一致病例：重采样后体素级核对")
    mism = [r for r in recs if r["status"] == "OK" and not r["affine_match"]]
    if not mism:
        print("所有 case 的 volume 与 segmentation 仿射一致，无需特殊处理。")
        return
    print(f"仿射不一致 case 数 = {len(mism)} -> {[r['case'] for r in mism]}")
    print("case | vol_axcodes | seg_axcodes | affine_maxdiff | resampled | 重采样后 label 分布")
    for r in mism:
        print(f"{r['case']} | {r['vol_axcodes']} | {r['seg_axcodes']} | {fmt_num(r['affine_maxdiff'])} | "
              f"{fmt_num(r['resampled'])} | {r['labels']}")
    print()
    print("判读：shape 相同但 affine 不同，且 axcodes 一个是 LAS 一个是 RAS，说明掩膜与影像的 x 轴翻转，")
    print("      物理坐标相差几十厘米；必须按 affine 对齐（nibabel resample_from_to / MONAI 的 spacing 变换），")
    print("      不能直接按数组下标配对。本节的 label 分布即为对齐后重新统计的结果。")


def report_duplicates(recs: list) -> None:
    hr("[6] 疑似重复病例：体素指纹比对（防止数据泄漏）")
    vol_sha = collections.defaultdict(list)
    seg_sha = collections.defaultdict(list)
    for r in recs:
        if r["status"] != "OK":
            continue
        vol_sha[r["vol_sha1"]].append(r["case"])
        seg_sha[r["seg_sha1"]].append(r["case"])
    dup_v = {k: v for k, v in vol_sha.items() if len(v) > 1}
    dup_s = {k: v for k, v in seg_sha.items() if len(v) > 1}
    print("case | vol_sha1(前12) | seg_sha1(前12)")
    for r in recs:
        if r["status"] != "OK":
            continue
        print(f"{r['case']} | {r['vol_sha1'][:12]} | {r['seg_sha1'][:12]}")
    print()
    print(f"体素完全相同的 volume 组数 = {len(dup_v)} -> {list(dup_v.values())}")
    print(f"体素完全相同的 segmentation 组数 = {len(dup_s)} -> {list(dup_s.values())}")
    print("判读：若某组出现在同一折训练/验证两侧会造成数据泄漏；划分数据前必须先合并或剔除。")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CT 标签语义探针：判定 label 1/2 的器官语义并核对强度定标与重复病例")
    ap.add_argument("--data-dir", default="data", help="数据目录（相对项目根或绝对路径），默认 data")
    ap.add_argument("--vol-pattern", default=r"^volume[-_](\d+)$", help="影像文件名正则")
    ap.add_argument("--seg-pattern", default=r"^segmentation[-_](\d+)$", help="掩膜文件名正则")
    ap.add_argument("--limit-cases", type=int, default=0, help="只检查前 N 个 case（0 = 全部）")
    ap.add_argument("--skip-affine-fix", action="store_true", help="跳过仿射不一致病例的重采样核对")
    args = ap.parse_args(argv)

    global VOLUME_PATTERNS, SEG_PATTERNS
    try:
        VOLUME_PATTERNS = (re.compile(args.vol_pattern),)
        SEG_PATTERNS = (re.compile(args.seg_pattern),)
    except re.error as exc:
        eprint(f"[错误] 文件名正则无法编译：{exc}")
        return 2

    hr("CT 标签语义探针报告")
    print(f"生成时间(UTC)={datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"python={sys.version.split()[0]} numpy={np.__version__} nibabel={nib.__version__}")
    print(f"cwd={os.getcwd()}")
    print("脚本只读数据，不写入任何文件；进度与警告走 stderr。")

    vols, segs, unmatched = collect_pairs(args.data_dir)
    print(f"data_dir={os.path.abspath(args.data_dir)}")
    print(f"volume={len(vols)} segmentation={len(segs)} unmatched={len(unmatched)}")
    if unmatched:
        print(f"unmatched 明细={unmatched[:10]}")

    cases = sort_cases(set(vols) & set(segs))
    if args.limit_cases > 0:
        cases = cases[: args.limit_cases]
        print(f"[限制] --limit-cases={args.limit_cases}，只检查 {len(cases)} 个 case。")

    recs = []
    total = len(cases)
    for i, case in enumerate(cases, 1):
        eprint(f"[{i}/{total}] case {case} ...")
        try:
            recs.append(inspect_case(case, os.path.join(args.data_dir, vols[case]),
                                     os.path.join(args.data_dir, segs[case]), args.skip_affine_fix))
        except Exception as exc:  # noqa: BLE001 - 单 case 失败不中断整体
            eprint(f"[warn] case {case} 失败: {type(exc).__name__}: {exc}")
            recs.append({"case": case, "status": "READ_ERROR", "error": f"{type(exc).__name__}: {exc}",
                         "label_info": {}, "label_set": []})

    if not recs:
        hr("没有可用 case，报告结束", "-")
        return 0

    report_labels(recs)
    report_nesting(recs)
    report_scale_task(recs)
    report_intensity_scaling(recs)
    report_affine(recs)
    report_duplicates(recs)

    hr("报告结束", "-")
    print("请把以上全部 stdout 原样贴回，用于撰写 docs/data.md。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
