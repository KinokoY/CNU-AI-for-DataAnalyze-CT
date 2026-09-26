"""整卷评估指标：体素级（Dice/IoU/精确率/召回率）、病灶级检出、假阳性统计（第 5 轮新增）。

整体功能（全部是纯函数：吃 numpy 数组 + spacing，返回可 JSON 序列化的 dict）：
    1. ``dice`` / ``iou`` / ``precision`` / ``recall`` —— **唯一的指标口径**；
    2. ``voxel_metrics`` —— 一次算完 TP/FP/FN 与四个指标（训练期整卷验证与评估报告共用）；
    3. ``confusion_summary`` —— 直接由 TP/FP/FN 汇总（跨病例合并时**必须先合并再算比率**，
       不能把逐例的比率平均，否则大病灶与小病灶会被等权，见 ``detection_stats`` 的说明）；
    4. ``lesion_detection`` —— 病灶级检出：逐 GT 病灶判「检出 / 漏检」；
    5. ``detection_stats`` —— 把所有病例的病灶级结果合并成分层的检出率；
    6. ``case_metrics`` —— 一例的全套指标（后处理前 / 后各一份体素级 + 病灶级 + 假阳性）。

**公式口径（必须与训练期完全一致，改前先读 ``docs/preprocess_notes.md`` 第三节的指标行）**：
    * ``dice = (2|A∩B| + eps) / (|A| + |B| + eps)``，``eps = 1e-6``，**两边都空记 1.0**
      （用未加 eps 的 TP/FP/FN 形式写就是 ``(2TP + eps) / (2TP + FP + FN + eps)``，
      与 ``(2|A∩B|+eps)/(|A|+|B|+eps)`` 数学上等价，因为 ``|A|+|B| = 2TP + FP + FN``）；
    * ``iou = (TP + eps) / (TP + FP + FN + eps)``，两边都空记 1.0；
    * ``precision = TP / (TP + FP)``：**预测为空时记 0.0 并标 ``precision_defined=False``**
      —— 记 1.0 会让「全预测背景」在报告里看起来完美（第 3 轮就是这么骗过眼睛的）；
    * ``recall = TP / (TP + FN)``：GT 为空时记 0.0（5 例仅肝脏病人走这条）。

**两类指标不可互相替代**：体素级 Dice 由大病灶主导（肿瘤体积跨 3 个数量级），
小病灶长期为 0 也能拿到不低的均值；病灶级检出率才是「有没有找到病灶」的答案。
报告一律两者的逐例值一起给（口径见 ``docs/preprocess_notes.md`` 第三节）。

依赖：numpy / ``src.postprocess``（连通域口径只在那里定义一份）。
前后接口：上游是 ``src.infer`` 的预测卷 / GT 卷与 ``src.postprocess`` 的后处理结果；
    下游是 ``src.train``（每轮整卷验证）与 ``src.evaluate``（整卷评估报告）。
用法：``m = case_metrics(pred_bin, gt_bin, spacing, cfg)``；或
    ``voxel_metrics(pred, gt)["dice"]`` 取单个指标。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from src.postprocess import (
        DEFAULT_SIZE_BINS,
        label_lesions,
        lesion_size_bin,
        to_mm3,
        unmatched_lesion_stats,
    )
except ModuleNotFoundError:  # pragma: no cover - 兜底：把仓库根塞进 sys.path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.postprocess import (  # type: ignore
        DEFAULT_SIZE_BINS,
        label_lesions,
        lesion_size_bin,
        to_mm3,
        unmatched_lesion_stats,
    )

#: 指标公式里的平滑项（**与 src/train.py 的既往口径一致，不要改**）
EPS = 1e-6

#: 病灶级检出的默认体积阈值（mm³）：单病灶 >= 该体积即算检出（``eval.detect_min_mm3``）
DEFAULT_DETECT_MIN_MM3 = 10.0

#: 体素级混淆矩阵的键（跨病例汇总时按这些键累加）
CONFUSION_KEYS = ("tp", "fp", "fn")


# --------------------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------------------

def _as_bool3d(array, name: str = "array") -> np.ndarray:
    """把输入规整成 3D bool 数组（``> 0`` 口径：预测是 uint8 {0,1}，GT 也是二值）。"""
    values = np.asarray(array)
    if values.ndim != 3:
        raise ValueError(f"{name} 必须是 3D（本项目是 (H,W,Z) 的 nibabel 轴序），"
                         f"收到 shape={values.shape}")
    return values > 0


def confusion_counts(pred_bin, gt_bin) -> tuple:
    """返回 ``(tp, fp, fn)``（体素数，int）。

    预测与 GT 都按 ``> 0`` 取前景。只做三个逻辑运算与 ``count_nonzero``，
    整卷 512×512×488 上是毫秒级。
    """
    pred = _as_bool3d(pred_bin, "pred_bin")
    gt = _as_bool3d(gt_bin, "gt_bin")
    if pred.shape != gt.shape:
        raise ValueError(f"预测卷 {pred.shape} 与 GT 卷 {gt.shape} 形状不一致："
                         f"轴序或 pad_offset 裁回出错（先查 src/infer.predict_volume）")
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    return tp, fp, fn


# --------------------------------------------------------------------------------------
# 体素级指标
# --------------------------------------------------------------------------------------

def dice(pred_bin, gt_bin, eps: float = EPS) -> float:
    """二值 Dice：``(2|A∩B| + eps) / (|A| + |B| + eps)``，两边都空记 **1.0**。

    与 ``src.train`` 选 ``best.pt`` 用的整卷 Dice 是同一个函数（第 5 轮起训练期改为直接调用本函数），
    因此评估报告的 Dice 与训练日志里的 Dice 可以直接对照。
    """
    tp, fp, fn = confusion_counts(pred_bin, gt_bin)
    if (tp + fp + fn) == 0:
        return 1.0
    return float((2.0 * tp + float(eps)) / (2.0 * tp + fp + fn + float(eps)))


def iou(pred_bin, gt_bin, eps: float = EPS) -> float:
    """二值 IoU（Jaccard）：``(TP + eps) / (TP + FP + FN + eps)``，两边都空记 1.0。"""
    tp, fp, fn = confusion_counts(pred_bin, gt_bin)
    if (tp + fp + fn) == 0:
        return 1.0
    return float((tp + float(eps)) / (tp + fp + fn + float(eps)))


def precision(pred_bin, gt_bin) -> float:
    """精确率 ``TP / (TP + FP)``；**预测为空时记 0.0**（数学上未定义，见模块 docstring）。"""
    tp, fp, _ = confusion_counts(pred_bin, gt_bin)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(pred_bin, gt_bin) -> float:
    """召回率 ``TP / (TP + FN)``；GT 为空时记 0.0（5 例仅肝脏病人走这条）。"""
    tp, _, fn = confusion_counts(pred_bin, gt_bin)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def voxel_metrics(pred_bin, gt_bin, spacing=None, eps: float = EPS) -> dict:
    """一次算完体素级指标，返回与 ``src.train.validate`` 完全同构的 dict：

        ``dice / iou / precision / recall / precision_defined / tp / fp / fn /
        pred_voxels / gt_voxels``

    ``spacing`` 给了就额外带上 ``tp_mm3 / fp_mm3 / fn_mm3 / pred_mm3 / gt_mm3``
    （预处理后 spacing=(1,1,1)，两者数值相同；带上它只是为了报告里能直接谈体积）。
    """
    tp, fp, fn = confusion_counts(pred_bin, gt_bin)
    both_empty = (tp + fp + fn) == 0
    out = {
        "dice": 1.0 if both_empty else float((2.0 * tp + float(eps)) / (2.0 * tp + fp + fn + float(eps))),
        "iou": 1.0 if both_empty else float((tp + float(eps)) / (tp + fp + fn + float(eps))),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "precision_defined": bool((tp + fp) > 0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "pred_voxels": int(tp + fp), "gt_voxels": int(tp + fn),
    }
    if spacing is not None:
        out.update({
            "tp_mm3": round(to_mm3(tp, spacing), 3),
            "fp_mm3": round(to_mm3(fp, spacing), 3),
            "fn_mm3": round(to_mm3(fn, spacing), 3),
            "pred_mm3": round(to_mm3(tp + fp, spacing), 3),
            "gt_mm3": round(to_mm3(tp + fn, spacing), 3),
        })
    return out


def confusion_summary(totals: dict, spacing=None, eps: float = EPS) -> dict:
    """由**汇总后的** ``{"tp","fp","fn","pred_voxels","gt_voxels"}`` 重算比率指标。

    为什么必须「先合并再算比率」，而不是把逐例比率平均：肿瘤体积跨 3 个数量级，
    逐例平均等于给 652 mm³ 的小病灶与 434721 mm³ 的大病灶同样的权重；报告里需要**两种都有**：
      * ``*_mean``（macro，逐例平均）—— 回答「典型病例表现如何」，与训练期早停指标同口径；
      * 池化指标（本函数）—— 回答「所有体素合起来表现如何」，由大病灶主导。
    两者一起给，读者才不会被其中任一个误导（见 ``docs/preprocess_notes.md`` 第三节）。
    """
    tp = int(totals.get("tp", 0))
    fp = int(totals.get("fp", 0))
    fn = int(totals.get("fn", 0))
    pred_voxels = int(totals.get("pred_voxels", tp + fp))
    gt_voxels = int(totals.get("gt_voxels", tp + fn))
    both_empty = (tp + fp + fn) == 0
    out = {
        "cases": int(totals.get("cases", 0)),
        "tp": tp, "fp": fp, "fn": fn,
        "pred_voxels": pred_voxels, "gt_voxels": gt_voxels,
        "dice": 1.0 if both_empty else float((2.0 * tp + float(eps)) / (2.0 * tp + fp + fn + float(eps))),
        "iou": 1.0 if both_empty else float((tp + float(eps)) / (tp + fp + fn + float(eps))),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "precision_defined": bool((tp + fp) > 0),
    }
    if spacing is not None:
        out.update({
            "pred_mm3": round(to_mm3(pred_voxels, spacing), 3),
            "gt_mm3": round(to_mm3(gt_voxels, spacing), 3),
        })
    return out


def summarize(values: Sequence[float]) -> dict:
    """把一串逐例指标压成 ``{"mean","std","min","max","n"}``（样本 <2 时 std 记 0.0）。

    ``std`` 用**样本标准差**（``ddof=1``）：报告里的小样本（4 例/折）用总体标准差会系统性偏小。
    """
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "n": 0}
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {"mean": float(array.mean()), "std": std, "min": float(array.min()),
            "max": float(array.max()), "n": int(array.size)}


# --------------------------------------------------------------------------------------
# 病灶级检出
# --------------------------------------------------------------------------------------

def lesion_detection(pred_bin, gt_bin, spacing, detect_min_mm3: float = DEFAULT_DETECT_MIN_MM3,
                     gt_slices: Sequence[dict] | None = None, return_detected: bool = False):
    """逐 GT 病灶判定「检出 / 漏检」，返回统计 dict（``return_detected=True`` 时多返回一个列表）。

    判据是**两条 OR**（第 5 轮定稿，见 ``docs/preprocess_notes.md`` 第三节）：
      1. **有重叠**：该 GT 病灶与预测前景有任何重叠（哪怕 1 个体素）即算检出；
      2. **体积达标**：该 GT 病灶自身体积 ``>= detect_min_mm3``（默认 10 mm³）即算检出
         —— 「这么小的目标本来就难定位，模型没标出来不应算失败」。

    返回：
        ``{"n_gt", "n_detected", "n_missed", "n_hit_overlap", "n_hit_big", "n_gt_big",
           "detection_rate", "detection_rate_big", "per_lesion": [...]}``
        ``detection_rate`` = 检出的 GT 病灶数 / GT 病灶总数（GT 为空时记 0.0、``n_gt=0``）；
        ``detection_rate_big`` 只在 ``>= detect_min_mm3`` 的病灶上算，是与阈值无关的那个分母。

    ``gt_slices`` 允许直接传入已经算好的 ``lesion_stats(...)[2]``，避免对同一张 GT 卷重复标记
    （``case_metrics`` 就是这么用的）。
    """
    pred = _as_bool3d(pred_bin, "pred_bin")
    gt = _as_bool3d(gt_bin, "gt_bin")
    if pred.shape != gt.shape:
        raise ValueError(f"预测卷 {pred.shape} 与 GT 卷 {gt.shape} 形状不一致")
    threshold = float(detect_min_mm3)
    if gt_slices is None:
        gt_marks, n_gt, gt_volumes, gt_slices = label_lesions(gt, spacing, return_slices=True)
    else:
        gt_slices = list(gt_slices)
        n_gt = len(gt_slices)
        gt_volumes = [float(rec.get("volume_mm3", 0.0)) for rec in gt_slices]
        gt_marks = None

    empty = {"n_gt": 0, "n_detected": 0, "n_missed": 0, "n_hit_overlap": 0, "n_hit_big": 0,
             "n_gt_big": 0, "detection_rate": 0.0, "detection_rate_big": 0.0,
             "detect_min_mm3": threshold, "per_lesion": []}
    if n_gt <= 0:
        return (empty, []) if return_detected else empty

    if gt_marks is None:
        gt_marks, _, _ = label_lesions(gt, spacing)
    # 一次 bincount 判出「哪些 GT 病灶被预测碰到过」：比逐病灶切片快得多
    hit = np.bincount(gt_marks[pred].ravel(), minlength=n_gt + 1) > 0
    per_lesion: list = []
    detected_flags: list = []
    for index in range(1, n_gt + 1):
        volume = float(gt_volumes[index - 1])
        hit_overlap = bool(hit[index])
        hit_big = bool(volume >= threshold)
        detected = bool(hit_overlap or hit_big)
        detected_flags.append(detected)
        record = gt_slices[index - 1] if index - 1 < len(gt_slices) else {}
        per_lesion.append({
            "label": int(index),
            "voxels": int(record.get("voxels", 0)),
            "volume_mm3": round(volume, 3),
            "size_bin": lesion_size_bin(volume),
            "bbox": record.get("bbox"),
            "hit_overlap": hit_overlap,
            "hit_big": hit_big,
            "detected": detected,
        })
    n_detected = int(sum(1 for flag in detected_flags if flag))
    n_hit_overlap = int(sum(1 for rec in per_lesion if rec["hit_overlap"]))
    n_hit_big = int(sum(1 for rec in per_lesion if rec["hit_big"]))
    big_flags = [flag for flag, volume in zip(detected_flags, gt_volumes) if float(volume) >= threshold]
    stats = {
        "n_gt": int(n_gt),
        "n_detected": n_detected,
        "n_missed": int(n_gt - n_detected),
        "n_hit_overlap": n_hit_overlap,
        "n_hit_big": n_hit_big,
        "n_gt_big": int(len(big_flags)),
        "detection_rate": float(n_detected / n_gt),
        "detection_rate_big": (float(sum(1 for f in big_flags if f) / len(big_flags))
                               if big_flags else 0.0),
        "detect_min_mm3": threshold,
        "per_lesion": per_lesion,
    }
    return (stats, detected_flags) if return_detected else stats


def detection_stats(records: Sequence[dict], bins: Sequence = DEFAULT_SIZE_BINS) -> dict:
    """把多个病例的病灶级结果合并成分层检出率。

    ``records`` 的每一项需要含 ``n_gt`` / ``n_detected``（``lesion_detection`` 的返回值即可），
    含 ``detected``（bool 列表，逐病灶）时会额外给出**按体积分档**的检出率与逐个 GT 病灶的身高。

    口径说明：这里同样**先合并再算比率**（分母是所有 GT 病灶），并同时给出
    ``detection_rate_per_case_mean``（逐例检出率的平均，macro）——两者在小病灶占比重时差别很大，
    一起给才不会误导。
    """
    usable = [rec for rec in (records or []) if isinstance(rec, dict) and rec.get("detected") is not None]
    if not usable:
        return {"status": "无病灶级结果（本折没有含肿瘤的病例）", "n_cases": 0}
    n_gt = int(sum(len(rec["detected"]) for rec in usable))
    n_detected = int(sum(1 for rec in usable for flag in rec["detected"] if flag))
    per_case_rates = [float(rec.get("detection_rate", 0.0)) for rec in usable]
    binned: dict = {}
    for rec in usable:
        lesions = rec.get("per_lesion") or []
        for lesion in lesions:
            name = str(lesion.get("size_bin") or lesion_size_bin(lesion.get("volume_mm3", 0.0), bins))
            slot = binned.setdefault(name, {"n_gt": 0, "n_detected": 0})
            slot["n_gt"] += 1
            if lesion.get("detected"):
                slot["n_detected"] += 1
    by_bin = {}
    for name, slot in sorted(binned.items(),
                             key=lambda item: min(float(b[0]) for b in bins if b[2] == item[0])
                             if any(b[2] == item[0] for b in bins) else float("inf")):
        by_bin[name] = {
            "n_gt": int(slot["n_gt"]),
            "n_detected": int(slot["n_detected"]),
            "rate": float(slot["n_detected"] / slot["n_gt"]) if slot["n_gt"] else 0.0,
        }
    big_n = int(sum(1 for rec in usable for lesion in (rec.get("per_lesion") or [])
                    if float(lesion.get("volume_mm3", 0.0)) >= float(rec.get("detect_min_mm3", 0.0))))
    big_hit = int(sum(1 for rec in usable for lesion in (rec.get("per_lesion") or [])
                      if float(lesion.get("volume_mm3", 0.0)) >= float(rec.get("detect_min_mm3", 0.0))
                      and lesion.get("detected")))
    return {
        "status": "ok",
        "n_cases": int(len(usable)),
        "n_gt": n_gt,
        "n_detected": n_detected,
        "n_missed": int(n_gt - n_detected),
        "detection_rate": float(n_detected / n_gt) if n_gt else 0.0,
        "detection_rate_per_case_mean": float(np.mean(per_case_rates)) if per_case_rates else 0.0,
        "n_gt_big": big_n,
        "n_detected_big": big_hit,
        "detection_rate_big": float(big_hit / big_n) if big_n else 0.0,
        "by_size_bin": by_bin,
    }


# --------------------------------------------------------------------------------------
# 一例的全套指标
# --------------------------------------------------------------------------------------

def false_positive_stats(pred_bin, gt_bin, spacing) -> dict:
    """假阳性统计：整体 FP 体积 + 与 GT 零重叠的孤立病灶数 / 体积 / 最大块。

    返回的键分三层（报告里那 5 例仅肝脏病人用的就是这几个数）：
      * ``fp_voxels`` / ``fp_mm3`` —— 预测落在 GT 之外的**体素**总量（含与 GT 相接但溢出的部分）；
      * ``n_lesions`` / ``lesion_volume_mm3`` / ``largest_mm3`` / ``largest_bbox``
        —— 与 GT **完全没有重叠**的连通块（这才是严格意义的"凭空多出来的病灶"）。

    ``gt_bin`` 全空时（5 例仅肝脏病人）前者等于预测总体积、后者等于预测的病灶总数，
    所以同一个函数可以直接用于两类病例，报告里不必分两套口径。
    """
    pred = _as_bool3d(pred_bin, "pred_bin")
    gt = _as_bool3d(gt_bin, "gt_bin")
    if pred.shape != gt.shape:
        raise ValueError(f"预测卷 {pred.shape} 与 GT 卷 {gt.shape} 形状不一致")
    fp_voxels = int(np.count_nonzero(pred & ~gt))
    unmatched = unmatched_lesion_stats(pred, gt, spacing)
    return {
        "fp_voxels": fp_voxels,
        "fp_mm3": round(to_mm3(fp_voxels, spacing), 3),
        "n_lesions": int(unmatched["n_lesions"]),
        "lesion_volume_mm3": float(unmatched["volume_mm3"]),
        "largest_mm3": float(unmatched["largest_mm3"]),
        "largest_bbox": unmatched["largest_bbox"],
    }


def case_metrics(pred_raw, pred_clean, gt_bin, spacing, cfg: dict | None = None) -> dict:
    """一例的**全套**指标：后处理前（``raw``）与后处理**后**（``clean``）各一份。

    ``pred_raw`` / ``pred_clean`` 都是 ``(H,W,Z)`` 的二值卷（``pred_clean`` 由
    ``src.postprocess.remove_small_lesions`` 产出）；``gt_bin`` 是同一轴序的 GT。
    报告主表用 ``clean``（第 5 轮拍板），``raw`` 一并写进 JSON 供对照。

    返回的键：
        ``voxel_raw / voxel_clean``（体素级指标，含 mm³）、
        ``lesion_raw / lesion_clean``（病灶级检出，含逐病灶明细）、
        ``fp_raw / fp_clean``（假阳性统计）、
        ``gt_lesion_*``（GT 的病灶数与体积，用于报告里的分层表）、
        ``postprocess``（删掉的病灶数与体积，便于判断后处理是否砍掉了真病灶）。
    """
    eval_cfg = ((cfg or {}).get("eval") or {})
    min_lesion_mm3 = float(eval_cfg.get("min_lesion_mm3", 50.0) or 0.0)
    detect_min_mm3 = float(eval_cfg.get("detect_min_mm3", DEFAULT_DETECT_MIN_MM3) or 0.0)

    gt = _as_bool3d(gt_bin, "gt_bin")
    gt_marks, n_gt, gt_volumes, gt_slices = label_lesions(gt, spacing, return_slices=True)
    del gt_marks   # 只用于统计，标记图不必留着（整卷 int32 很占内存）

    raw = np.asarray(pred_raw) > 0
    clean = np.asarray(pred_clean) > 0
    # 注意解包顺序：``label_lesions`` 返回的是 ``(marks, n_lesions, volumes)``（与
    # ``lesion_stats`` 的 ``(n, volumes, slices)`` 不同）；标记图这里用不到，只取个数与体积。
    _, n_pred_raw, pred_raw_volumes = label_lesions(raw, spacing)
    _, n_pred_clean, pred_clean_volumes = label_lesions(clean, spacing)

    lesion_raw, detected_raw = lesion_detection(raw, gt, spacing, detect_min_mm3,
                                                gt_slices=gt_slices, return_detected=True)
    lesion_clean, detected_clean = lesion_detection(clean, gt, spacing, detect_min_mm3,
                                                    gt_slices=gt_slices, return_detected=True)
    lesion_raw["detected"] = detected_raw
    lesion_clean["detected"] = detected_clean
    lesion_raw["size_bin"] = "raw"
    lesion_clean["size_bin"] = "clean"

    return {
        "spacing": [float(v) for v in spacing],
        "voxel_raw": voxel_metrics(raw, gt, spacing),
        "voxel_clean": voxel_metrics(clean, gt, spacing),
        "lesion_raw": lesion_raw,
        "lesion_clean": lesion_clean,
        "fp_raw": false_positive_stats(raw, gt, spacing),
        "fp_clean": false_positive_stats(clean, gt, spacing),
        "gt_lesions": {
            "n": int(len(gt_volumes)),
            "volumes_mm3": [round(float(v), 3) for v in gt_volumes],
            "total_mm3": round(float(sum(float(v) for v in gt_volumes)), 3),
            "by_size_bin": _bin_counts(gt_volumes),
            "slices": gt_slices,
        },
        "pred_lesions_raw": {"n": int(n_pred_raw), "volumes_mm3": [round(float(v), 3) for v in pred_raw_volumes]},
        "pred_lesions_clean": {"n": int(n_pred_clean),
                               "volumes_mm3": [round(float(v), 3) for v in pred_clean_volumes]},
        "postprocess": {
            "min_lesion_mm3": min_lesion_mm3,
            "n_removed": int(n_pred_raw - n_pred_clean),
            "volume_removed_mm3": round(float(sum(pred_raw_volumes)) - float(sum(pred_clean_volumes)), 3),
        },
    }


def _bin_counts(volumes: Sequence[float], bins: Sequence = DEFAULT_SIZE_BINS) -> dict:
    """逐档统计病灶个数（``{档名: 个数}``，空的档也给 0，报告表列固定）。"""
    counts = {str(name): 0 for _, _, name in bins}
    for volume in volumes or []:
        name = lesion_size_bin(float(volume), bins)
        counts[name] = counts.get(name, 0) + 1
    return counts


if __name__ == "__main__":  # pragma: no cover - 直接运行本文件时给出正确入口
    raise SystemExit("本模块是库，不是脚本；整卷评估由 python -m src.evaluate 调用。")
