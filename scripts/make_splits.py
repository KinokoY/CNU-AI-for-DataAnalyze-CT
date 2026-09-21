"""划分：按肿瘤体积把 20 例含肿瘤病人分层成 5 折，5 例仅肝脏病人只进训练集。

整体功能：读 reports/preprocess_stats.json（或 cache 清单）拿到逐 case 肿瘤体积，先按体积排秩分 5 个体积档，
        再把每档 4 例贪心分给 5 折，保证每折验证集恰好 4 例且各折体积总和接近；仅肝脏病例进每折的 train。
前后接口：上游是 scripts/preprocess.py 产出的统计/清单；下游给 src/dataset.py 与 src/train.py 提供 data/splits.json（纳入 git）。
用法：在仓库根目录执行 ``python scripts/make_splits.py``，自检通过后把 data/splits.json 提交入库。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

try:
    from src.utils import (
        load_config,
        load_json,
        rel_to_root,
        resolve_path,
        save_json,
        setup_logger,
    )
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

LOGGER = setup_logger("make_splits")


def collect_tumor_volumes(cfg: dict) -> dict:
    """返回 case_id -> 肿瘤体积(mm3)；优先用预处理统计，缺失时退到 cache 清单。"""
    paths = cfg.get("paths", {}) or {}
    stats = load_json(paths.get("preprocess_stats", "reports/preprocess_stats.json"))
    if stats and stats.get("cases"):
        out = {}
        for rec in stats["cases"]:
            if rec.get("status") != "OK":
                continue
            out[int(rec["case"])] = float(rec.get("tumor_volume_mm3", 0.0) or 0.0)
        if out:
            return out

    manifest = load_json(paths.get("cache_manifest", "cache/cache_manifest.json"))
    if manifest and manifest.get("cases"):
        LOGGER.warning("未找到预处理统计，改用 cache 清单 %s", rel_to_root(paths.get("cache_manifest", "")))
        return {int(r["case"]): float(r.get("tumor_volume_mm3", 0.0) or 0.0) for r in manifest["cases"]}

    raise FileNotFoundError(
        "既没有 reports/preprocess_stats.json 也没有 cache/cache_manifest.json，请先运行 python scripts/preprocess.py")


def stratify_assign(volumes: dict, n_folds: int, seed: int, verbose: bool = True) -> tuple:
    """按肿瘤体积分层把病例分给各折。

    做法：体积降序后每 ``n_folds`` 例构成一个「体积档」（档内是相邻量级的病例）；档内按体积降序
    （最重的先排），依次配给「每档平均体积预算最接近」的折（预算 = 该折已分配总体积 / 已分配例数，
    空折预算记为 +inf 所以最重的病例一定落在空折）。这样每个档内的病例一定落在不同折 → 每折都同时
    拿到大、中、小病灶；按「例数」而不是「体积余量」做归一，避免了余量变负后退化成任意绑扎。
    同体积并列者用固定 seed 随机排序，消除 id 与体积的隐性相关。
    返回 (fold -> [case,...], fold -> 体积合计)。
    """
    rng = random.Random(seed)
    ordered = sorted(volumes, key=lambda c: (-volumes[c], c))  # 体积降序，并列按 id

    n_total = len(ordered)
    k = n_total // n_folds
    if n_total % n_folds:
        LOGGER.warning("%d 例无法被 %d 折均分，最后 %d 例将造成各折验证数不等。",
                       n_total, n_folds, n_total % n_folds)
    if k < 1:
        raise ValueError(f"病例数 {n_total} 少于折数 {n_folds}，无法分层划分。")

    # 只对体积完全相同的并列病例做随机化，打破 id 与体积的隐性相关
    tie_order = list(ordered)
    rng.shuffle(tie_order)
    tie_rank = {c: i for i, c in enumerate(tie_order)}
    ordered.sort(key=lambda c: (-volumes[c], tie_rank[c]))

    strata: list = [ordered[i:i + n_folds] for i in range(0, n_folds * k, n_folds)]
    leftover = ordered[n_folds * k:]

    folds: list = [[] for _ in range(n_folds)]
    fold_volume = [0.0] * n_folds

    def budget(fold: int) -> float:
        """该折「每例平均已分配体积」；空折记为 inf，排序时会被放在最前面（最优先接收病例）。"""
        if not folds[fold]:
            return float("inf")
        return fold_volume[fold] / len(folds[fold])

    for stratum in strata:
        # 一个档里恰好有 n_folds 例（数量少于 n_folds 时就是全部），按预算升序各折取一例：
        # 空折（inf）优先，保证每折都拿到一个本档病例；同时最重的病例落进预算最小的折。
        order = sorted(range(n_folds), key=lambda f: (budget(f), len(folds[f]), f))
        for case_id, fold in zip(stratum, order):
            folds[fold].append(case_id)
            fold_volume[fold] += volumes[case_id]
    for case_id in leftover:  # 不足一档的余数：优先补例数少、预算低的折
        fold = min(range(n_folds), key=lambda f: (len(folds[f]), budget(f), f))
        folds[fold].append(case_id)
        fold_volume[fold] += volumes[case_id]

    fold_map = {f: sorted(folds[f]) for f in range(n_folds)}
    if verbose:
        LOGGER.debug("体积档：%s；各折体积：%s", strata, [round(x, 1) for x in fold_volume])
    return fold_map, {f: fold_volume[f] for f in range(n_folds)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="按肿瘤体积分层生成 5 折划分（病人级，不按切片）")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件")
    parser.add_argument("--n-folds", type=int, default=5, help="折数，默认 5")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 train.seed")
    parser.add_argument("--out", default=None, help="输出路径，默认 paths.splits（data/splits.json）")
    parser.add_argument("--set", dest="overrides", action="append", default=None, help="覆盖配置项，可多次")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    paths = cfg.get("paths", {}) or {}
    seed = int(args.seed if args.seed is not None else (cfg.get("train", {}) or {}).get("seed", 42))
    out_path = args.out or paths.get("splits", "data/splits.json")

    exclude_doc = load_json(paths.get("exclude", "data/exclude_cases.json"), default={}) or {}
    excluded = sorted(int(item["case"]) for item in exclude_doc.get("exclude", []))

    volumes = collect_tumor_volumes(cfg)
    tumor_cases = sorted([c for c, v in volumes.items() if v > 0.0])
    liver_only = sorted([c for c, v in volumes.items() if v <= 0.0])
    LOGGER.info("可用 %d 例：含肿瘤 %d 例 %s", len(volumes), len(tumor_cases), tumor_cases)
    LOGGER.info("仅肝脏 %d 例 %s（只进训练集，不出现在任何验证集）", len(liver_only), liver_only)

    if excluded:
        LOGGER.info("排除清单 %s 不在划分中出现（预处理阶段已剔除）", excluded)

    folds_map, fold_volume = stratify_assign({c: volumes[c] for c in tumor_cases}, int(args.n_folds), seed)

    fold_records = []
    for fold, val_cases in folds_map.items():
        train_cases = sorted(set(tumor_cases) - set(val_cases)) + liver_only
        fold_records.append({
            "fold": fold,
            "train": train_cases,
            "val": val_cases,
            "n_train": len(train_cases),
            "n_val": len(val_cases),
            "val_tumor_volume_mm3_sum": round(float(fold_volume[fold]), 3),
            "val_tumor_volume_mm3": {str(c): round(float(volumes[c]), 3) for c in val_cases},
        })

    val_sums = np.array([rec["val_tumor_volume_mm3_sum"] for rec in fold_records], dtype=np.float64)
    balance = {
        "val_volume_mean": round(float(val_sums.mean()), 3),
        "val_volume_std": round(float(val_sums.std()), 3),
        "val_volume_cv": round(float(val_sums.std() / max(1e-9, val_sums.mean())), 4),
        "val_volume_min": round(float(val_sums.min()), 3),
        "val_volume_max": round(float(val_sums.max()), 3),
    }

    doc = {
        "seed": seed,
        "n_folds": int(args.n_folds),
        "stratify_by": "tumor_volume_mm3",
        "split_level": "patient",
        "excluded_cases": excluded,
        "liver_only_cases": liver_only,
        "cases_with_tumor": tumor_cases,
        "n_cases_with_tumor": len(tumor_cases),
        "n_cases_liver_only": len(liver_only),
        "fold_size_target": {"train": len(tumor_cases) - len(tumor_cases) // int(args.n_folds),
                             "val": len(tumor_cases) // int(args.n_folds)},
        "val_volume_balance": balance,
        "tumor_volume_mm3": {str(c): round(float(v), 3) for c, v in sorted(volumes.items())},
        "folds": fold_records,
        "notes": [
            "划分粒度是病人，不是切片：同一病人的所有切片只会出现在同一侧，避免相邻层泄漏导致 Dice 虚高。",
            "5 例仅肝脏病人固定进每折的 train，用于约束模型假阳性；验证集分母恒为 4 例含肿瘤病人。",
            "分层做法：按肿瘤体积降序每 5 例一档，档内最重的先配给体积余量最大的折，"
            "使每折都同时含大、中、小病灶且体积总和接近（见 val_volume_balance）。",
            "本文件纳入 git，保证跨轮次、跨机器复现同一划分。",
        ],
    }

    # ---- 自检 ----
    problems: list = []
    all_val: list = []
    for rec in fold_records:
        if rec["n_val"] != len(tumor_cases) // int(args.n_folds):
            problems.append(f"fold {rec['fold']} 验证集 {rec['n_val']} 例，期望 {len(tumor_cases) // int(args.n_folds)} 例")
        overlap = set(rec["train"]) & set(rec["val"])
        if overlap:
            problems.append(f"fold {rec['fold']} 训练/验证重叠：{sorted(overlap)}")
        missing = set(liver_only) - set(rec["train"])
        if missing:
            problems.append(f"fold {rec['fold']} 训练集缺少仅肝病例：{sorted(missing)}")
        leaked = set(liver_only) & set(rec["val"])
        if leaked:
            problems.append(f"fold {rec['fold']} 验证集混入仅肝病例：{sorted(leaked)}")
        if set(rec["train"]) | set(rec["val"]) != set(volumes):
            problems.append(f"fold {rec['fold']} 训练+验证未覆盖全部 %d 例" % len(volumes))
        all_val.extend(rec["val"])

    if sorted(all_val) != tumor_cases:
        problems.append(f"5 折验证集并集 {sorted(all_val)} != 全部含肿瘤病例 {tumor_cases}")

    if problems:
        LOGGER.error("划分自检失败：")
        for item in problems:
            LOGGER.error("  - %s", item)
        return 2

    path = save_json(doc, out_path)
    LOGGER.info("自检通过：%d 折各 %d 例验证，验证集并集覆盖全部 %d 例含肿瘤病人，无重叠，仅肝病例只在 train。",
                int(args.n_folds), len(tumor_cases) // int(args.n_folds), len(tumor_cases))
    LOGGER.info("验证集体积均衡：mean=%.1f std=%.1f CV=%.3f（min=%.1f max=%.1f）",
                balance["val_volume_mean"], balance["val_volume_std"], balance["val_volume_cv"],
                balance["val_volume_min"], balance["val_volume_max"])
    if balance["val_volume_cv"] > 0.6:
        LOGGER.warning("验证集体积 CV=%.2f 偏大（肿瘤体积跨 3 个数量级时难以更均衡），"
                       "跨折 Dice 会比较抖，报告里请以均值±标准差解读。", balance["val_volume_cv"])
    for rec in fold_records:
        LOGGER.info("  fold %d: train=%d val=%d %s（验证体积合计 %.1f mm3）",
                    rec["fold"], rec["n_train"], rec["n_val"], rec["val"], rec["val_tumor_volume_mm3_sum"])
    LOGGER.info("划分已写入：%s（请提交入库）", rel_to_root(path))
    LOGGER.info("下一步：python -m src.train --fold 0 --debug")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
