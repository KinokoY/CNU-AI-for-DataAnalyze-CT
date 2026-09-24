"""数据自检：在写训练代码之前，先把「数据进模型的形态」用真实 cache 验一遍。

整体功能（全部只读 cache，不依赖网络、不依赖模型）：
    1. 环境与前置：打印 torch / nibabel / numpy / CUDA 版本（MONAI 只是**顺带**报告——增强已全部
       自实现，本脚本与 src/dataset.py 都不依赖它）；核对 ``data/splits.json``、
       ``cache/cache_manifest.json`` 是否存在，清单里的预处理指纹是否与当前配置一致；
    2. 数据集形态：``CTSliceDataset``（train / val 两侧）的病例数、切片数、含肿瘤切片比例、
       面内尺寸分桶明细，以及每例的 ``nz`` / 含肿瘤层数 / 桶键；
    3. 3 个 batch 的实测：image/label 的 shape、dtype、值域、label 取值集合、
       **batch 内实测肿瘤切片比例**、batch 的桶键与病例构成、单 batch 耗时、CUDA 显存；
    4. 切片轴校验：对指定病例打印逐层肿瘤体素数曲线（含 ASCII 直方图）与「首尾各 10% 层的
       阳性占比」，切片轴若被搞反，曲线会贴到某一端而不是集中在中间层；
    5. 增强自检：随机抽若干样本做「同一 idx 分别用 无增强/增强」的对比，确认形状不变、
       label 仍是 {0,1}、image 仍在 [0,1]，并给出被改动的样本比例；
    6. 采样器自检：同一 epoch 用同一 seed 采样两次必须完全一致（可复现），并统计
       正/负切片对 batch 的覆盖情况。

前后接口：上游是 scripts/preprocess.py 的产物（cache/ + cache_manifest.json）与 data/splits.json；
        下游是 src/train.py（第 3 轮）——本脚本通过后再往上搭训练。
用法：仓库根目录执行 ``python -m src.selfcheck_data``（默认 fold 0、3 个 batch）。
     产物 reports/selfcheck_data.json 在远程（不入库），把终端输出贴回本地即可。
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

try:
    from src.dataset import (
        UINT16_SCALE,
        BinarizeLabel,
        BucketBatchSampler,
        CTSliceDataset,
        ClampImageToUnit,
        FlipSlice2D,
        GammaSlice2D,
        GaussianNoiseSlice2D,
        GridAffine2D,
        RandAffineSlice2D,
        Rotate90Slice2D,
        bucket_key,
        build_transforms,
        data_config,
        format_bucket,
        make_augment_steps,
        make_train_loader,
        make_val_loader,
        open_volume,
        reset_open_cache,
    )
    from src.utils import (
        config_fingerprint,
        load_config,
        load_json,
        rel_to_root,
        resolve_path,
        save_json,
        setup_logger,
    )
except ModuleNotFoundError:  # pragma: no cover - 兜底：把仓库根塞进 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.dataset import (  # type: ignore
        UINT16_SCALE,
        BinarizeLabel,
        BucketBatchSampler,
        CTSliceDataset,
        ClampImageToUnit,
        FlipSlice2D,
        GammaSlice2D,
        GaussianNoiseSlice2D,
        GridAffine2D,
        RandAffineSlice2D,
        Rotate90Slice2D,
        bucket_key,
        build_transforms,
        data_config,
        format_bucket,
        make_augment_steps,
        make_train_loader,
        make_val_loader,
        open_volume,
        reset_open_cache,
    )
    from src.utils import (  # type: ignore
        config_fingerprint,
        load_config,
        load_json,
        rel_to_root,
        resolve_path,
        save_json,
        setup_logger,
    )

LOGGER = setup_logger("selfcheck_data")

#: 逐层曲线最多打印多少行（超长卷只打印首尾与峰值附近）
MAX_CURVE_ROWS = 60


# --------------------------------------------------------------------------------------
# 前置检查
# --------------------------------------------------------------------------------------

def check_environment() -> dict:
    """打印并返回版本信息；CUDA 不可用不报错（本脚本不需要 GPU）。"""
    import nibabel as nib
    import torch

    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "nibabel": nib.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:   # MONAI 只用于报告版本（增强已自实现，装卸都不影响本链路）
        import monai

        env["monai"] = monai.__version__
    except Exception:  # noqa: BLE001
        env["monai"] = None
    LOGGER.info("环境：python %s / torch %s / monai %s（仅报告）/ nibabel %s / numpy %s",
                env["python"], env["torch"], env["monai"], env["nibabel"], env["numpy"])
    LOGGER.info("CUDA：%s%s", env["cuda_available"],
                f"（{env['device']}）" if env["device"] else "（本脚本不需要 GPU，仅报告）")
    return env


def check_prerequisites(cfg: dict) -> tuple:
    """核对 splits / manifest / cache 目录，返回 (splits, manifest, 问题列表)。"""
    paths = (cfg or {}).get("paths") or {}
    problems: list = []

    splits_path = resolve_path(paths.get("splits", "data/splits.json"))
    splits = load_json(splits_path, default=None)
    if not splits:
        problems.append(f"读不到划分文件：{rel_to_root(splits_path)}（远程需先跑 make_splits.py 并 push 回本地）")
        splits = {}
    else:
        folds = splits.get("folds") or []
        LOGGER.info("划分：%s，%d 折，每折 train=%s / val=%s",
                    rel_to_root(splits_path), len(folds),
                    [len(r.get("train", [])) for r in folds], [len(r.get("val", [])) for r in folds])
        if len(folds) != int(splits.get("n_folds", len(folds))):
            problems.append(f"folds 条数 {len(folds)} 与 n_folds={splits.get('n_folds')} 不一致")

    cache_dir = resolve_path(paths.get("cache", "cache"))
    for sub in ("image", "label"):
        if not (cache_dir / sub).is_dir():
            problems.append(f"cache 子目录不存在：{rel_to_root(cache_dir / sub)}（先跑 preprocess.py）")
    n_img = len(list((cache_dir / "image").glob("*.nii.gz"))) if (cache_dir / "image").is_dir() else 0
    n_lab = len(list((cache_dir / "label").glob("*.nii.gz"))) if (cache_dir / "label").is_dir() else 0
    LOGGER.info("cache：%s（image=%d，label=%d）", rel_to_root(cache_dir), n_img, n_lab)
    if n_img != n_lab:
        problems.append(f"cache 里 image({n_img}) 与 label({n_lab}) 数量不一致")

    manifest_path = paths.get("cache_manifest", "cache/cache_manifest.json")
    manifest = load_json(manifest_path, default=None)
    if not manifest or not manifest.get("cases"):
        LOGGER.warning("读不到 cache 清单 %s —— 跳过「与清单逐例比对」，"
                       "建议先跑 python scripts/fetch_manifest.py", rel_to_root(manifest_path))
        manifest = {}
    else:
        expect_hash = config_fingerprint(cfg, drop=["paths", "train", "model", "eval", "data"])
        got_hash = manifest.get("cfg_hash")
        LOGGER.info("清单：%d 例，预处理指纹 %s（当前配置 %s）%s",
                    len(manifest["cases"]), got_hash, expect_hash,
                    "一致" if got_hash == expect_hash else "**不一致**")
        if got_hash and got_hash != expect_hash:
            problems.append(f"cache_manifest.cfg_hash={got_hash} 与当前配置算出的 {expect_hash} 不一致："
                            f"缓存可能来自不同的预处理参数，需要重跑 preprocess.py")
        agg = manifest.get("aggregate") or {}
        if agg:
            LOGGER.info("清单汇总：面内尺寸 %s；分桶 %s；切片数 %s-%s；含肿瘤切片 %s/%s = %.4f",
                        agg.get("inplane_shapes"), agg.get("size_buckets"),
                        agg.get("n_slices_min"), agg.get("n_slices_max"),
                        agg.get("tumor_slices"), agg.get("total_slices"),
                        float(agg.get("tumor_slice_ratio_overall") or 0.0))
    return splits, manifest, problems


def check_bucket_consistency(ds: CTSliceDataset, manifest: dict, problems: list) -> None:
    """把数据集的桶与清单里的 ``size_buckets`` 对齐（尺寸不一致说明轴序或 pad_multiple 变了）。"""
    agg = (manifest or {}).get("aggregate") or {}
    recorded = agg.get("size_buckets") or {}
    if not recorded:
        return
    mine = {format_bucket(k): len(v) for k, v in ds.buckets.items()}
    if dict(mine) != dict(recorded):
        LOGGER.warning("本脚本按 (H,W)+pad_multiple=%d 分桶得到 %s，清单记录 %s",
                       ds.pad_multiple, mine, recorded)
    else:
        LOGGER.info("分桶与清单一致：%s", mine)


# --------------------------------------------------------------------------------------
# batch 实测
# --------------------------------------------------------------------------------------

def describe_batch(batch: dict) -> dict:
    """汇总一个 batch 的形态信息（不落盘，供打印与报告使用）。"""
    image, label = batch["image"], batch["label"]
    cases = sorted(set(batch["case"]))
    per_sample_ratio = [float((label[i] > 0).float().mean()) for i in range(label.shape[0])]
    return {
        "batch_size": int(image.shape[0]),
        "image_shape": [int(s) for s in image.shape],
        "label_shape": [int(s) for s in label.shape],
        "image_dtype": str(image.dtype),
        "label_dtype": str(label.dtype),
        "image_min": round(float(image.min()), 6),
        "image_max": round(float(image.max()), 6),
        "label_values": sorted(int(v) for v in np.unique(label.numpy()).tolist()),
        "cases": cases,
        "z": [int(z) for z in batch["z"]],
        "orig_hw": [[int(h), int(w)] for h, w in batch["orig_hw"]],
        "pos_slices_in_batch": int(sum(1 for i in range(label.shape[0]) if bool((label[i] > 0).any()))),
        "pos_ratio_measured": round(
            float(sum(1 for i in range(label.shape[0]) if bool((label[i] > 0).any())) / max(1, image.shape[0])), 4),
        "pos_pixels_per_slice": [round(v, 6) for v in per_sample_ratio],
        "bucket": format_bucket(bucket_key(int(image.shape[2]), int(image.shape[3]))),
    }


def check_batch(idx: int, batch: dict, problems: list, tolerance: float) -> dict:
    """校验一个 batch：尺寸一致、值域、label 取值、阳性比例，返回汇总 dict。"""
    info = describe_batch(batch)
    image, label = batch["image"], batch["label"]
    n = int(image.shape[0])

    # 1) 同 batch 内尺寸必须一致（分桶采样器的硬约束）
    shapes = {tuple(int(s) for s in image[i].shape) for i in range(n)}
    if len(shapes) != 1:
        problems.append(f"batch {idx}：同 batch 内 image 尺寸不一致 {sorted(shapes)}（分桶失效）")
    if tuple(image.shape[1:]) != tuple(label.shape[1:]) or int(image.shape[0]) != int(label.shape[0]):
        problems.append(f"batch {idx}：image {tuple(image.shape)} 与 label {tuple(label.shape)} 不匹配")
    if info["bucket"] != format_bucket(bucket_key(int(image.shape[2]), int(image.shape[3]))):
        problems.append(f"batch {idx}：桶键与 shape 不符")

    # 2) 值域与标签取值
    if info["image_min"] < -1e-6 or info["image_max"] > 1.0 + 1e-6:
        problems.append(f"batch {idx}：image 值域 [{info['image_min']}, {info['image_max']}] 超出 [0,1]")
    bad = [v for v in info["label_values"] if v not in (0, 1)]
    if bad:
        problems.append(f"batch {idx}：label 出现了 {bad}，不是 {0,1}")
    if info["image_dtype"] != "torch.float32":
        problems.append(f"batch {idx}：image dtype={info['image_dtype']}，期望 float32")
    if info["label_dtype"] not in ("torch.int64", "torch.int32"):
        problems.append(f"batch {idx}：label dtype={info['label_dtype']}，期望整型")

    # 3) 阳性比例
    if abs(info["pos_ratio_measured"] - tolerance["target"]) > tolerance["limit"]:
        problems.append(f"batch {idx}：batch 内含肿瘤切片比例 {info['pos_ratio_measured']:.3f} "
                        f"偏离目标 {tolerance['target']:.2f} 超过 {tolerance['limit']:.2f}")

    LOGGER.info("batch %d：image %s / label %s；桶 %s；病例 %s；z=%s",
                idx, tuple(info["image_shape"]), tuple(info["label_shape"]),
                info["bucket"], info["cases"], info["z"])
    LOGGER.info("        值域 [%.4f, %.4f]；label 取值 %s；**含肿瘤切片 %d/%d = %.3f**（目标 %.2f）；"
                "本 batch 消耗 %s",
                info["image_min"], info["image_max"], info["label_values"],
                info["pos_slices_in_batch"], info["batch_size"], info["pos_ratio_measured"],
                tolerance["target"], tolerance.get("_note", ""))
    return info


def run_batches(loader, n_batches: int, tolerance: dict, problems: list, tag: str,
                max_batches: int = 0) -> list:
    """从 loader 依次取 ``n_batches`` 个 batch，逐个自检并计时。

    ``max_batches>0`` 时只统计前面若干个 batch 的平均耗时（第一个 batch 含冷读，单独报）。
    """
    infos: list = []
    times: list = []
    iterator = iter(loader)
    total = len(loader) if hasattr(loader, "__len__") else -1
    LOGGER.info("[%s] 开始取 %d 个 batch（该 loader 一轮共 %s 个 batch）", tag, n_batches, total)
    for i in range(1, n_batches + 1):
        t0 = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            LOGGER.warning("[%s] loader 提前结束（只有 %d 个 batch）", tag, i - 1)
            break
        dt = time.perf_counter() - t0
        times.append(dt)
        info = check_batch(i, batch, problems, tolerance)
        info["seconds"] = round(dt, 4)
        infos.append(info)
        LOGGER.info("        取该 batch 耗时 %.3f s%s", dt, "（含首次冷读）" if i == 1 else "")
    if len(times) > 1:
        warm = times[1:]
        LOGGER.info("[%s] 单 batch 耗时：首个 %.3f s，其余平均 %.3f s（min %.3f / max %.3f）",
                    tag, times[0], float(np.mean(warm)), float(np.min(warm)), float(np.max(warm)))
    return infos


# --------------------------------------------------------------------------------------
# 切片轴校验
# --------------------------------------------------------------------------------------

def pick_probe_case(ds: CTSliceDataset, preferred: int | None) -> int:
    """挑一个用于「逐层曲线」的病例：优先 ``--probe-case``，否则取含肿瘤层最多的那一例。"""
    if preferred is not None:
        if preferred not in ds.case_ids:
            raise ValueError(f"--probe-case {preferred} 不在 split={ds.split} 的病例里：{ds.case_ids}")
        return int(preferred)
    if not ds.case_pos_slices:
        raise ValueError("数据集里没有任何病例")
    best = max(ds.case_ids, key=lambda c: (ds.case_pos_slices.get(c, 0), -c))
    if ds.case_pos_slices.get(best, 0) == 0:
        LOGGER.warning("split=%s 里没有任何含肿瘤的病例，逐层曲线只能展示全 0", ds.split)
    return int(best)


def curve_rows(values: list) -> list:
    """把逐层曲线压缩成不超过 ``MAX_CURVE_ROWS`` 行的 ``(start, stop, 区间体素合计, 区间峰值)`` 列表。"""
    n = len(values)
    if n <= MAX_CURVE_ROWS:
        return [(i, i + 1, int(values[i]), int(values[i])) for i in range(n)]
    step = int(np.ceil(n / MAX_CURVE_ROWS))
    rows = []
    for start in range(0, n, step):
        chunk = values[start:start + step]
        rows.append((start, start + len(chunk), int(sum(chunk)), int(max(chunk))))
    return rows


def check_slice_axis(ds: CTSliceDataset, case: int, manifest: dict, problems: list) -> dict:
    """打印某一病例的逐层肿瘤体素数曲线，并用「首尾端阳性占比」判断切片轴是否搞反。

    曲线应集中在肝脏所在的中段；若阳性全挤在 ``z≈0`` 或 ``z≈nz-1``，说明切片轴被搞反了
    （例如把 ``a[:, :, k]`` 写成了 ``a[k]``）。
    """
    with open_volume(ds.label_dir / f"{case}.nii.gz") as lab_img:
        lab = np.asanyarray(lab_img.dataobj)
        nx, ny, nz = (int(lab.shape[0]), int(lab.shape[1]), int(lab.shape[2]))
        per_slice = (lab > 0).sum(axis=(0, 1))      # 对 (x, y) 求和 → 长度 nz
        counts = [int(v) for v in per_slice.tolist()]
    n_pos = int(sum(1 for v in counts if v > 0))
    total_voxels = int(sum(counts))
    peak = int(np.argmax(counts)) if counts else -1

    LOGGER.info("逐层肿瘤体素数（case %d）：nibabel shape=(nx,ny,nz)=%s，切片轴=最后一维，"
                "含肿瘤层 %d/%d，肿瘤体素合计 %d，峰值在第 %d 层（z 从 0 开始）",
                case, (nx, ny, nz), n_pos, nz, total_voxels, peak)
    LOGGER.info("   z 范围 | 体素合计 | 区间峰值 | 直方图（每格 1 层，按区间峰值画）")
    scale = max(1, max(counts) if counts else 1)
    for start, stop, total, top in curve_rows(counts):
        bar = "#" * max(1, int(round(top / scale * 30))) if top > 0 else ""
        LOGGER.info("  %4d-%4d | %8d | %8d | %s", start, stop - 1, total, top, bar)

    head_n = max(1, int(round(nz * 0.1)))
    tail = counts[-head_n:]
    head = counts[:head_n]
    head_pos = sum(1 for v in head if v > 0) / max(1, len(head))
    tail_pos = sum(1 for v in tail if v > 0) / max(1, len(tail))
    manifest_slices = None
    rec = (manifest or {}).get("cases") and next(
        (r for r in manifest["cases"] if int(r.get("case", -1)) == int(case)), None)
    if rec:
        manifest_slices = int(rec.get("tumor_slices", -1))

    info = {
        "case": int(case),
        "nib_shape_xyz": [nx, ny, nz],
        "n_slices": nz,
        "tumor_slices_measured": n_pos,
        "tumor_slices_manifest": manifest_slices,
        "peak_slice": peak,
        "head10_pos_ratio": round(head_pos, 4),
        "tail10_pos_ratio": round(tail_pos, 4),
        "total_tumor_voxels": total_voxels,
    }
    LOGGER.info("  首 10%% 层（z<%d）阳性占比 %.3f，尾 10%% 层阳性占比 %.3f", head_n, head_pos, tail_pos)
    LOGGER.info("  判读：曲线应集中在肝脏所在的中段；若阳性全挤在 z≈0 或 z≈nz-1，说明切片轴被搞反了。")

    if n_pos > nz:
        problems.append(f"case {case}：含肿瘤层数 {n_pos} 超过总层数 {nz}（per-slice 轴用错）")
    if manifest_slices is not None and manifest_slices != n_pos:
        problems.append(f"case {case}：实测含肿瘤层 {n_pos} 与清单 {manifest_slices} 不一致")
    return info


# --------------------------------------------------------------------------------------
# 增强与采样器自检
# --------------------------------------------------------------------------------------

def check_augment(cfg: dict, ds: CTSliceDataset, n_samples: int, problems: list) -> dict:
    """对若干样本比较「无增强 / 有增强」的输出：形状、值域、标签取值、被改动的比例。

    传入的 ``ds`` 必须是**无增强**的数据集（``augment=False``），这样取到的是原始切片；
    增强流水线单独用 ``build_transforms`` 构造（自实现，随机源由 seed 固定）。
    """
    transforms = build_transforms(cfg, train=True, seed=int(((cfg or {}).get("train") or {}).get("seed", 42)))
    if transforms is None:
        problems.append("build_transforms(cfg, train=True) 返回了 None，训练增强没生效")
        return {}

    rng = np.random.default_rng(int(((cfg or {}).get("train") or {}).get("seed", 42)))
    n = len(ds)
    indices = [int(i) for i in rng.choice(n, size=min(n_samples, n), replace=False)]
    changed = 0
    label_changed = 0
    shapes_ok = True
    ranges_ok = True
    labels_ok = True

    for i in indices:
        sample = ds[i]
        raw_image = sample["image"][0].numpy()
        raw_label = sample["label"].numpy().astype(np.uint8)
        out = transforms({"image": raw_image, "label": raw_label})
        aug_image = np.asarray(out["image"], dtype=np.float32)
        aug_label = np.asarray(out["label"]).astype(np.uint8)
        if aug_image.ndim == 3 and aug_image.shape[0] == 1:
            aug_image = aug_image[0]
        if aug_label.ndim == 3 and aug_label.shape[0] == 1:
            aug_label = aug_label[0]
        if aug_image.shape != raw_image.shape or aug_label.shape != raw_label.shape:
            shapes_ok = False
            problems.append(f"增强改变了形状：idx={i} image {aug_image.shape}（原 {raw_image.shape}）"
                            f" label {aug_label.shape}（原 {raw_label.shape}）")
        if float(aug_image.min()) < -1e-6 or float(aug_image.max()) > 1.0 + 1e-6:
            ranges_ok = False
            problems.append(f"增强后 image 值域 [{float(aug_image.min()):.4f}, {float(aug_image.max()):.4f}] 超出 [0,1]")
        bad = sorted({int(v) for v in np.unique(aug_label).tolist()} - {0, 1})
        if bad:
            labels_ok = False
            problems.append(f"增强后 label 出现 {bad}，不是 {{0,1}}")
        if not np.array_equal(aug_image, raw_image):
            changed += 1
        if not np.array_equal(aug_label, raw_label):
            label_changed += 1

    info = {
        "n_samples": len(indices),
        "image_changed": changed,
        "image_changed_ratio": round(changed / max(1, len(indices)), 4),
        "label_changed": label_changed,
        "label_changed_ratio": round(label_changed / max(1, len(indices)), 4),
        "shapes_preserved": shapes_ok,
        "value_range_ok": ranges_ok,
        "label_binary_ok": labels_ok,
    }
    LOGGER.info("增强自检（%d 个样本）：image 被改动 %d 个（%.2f）、label 被改动 %d 个（%.2f）；"
                "形状/值域/标签二值 = %s/%s/%s",
                info["n_samples"], changed, info["image_changed_ratio"],
                label_changed, info["label_changed_ratio"],
                shapes_ok, ranges_ok, labels_ok)
    if changed == 0:
        LOGGER.warning("所有样本的 image 都没被改动：增强概率可能全为 0，或 transforms 没接上")
    if label_changed == 0:
        LOGGER.warning("所有样本的 label 都没被改动：几何增强可能没生效（翻转/旋转/仿射应当会改到 label）")
    return info


def check_affine_geometry(problems: list) -> dict:
    """用一个人造图案验证 ``GridAffine2D`` 的几何口径（纯 CPU、毫秒级）。

    图案：正中心一个方块 + 右上角一个标记。检查
      * 旋转 90°：右上角的标记应转到左上角（正角度 = 内容逆时针转）；
      * 平移：整体位移方向与给定量一致；
      * 缩放 0.5：前景体素变少、但仍集中在中心。
    """
    size = 64
    canvas = np.zeros((size, size), dtype=np.float32)
    canvas[size // 2 - 8:size // 2 + 8, size // 2 - 8:size // 2 + 8] = 1.0          # 中心方块
    canvas[6:14, size - 14:size - 6] = 2.0                                          # 右上角标记（值 2 便于追踪）
    checks: dict = {}

    rotated = GridAffine2D(rotate_deg=90.0).warp(canvas)
    ys, xs = np.nonzero(rotated > 1.5)
    checks["rotate90_marker_center_xy"] = [round(float(xs.mean()), 2), round(float(ys.mean()), 2)]
    checks["rotate90_expected_xy"] = [round(size - 1 - (6 + 13) / 2, 2), round((6 + 13) / 2, 2)]
    ok_rot = abs(checks["rotate90_marker_center_xy"][0] - checks["rotate90_expected_xy"][0]) <= 2 \
        and abs(checks["rotate90_marker_center_xy"][1] - checks["rotate90_expected_xy"][1]) <= 2
    checks["rotate90_ok"] = bool(ok_rot)
    if not ok_rot:
        problems.append(f"GridAffine2D 旋转 90° 的几何不对：标记落在 "
                        f"{checks['rotate90_marker_center_xy']}，期望 {checks['rotate90_expected_xy']}")

    shifted = GridAffine2D(rotate_deg=0.0, shift_xy=(0.125, 0.0)).warp(canvas)
    ys2, xs2 = np.nonzero(shifted > 1.5)
    checks["shift_marker_center_x"] = round(float(xs2.mean()), 2)
    checks["shift_expected_x"] = round((size - 1 - (6 + 13) / 2) + 0.125 * size, 2)
    ok_shift = abs(checks["shift_marker_center_x"] - checks["shift_expected_x"]) <= 2
    checks["shift_ok"] = bool(ok_shift)
    if not ok_shift:
        problems.append(f"GridAffine2D 平移方向不对：标记 x={checks['shift_marker_center_x']}，"
                        f"期望 {checks['shift_expected_x']}（+x 应向右）")

    small = GridAffine2D(rotate_deg=0.0, scale=0.5).warp(canvas)
    area_raw = float((canvas > 0.5).sum())
    area_small = float((small > 0.5).sum())
    checks["scale_area_ratio"] = round(area_small / max(1.0, area_raw), 4)
    checks["scale_ok"] = bool(area_small < area_raw)
    if not checks["scale_ok"]:
        problems.append(f"GridAffine2D 缩放 0.5 后前景没变小：{area_small} vs {area_raw}")
    LOGGER.info("仿射几何自检：旋转 90° %s（标记 %s，期望 %s）；平移 %s；缩放面积比 %.3f",
                "通过" if checks["rotate90_ok"] else "失败",
                checks["rotate90_marker_center_xy"], checks["rotate90_expected_xy"],
                "通过" if checks["shift_ok"] else "失败", checks["scale_area_ratio"])
    return checks


def check_custom_transforms(problems: list) -> dict:
    """逐个验证自实现的增强步骤（不依赖 MONAI，逻辑可在本地离线推演）。

    检查项：
      * `RandAffineSlice2D`（prob=1）后形状不变、image 落在 [0,1]、label 仍是 {0,1}；prob=0 恒等；
      * `FlipSlice2D` 沿 axis 翻转且 image/label 同步；
      * `Rotate90Slice2D` 旋转后形状与内容总量不变；
      * `GammaSlice2D` 单调保序、gamma<1 提亮；
      * `GaussianNoiseSlice2D` 的 sigma 不超过配置上界、输出夹在 [0,1]；
      * `ClampImageToUnit` / `BinarizeLabel` 的边界行为。
    """
    import random as _random

    size = 64
    canvas = np.zeros((size, size), dtype=np.float32)
    canvas[20:44, 20:44] = 1.0
    canvas[2:6, 58:62] = 1.0
    label = (canvas > 0).astype(np.uint8)

    step = RandAffineSlice2D(prob=1.0, aug={"rotation_deg": 15.0, "scale_range": [0.9, 1.1],
                                            "shift_frac": 0.1}, rng=_random.Random(0))
    out = step({"image": canvas, "label": label})
    aug_image = np.asarray(out["image"], dtype=np.float32)
    aug_label = np.asarray(out["label"])

    # 翻转方向：只在上边一行有内容的图案，翻行后内容应落到最后一行
    marked = np.zeros((8, 8), dtype=np.float32)
    marked[0, :] = 1.0
    flip0 = FlipSlice2D(prob=1.0, axis=0, rng=_random.Random(0))({"image": marked, "label": marked})
    flip1 = FlipSlice2D(prob=1.0, axis=1, rng=_random.Random(0))({"image": marked, "label": marked})
    flip0_ok = bool(float(np.asarray(flip0["image"])[-1, :].sum()) == 8.0
                    and float(np.asarray(flip0["image"])[0, :].sum()) == 0.0)
    flip1_ok = bool(np.array_equal(np.asarray(flip1["image"]), marked))
    flip_sync = bool(np.array_equal(np.asarray(flip0["image"]).astype(np.uint8),
                                    np.asarray(flip0["label"])))

    rot = Rotate90Slice2D(prob=1.0, max_k=1, rng=_random.Random(1))({"image": canvas, "label": label})
    rot_ok = bool(np.asarray(rot["image"]).shape == canvas.shape
                  and abs(float(np.asarray(rot["image"]).sum()) - float(canvas.sum())) < 1e-3)

    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    gam = GammaSlice2D(prob=1.0, gamma_range=(0.5, 0.5), rng=_random.Random(2))({"image": ramp})
    gam_img = np.asarray(gam["image"], dtype=np.float32)
    gamma_monotonic = bool(np.all(np.diff(gam_img.ravel()) >= -1e-7))
    gamma_brighter = bool(gam_img[0, 4] > ramp[0, 4])

    noise_step = GaussianNoiseSlice2D(prob=1.0, std=0.02, rng=_random.Random(3))
    noisy = noise_step({"image": canvas})
    noise_in_range = bool(float(np.asarray(noisy["image"]).min()) >= -1e-6
                          and float(np.asarray(noisy["image"]).max()) <= 1.0 + 1e-6)

    clamp_out = ClampImageToUnit()({"image": np.asarray([-0.5, 0.5, 1.5], dtype=np.float32)})
    bin_out = BinarizeLabel()({"label": np.asarray([0, 1, 2, 3], dtype=np.uint8)})

    info = {
        "affine_prob1_shape_preserved": bool(aug_image.shape == canvas.shape
                                             and aug_label.shape == label.shape),
        "affine_prob1_changed_image": bool(not np.array_equal(aug_image, canvas)),
        "affine_prob1_image_range": [round(float(aug_image.min()), 4), round(float(aug_image.max()), 4)],
        "affine_prob1_image_in_range": bool(aug_image.min() >= -1e-6 and aug_image.max() <= 1.0 + 1e-6),
        "affine_prob1_label_values": sorted(int(v) for v in np.unique(aug_label).tolist()),
        "affine_prob0_identity": bool(np.array_equal(
            RandAffineSlice2D(prob=0.0, rng=_random.Random(0))(
                {"image": canvas, "label": label})["image"], canvas)),
        "flip_axis0_moves_top_row_to_bottom": flip0_ok,
        "flip_axis1_identity_on_constant_columns": flip1_ok,
        "flip_image_label_synced": flip_sync,
        "rotate90_shape_and_sum_kept": rot_ok,
        "gamma_monotonic": gamma_monotonic,
        "gamma_half_brightens": gamma_brighter,
        "noise_sigma": round(float(noise_step.last_sigma or 0.0), 6),
        "noise_in_range": noise_in_range,
        "clamp_values": [round(float(v), 4) for v in np.asarray(clamp_out["image"]).tolist()],
        "clamp_ok": bool(np.allclose(np.asarray(clamp_out["image"]), [0.0, 0.5, 1.0], atol=1e-6)),
        "binarize_values": [int(v) for v in np.asarray(bin_out["label"]).tolist()],
        "binarize_ok": bool(np.array_equal(np.asarray(bin_out["label"]), np.asarray([0, 1, 1, 1]))),
    }

    for flag, msg in [
        (info["affine_prob1_shape_preserved"], "RandAffineSlice2D(prob=1) 改变了形状"),
        (info["affine_prob1_changed_image"], "RandAffineSlice2D(prob=1) 没有改动 image（prob 语义可能反了）"),
        (info["affine_prob1_image_in_range"],
         f"RandAffineSlice2D 后 image 值域 {info['affine_prob1_image_range']} 超出 [0,1]"),
        (info["affine_prob0_identity"], "RandAffineSlice2D(prob=0) 不应该改动数据"),
        (flip0_ok, "FlipSlice2D(axis=0) 没有把第一行翻到最后一行"),
        (flip1_ok, "FlipSlice2D(axis=1) 在列方向常数的图上应保持不变"),
        (flip_sync, "FlipSlice2D 没有让 image/label 同步"),
        (rot_ok, "Rotate90Slice2D 改变了形状或内容总量"),
        (gamma_monotonic, "GammaSlice2D 不是单调映射（会破坏亮暗关系）"),
        (gamma_brighter, "GammaSlice2D(gamma=0.5) 应该提亮"),
        (noise_in_range, "GaussianNoiseSlice2D 输出超出 [0,1]"),
        (info["clamp_ok"], f"ClampImageToUnit 结果 {info['clamp_values']}，期望 [0, 0.5, 1]"),
        (info["binarize_ok"], f"BinarizeLabel 结果 {info['binarize_values']}，期望 [0, 1, 1, 1]"),
    ]:
        if not flag:
            problems.append(msg)
    if info["affine_prob1_label_values"] not in ([0], [0, 1], [1]):
        problems.append(f"RandAffineSlice2D 后 label 取值异常：{info['affine_prob1_label_values']}")
    if info["noise_sigma"] > 0.02 + 1e-9:
        problems.append(f"GaussianNoiseSlice2D 的 sigma {info['noise_sigma']} 超过配置上界 0.02")

    LOGGER.info("自实现增强自检：仿射(prob=1) 形状保持 %s / 有改动 %s / 值域 %s / label 取值 %s；"
                "prob=0 恒等 %s；翻转(axis0 %s, axis1 %s, 同步 %s)；旋转 90° %s；"
                "gamma 单调 %s / 提亮 %s；噪声 sigma=%.5f 值域 %s；夹取 %s；二值化 %s",
                info["affine_prob1_shape_preserved"], info["affine_prob1_changed_image"],
                info["affine_prob1_image_range"], info["affine_prob1_label_values"],
                info["affine_prob0_identity"], flip0_ok, flip1_ok, flip_sync, rot_ok,
                gamma_monotonic, gamma_brighter, info["noise_sigma"], noise_in_range,
                info["clamp_ok"], info["binarize_ok"])
    return info


def check_augment_pipeline(cfg: dict, problems: list) -> dict:
    """确认增强步骤的数量、顺序与作用对象（防止有人把某一步删掉却没发现）。"""
    try:
        steps = make_augment_steps(cfg)
    except Exception as exc:  # noqa: BLE001 - 构造失败要显式暴露
        problems.append(f"make_augment_steps 构造失败：{type(exc).__name__}: {exc}")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    names = [type(step).__name__ for step in steps]
    expected = ["FlipSlice2D", "FlipSlice2D", "Rotate90Slice2D", "RandAffineSlice2D",
                "GammaSlice2D", "GaussianNoiseSlice2D", "ClampImageToUnit", "BinarizeLabel"]
    ok = names == expected
    if not ok:
        problems.append(f"增强步骤与约定不一致：实际 {names}，期望 {expected}")
    geo_ok = all(set(getattr(s, "keys", ())) == {"image", "label"} for s in steps[:4])
    photo_ok = all(set(getattr(s, "keys", ())) == {"image"} for s in steps[4:6])
    if not geo_ok:
        problems.append(f"几何增强的 keys 不是 (image,label)：{[getattr(s, 'keys', None) for s in steps[:4]]}")
    if not photo_ok:
        problems.append(f"强度增强的 keys 不是 (image,)：{[getattr(s, 'keys', None) for s in steps[4:6]]}")
    LOGGER.info("增强流水线：%d 步 %s（几何同步 image+label：%s；强度仅 image：%s）",
                len(names), names, geo_ok, photo_ok)
    return {"ok": bool(ok and geo_ok and photo_ok), "steps": names, "expected": expected,
            "geo_synced": geo_ok, "photo_image_only": photo_ok}
    if not ok:
        problems.append(f"增强步骤与约定不一致：实际 {names}，期望 {expected}")
    LOGGER.info("增强流水线：%d 步 %s", len(names), names)
    return {"ok": bool(ok), "steps": names, "expected": expected}


def check_sampler(ds: CTSliceDataset, cfg: dict, problems: list) -> dict:
    """采样器自检：批数与配额、正/负覆盖、以及「同 seed 两轮完全一致」的可复现性。"""
    sampler = BucketBatchSampler(ds, batch_size=int(((cfg or {}).get("train") or {}).get("batch_size", 8)),
                                 pos_ratio_target=float((((cfg or {}).get("train") or {})
                                                         .get("pos_ratio_target", 0.30))),
                                 seed=int(((cfg or {}).get("train") or {}).get("seed", 42)))
    LOGGER.info("%s", sampler.describe())

    first = [list(b) for b in sampler]
    second = [list(b) for b in sampler]
    n_batches = len(first)
    reproducible = bool(first == second)
    info = {
        "n_batches_per_epoch": n_batches,
        "expected_batches": round(len(ds) / max(1, sampler.batch_size), 2),
        "reproducible": reproducible,
        "epoch1_seed": sampler.epoch_seed(1),
        "n_pos": sampler.bucket_counts and int(sum(v[0] for v in sampler.bucket_counts.values())),
        "n_neg": sampler.bucket_counts and int(sum(v[1] for v in sampler.bucket_counts.values())),
    }
    if not reproducible:
        problems.append("同一个 epoch 采样两次结果不同：采样器不可复现（检查种子派生是否用了 hash()）")

    # 逐 batch 校验：尺寸一致 + 阳性数量符合目标
    n_pos_target = int(round(sampler.batch_size * sampler.pos_ratio_target))
    bad_sizes = 0
    bad_pos = 0
    for k, batch in enumerate(first, 1):
        sizes = {ds.case_hw[ds.index[i][0]] for i in batch}
        if len(sizes) != 1:
            bad_sizes += 1
        got = int(ds.pos_flags[batch].sum())
        if got != n_pos_target:
            bad_pos += 1
    info["batches_with_mixed_sizes"] = bad_sizes
    info["batches_with_wrong_pos_count"] = bad_pos
    info["pos_count_target"] = n_pos_target
    if bad_sizes:
        problems.append(f"{bad_sizes}/{n_batches} 个 batch 混了不同面内尺寸（分桶失效）")
    if bad_pos:
        LOGGER.warning("%d/%d 个 batch 的阳性切片数不等于目标 %d（桶内样本不足时会如此，"
                       "若数量很多需要调低 batch_size 或改采样策略）", bad_pos, n_batches, n_pos_target)
    LOGGER.info("采样器自检：每轮 %d 个 batch（期望约 %.1f）；同 epoch 两轮一致=%s；"
                "混尺寸 batch=%d；阳性数偏离目标的 batch=%d",
                n_batches, info["expected_batches"], reproducible, bad_sizes, bad_pos)
    return info


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="数据集自检（只读 cache，不训练）")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件")
    parser.add_argument("--cache-dir", default=None, help="覆盖 cache/ 位置")
    parser.add_argument("--fold", type=int, default=0, help="用哪一折的 train/val（默认 0）")
    parser.add_argument("--batches", type=int, default=3, help="实测多少个 batch（默认 3）")
    parser.add_argument("--probe-case", type=int, default=None, help="逐层曲线用哪个病例（默认含肿瘤层最多的一例）")
    parser.add_argument("--aug-samples", type=int, default=8, help="增强自检抽多少个样本（默认 8）")
    parser.add_argument("--skip-sampler", action="store_true", help="跳过采样器自检（它要跑完整轮采样）")
    parser.add_argument("--set", dest="overrides", action="append", default=None, help="覆盖配置项，可多次")
    parser.add_argument("--out", default=None, help="自检报告路径，默认 reports/selfcheck_data.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    paths = cfg.get("paths") or {}
    cache_dir = args.cache_dir or paths.get("cache", "cache")
    train_cfg = cfg.get("train") or {}
    data_cfg = data_config(cfg)
    problems: list = []

    LOGGER.info("=" * 78)
    LOGGER.info("第 2 轮数据自检：Dataset + 分桶采样器 + 增强（fold=%d）", int(args.fold))
    LOGGER.info("=" * 78)

    env = check_environment()
    splits, manifest, prereq_problems = check_prerequisites(cfg)
    problems.extend(prereq_problems)

    # ---- 数据集 ----
    LOGGER.info("-" * 78)
    LOGGER.info("1) 构建数据集（train 侧带增强、val 侧无增强）")
    train_ds = CTSliceDataset("train", cache_dir, "train", cfg, augment=True,
                              debug=True, fold=int(args.fold))
    val_ds = CTSliceDataset("val", cache_dir, "val", cfg, augment=False, fold=int(args.fold))
    LOGGER.info("\n%s", train_ds.describe())
    LOGGER.info("\n%s", val_ds.describe())
    check_bucket_consistency(train_ds, manifest, problems)

    tolerance = {
        "target": float(train_cfg.get("pos_ratio_target", 0.30)),
        "limit": float(data_cfg.get("pos_ratio_tolerance", 0.10)),
    }
    LOGGER.info("整体阳性率：train 侧 %.4f（%d/%d），val 侧 %.4f（%d/%d）；"
                "目标 batch 内比例 %.2f（≈ %.1f 倍过采样）",
                train_ds.pos_ratio, train_ds.n_positive, len(train_ds),
                val_ds.pos_ratio, val_ds.n_positive, len(val_ds),
                tolerance["target"], tolerance["target"] / max(1e-9, train_ds.pos_ratio))

    # ---- 3 个 batch 实测 ----
    LOGGER.info("-" * 78)
    LOGGER.info("2) 实测 batch（第 1 个 batch 含冷读，耗时单独看）")
    train_loader = make_train_loader(splits, int(args.fold), cfg)
    batch_infos = run_batches(train_loader, int(args.batches), tolerance, problems, tag="train")

    val_loader = make_val_loader(splits, int(args.fold), cfg)
    val_infos = run_batches(val_loader, min(2, int(args.batches)), tolerance, problems, tag="val")
    if val_infos and any(info["pos_ratio_measured"] > 0 for info in val_infos):
        LOGGER.info("提示：val 侧顺序读取，batch 内阳性比例天然接近整体比例（%.4f），"
                    "不做过采样是正确行为。", val_ds.pos_ratio)

    if batch_infos:
        sizes = {tuple(info["image_shape"][2:]) for info in batch_infos}
        LOGGER.info("%d 个 train batch 涉及 %d 种面内尺寸：%s（每个 batch 内部只有 1 种）",
                    len(batch_infos), len(sizes), sorted(sizes))
    if env.get("cuda_available"):
        import torch

        LOGGER.info("CUDA 显存：allocated %.1f MB / reserved %.1f MB（只做数据加载，未建模型）",
                    torch.cuda.memory_allocated() / 1024 ** 2,
                    torch.cuda.memory_reserved() / 1024 ** 2)

    # ---- 切片轴 ----
    LOGGER.info("-" * 78)
    LOGGER.info("3) 切片轴校验（逐层肿瘤体素数曲线）")
    probe_case = pick_probe_case(train_ds, args.probe_case)
    axis_info = check_slice_axis(train_ds, probe_case, manifest, problems)

    # ---- 增强 ----
    LOGGER.info("-" * 78)
    LOGGER.info("4) 增强自检（为了对比原始切片，额外建一个无增强的数据集）")
    raw_ds = CTSliceDataset("train", cache_dir, "train", cfg, augment=False, fold=int(args.fold))
    augment_info = check_augment(cfg, raw_ds, int(args.aug_samples), problems)
    pipeline_info = check_augment_pipeline(cfg, problems)
    affine_info = check_affine_geometry(problems)
    custom_info = check_custom_transforms(problems)

    # ---- 采样器 ----
    sampler_info = {}
    if not args.skip_sampler:
        LOGGER.info("-" * 78)
        LOGGER.info("5) 分桶采样器自检")
        sampler_info = check_sampler(train_ds, cfg, problems)

    # ---- 收尾：评估是否可进入第 3 轮 ----
    del train_loader, val_loader
    reset_open_cache()

    report = {
        "fold": int(args.fold),
        "config": rel_to_root(resolve_path(args.config)),
        "cache_dir": rel_to_root(resolve_path(cache_dir)),
        "env": env,
        "train_dataset": {
            "n_cases": len(train_ds.case_ids),
            "cases": train_ds.case_ids,
            "n_slices": len(train_ds),
            "n_positive": train_ds.n_positive,
            "pos_ratio": round(train_ds.pos_ratio, 6),
            "image_dtype": train_ds.image_dtype,
            "pad_multiple": train_ds.pad_multiple,
            "buckets": {format_bucket(k): v for k, v in train_ds.bucket_stats().items()},
            "per_case": {str(c): {"nz": train_ds.case_n_slices[c],
                                  "pos_slices": train_ds.case_pos_slices[c],
                                  "hw": list(train_ds.case_hw[c]),
                                  "bucket": format_bucket(bucket_key(*train_ds.case_hw[c],
                                                                    train_ds.pad_multiple))}
                         for c in train_ds.case_ids},
        },
        "val_dataset": {
            "n_cases": len(val_ds.case_ids),
            "cases": val_ds.case_ids,
            "n_slices": len(val_ds),
            "n_positive": val_ds.n_positive,
            "pos_ratio": round(val_ds.pos_ratio, 6),
        },
        "batches_train": batch_infos,
        "batches_val": val_infos,
        "slice_axis": axis_info,
        "augment": augment_info,
        "augment_pipeline": pipeline_info,
        "affine_geometry": affine_info,
        "custom_transforms": custom_info,
        "sampler": sampler_info,
        "problems": problems,
    }
    out_path = args.out or (Path(paths.get("reports", "reports")) / "selfcheck_data.json")
    report_path = save_json(report, out_path)

    LOGGER.info("=" * 78)
    if problems:
        LOGGER.error("自检发现 %d 个问题：", len(problems))
        for item in problems:
            LOGGER.error("  - %s", item)
        LOGGER.error("报告：%s", rel_to_root(report_path))
        LOGGER.error("请把上面带 - 的行贴回本地，修好后再进入第 3 轮。")
        return 1

    LOGGER.info("自检通过：%d 例 / %d 层切片的形态、值域、标签、分桶、增强与采样比例均符合约定。",
                len(train_ds.case_ids), len(train_ds))
    LOGGER.info("报告：%s（远程产物，不入库，贴回终端输出即可）", rel_to_root(report_path))
    LOGGER.info("下一步：第 3 轮 python -m src.train --fold %d --debug", int(args.fold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
