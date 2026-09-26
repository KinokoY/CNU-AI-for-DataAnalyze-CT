"""数据自检：在写训练代码之前，先把「数据进模型的形态」用真实 cache 验一遍。

整体功能（全部只读 cache，不依赖网络、不依赖模型）：
    1. 环境与前置：打印 torch / nibabel / numpy / CUDA 版本（MONAI 只是**顺带**报告——增强已全部
       自实现，本脚本与 src/dataset.py 都不依赖它）；核对 ``data/splits.json``、
       ``cache/cache_manifest.json`` 是否存在，清单里的预处理指纹是否与当前配置一致；
    2. 数据集形态：``CTSliceDataset``（train / val 两侧）的病例数、切片数、含肿瘤切片比例、
       原始面内尺寸分布与补边占比，以及每例的 ``nz`` / 含肿瘤层数 / 原始尺寸；
    3. 3 个 batch 的实测：image/label 的 shape（应恒为 ``(B,C,512,512)`` / ``(B,512,512)``，
       C = 2×z_context+1）、dtype、中心通道值域、label 取值集合、**batch 内实测肿瘤切片比例**、
       2.5D 三层窗、batch 的病例构成、取 batch 耗时、CUDA 显存；
    4. 切片轴校验：对指定病例打印逐层肿瘤体素数曲线（含 ASCII 直方图）与「首尾各 10% 层的
       阳性占比」，切片轴若被搞反，曲线会贴到某一端而不是集中在中间层；
    5. 增强自检：分「含肿瘤组 / 全背景组」抽样做「无增强 vs 有增强」对比，确认形状不变、
       label 仍是 {0,1}、image 仍在 [0,1]，且含肿瘤组的 label 真的被改动过；
    6. **2.5D 三层窗自检（第 4 轮新增）**：窗口层号必须全在本病例内、中心通道逐像素等于
       「单独读那一层」、越界处用端点复制、几何增强同时改动三层（详见 ``check_slice_window``）；
    7. 平衡采样器自检：同一 epoch 用同一 seed 采样两次必须完全一致（可复现）、相邻 epoch 必须不同、
       每批阳性数恰好是计划值、阳性层重复不超过 ``max_pos_repeat``、抽到的病例全在本折 train 内。

判阈值的口径（旧版在这里误报过，别改回去）：
    * 训练侧每个 batch 的阳性数是**平衡采样计划里的定值**（``batch_targets()`` 的
      ``n_pos_per_batch``），直接等号比对，不是「上限」也不是「区间」；
    * **验证侧不判阳性数**：val loader 是顺序读、不做平衡采样，某个 batch 恰好落在「前 8 层无肿瘤」的
      病例上（如 case 33）就是正常现象。旧版把训练侧的预算套到 val batch 上，于是误报
      「阳性切片 0 超出该桶预算 [1,2]」。
    * 比较 image 与 label 的空间维要**错开一个通道维**（``image (B,C,H,W)`` vs ``label (B,H,W)``）。

前后接口：上游是 scripts/preprocess.py 的产物（cache/ + cache_manifest.json）与 data/splits.json；
        下游是 src/train.py——本脚本通过后再往上搭训练（这是**长期保留的回归自检**，
        任何一轮改了预处理 / dataset / 配置之后都应当重跑一遍）。
用法：仓库根目录执行 ``python -m src.selfcheck_data``（默认 fold 0、3 个 batch）。
     想核对某个特定病例的切片轴（它不在当前折里也行）：``--probe-case 56``。
     产物 reports/selfcheck_data.json 在远程（不入库），把终端输出贴回本地即可。
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

try:
    from src.dataset import (
        UINT16_SCALE,
        BalancedBatchSampler,
        BinarizeLabel,
        CTSliceDataset,
        ClampImageToUnit,
        FlipSlice2D,
        GammaSlice2D,
        GaussianNoiseSlice2D,
        GridAffine2D,
        RandAffineSlice2D,
        Rotate90Slice2D,
        build_transforms,
        data_config,
        format_hw,
        make_augment_steps,
        make_train_loader,
        make_val_loader,
        open_volume,
        pad_offset_of,
        plan_balanced_slots,
        pad_to_target,
        reset_open_cache,
        resolve_z_context,
        slice_usage,
        window_channels,
        window_index,
    )
    from src.utils import (
        cache_cases,
        cache_file,
        cache_fingerprint,
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
        BalancedBatchSampler,
        BinarizeLabel,
        CTSliceDataset,
        ClampImageToUnit,
        FlipSlice2D,
        GammaSlice2D,
        GaussianNoiseSlice2D,
        GridAffine2D,
        RandAffineSlice2D,
        Rotate90Slice2D,
        build_transforms,
        data_config,
        format_hw,
        make_augment_steps,
        make_train_loader,
        make_val_loader,
        open_volume,
        pad_offset_of,
        plan_balanced_slots,
        pad_to_target,
        reset_open_cache,
        resolve_z_context,
        slice_usage,
        window_channels,
        window_index,
    )
    from src.utils import (  # type: ignore
        cache_cases,
        cache_file,
        cache_fingerprint,
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
    n_img = len(cache_cases(cache_dir, "image"))
    n_lab = len(cache_cases(cache_dir, "label"))
    LOGGER.info("cache：%s（image=%d，label=%d；.nii 优先、兼容 .nii.gz）",
                rel_to_root(cache_dir), n_img, n_lab)
    if n_img != n_lab:
        problems.append(f"cache 里 image({n_img}) 与 label({n_lab}) 数量不一致")
    for kind in ("image", "label"):
        compressed = [c for c in cache_cases(cache_dir, kind)
                      if str(cache_file(cache_dir, kind, c)).endswith(".gz")]
        if compressed:
            LOGGER.warning("cache/%s 里有 %d 例仍是压缩的 .nii.gz（如 %s）：nibabel 每读一层都会"
                           "整卷解压，训练取数会慢到 ~9 分钟/epoch。跑一次 "
                           "python scripts/inflate_cache.py 生成同名未压缩 .nii 即可。",
                           kind, len(compressed), compressed[:5])

    manifest_path = paths.get("cache_manifest", "cache/cache_manifest.json")
    manifest = load_json(manifest_path, default=None)
    if not manifest or not manifest.get("cases"):
        LOGGER.warning("读不到 cache 清单 %s —— 跳过「与清单逐例比对」，"
                       "建议先跑 python scripts/fetch_manifest.py", rel_to_root(manifest_path))
        manifest = {}
    else:
        expect_hash = cache_fingerprint(cfg)
        got_hash = manifest.get("cfg_hash")
        LOGGER.info("清单：%d 例，预处理指纹 %s（当前配置 %s）%s",
                    len(manifest["cases"]), got_hash, expect_hash,
                    "一致" if got_hash == expect_hash else "**不一致**")
        if got_hash and got_hash != expect_hash:
            problems.append(f"cache_manifest.cfg_hash={got_hash} 与当前配置算出的 {expect_hash} 不一致："
                            f"缓存可能来自不同的预处理参数，需要重跑 preprocess.py")
        agg = manifest.get("aggregate") or {}
        if agg:
            shapes = agg.get("inplane_shapes") or {}
            LOGGER.info("清单汇总：原始面内尺寸 %s（%d 种）；切片数 %s-%s；含肿瘤切片 %s/%s = %.4f",
                        shapes, len(shapes), agg.get("n_slices_min"), agg.get("n_slices_max"),
                        agg.get("tumor_slices"), agg.get("total_slices"),
                        float(agg.get("tumor_slice_ratio_overall") or 0.0))
    return splits, manifest, problems


def check_inplane_consistency(ds: CTSliceDataset, manifest: dict, problems: list) -> None:
    """核对**原始面内尺寸**与补边口径：逐例尺寸必须与清单一致，且不能超过 ``data.target_hw``。

    已废弃的口径（别改回去）：早期版本按「对齐 ``pad_multiple`` 之后」的尺寸分桶，要求同 batch
    精确同尺寸；现在统一补边到 ``data.target_hw``，分桶概念整条链路都不存在了。
    这里只做两件事：用清单的逐例 ``inplane_hw`` 交叉验证数据集读到的尺寸，并把补边占比列出来
    （补边区的像素是白算的，第 3 轮 loss 可考虑忽略）。
    """
    cases = (manifest or {}).get("cases") or []
    if not cases:
        return
    by_case = {int(r["case"]): r for r in cases if "case" in r}
    mismatch: list = []
    for case in ds.case_ids:
        rec = by_case.get(int(case))
        if not rec:
            continue
        recorded = tuple(int(x) for x in rec["inplane_hw"])
        if recorded != tuple(ds.case_hw[int(case)]):
            mismatch.append(f"case {case}：{tuple(ds.case_hw[int(case)])} vs 清单 {recorded}")
    if mismatch:
        problems.append(f"原始面内尺寸与清单逐例不一致（{len(mismatch)} 例）：{mismatch[:5]}")
    else:
        LOGGER.info("原始面内尺寸与清单逐例一致（%d 例）：%s", len(ds.case_ids),
                    {format_hw(k): v["n_cases"] for k, v in sorted(ds.inplane_stats().items())})
    over = [c for c in ds.case_ids
            if ds.case_hw[int(c)][0] > ds.target_hw[0] or ds.case_hw[int(c)][1] > ds.target_hw[1]]
    if over:
        problems.append(f"这些病例的面内尺寸超过 data.target_hw {format_hw(ds.target_hw)}：{over}")
    LOGGER.info("补边口径：统一补到 %s（%s）；逐种原始尺寸的补边像素占比 %s",
                format_hw(ds.target_hw), ds.pad_align,
                {format_hw(k): f"{v['pad_fraction']:.1%}" for k, v in sorted(ds.inplane_stats().items())})


# --------------------------------------------------------------------------------------
# batch 实测
# --------------------------------------------------------------------------------------

def make_sampler(ds: CTSliceDataset, cfg: dict) -> BalancedBatchSampler:
    """按配置构造一个平衡采样器（不读任何切片），供自检判阈值与复现采样顺序。"""
    train_cfg = (cfg or {}).get("train") or {}
    return BalancedBatchSampler(ds,
                                batch_size=int(train_cfg.get("batch_size", 8)),
                                data_cfg=data_config(cfg),
                                seed=int(train_cfg.get("seed", 42)))


def describe_batch(batch: dict) -> dict:
    """汇总一个 batch 的形态信息（不落盘，供打印与报告使用）。

    取值依据 ``src.dataset.collate_samples`` 定义的 batch 契约：
    ``image``/``label`` 是张量，``case``/``z``/``window``/``orig_hw``/``pad_offset`` 是长度 B 的 list
    （``orig_hw`` / ``pad_offset`` 是 tuple 列表，**不是**被转置的两行列表——默认 collate 会踩这个坑）。

    ``image`` 是 ``(B, C, H, W)``（2.5D：C=3 = [z-1,z,z+1]），所以**逐像素统计取中心通道**、
    通道数单独报；``label`` 是 ``(B, H, W)``。统计一律先 ``.numpy()`` 再算：张量的逐元素比较/归约
    在本地 stubs 里没有实现，而 numpy 侧本来就够用（形状、值域、取值集合）。
    """
    image, label = batch["image"], batch["label"]
    img = np.asarray(image.numpy(), dtype=np.float32)
    lab = np.asarray(label.numpy())
    channels = int(img.shape[1])
    center = int(channels) // 2                    # 2.5D 的中心通道 = 标签所在的层
    target_h, target_w = (int(img.shape[-2]), int(img.shape[-1]))
    per_sample_ratio = [float((lab[i] > 0).mean()) for i in range(lab.shape[0])]
    per_sample_pos = [bool((lab[i] > 0).any()) for i in range(lab.shape[0])]
    per_sample_pad = [
        1.0 - (int(hw[0]) * int(hw[1])) / float(max(1, target_h * target_w)) for hw in batch["orig_hw"]
    ]
    return {
        "batch_size": int(image.shape[0]),
        "image_shape": [int(s) for s in image.shape],
        "image_channels": channels,
        "label_shape": [int(s) for s in label.shape],
        "image_dtype": str(image.dtype),
        "label_dtype": str(label.dtype),
        "image_min": round(float(img[:, center].min()), 6),
        "image_max": round(float(img[:, center].max()), 6),
        "label_values": sorted(int(v) for v in np.unique(lab).tolist()),
        "cases": sorted(set(batch["case"])),
        "z": [int(z) for z in batch["z"]],
        "window": [[int(v) for v in w] for w in batch.get("window", [])],
        "orig_hw": [[int(hw[0]), int(hw[1])] for hw in batch["orig_hw"]],
        "pad_offset": [[int(o[0]), int(o[1])] for o in batch["pad_offset"]],
        "pos_slices_in_batch": int(sum(1 for v in per_sample_pos if v)),
        "pos_ratio_measured": round(float(sum(1 for v in per_sample_pos if v) / max(1, img.shape[0])), 4),
        "pos_pixels_per_slice": [round(v, 6) for v in per_sample_ratio],
        # 补边区在 batch 里占的像素比例（居中补边；512 病例为 0）——loss 是否忽略它看这个数
        "pad_fraction_mean": round(float(np.mean(per_sample_pad)) if per_sample_pad else 0.0, 6),
        "n_padded_samples": int(sum(1 for v in per_sample_pad if v > 0)),
    }


def check_batch(idx: int, batch: dict, problems: list, tolerance: dict, tag: str = "train",
                sampler: "BalancedBatchSampler | None" = None) -> dict:
    """校验一个 batch：形状/通道契约、值域、label 取值，以及（**仅训练侧**）每批阳性数。

    阳性数只在训练侧判，且现在判的是**等号**：平衡采样器每批固定 ``n_pos`` 个含肿瘤切片
    （旧口径是"均摊区间"，因为阳性层总数摊不满每批上限）。
    **验证侧不判阳性数**——val loader 顺序读、不做过采样，某个 batch 恰好落在病例前几层
    无肿瘤（如 case 33 的前 8 层）本来就是正常的；旧版拿训练侧的桶预算去判 val batch，
    于是误报「阳性切片 0 超出该桶预算 [1,2]」。
    """
    info = describe_batch(batch)
    image, label = batch["image"], batch["label"]
    n = int(image.shape[0])

    # 1) 形状契约：所有样本补边到同一个 target_hw，因此同 batch 内必然逐像素一致
    shapes = {tuple(int(s) for s in image[i].shape) for i in range(n)}
    if len(shapes) != 1:
        problems.append(f"batch {idx}：同 batch 内 image 尺寸不一致 {sorted(shapes)}"
                        f"（补边失效：所有样本都应补到 data.target_hw）")
    # image 是 (B,C,H,W)、label 是 (B,H,W)：比较空间维时要错开一个通道维
    if tuple(image.shape[2:]) != tuple(label.shape[1:]) or int(image.shape[0]) != int(label.shape[0]):
        problems.append(f"batch {idx}：image {tuple(image.shape)} 与 label {tuple(label.shape)} 空间维不匹配"
                        f"（期望 image (B,C,H,W) / label (B,H,W)）")
    expect_channels = int(tolerance["in_channels"])
    if int(image.shape[1]) != expect_channels:
        problems.append(f"batch {idx}：image 通道维是 {int(image.shape[1])}，期望 {expect_channels}"
                        f"（= 2×data.z_context+1，由 {tolerance['z_context']} 推出）")
    if label.ndim != 3:
        problems.append(f"batch {idx}：label 维度是 {label.ndim}，期望 3（(B,H,W)，**不带通道维**）")
    expect_hw = tuple(int(v) for v in tolerance["target_hw"])
    if tuple(image.shape[2:]) != expect_hw:
        problems.append(f"batch {idx}：image 空间维 {tuple(image.shape[2:])} != data.target_hw "
                        f"{format_hw(expect_hw)}（补边口径不对）")
    bad_orig = [hw for hw in info["orig_hw"] if hw[0] > expect_hw[0] or hw[1] > expect_hw[1]]
    if bad_orig:
        problems.append(f"batch {idx}：orig_hw {bad_orig} 超过 target_hw {format_hw(expect_hw)}")
    # 补边偏移必须与 orig_hw、张量形状自洽（第 4 轮裁回原始尺寸就用它）
    for orig, offset in zip(info["orig_hw"], info["pad_offset"]):
        if tuple(offset) != pad_offset_of(orig, expect_hw):
            problems.append(f"batch {idx}：orig_hw {orig} 的 pad_offset {offset} 与推导值 "
                            f"{pad_offset_of(orig, expect_hw)} 不一致")
            break

    # 1b) 2.5D 窗口的结构检查（只查结构，三层内容是否同属一个病人在 check_slice_window 里单独查）
    if expect_channels > 1:
        for sample_z, window in zip(info["z"], info["window"]):
            if len(window) != expect_channels:
                problems.append(f"batch {idx}：样本 z={sample_z} 的 2.5D 窗口 {window} 长度不是 "
                                f"{expect_channels}（应等于 2×z_context+1）")
                break
            if window[int(expect_channels) // 2] != int(sample_z):
                problems.append(f"batch {idx}：样本 z={sample_z} 的窗口中心是 "
                                f"{window[int(expect_channels) // 2]}，与 z 不一致（中心通道必须是被"
                                f"监督的那一层）")
                break
            if any(b - a != 1 for a, b in zip(window, window[1:])):
                problems.append(f"batch {idx}：样本 z={sample_z} 的窗口 {window} 层号不连续"
                                f"（越界应当用**端点复制**，不是跳过该层）")
                break

    # 2) 值域与标签取值
    if info["image_min"] < -1e-6 or info["image_max"] > 1.0 + 1e-6:
        problems.append(f"batch {idx}：image 值域 [{info['image_min']}, {info['image_max']}] 超出 [0,1]")
    bad = [v for v in info["label_values"] if v not in (0, 1)]
    if bad:
        problems.append(f"batch {idx}：label 出现了 {bad}，不是 {{0,1}}")
    if info["image_dtype"] != "torch.float32":
        problems.append(f"batch {idx}：image dtype={info['image_dtype']}，期望 float32")
    if info["label_dtype"] not in ("torch.int64", "torch.int32"):
        problems.append(f"batch {idx}：label dtype={info['label_dtype']}，期望整型")

    # 3) 阳性数：**仅训练侧**，且平衡采样下应当是定值（val 侧不判，见 docstring）
    if sampler is not None and str(tag).startswith("train"):
        targets = sampler.batch_targets()
        want = int(targets["n_pos_per_batch"])
        got = int(info["pos_slices_in_batch"])
        if int(targets["n_pos_slices"]) <= 0:
            LOGGER.warning("        注：本折训练集里没有含肿瘤切片，无法判每批阳性数")
        elif got != want:
            problems.append(f"batch {idx}：阳性切片 {got} 与平衡采样计划的 {want} 不符"
                            f"（数据集阳性 {targets['n_pos_slices']} 层 / 每批 {want} 正 + "
                            f"{targets['n_neg_per_batch']} 阴）")

    LOGGER.info("batch %d：image %s / label %s（2.5D %d 通道 = 层 %s 为中心）；病例 %s；z=%s",
                idx, tuple(info["image_shape"]), tuple(info["label_shape"]),
                info["image_channels"] if info["window"] else 1,
                info["window"][0] if info["window"] else "单层",
                info["cases"], info["z"])
    LOGGER.info("        中心通道值域 [%.4f, %.4f]；label 取值 %s；**含肿瘤切片 %d/%d = %.3f**"
                "（平衡采样目标 %.2f）；补边样本 %d/%d（补边像素均值 %.1f%%）",
                info["image_min"], info["image_max"], info["label_values"],
                info["pos_slices_in_batch"], info["batch_size"], info["pos_ratio_measured"],
                tolerance["target"], info["n_padded_samples"], info["batch_size"],
                100.0 * info["pad_fraction_mean"])
    LOGGER.info("        orig_hw %s；补边偏移 %s；三层窗 %s",
                info["orig_hw"][:3] + (["..."] if len(info["orig_hw"]) > 3 else []),
                info["pad_offset"][:3] + (["..."] if len(info["pad_offset"]) > 3 else []),
                info["window"][:3] + (["..."] if len(info["window"]) > 3 else []))
    return info


def run_batches(loader, n_batches: int, tolerance: dict, problems: list, tag: str,
                max_batches: int = 0, sampler=None) -> list:
    """从 loader 依次取 ``n_batches`` 个 batch，逐个自检并计时。

    ``sampler`` 只在训练侧传：传了才判「阳性数落在采样器计划区间内」，验证侧不判（见 ``check_batch``）。
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
        info = check_batch(i, batch, problems, tolerance, tag=tag, sampler=sampler)
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
    with open_volume(ds.label_path(case)) as lab_img:
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

    **抽样是刻意的**：训练集里含肿瘤的切片只占 13.8%，随机抽 8 个样本通常有 ~7 个 label 全空，
    而全空 label 无论怎么翻转/旋转都还是全空——于是「label 被改动 0 个」这条判据会被噪声淹没
    （远程第一次跑就出现了这个假警报）。所以这里分两组取样：
      * ``--aug-samples`` 个**含肿瘤**的切片（``pos_flags`` 为真）→ 几何增强必须改到 label；
      * 同样数量的全背景切片 → label 应当保持全空（顺带验证增强不会凭空造出前景）。
    """
    transforms = build_transforms(cfg, train=True, seed=int(((cfg or {}).get("train") or {}).get("seed", 42)))
    if transforms is None:
        problems.append("build_transforms(cfg, train=True) 返回了 None，训练增强没生效")
        return {}

    rng = np.random.default_rng(int(((cfg or {}).get("train") or {}).get("seed", 42)))
    n = len(ds)
    pos_idx = np.flatnonzero(np.asarray(ds.pos_flags))
    neg_idx = np.flatnonzero(~np.asarray(ds.pos_flags))
    take = min(int(n_samples), max(1, len(pos_idx)))
    if len(pos_idx):
        picked_pos = [int(i) for i in rng.choice(pos_idx, size=take, replace=False)]
    else:
        picked_pos = [int(i) for i in rng.choice(n, size=min(int(n_samples), n), replace=False)]
    picked_neg = ([int(i) for i in rng.choice(neg_idx, size=min(int(n_samples), len(neg_idx)), replace=False)]
                  if len(neg_idx) else [])
    indices = picked_pos + picked_neg
    n_pos_group = len(picked_pos)
    pos_group = set(picked_pos)

    changed = 0
    label_changed = 0
    label_changed_pos = 0        # 含肿瘤切片这一组里 label 被改动的个数
    empty_label_kept = 0         # 全背景切片里 label 仍为全空的个数
    shapes_ok = True
    ranges_ok = True
    labels_ok = True

    for i in indices:
        sample = ds[i]
        # 增强只吃**中心层的 2D 平面**（与 __getitem__ 里的调用完全一致）；
        # 2.5D 的三个通道由 dataset 在增强**之后**叠出来，所以这里看的是中心通道。
        raw_image = sample["image"].array[int(sample["image"].array.shape[0]) // 2]
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
            if i in pos_group:
                label_changed_pos += 1
        elif i not in pos_group:
            empty_label_kept += 1

    info = {
        "n_samples": len(indices),
        "n_pos_group": n_pos_group,
        "image_changed": changed,
        "image_changed_ratio": round(changed / max(1, len(indices)), 4),
        "label_changed": label_changed,
        "label_changed_ratio": round(label_changed / max(1, len(indices)), 4),
        "label_changed_in_pos_group": label_changed_pos,
        "pos_group_label_changed_ratio": round(label_changed_pos / max(1, n_pos_group), 4),
        "empty_label_kept_empty": empty_label_kept,
        "shapes_preserved": shapes_ok,
        "value_range_ok": ranges_ok,
        "label_binary_ok": labels_ok,
    }
    LOGGER.info("增强自检（%d 个样本 = %d 含肿瘤 + %d 全背景）：image 被改动 %d 个（%.2f）；"
                "含肿瘤组的 label 被改动 %d/%d（%.2f）——几何增强生效与否看这一项；"
                "全背景组 label 仍全空 %d/%d；形状/值域/标签二值 = %s/%s/%s",
                info["n_samples"], n_pos_group, len(indices) - n_pos_group,
                changed, info["image_changed_ratio"],
                label_changed_pos, n_pos_group, info["pos_group_label_changed_ratio"],
                empty_label_kept, len(indices) - n_pos_group,
                shapes_ok, ranges_ok, labels_ok)
    if changed == 0:
        problems.append("增强自检：所有样本的 image 都没被改动（增强概率可能全为 0 或流水线没接上）")
    if n_pos_group and label_changed_pos == 0:
        problems.append(f"增强自检：{n_pos_group} 个含肿瘤切片的 label 一个都没被改动"
                        f"（几何增强没生效）")
    if empty_label_kept != len(indices) - n_pos_group:
        problems.append("增强自检：全背景切片的 label 被增强改成了非空（不应发生）")
    return info


def check_affine_geometry(problems: list) -> dict:
    """用一个人造图案验证 ``GridAffine2D`` 的几何口径（纯 CPU、毫秒级）。

    图案：正中心一个方块 + **右上角**一个标记，标记质心为 ``(row=9.5, col=53.5)``。

    期望值一律按 ``GridAffine2D`` 的坐标约定实测/推导（theta 作用在**像素**坐标、最后统一归一化）：

      * **旋转 +90°**（内容逆时针）：右上角标记转到**左上角**，列 53.5 → 63-53.5 = 9.5、行保持 9.5；
      * **shift_xy=(+8, 0) 像素**：这是**内容**位移（正 = 向右），标记 x 53.5 → 61.5；
      * **scale=0.5**：采样点只铺一半范围 → 视野放大 2 倍 → 内容放大，前景面积约为 **4 倍**。

    历史教训：这里的期望值我曾两次写反（`shift_xy` 还在用「比例」口径时就按像素写、以及把
    scale 的面积比方向写反），两次都是远程自检把它抓出来的——所以现在期望值都按实测口径钉死，
    并把具体数值打进日志便于对照；断言只判方向与数量级（±2 像素 / 区间），不追求逐位相等。
    """
    size = 64
    canvas = np.zeros((size, size), dtype=np.float32)
    canvas[size // 2 - 8:size // 2 + 8, size // 2 - 8:size // 2 + 8] = 1.0          # 中心方块
    canvas[6:14, size - 14:size - 6] = 2.0                                          # 右上角标记（值 2 便于追踪）
    ys0, xs0 = np.nonzero(canvas > 1.5)
    marker_row, marker_col = float(ys0.mean()), float(xs0.mean())                   # (9.5, 53.5)
    checks: dict = {"marker_center_xy": [round(marker_col, 2), round(marker_row, 2)]}

    rotated = GridAffine2D(rotate_deg=90.0).warp(canvas)
    ys, xs = np.nonzero(rotated > 1.5)
    checks["rotate90_marker_center_xy"] = [round(float(xs.mean()), 2), round(float(ys.mean()), 2)]
    checks["rotate90_expected_xy"] = [round(size - 1 - marker_col, 2), round(marker_row, 2)]
    ok_rot = abs(checks["rotate90_marker_center_xy"][0] - checks["rotate90_expected_xy"][0]) <= 2 \
        and abs(checks["rotate90_marker_center_xy"][1] - checks["rotate90_expected_xy"][1]) <= 2
    checks["rotate90_ok"] = bool(ok_rot)
    if not ok_rot:
        problems.append(f"GridAffine2D 旋转 90° 的几何不对：标记落在 "
                        f"{checks['rotate90_marker_center_xy']}，期望 {checks['rotate90_expected_xy']}"
                        f"（右上角标记应转到左上角）")

    shift_px = 8.0     # 单位是像素（不是比例）
    shifted = GridAffine2D(rotate_deg=0.0, shift_xy=(shift_px, 0.0)).warp(canvas)
    ys2, xs2 = np.nonzero(shifted > 1.5)
    checks["shift_marker_center_x"] = round(float(xs2.mean()), 2)
    checks["shift_expected_x"] = round(marker_col + shift_px, 2)
    ok_shift = abs(checks["shift_marker_center_x"] - checks["shift_expected_x"]) <= 2
    checks["shift_ok"] = bool(ok_shift)
    if not ok_shift:
        problems.append(f"GridAffine2D 平移方向/幅度不对：标记 x={checks['shift_marker_center_x']}，"
                        f"期望 {checks['shift_expected_x']}（shift_xy 单位是像素、正 = 内容向右）")

    small = GridAffine2D(rotate_deg=0.0, scale=0.5).warp(canvas)
    area_raw = float((canvas > 0.5).sum())
    area_small = float((small > 0.5).sum())
    checks["scale_area_ratio"] = round(area_small / max(1.0, area_raw), 4)
    # scale<1 → 采样范围变小 → 视野放大 → 内容放大（面积比 ≈ 1/scale² = 4，即 > 1）
    checks["scale_ok"] = bool(area_small > area_raw)
    if not checks["scale_ok"]:
        problems.append(f"GridAffine2D 缩放 0.5 后前景没变大：{area_small} vs {area_raw}"
                        f"（scale<1 应放大内容，面积比 ≈ 1/scale²）")
    LOGGER.info("仿射几何自检：标记原点 %s；旋转 90° %s（实测 %s，期望 %s）；平移 %s（实测 x=%s，"
                "期望 %s）；缩放 0.5 面积比 %.3f（应 > 1）",
                checks["marker_center_xy"],
                "通过" if checks["rotate90_ok"] else "失败",
                checks["rotate90_marker_center_xy"], checks["rotate90_expected_xy"],
                "通过" if checks["shift_ok"] else "失败",
                checks["shift_marker_center_x"], checks["shift_expected_x"],
                checks["scale_area_ratio"])
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
    """平衡采样器自检：batch 数/大小、每批阳性数、阳性重复上限、阴性覆盖率、可复现性、病人级隔离。

    可复现性的正确口径（这里踩过坑）：
      * **同一个 epoch 用同一个 seed 必须完全一致** → 用两个**新建的**采样器各跑一次比较；
      * **相邻 epoch 必须不同**（否则每轮都固定同一顺序，过采样退化）→ 同一采样器连续两轮比较。

    早期版本直接把「同一个采样器连续迭代两次」当成可复现性检查，但那两次是 epoch 1 与 epoch 2，
    种子本来就不同（``_seed_for_epoch(1) != _seed_for_epoch(2)``），于是把一个**正常**行为报成了
    「采样器不可复现」。

    第 4 轮换口径后的断言（与 ``plan_balanced_slots`` 的计划值逐条比对）：
      * 每个 batch 的大小恒为 ``batch_size``，形状集合只有一个（=补边后的固定尺寸）；
      * 每个 batch 的阳性数**恰好** ``n_pos_per_batch``（不再有全阴性 batch，也不再是"上限"）；
      * 阳性层的重复次数 **<= data.max_pos_repeat**，且各层尽量均匀；
      * 一轮抽到的样本索引**全部落在本折 train 病例内**（病人级隔离；
        采样器只做重排与重复，不跨病人、不引入 val 病例）。
    """
    train_cfg = (cfg or {}).get("train") or {}
    kwargs = dict(batch_size=int(train_cfg.get("batch_size", 8)),
                  data_cfg=data_config(cfg),
                  seed=int(train_cfg.get("seed", 42)))
    sampler = BalancedBatchSampler(ds, **kwargs)
    plan = sampler.batch_targets()
    LOGGER.info("%s", sampler.describe())

    # 1) 可复现性：两个新建采样器的 epoch 1 必须逐 batch 相同
    run_a = [list(b) for b in BalancedBatchSampler(ds, **kwargs)]
    run_b = [list(b) for b in BalancedBatchSampler(ds, **kwargs)]
    reproducible = bool(run_a == run_b)
    # 2) 相邻 epoch 应当不同（同一 sampler 连续两轮）
    run_e1 = [list(b) for b in sampler]
    run_e2 = [list(b) for b in sampler]
    epochs_differ = bool(run_e1 != run_e2)

    n_batches = len(run_a)
    drawn = [i for b in run_a for i in b]
    pos_indices = set(sampler.pos_indices)
    neg_indices = set(sampler.neg_indices)
    drawn_pos = [i for i in drawn if i in pos_indices]
    drawn_neg = [i for i in drawn if i in neg_indices]
    pos_rep_min, pos_rep_max = slice_usage(drawn_pos)
    _, neg_rep_max = slice_usage(drawn_neg)
    size_bad = sorted({len(b) for b in run_a} - {sampler.batch_size})
    per_batch_pos = [sum(1 for i in b if i in pos_indices) for b in run_a]
    pos_counts = sorted(set(per_batch_pos))
    # 3) 病人级隔离：样本索引覆盖的病例必须全部来自本折 train
    cases_seen = {int(ds.index[i][0]) for i in drawn}
    cases_outside = sorted(cases_seen - {int(c) for c in ds.case_ids})
    # 4) 阳性层是否「尽量均摊」：每层出现次数应落在 {floor(repeat), ceil(repeat)} 内
    n_pos_slices = max(1, sampler.n_positive)
    repeat_lo = int(plan["pos_slots"] // n_pos_slices)
    repeat_hi = int(-(-plan["pos_slots"] // n_pos_slices))
    spread_ok = bool(pos_rep_min >= max(1, repeat_lo) and pos_rep_max <= max(repeat_lo, repeat_hi))

    info = {
        "n_batches_per_epoch": n_batches,
        "expected_batches": int(plan["batches"]),
        "slots": int(plan["slots"]),
        "batch_size": int(sampler.batch_size),
        "n_pos_per_batch_plan": int(plan["n_pos_per_batch"]),
        "n_neg_per_batch_plan": int(plan["n_neg_per_batch"]),
        "pos_per_batch_observed": pos_counts,
        "actual_pos_ratio": round(float(plan["actual_pos_ratio"]), 6),
        "pos_ratio_target": float(plan["pos_ratio_target"]),
        "pos_repeat_plan": round(float(plan["pos_repeat"]), 4),
        "pos_repeat_observed_min": int(pos_rep_min),
        "pos_repeat_observed_max": int(pos_rep_max),
        "max_pos_repeat": int(sampler.max_pos_repeat),
        "neg_coverage_plan": round(float(plan["neg_coverage"]), 4),
        "neg_repeat_observed_max": int(neg_rep_max),
        "pool_limited": bool(plan["pool_limited"]),
        "reproducible_same_epoch": reproducible,
        "adjacent_epochs_differ": epochs_differ,
        "epoch1_seed": sampler.epoch_seed(1),
        "epoch2_seed": sampler.epoch_seed(2),
        "n_pos": int(sampler.n_positive),
        "n_neg": int(sampler.n_negative),
        "pos_drawn_total": len(drawn_pos),
        "pos_drawn_unique": len(set(drawn_pos)),
        "neg_drawn_total": len(drawn_neg),
        "neg_drawn_unique": len(set(drawn_neg)),
        "slices_drawn_total": len(drawn),
        "slices_total": len(ds),
        "batch_sizes_out_of_spec": size_bad,
        "all_negative_batches": int(sum(1 for c in per_batch_pos if c == 0)),
        "cases_seen": sorted(cases_seen),
        "cases_outside_this_split": cases_outside,
        "pos_repeat_spread_ok": spread_ok,
    }
    if not reproducible:
        detail = f"两个采样器的 batch 数不同：{len(run_a)} vs {len(run_b)}"
        if len(run_a) == len(run_b):
            for k, (x, y) in enumerate(zip(run_a, run_b), 1):
                if x != y:
                    detail = f"首个不同的 batch 是第 {k} 个：run1={x} run2={y}"
                    break
        problems.append(f"同一 epoch 两次采样结果不同（采样器不可复现）：{detail}")
        LOGGER.error("采样器不可复现的细节：%s", detail)
    if not epochs_differ:
        problems.append("相邻两个 epoch 的采样顺序完全相同：过采样会退化成每轮固定同一批样本"
                        f"（epoch1_seed={sampler.epoch_seed(1)} epoch2_seed={sampler.epoch_seed(2)}）")
    if size_bad:
        problems.append(f"有 batch 的大小不等于 batch_size={sampler.batch_size}：出现 {size_bad}"
                        f"（池子够大时不该发生）")
    if cases_outside:
        problems.append(f"采样器抽到了**不属于本折 train 的病例** {cases_outside}："
                        f"病人级隔离被破坏（样本只能来自 {ds.case_ids}）")
    if sampler.n_positive > 0 and not plan["pool_limited"]:
        want = int(plan["n_pos_per_batch"])
        wrong = sorted(c for c in pos_counts if c != want)
        if wrong:
            problems.append(f"每批阳性数出现了 {wrong}，与平衡采样计划 {want} 不符"
                            f"（应为定值，见 dataset.plan_balanced_slots）")
    if pos_rep_max > int(sampler.max_pos_repeat):
        problems.append(f"阳性层重复次数 {pos_rep_max} 超过 data.max_pos_repeat="
                        f"{sampler.max_pos_repeat}（同一层被喂太多次，容易过拟合到少数层）")
    if not spread_ok and sampler.n_positive > 0:
        problems.append(f"阳性层重复不均匀：出现次数落在 [{pos_rep_min}, {pos_rep_max}]，"
                        f"计划把 {plan['pos_slots']} 个槽位分给 {sampler.n_positive} 个阳性层"
                        f"（应落在 [{repeat_lo}, {repeat_hi}]）")
    if sampler.n_negative and float(plan["neg_coverage"]) < 1.0:
        LOGGER.info("采样器说明：阴性层一轮只覆盖 %.1f%%（%d/%d 个）——平衡采样下这是**预期**的，"
                    "阴性层不再要求一轮全过一遍，改用轮转池重复抽取。",
                    100.0 * float(plan["neg_coverage"]), int(plan["neg_slots"]), sampler.n_negative)
    LOGGER.info("采样器自检：每轮 %d 个 batch（计划 %d）× %d = %d 个槽位；每批阳性数 %s"
                "（计划 %d 正 + %d 阴）；同 epoch 可复现=%s；相邻 epoch 不同=%s；"
                "batch 大小越界=%d 种；阳性层出现 %d 次/去重 %d 个（重复 %d~%d 次，上限 %d）；"
                "阴性层出现 %d 次/去重 %d 个（覆盖率 %.1f%%）；全阴性 batch=%d；抽到的病例 %s"
                "（越界 %d 个）",
                n_batches, plan["batches"], sampler.batch_size, plan["slots"], pos_counts,
                plan["n_pos_per_batch"], plan["n_neg_per_batch"], reproducible, epochs_differ,
                len(size_bad), len(drawn_pos), len(set(drawn_pos)), pos_rep_min, pos_rep_max,
                sampler.max_pos_repeat, len(drawn_neg), len(set(drawn_neg)),
                100.0 * (len(set(drawn_neg)) / max(1, sampler.n_negative)),
                info["all_negative_batches"], sorted(cases_seen), len(cases_outside))
    return info


def check_slice_window(raw_ds: CTSliceDataset, enhanced_ds: CTSliceDataset, problems: list,
                       n_cases: int = 3) -> dict:
    """2.5D 三层窗自检：**三层同属一个病人 + 中心通道等于原单层读取 + 端点复制 + 上下文未被增强**。

    三个数据源的分工（写错就会误报，这里踩过一次）：
      * ``raw_ds``（``augment=False``）：查**内容**——窗口层号、中心通道与「单独读那一层」是否逐像素
        相同、端点复制是否成立、上下文通道是否仍是**未经增强的真实邻居层**。
        绝不能用增强过的数据集查内容：增强是随机的，中心通道必然与原始层不同。
      * ``enhanced_ds``（``augment=True``，同一批病例、index 逐元素对应）：只用来确认「增强确实发生了」，
        即中心通道与 raw 不同（下面的 ``center_augmented`` 计数）。

    硬断言（任何一条坏了，2.5D 学到的都是假上下文）：
      1. 窗口层号全部落在本病例 ``[0, nz-1]`` 内，且与 ``window_index`` 推出的完全一致；
      2. ``z=0`` 的窗口是 ``[0,0,1]``（第 0、1 通道像素相同）、``z=nz-1`` 是 ``[nz-2,nz-1,nz-1]``
         —— 越界必须**端点复制**，不能去读邻居病人的层；
      3. 中心通道逐像素等于「``(case,z)`` 单独读一层再补边」的结果（同时钉住轴序与归一化口径）；
      4. 上下文通道逐像素等于「同一个病人的 ``z±1`` 层单独读出来的样子」
         —— 强度增强（gamma/噪声）只该动中心层，动了上下文就是伪影。
    """
    z_context = int(raw_ds.z_context)
    n_cases = max(1, int(n_cases))
    info: dict = {"z_context": z_context, "in_channels": int(raw_ds.in_channels), "cases": []}
    if z_context <= 0:
        LOGGER.info("2.5D 窗口自检：data.z_context=0（单层 2D），跳过（窗口就是 [z] 本身）")
        info["skipped"] = True
        return info

    center = int(raw_ds.in_channels) // 2
    checked = 0
    mismatch = 0
    context_mismatch = 0
    clamped_ok = 0
    augmented_cases = 0
    for case in list(raw_ds.case_ids)[:n_cases]:
        case = int(case)
        nz = int(raw_ds.case_n_slices[case])
        start = raw_ds.index.index((case, 0))          # 该病例第 0 层在 index 里的位置
        record = {"case": case, "nz": nz, "positions": {}, "clamped_match": True,
                  "center_equal": True, "context_clean": True}
        for name, z in (("z0", 0), ("zn", nz - 1), ("mid", nz // 2)):
            i = start + int(z)
            sample = raw_ds[i]
            got = [int(v) for v in sample["window"]]
            want = [int(v) for v in window_index(int(z), nz, z_context)]
            if got != want:
                problems.append(f"case {case} z={z}：三层窗 {got} 与 window_index 推出的 {want} 不一致")
            if not all(0 <= v < nz for v in got):
                problems.append(f"case {case} z={z}：三层窗 {got} 越过本病例范围 [0,{nz - 1}]"
                                f"——**跨病人取层**（端点复制失效）")
            image = np.asarray(sample["image"].numpy(), dtype=np.float32)
            # 3) 中心通道 vs「独立重读该层」（不复用窗口逻辑，避免用自己的错误证明自己）
            raw_center = raw_ds.window_planes(case, int(z))
            if not np.array_equal(image[center], raw_center):
                mismatch += 1
                record["center_equal"] = False
                problems.append(f"case {case} z={z}：中心通道与「单独读第 {z} 层并补边」的结果不一致"
                                f"（轴序 / 归一化 / 中心位置有误）")
            # 4) 上下文通道 vs「独立重读 z±1」（未增强的真实邻居层）
            for offset, zz in enumerate(got):
                if offset == center:
                    continue
                raw_ctx = raw_ds.window_planes(case, int(zz))
                if not np.array_equal(image[offset], raw_ctx):
                    context_mismatch += 1
                    record["context_clean"] = False
                    problems.append(f"case {case} z={z}：窗口第 {offset} 个通道（层 {zz}）与单独读该层"
                                    f"的结果不一致——上下文通道不该被强度增强改动")
            # 2) 端点复制：窗口里与中心层同为 z 的通道必须逐像素相同
            for offset, zz in enumerate(got):
                if zz == got[center] and offset != center:
                    if not np.array_equal(image[offset], image[center]):
                        record["clamped_match"] = False
                        problems.append(f"case {case} z={z}：窗口 {got} 的第 {offset} 个通道与中心层"
                                        f"同为 z={zz} 但像素不同（端点复制口径坏了）")
            record["positions"][name] = {"z": int(z), "window": got, "want": want}
            checked += 1
        # 增强数据集：只确认「中心通道确实被增强改动过」（否则增强没接上）
        a = np.asarray(enhanced_ds[start]["image"].numpy(), dtype=np.float32)
        r = np.asarray(raw_ds[start]["image"].numpy(), dtype=np.float32)
        record["center_augmented"] = bool(not np.array_equal(a[center], r[center]))
        if record["center_augmented"]:
            augmented_cases += 1
        info["cases"].append(record)
        if record["clamped_match"] and record["center_equal"] and record["context_clean"]:
            clamped_ok += 1

    info.update({"checked_positions": checked, "center_mismatch": mismatch,
                 "context_mismatch": context_mismatch, "cases_all_ok": clamped_ok,
                 "center_augmented_cases": augmented_cases,
                 "n_cases_checked": len(info["cases"])})
    LOGGER.info("2.5D 窗口自检（z_context=%d → %d 通道）：抽 %d 例查首/中/末层 —— 首层窗口 %s、"
                "末层窗口 %s；中心通道与单层读取不一致 %d 处；上下文通道与邻居层不一致 %d 处；"
                "端点复制+中心一致+上下文干净 %d/%d 例；增强确实改了中心层 %d/%d 例",
                z_context, int(raw_ds.in_channels), len(info["cases"]),
                info["cases"][0]["positions"]["z0"]["window"] if info["cases"] else None,
                info["cases"][0]["positions"]["zn"]["window"] if info["cases"] else None,
                mismatch, context_mismatch, clamped_ok, len(info["cases"]),
                augmented_cases, len(info["cases"]))
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
    LOGGER.info("第 4 轮数据自检：Dataset（2.5D %d 通道 + 统一补边到 %s）+ 平衡采样器 + 增强（fold=%d）",
                window_channels(resolve_z_context(data_cfg)), format_hw(data_cfg["target_hw"]),
                int(args.fold))
    LOGGER.info("=" * 78)

    env = check_environment()
    splits, manifest, prereq_problems = check_prerequisites(cfg)
    problems.extend(prereq_problems)

    # ---- 数据集 ----
    LOGGER.info("-" * 78)
    LOGGER.info("1) 构建数据集（train 侧带增强、val 侧无增强；两侧都补边到 %s）",
                format_hw(data_cfg["target_hw"]))
    train_ds = CTSliceDataset("train", cache_dir, "train", cfg, augment=True,
                              debug=True, fold=int(args.fold))
    val_ds = CTSliceDataset("val", cache_dir, "val", cfg, augment=False, fold=int(args.fold))
    LOGGER.info("\n%s", train_ds.describe())
    LOGGER.info("\n%s", val_ds.describe())
    check_inplane_consistency(train_ds, manifest, problems)

    tolerance = {
        # 平衡采样的每批阳性比例目标（旧的 train.pos_ratio_target 已不生效）
        "target": float(data_cfg.get("pos_ratio_train", 0.5)),
        "limit": float(data_cfg.get("pos_ratio_tolerance", 0.20)),
        "target_hw": tuple(int(v) for v in train_ds.target_hw),
        "z_context": int(train_ds.z_context),
        "in_channels": int(train_ds.in_channels),
    }
    LOGGER.info("整体阳性率：train 侧 %.4f（%d/%d），val 侧 %.4f（%d/%d）；"
                "**训练侧平衡采样目标 %.2f**（≈ %.1f 倍过采样），验证侧保持原始分布不做平衡",
                train_ds.pos_ratio, train_ds.n_positive, len(train_ds),
                val_ds.pos_ratio, val_ds.n_positive, len(val_ds),
                tolerance["target"], tolerance["target"] / max(1e-9, train_ds.pos_ratio))

    # ---- 3 个 batch 实测 ----
    LOGGER.info("-" * 78)
    LOGGER.info("2) 实测 batch（第 1 个 batch 含冷读，耗时单独看）")
    # 阳性数的判据是**平衡采样器自己的计划值**（定值），不是任何"上限"
    train_sampler = make_sampler(train_ds, cfg)
    targets = train_sampler.batch_targets()
    LOGGER.info("训练侧每批阳性数（计划值，定值）：%d 正 + %d 阴 = %d；一轮 %d 个 batch / %d 个槽位；"
                "阳性层重复 %.2f 次（上限 %d）、阴性层覆盖率 %.1f%%",
                targets["n_pos_per_batch"], targets["n_neg_per_batch"], targets["batch_size"],
                targets["batches"], targets["slots"], targets["pos_repeat"],
                train_sampler.max_pos_repeat, 100.0 * targets["neg_coverage"])
    train_loader = make_train_loader(splits, int(args.fold), cfg)
    batch_infos = run_batches(train_loader, int(args.batches), tolerance, problems, tag="train",
                              sampler=train_sampler)

    # 验证侧顺序读、不做过采样：**不判阳性数**（某个 batch 落在病例前几层无肿瘤是正常的）
    val_loader = make_val_loader(splits, int(args.fold), cfg)
    val_infos = run_batches(val_loader, min(2, int(args.batches)), tolerance, problems, tag="val")
    if val_infos:
        LOGGER.info("提示：val 侧顺序读取，batch 内阳性比例天然接近整体比例（%.4f），"
                    "不做过采样也不判阳性数——出现全阴性 batch 是正常的。", val_ds.pos_ratio)

    if batch_infos:
        sizes = {tuple(info["image_shape"][2:]) for info in batch_infos}
        LOGGER.info("%d 个 train batch 的空间维只有 %d 种：%s（统一补边后应恒为 1 种）",
                    len(batch_infos), len(sizes), sorted(sizes))
    if env.get("cuda_available"):
        import torch

        LOGGER.info("CUDA 显存：allocated %.1f MB / reserved %.1f MB（只做数据加载，未建模型）",
                    torch.cuda.memory_allocated() / 1024 ** 2,
                    torch.cuda.memory_reserved() / 1024 ** 2)

    # ---- 切片轴 ----
    LOGGER.info("-" * 78)
    LOGGER.info("3) 切片轴校验（逐层肿瘤体素数曲线）")
    # 用「全部 25 例」建一个数据集：这样 --probe-case 可以指定任意病例（它不一定在当前折里，
    # 例如 case 56 在 fold 2 的 val 里）。曲线只看单例逐层体素数，与折划分无关。
    probe_ds = CTSliceDataset("all", cache_dir, "all", cfg, augment=False)
    probe_case = pick_probe_case(probe_ds, args.probe_case)
    LOGGER.info("（曲线用病例 %d；它在 fold %d 的 %s 侧）", probe_case,
                int(args.fold),
                "train" if probe_case in train_ds.case_ids else
                ("val" if probe_case in val_ds.case_ids else "其它折"))
    axis_info = check_slice_axis(probe_ds, probe_case, manifest, problems)

    # ---- 增强 ----
    LOGGER.info("-" * 78)
    LOGGER.info("4) 增强自检 + 2.5D 三层窗自检（都用一个无增强的数据集做对照）")
    raw_ds = CTSliceDataset("train", cache_dir, "train", cfg, augment=False, fold=int(args.fold))
    augment_info = check_augment(cfg, raw_ds, int(args.aug_samples), problems)
    pipeline_info = check_augment_pipeline(cfg, problems)
    affine_info = check_affine_geometry(problems)
    custom_info = check_custom_transforms(problems)
    window_info = check_slice_window(raw_ds, train_ds, problems)

    # ---- 采样器 ----
    sampler_info = {}
    if not args.skip_sampler:
        LOGGER.info("-" * 78)
        LOGGER.info("5) 平衡采样器自检（每批阳性数 / 重复上限 / 可复现性 / 病人级隔离）")
        sampler_info = check_sampler(train_ds, cfg, problems)

    # ---- 收尾：评估是否可进入第 3 轮 ----
    del train_loader, val_loader, probe_ds
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
            "target_hw": list(train_ds.target_hw),
            "pad_align": train_ds.pad_align,
            "z_context": int(train_ds.z_context),
            "in_channels": int(train_ds.in_channels),
            "inplane": {format_hw(k): v for k, v in sorted(train_ds.inplane_stats().items())},
            "per_case": {str(c): {"nz": train_ds.case_n_slices[c],
                                  "pos_slices": train_ds.case_pos_slices[c],
                                  "orig_hw": list(train_ds.case_hw[c]),
                                  "pad_offset": list(train_ds.pad_offset_of_case(c))}
                         for c in train_ds.case_ids},
        },
        "val_dataset": {
            "n_cases": len(val_ds.case_ids),
            "cases": val_ds.case_ids,
            "n_slices": len(val_ds),
            "n_positive": val_ds.n_positive,
            "pos_ratio": round(val_ds.pos_ratio, 6),
            "z_context": int(val_ds.z_context),
            "in_channels": int(val_ds.in_channels),
        },
        "batches_train": batch_infos,
        "batches_val": val_infos,
        "slice_axis": axis_info,
        "slice_window": window_info,
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
        LOGGER.error("请把上面带 - 的行贴回本地，修好后再进入训练。")
        return 1

    LOGGER.info("自检通过：%d 例 / %d 层切片的形态、值域、标签、2.5D 三层窗、补边、增强与"
                "平衡采样比例均符合约定。", len(train_ds.case_ids), len(train_ds))
    LOGGER.info("报告：%s（远程产物，不入库，贴回终端输出即可）", rel_to_root(report_path))
    LOGGER.info("下一步：python -m src.train --fold %d --debug（冒烟自检，不落盘）", int(args.fold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
