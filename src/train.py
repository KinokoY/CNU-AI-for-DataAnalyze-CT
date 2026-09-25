"""训练入口：单折训练 + 每轮**整卷验证** + 早停 + checkpoint / metrics.csv / TensorBoard。

整体流程（``python -m src.train --fold k``）：
    1. **前置校验**：读 ``data/splits.json`` 与 ``cache/cache_manifest.json``，核对
       病例集合、预处理指纹、cache 文件是否齐、模型与损失的搭配；任何一项不对就直接退出（码 2），
       不浪费一次 GPU 启动；
    2. 建 ``CTSliceDataset``（训练侧过采样+增强）→ ``UNet2D`` → ``DiceCELoss`` → AdamW +
       CosineAnnealingLR；bf16 autocast（**bf16 不需要 GradScaler**，见 ``src/utils.make_grad_scaler``）；
    3. 每个 epoch 训练完，按 ``train.val_every`` 用 ``src.infer.predict_volume`` 对验证集的
       每一例做**整卷推理**，逐例算整卷肿瘤 Dice 再取平均（macro 口径，用户拍板）——
       不做「逐层平均」，否则大量空切片会把指标稀释得看不出来；
    4. 以该 macro Dice 选 ``runs/fold<k>/best.pt`` 并按 ``train.early_stop_patience`` 早停；
       每轮都覆盖 ``last.pt``（``--resume`` 靠它续跑），追加 ``metrics.csv``、写 TensorBoard；
    5. 产物全部在 ``runs/fold<k>/``（不入库）：``best.pt`` / ``last.pt`` / ``metrics.csv`` /
       ``run.json`` / ``train.log`` / ``tensorboard/``。

``--debug``（冒烟自检，**不落盘**）：训练侧只取少量病例、默认 ``batch_size=2``、跑几个 iteration，
    打印每个 batch 的 shape / 值域 / 实测阳性比例、前向-反向-优化器分段耗时、
    ``torch.cuda.max_memory_allocated()/max_memory_reserved()``，再对 1 例验证病人做一次整卷推理，
    确认 ``predict_volume`` 的形状/轴序/裁回都对，然后退出。这是标定 ``batch_size`` 的依据：
    想测某个 batch_size 就显式给 ``--set train.batch_size=8``（给了就不再用默认的 2）。

进度与随机性：
    * 采样顺序由 ``ProportionalBatchSampler`` 按 ``(train.seed, epoch, 病例集合)`` 派生，与
      ``num_workers`` 无关；``--resume`` 时用 ``sampler.set_epoch(已完成轮数)`` 把序列接上；
    * 增强的随机源在 ``CTSliceDataset`` 构造时按 ``(seed, fold, split)`` 固定；
    * ``set_seed(train.seed)`` 固定 random/numpy/torch/cuda，DataLoader 的 worker 种子由
      ``src.dataset._loader_kwargs`` 的 ``worker_init_fn`` 派生。

退出码：``0`` 正常；``2`` 前置校验失败；``3`` 训练出现 NaN/Inf 损失；``4`` ``--resume`` 的
    checkpoint 与当前配置不一致（或缺文件）；``130`` 被 Ctrl-C 中断（``last.pt`` 可用于续跑）。

前后接口：上游是 ``scripts/preprocess.py`` / ``scripts/make_splits.py`` 的产物；
    下游是第 4 轮的 ``python -m src.evaluate --fold <k>``（读 ``best.pt``）。
用法：
    ```bash
    python -m src.train --fold 0 --debug                          # 冒烟：形状/显存/耗时
    python -m src.train --fold 0 --debug --set train.batch_size=8 # 标定 batch_size
    python -m src.train --fold 0                                  # 正式训练
    python -m src.train --fold 0 --resume                         # 中断后续跑
    for f in 0 1 2 3 4; do python -m src.train --fold $f; done     # 5 折
    ```
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from src.dataset import data_config, format_hw, make_train_loader, reset_open_cache
    from src.infer import load_label_volume, predict_volume
    from src.losses import build_loss
    from src.unet import build_unet, count_parameters, load_encoder_pretrained
    from src.utils import (
        autocast_context,
        cache_file,
        cache_fingerprint,
        config_fingerprint,
        load_config,
        load_json,
        make_grad_scaler,
        rel_to_root,
        resolve_amp,
        resolve_path,
        save_json,
        set_seed,
        setup_logger,
    )
except ModuleNotFoundError:  # pragma: no cover - 兜底：把仓库根塞进 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.dataset import data_config, format_hw, make_train_loader, reset_open_cache  # type: ignore
    from src.infer import load_label_volume, predict_volume  # type: ignore
    from src.losses import build_loss  # type: ignore
    from src.unet import build_unet, count_parameters, load_encoder_pretrained  # type: ignore
    from src.utils import (  # type: ignore
        autocast_context,
        cache_file,
        cache_fingerprint,
        config_fingerprint,
        load_config,
        load_json,
        make_grad_scaler,
        rel_to_root,
        resolve_amp,
        resolve_path,
        save_json,
        set_seed,
        setup_logger,
    )

LOGGER = setup_logger("train")

#: 退出码（docs/baseline.md 第 4 节有对照表）
EXIT_OK = 0
EXIT_PREREQ = 2
EXIT_NUMERIC = 3
EXIT_RESUME = 4
EXIT_INTERRUPTED = 130

#: metrics.csv 的列（顺序固定；续跑时沿用同一表头，便于直接 pandas.read_csv）
CSV_COLUMNS = [
    "epoch", "lr", "train_loss", "train_dice", "train_ce", "train_batches",
    "train_pos_ratio", "train_seconds", "val_dice_mean", "val_dice_min", "val_dice_max",
    "val_seconds", "is_best", "best_dice", "patience", "epoch_seconds", "gpu_peak_mb",
]


class NonFiniteLoss(RuntimeError):
    """训练中出现了 NaN/Inf 损失（AMP 溢出或数据异常），直接停而不是把 NaN 权重存进 checkpoint。"""


# --------------------------------------------------------------------------------------
# 命令行与配置
# --------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CT 肝脏肿瘤 2D U-Net 训练（单折）；运行手册见 docs/baseline.md 第 4 节")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件")
    parser.add_argument("--fold", type=int, default=0, help="第几折（0-4），取自 data/splits.json")
    parser.add_argument("--debug", action="store_true",
                        help="冒烟自检：少量病例/少量 iteration，打印形状、阳性比例、显存与分段耗时后退出，**不落盘**")
    parser.add_argument("--debug-cases", type=int, default=2, help="--debug 时训练侧用几例（默认 2）")
    parser.add_argument("--debug-val-cases", type=int, default=1,
                        help="--debug 时用几例做整卷推理自检（默认 1；0 = 跳过）")
    parser.add_argument("--debug-iters", type=int, default=3, help="--debug 跑几个 iteration（默认 3）")
    parser.add_argument("--resume", action="store_true", help="从 runs/fold<k>/last.pt 续跑")
    parser.add_argument("--device", default=None, help="cuda / cpu；默认自动（有 CUDA 就用 cuda）")
    parser.add_argument("--out-dir", default=None, help="产物目录，默认 <paths.runs>/fold<k>")
    parser.add_argument("--set", dest="overrides", action="append", default=None,
                        help="覆盖配置项，可多次：--set train.batch_size=8")
    return parser.parse_args(argv)


def cfg_for_hash(cfg: dict) -> dict:
    """参与 checkpoint 指纹的配置视图（去掉 ``paths`` 与 ``train.epochs``）。

    ``paths`` 与本机路径有关、``train.epochs`` 允许续跑时改（比如先跑 5 轮试水再续到 200），
    其余任何一项变了都认为「不是同一场训练」，``--resume`` 会拒绝并列出差异。
    """
    view = copy.deepcopy(cfg or {})
    view.pop("paths", None)
    if isinstance(view.get("train"), dict):
        view["train"].pop("epochs", None)
    return view


def subset_splits(splits: dict, fold: int, n_train: int = 2, n_val: int = 1) -> dict:
    """``--debug`` 用：深拷贝 splits 并把该折的 train/val 截断到前 N 例。

    截断取排序后的前 N 例，保证同一 fold 下 ``--debug`` 用的病例固定、日志可逐次比较。
    """
    out = copy.deepcopy(splits or {})
    for record in out.get("folds", []):
        if int(record.get("fold", -1)) != int(fold):
            continue
        train_cases = sorted(int(c) for c in record.get("train", []))
        val_cases = sorted(int(c) for c in record.get("val", []))
        record["train"] = train_cases[:max(1, int(n_train))] if int(n_train) > 0 else []
        record["val"] = val_cases[:max(0, int(n_val))]
    return out


def _overrides_batch_size(overrides) -> bool:
    """命令行里是否显式给了 ``train.batch_size``（--debug 时给了就用它，不再强制成 2）。"""
    for item in overrides or []:
        key = str(item).split("=", 1)[0].strip()
        if key == "train.batch_size":
            return True
    return False


def file_digest(path, length: int = 16) -> str:
    """文件内容的 sha256 前若干位（把 splits 的版本写进产物，便于跨轮次对照）。"""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:int(length)]
    except OSError:
        return ""


def resolve_device(name: str | None = None) -> torch.device:
    """选设备：显式入参 > 有 CUDA 就用 cuda > 否则 cpu（并告警）。"""
    if name:
        device = torch.device(str(name))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但 torch.cuda.is_available() 为 False")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    LOGGER.warning("没有可用 CUDA：退到 CPU（正式训练请在 A100 上跑；CPU 只适合 --debug 验证链路）")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """CUDA 上同步一次，让分段计时准确；CPU 上是空操作。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float:
    """CUDA 峰值显存（MB，allocated）；非 CUDA 返回 0。"""
    if device.type != "cuda":
        return 0.0
    return round(float(torch.cuda.max_memory_allocated(device)) / 1024 ** 2, 1)


def environment_summary(device: torch.device) -> dict:
    """把环境版本写进 run.json（跨机器复现时要看的就是这些）。"""
    summary = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "cudnn": (torch.backends.cudnn.version()
                  if torch.backends.cudnn.is_available() else None),
        "device": str(device),
    }
    try:
        summary["numpy"] = np.__version__
    except Exception:  # noqa: BLE001 - 版本信息拿不到就算了
        pass
    if device.type == "cuda":
        summary["gpu"] = torch.cuda.get_device_name(device)
        summary["gpu_total_mb"] = round(
            float(torch.cuda.get_device_properties(device).total_memory) / 1024 ** 2, 0)
    return summary


# --------------------------------------------------------------------------------------
# 前置校验
# --------------------------------------------------------------------------------------

def check_prerequisites(cfg: dict, fold: int) -> tuple:
    """训练启动前的硬校验，返回 ``(info, problems)``；``problems`` 非空时 main 以码 2 退出。

    核对的四件事（对应 docs/todo.md 第 3 轮的要求）：
      1. ``data/splits.json`` 里有 fold、train/val 都非空；
      2. ``cache/cache_manifest.json`` 存在、预处理指纹与当前配置一致、病例集合与划分一致；
      3. 本折每个病例的 ``cache/image`` 与 ``cache/label`` 文件都在；
      4. 模型与损失的搭配（``out_channels`` 与 ``loss.softmax``）不会算出错误结果。
    另外校验「验证集里不出现不含肿瘤的病例」：5 例仅肝脏病例按约定只进训练集。
    """
    paths = (cfg or {}).get("paths") or {}
    problems: list = []
    info: dict = {"splits": {}, "manifest": {}, "train_cases": [], "val_cases": []}

    splits_path = resolve_path(paths.get("splits", "data/splits.json"))
    cache_dir = resolve_path(paths.get("cache", "cache"))
    manifest_path = resolve_path(paths.get("cache_manifest", "cache/cache_manifest.json"))
    info.update({
        "splits_path": rel_to_root(splits_path),
        "cache_dir": str(cache_dir),          # 绝对路径供 dataset/infer 使用
        "cache_dir_rel": rel_to_root(cache_dir),
        "manifest_path": rel_to_root(manifest_path),
        "splits_digest": file_digest(splits_path),
    })

    splits = load_json(splits_path, default=None)
    if not splits or not splits.get("folds"):
        problems.append(f"读不到划分文件 {info['splits_path']}（远程先跑 python scripts/make_splits.py）")
        return info, problems
    info["splits"] = splits
    folds = splits.get("folds") or []
    record = next((r for r in folds if int(r.get("fold", -1)) == int(fold)), None)
    if record is None:
        problems.append(f"{info['splits_path']} 里没有 fold={fold}"
                        f"（可选：{[r.get('fold') for r in folds]}）")
        return info, problems
    train_cases = sorted(int(c) for c in record.get("train", []))
    val_cases = sorted(int(c) for c in record.get("val", []))
    info["train_cases"], info["val_cases"] = train_cases, val_cases
    if not train_cases:
        problems.append(f"fold {fold} 的 train 为空")
    if not val_cases:
        problems.append(f"fold {fold} 的 val 为空（best.pt 与早停都依赖验证集）")
    all_cases: set = set()
    for rec in folds:
        all_cases.update(int(c) for c in rec.get("train", []))
        all_cases.update(int(c) for c in rec.get("val", []))

    manifest = load_json(manifest_path, default=None)
    if not manifest or not manifest.get("cases"):
        problems.append(f"读不到 cache 清单 {info['manifest_path']}：先跑 python scripts/fetch_manifest.py"
                        f"（或重跑 python scripts/preprocess.py）")
    else:
        info["manifest"] = manifest
        expect_hash = cache_fingerprint(cfg)
        got_hash = manifest.get("cfg_hash")
        if got_hash and got_hash != expect_hash:
            problems.append(f"cache_manifest.cfg_hash={got_hash} 与当前配置算出的 {expect_hash} 不一致："
                            f"缓存来自别的预处理参数，请重跑 python scripts/preprocess.py")
        manifest_cases = {int(r["case"]) for r in manifest.get("cases", []) if "case" in r}
        if manifest_cases != all_cases:
            problems.append(f"cache 清单与 {info['splits_path']} 的病例集合不一致："
                            f"只在清单里 {sorted(manifest_cases - all_cases)}，"
                            f"只在划分里 {sorted(all_cases - manifest_cases)}")
        missing = [c for c in train_cases + val_cases if c not in manifest_cases]
        if missing:
            problems.append(f"本折有 {len(missing)} 例不在 cache 清单里：{missing}")
        no_tumor_val = [int(r.get("case", -1)) for r in manifest.get("cases", [])
                        if int(r.get("case", -1)) in set(val_cases)
                        and int(r.get("tumor_slices", 0)) <= 0]
        if no_tumor_val:
            problems.append(f"验证集里出现不含肿瘤的病例 {no_tumor_val}：5 例仅肝脏病例"
                            f"（32/34/38/41/47）按约定只进训练集，请检查 {info['splits_path']}")

    for sub in ("image", "label"):
        missing_files = [c for c in train_cases + val_cases
                         if not cache_file(cache_dir, sub, c).exists()]
        if missing_files:
            problems.append(f"cache/{sub}/ 下缺 {len(missing_files)} 例：{missing_files[:8]}"
                            f"（先跑 python scripts/preprocess.py）")

    model_cfg = (cfg or {}).get("model") or {}
    loss_cfg = (cfg or {}).get("loss") or {}
    out_channels = int(model_cfg.get("out_channels", 2))
    if out_channels < 2 and bool(loss_cfg.get("softmax", True)):
        problems.append(f"model.out_channels={out_channels} 时 loss.softmax 必须为 false"
                        f"（单通道是 sigmoid 口径；本项目的标签只有背景/肿瘤两类）")
    if int(model_cfg.get("in_channels", 1)) != 1:
        problems.append(f"model.in_channels={model_cfg.get('in_channels')}："
                        f"数据侧给的是单通道切片（data 的 image 是 (B,1,H,W)）")
    return info, problems


# --------------------------------------------------------------------------------------
# 模型 / 优化器 / AMP
# --------------------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    """按配置建模型并处理预训练接口（基础版 ``encoder_pretrained=null`` → 随机初始化）。"""
    model = build_unet(cfg)
    model_cfg = (cfg or {}).get("model") or {}
    load_encoder_pretrained(model, model_cfg.get("encoder_pretrained"),
                            model_cfg.get("pretrained_weights_path"))
    LOGGER.info("模型：%s", model.describe())
    return model


def build_optimizer_scheduler(model: nn.Module, cfg: dict, epochs: int) -> tuple:
    """AdamW + CosineAnnealingLR（``train.scheduler: none`` 时返回 ``None`` 调度器）。"""
    train_cfg = (cfg or {}).get("train") or {}
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    name = str(train_cfg.get("scheduler", "cosine") or "none").strip().lower()
    if name in ("cosine", "cosineannealing", "cosineannealinglr"):
        eta_min = lr * float(train_cfg.get("min_lr_ratio", 0.0) or 0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(epochs)), eta_min=eta_min)
        LOGGER.info("优化器：AdamW(lr=%g, weight_decay=%g)；调度器：CosineAnnealingLR"
                    "(T_max=%d, eta_min=%g)", lr, weight_decay, max(1, int(epochs)), eta_min)
    elif name in ("none", "off", ""):
        scheduler = None
        LOGGER.info("优化器：AdamW(lr=%g, weight_decay=%g)；调度器：无（配置为 none）",
                    lr, weight_decay)
    else:
        raise ValueError(f"train.scheduler 只支持 cosine / none，收到 {train_cfg.get('scheduler')!r}")
    return optimizer, scheduler


def make_amp_state(cfg: dict, device: torch.device) -> dict:
    """解析 AMP 口径并建 GradScaler，返回 ``{"name", "dtype", "enabled", "scaler"}``。

    * ``bf16``：只用 autocast，**不启用 GradScaler**——bf16 与 fp32 指数范围相同，不需要
      loss scaling（PyTorch 官方口径）；多余地开 scaler 只会给 checkpoint 添一份无用状态。
    * ``fp16``：autocast + GradScaler（fp16 需要 loss scaling）。
    * 非 CUDA 设备（本地 CPU 调试）一律关掉 autocast，并给一个直通 scaler。
    """
    train_cfg = (cfg or {}).get("train") or {}
    name, dtype = resolve_amp(train_cfg.get("amp", "bf16"))
    if dtype is not None and device.type != "cuda":
        LOGGER.warning("设备是 %s（非 CUDA）：关闭 autocast（%s 只在 CUDA 上启用；"
                       "CPU 上 bf16 更慢且没有收益）", device.type, name)
        name, dtype = "off", None
    scaler = make_grad_scaler(device.type, enabled=(name == "fp16"))
    if name == "bf16":
        LOGGER.info("AMP：bf16 autocast（**不启用 GradScaler**：bf16 与 fp32 同指数范围，"
                    "不需要 loss scaling）")
    elif name == "fp16":
        LOGGER.info("AMP：fp16 autocast + GradScaler（fp16 需要 loss scaling）")
    else:
        LOGGER.info("AMP：关闭（纯 fp32）")
    return {"name": name, "dtype": dtype, "enabled": dtype is not None, "scaler": scaler}


# --------------------------------------------------------------------------------------
# 训练 / 验证
# --------------------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scaler, device: torch.device,
                    amp: dict, cfg: dict, epoch: int, log_every: int = 0,
                    max_iters: int = 0) -> dict:
    """训练一个 epoch；返回统计（损失、dice/ce 两项、实测阳性比例、耗时、峰值显存、首个 batch 形态）。

    梯度路径只写一条：``scaler`` 在 bf16/fp32 下是直通替身（``is_enabled() == False``），
    只有 fp16 时才真的缩放。

    耗时分成两段分别累计（``data_seconds`` = 等 DataLoader 出 batch，``compute_seconds`` = 前向+反向+优化器）：
    缓存是 ``.nii.gz`` 时取数会成为瓶颈（远程实测 557 ms/层），这两段能一眼看出是「数据等 GPU」
    还是「GPU 等数据」——第 3 轮就是靠它定位到压缩缓存的。
    """
    train_cfg = (cfg or {}).get("train") or {}
    clip = float(train_cfg.get("grad_clip_norm", 0.0) or 0.0)
    model.train()
    loss_sum = dice_sum = ce_sum = 0.0
    n_batches = n_slices = n_pos = 0
    data_seconds = compute_seconds = 0.0
    first_batch: dict = {}
    started = time.perf_counter()
    iterator = iter(loader)
    while True:
        data_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        data_seconds += time.perf_counter() - data_started
        compute_started = time.perf_counter()
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if images.ndim != 4 or labels.ndim != 3:
            raise RuntimeError(f"batch 形状异常：image {tuple(images.shape)} / "
                               f"label {tuple(labels.shape)}（期望 (B,1,H,W) / (B,H,W)）")
        if not first_batch:
            first_batch = {
                "image": [int(v) for v in images.shape],
                "label": [int(v) for v in labels.shape],
                "cases": sorted({str(c) for c in batch["case"]}),
                "z": [int(v) for v in batch["z"]],
                "image_range": [round(float(images.min()), 6), round(float(images.max()), 6)],
                "orig_hw": [[int(hw[0]), int(hw[1])] for hw in batch["orig_hw"]],
                "pad_offset": [[int(o[0]), int(o[1])] for o in batch["pad_offset"]],
            }

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device.type, amp["dtype"], enabled=amp["enabled"]):
            logits = model(images)
            loss = criterion(logits, labels)
        if not bool(torch.isfinite(loss).all()):
            raise NonFiniteLoss(f"epoch {epoch} 第 {n_batches + 1} 个 batch 的损失是 {float(loss)}"
                                f"（AMP 溢出或数据异常；可先用 --set train.amp=off 复现）")
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)      # 裁剪前先把梯度还原成真实尺度
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        parts = dict(getattr(criterion, "last_parts", {}) or {})
        loss_sum += float(loss.detach())
        dice_sum += float(parts.get("dice_loss", 0.0))
        ce_sum += float(parts.get("ce_loss", 0.0))
        n_batches += 1
        n_slices += int(labels.shape[0])
        n_pos += int((labels.reshape(int(labels.shape[0]), -1).max(dim=1).values > 0).sum())
        compute_seconds += time.perf_counter() - compute_started

        if log_every and n_batches % int(log_every) == 0:
            LOGGER.info("  epoch %d | batch %d | loss %.4f（dice %.4f + ce %.4f）| "
                        "已跑 %.1f s（取数 %.1f s / 计算 %.1f s）", epoch, n_batches,
                        loss_sum / n_batches, dice_sum / n_batches, ce_sum / n_batches,
                        time.perf_counter() - started, data_seconds, compute_seconds)
        if max_iters and n_batches >= int(max_iters):
            break

    seconds = time.perf_counter() - started
    return {
        "loss": loss_sum / max(1, n_batches),
        "dice_loss": dice_sum / max(1, n_batches),
        "ce_loss": ce_sum / max(1, n_batches),
        "batches": n_batches,
        "slices": n_slices,
        "pos_slices": n_pos,
        "pos_ratio": n_pos / max(1, n_slices),
        "seconds": seconds,
        "data_seconds": data_seconds,
        "compute_seconds": compute_seconds,
        "sec_per_batch": seconds / max(1, n_batches),
        "peak_memory_mb": peak_memory_mb(device),
        "first_batch": first_batch,
    }


def volume_dice(pred_bin, gt_bin, eps: float = 1e-6) -> float:
    """整卷二值 Dice：``(2|A∩B| + eps) / (|A| + |B| + eps)``。

    与第 4 轮 ``src/metrics.py::dice`` 用**同一公式**（含 eps、以及"两边都空记 1.0"的口径），
    这样训练期的早停指标与最终评估报告可以直接对照。第 4 轮交付后如需换成公共实现，
    只要公式不变，数值就一一对应。
    """
    pred = np.asarray(pred_bin) > 0
    gt = np.asarray(gt_bin) > 0
    intersection = float(np.count_nonzero(pred & gt))
    denominator = float(np.count_nonzero(pred)) + float(np.count_nonzero(gt))
    return float((2.0 * intersection + eps) / (denominator + eps))


def validate(model: nn.Module, cases, cache_dir, cfg: dict, device: torch.device,
             amp: dict, batch_slices: int | None = None) -> dict:
    """对验证集每例做**整卷推理**再逐例算 Dice，返回 macro 口径的均值。

    口径（用户拍板，别改成全局聚合）：先对每例算整卷 Dice，再对若干例取平均，每例等权。
    肿瘤体积跨 3 个数量级，聚合口径会被大病灶主导，看不出小病灶退化。
    """
    eval_cfg = (cfg or {}).get("eval") or {}
    model_cfg = (cfg or {}).get("model") or {}
    threshold = float(eval_cfg.get("threshold", 0.5) or 0.5)
    if batch_slices is None:
        batch_slices = int(eval_cfg.get("infer_batch_slices", 8) or 8)
    pad_to_multiple = int(model_cfg.get("pad_to_multiple", 16) or 16)
    per_case: dict = {}
    started = time.perf_counter()
    for case in cases:
        prob, pred, meta = predict_volume(model, case, cache_dir, cfg, device,
                                          pad_to_multiple=pad_to_multiple,
                                          batch_slices=int(batch_slices),
                                          threshold=threshold, amp=amp["name"])
        gt = load_label_volume(case, cache_dir, cfg)
        if tuple(pred.shape) != tuple(gt.shape):
            raise RuntimeError(f"case {case}：预测卷 {tuple(pred.shape)} 与 GT 卷 {tuple(gt.shape)} "
                               f"形状不一致（轴序或 pad_offset 裁回出错）")
        gt_bin = gt > 0
        per_case[int(case)] = {
            "dice": round(volume_dice(pred, gt_bin), 6),
            "gt_voxels": int(np.count_nonzero(gt_bin)),
            "pred_voxels": int(np.count_nonzero(pred)),
            "prob_peak": meta.get("prob_peak"),
            "seconds": meta.get("seconds"),
        }
    seconds = time.perf_counter() - started
    values = [float(v["dice"]) for v in per_case.values()]
    if not values:
        return {"cases": {}, "dice_mean": float("nan"), "dice_min": float("nan"),
                "dice_max": float("nan"), "seconds": seconds,
                "gt_voxels_total": 0, "pred_voxels_total": 0}
    return {
        "cases": per_case,
        "dice_mean": float(np.mean(values)),
        "dice_min": float(np.min(values)),
        "dice_max": float(np.max(values)),
        "seconds": seconds,
        "gt_voxels_total": int(sum(int(v["gt_voxels"]) for v in per_case.values())),
        "pred_voxels_total": int(sum(int(v["pred_voxels"]) for v in per_case.values())),
    }


# --------------------------------------------------------------------------------------
# 产物：checkpoint / metrics.csv / TensorBoard
# --------------------------------------------------------------------------------------

def checkpoint_payload(*, model, cfg: dict, fold: int, epoch: int, metrics: dict, best: dict,
                       patience: int, info: dict, amp_name: str, optimizer=None,
                       scheduler=None, scaler=None, include_optimizer: bool = False) -> dict:
    """组装 checkpoint。

    ``best.pt``（``include_optimizer=False``）只含模型权重与元信息，供第 4 轮评估加载；
    ``last.pt``（``include_optimizer=True``）额外含 optimizer/scheduler/scaler 状态供 ``--resume``。
    ``epoch`` 是**已完成的轮数**：``--resume`` 从这个数之后继续。
    """
    payload = {
        "format": "ct-liver-tumor-2d-unet/1",
        "fold": int(fold),
        "epoch": int(epoch),
        "metric": {"name": "val_dice_macro", "value": float(metrics.get("dice_mean", float("nan")))},
        "best": {"dice": float(best.get("dice", float("-inf"))), "epoch": int(best.get("epoch", 0))},
        "patience": int(patience),
        "model_state": model.state_dict(),
        "cfg": cfg_for_hash(cfg),
        "cfg_hash": config_fingerprint(cfg_for_hash(cfg)),
        "preprocess_cfg_hash": cache_fingerprint(cfg),
        "cases": {"train": [int(c) for c in info.get("train_cases", [])],
                  "val": [int(c) for c in info.get("val_cases", [])]},
        "splits_digest": info.get("splits_digest", ""),
        "seed": int(((cfg.get("train") or {}).get("seed", 42))),
        "amp": str(amp_name),
        "torch": torch.__version__,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if include_optimizer:
        payload["optimizer_state"] = optimizer.state_dict() if optimizer is not None else None
        payload["scheduler_state"] = scheduler.state_dict() if scheduler is not None else None
        payload["scaler_state"] = scaler.state_dict() if scaler is not None else {}
    return payload


def save_checkpoint(path, payload: dict) -> None:
    """写 checkpoint：先写 ``*.tmp`` 再原子改名，避免中断时留下半个文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(path)


def load_checkpoint(path, device: torch.device) -> dict:
    """读 checkpoint。

    ``weights_only=False``：这是**我们自己写的**文件，里面含 cfg 字典、病例列表等非张量对象；
    torch 2.6+ 的默认值已改成 True，显式关掉才能读回完整内容（老版本没有该参数，回退即可）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 checkpoint {path}（--resume 需要先有一次训练产出 last.pt）")
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - 老版本 torch 没有 weights_only 参数
        return torch.load(str(path), map_location=device)


def append_metrics_row(csv_path, row: dict) -> None:
    """把一行指标追加到 ``metrics.csv``（文件不存在时先写表头；列顺序固定）。"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def make_summary_writer(log_dir, enabled: bool = True):
    """建 TensorBoard ``SummaryWriter``；不可用时只告警并返回 ``None``（metrics.csv 照旧写）。"""
    if not enabled:
        LOGGER.info("TensorBoard：配置里已关闭（train.tensorboard=false）")
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001 - 缺包 / protobuf 版本冲突都归到这里
        LOGGER.warning("TensorBoard 不可用（%s: %s）：跳过 TensorBoard，metrics.csv 不受影响",
                       type(exc).__name__, exc)
        return None
    try:
        return SummaryWriter(log_dir=str(log_dir))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("TensorBoard SummaryWriter 建不起来（%s: %s）：跳过 TensorBoard",
                       type(exc).__name__, exc)
        return None


def cfg_diff(old_cfg: dict, new_cfg: dict, limit: int = 8) -> list:
    """列出两份配置里取值不同的点分键（``--resume`` 拒绝时用来解释原因）。"""
    def flatten(node, prefix: str = "") -> dict:
        out: dict = {}
        if isinstance(node, dict):
            for key, value in node.items():
                out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[prefix[:-1]] = node
        return out

    old_flat, new_flat = flatten(old_cfg or {}), flatten(new_cfg or {})
    keys = sorted(set(old_flat) | set(new_flat))
    diff = [f"{key}: {old_flat.get(key)!r} → {new_flat.get(key)!r}"
            for key in keys if old_flat.get(key) != new_flat.get(key)]
    if len(diff) > limit:
        return diff[:limit] + [f"...（共 {len(diff)} 处差异，只显示前 {limit} 处）"]
    return diff


# --------------------------------------------------------------------------------------
# --debug 冒烟自检
# --------------------------------------------------------------------------------------

def run_debug(*, model, criterion, optimizer, train_loader, val_cases, cache_dir, cfg: dict,
              device: torch.device, amp: dict, args, epochs: int, fold: int) -> int:
    """``--debug``：跑少量 iteration 打印标定信息 + 1 例整卷推理自检，**不写任何文件**。"""
    train_cfg = (cfg or {}).get("train") or {}
    eval_cfg = (cfg or {}).get("eval") or {}
    threshold = float(eval_cfg.get("threshold", 0.5) or 0.5)
    batch_size = int(train_cfg.get("batch_size", 8))
    dataset = train_loader.dataset
    train_cases = list(getattr(dataset, "case_ids", []))

    LOGGER.info("=" * 78)
    LOGGER.info("--debug 冒烟自检：训练侧 %d 例 %s / %d 层切片 / 一轮 %d 个 batch；"
                "batch_size=%d；跑 %d 个 iteration 后退出（**不写 runs/**）",
                len(train_cases), train_cases, len(dataset), len(train_loader), batch_size,
                int(args.debug_iters))
    LOGGER.info("采样器：\n%s", train_loader.batch_sampler.describe())
    LOGGER.info("=" * 78)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    iterator = iter(train_loader)
    records: list = []
    for step in range(1, int(args.debug_iters) + 1):
        sync(device)
        wait_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            LOGGER.warning("loader 提前结束（只有 %d 个 batch），--debug-iters 未跑满", step - 1)
            break
        wait_seconds = time.perf_counter() - wait_started

        host_started = time.perf_counter()
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        sync(device)
        host_seconds = time.perf_counter() - host_started

        optimizer.zero_grad(set_to_none=True)
        forward_started = time.perf_counter()
        with autocast_context(device.type, amp["dtype"], enabled=amp["enabled"]):
            logits = model(images)
            loss = criterion(logits, labels)
        sync(device)
        forward_seconds = time.perf_counter() - forward_started

        backward_started = time.perf_counter()
        if amp["scaler"].is_enabled():
            amp["scaler"].scale(loss).backward()
        else:
            loss.backward()
        sync(device)
        backward_seconds = time.perf_counter() - backward_started

        step_started = time.perf_counter()
        if amp["scaler"].is_enabled():
            amp["scaler"].step(optimizer)
            amp["scaler"].update()
        else:
            optimizer.step()
        sync(device)
        step_seconds = time.perf_counter() - step_started

        n_slices = int(labels.shape[0])
        pos_slices = int((labels.reshape(n_slices, -1).max(dim=1).values > 0).sum())
        parts = dict(getattr(criterion, "last_parts", {}) or {})
        allocated = (float(torch.cuda.memory_allocated(device)) / 1024 ** 2
                     if device.type == "cuda" else 0.0)
        LOGGER.info("iteration %d：image %s / label %s（logits %s）；病例 %s",
                    step, tuple(images.shape), tuple(labels.shape), tuple(logits.shape),
                    sorted({str(c) for c in batch["case"]}))
        LOGGER.info("  值域 [%.4f, %.4f]；label 取值 %s；**含肿瘤切片 %d/%d = %.3f**（配置目标 %.2f）；"
                    "orig_hw %s；补边偏移 %s",
                    float(images.min()), float(images.max()),
                    sorted(int(v) for v in torch.unique(labels).tolist()),
                    pos_slices, n_slices, pos_slices / max(1, n_slices),
                    float(train_cfg.get("pos_ratio_target", 0.30) or 0.0),
                    [[int(hw[0]), int(hw[1])] for hw in batch["orig_hw"]][:3],
                    [[int(o[0]), int(o[1])] for o in batch["pad_offset"]][:3])
        LOGGER.info("  loss %.4f（dice %.4f + ce %.4f）；各 batch 分段耗时：取 batch %.3f s / "
                    "搬到 GPU %.3f s / 前向+损失 %.3f s / 反向 %.3f s / 优化器 %.3f s；"
                    "显存 allocated %.0f MB / 峰值 %.0f MB",
                    float(loss.detach()), float(parts.get("dice_loss", 0.0)),
                    float(parts.get("ce_loss", 0.0)), wait_seconds, host_seconds,
                    forward_seconds, backward_seconds, step_seconds,
                    allocated, peak_memory_mb(device))
        records.append({"step": step, "loss": float(loss.detach()),
                        "pos_slices": pos_slices, "batch_size": n_slices,
                        "wait": wait_seconds, "host": host_seconds, "forward": forward_seconds,
                        "backward": backward_seconds, "step_time": step_seconds,
                        "allocated_mb": allocated, "peak_mb": peak_memory_mb(device)})

    if records:
        steady = records[1:] or records       # 第 1 个 iteration 含 worker 冷启动，单独看

        def mean(key: str) -> float:
            return float(np.mean([float(r[key]) for r in steady]))

        total = mean("forward") + mean("backward") + mean("step_time")
        LOGGER.info("-" * 78)
        LOGGER.info("平均（排除第 1 个 iteration 的冷启动）：前向+损失 %.3f s / 反向 %.3f s / "
                    "优化器 %.3f s → 单步合计 %.3f s（batch_size=%d）",
                    mean("forward"), mean("backward"), mean("step_time"), total, batch_size)

    if device.type == "cuda":
        peak_alloc = float(torch.cuda.max_memory_allocated(device)) / 1024 ** 2
        peak_reserved = float(torch.cuda.max_memory_reserved(device)) / 1024 ** 2
        LOGGER.info("峰值显存（batch_size=%d，512×512 输入，amp=%s）：allocated %.0f MB / "
                    "reserved %.0f MB（含约 4.7 GB 不随 batch 增长的静态开销：cuDNN autotune 工作区；"
                    "两点标定见 docs/baseline.md 4.1）",
                    batch_size, amp["name"], peak_alloc, peak_reserved)
        # 只给「下一个该测多少」的建议，**不做线性外推**：实测 bs=16 → 10920 MB，
        # 而把静态开销一起按 batch 缩放会算出 bs=8 ≈ 5460 MB（真实值 7671 MB），偏小得离谱。
        configured = int((cfg.get("train") or {}).get("batch_size", batch_size) or batch_size)
        suggest = configured if configured != batch_size else (batch_size * 2 if batch_size < 32 else 48)
        LOGGER.info("要标定更大的 batch 就直接复测（一次约 1 分钟，比任何外推都准）："
                    "python -m src.train --fold %d --debug --set train.batch_size=%d",
                    int(fold), int(suggest))
    else:
        LOGGER.info("设备是 CPU：显存统计不可用（正式训练在 A100 上进行，本机只验证链路）")

    if int(args.debug_val_cases) > 0 and val_cases:
        case = int(val_cases[0])
        LOGGER.info("-" * 78)
        LOGGER.info("整卷推理自检：case %d（用当前（随机）权重，只为验证 predict_volume 的"
                    "形状 / 轴序 / pad_offset 裁回，**指标数值本身没有意义**）", case)
        prob, pred, meta = predict_volume(model, case, cache_dir, cfg, device,
                                          pad_to_multiple=int(((cfg.get("model") or {})
                                                               .get("pad_to_multiple", 16) or 16)),
                                          batch_slices=int(eval_cfg.get("infer_batch_slices", 8) or 8),
                                          threshold=threshold, amp=amp["name"])
        gt = load_label_volume(case, cache_dir, cfg)
        if tuple(pred.shape) != tuple(gt.shape):
            LOGGER.error("预测卷 %s 与 GT 卷 %s 形状不一致：轴序或 pad_offset 裁回出错",
                         tuple(pred.shape), tuple(gt.shape))
            return EXIT_NUMERIC
        LOGGER.info("  prob %s（峰值 %.4f）/ pred %s（前景 %d 体素）/ GT %s（前景 %d 体素）；"
                    "整卷 Dice = %.4f（随机权重）",
                    tuple(prob.shape), float(prob.max()), tuple(pred.shape), int(pred.sum()),
                    tuple(gt.shape), int(gt.sum()), volume_dice(pred, gt))
        LOGGER.info("  原始面内 %s → 补边画布 %s，pad_offset=%s，nz=%d；推理耗时 %.1f s"
                    "（infer_batch_slices=%d，阈值 %.2f）",
                    format_hw(meta["orig_hw"]), format_hw(meta["target_hw"]),
                    tuple(meta["pad_offset"]), int(meta["n_slices"]), float(meta["seconds"]),
                    int(meta["batch_slices"]), float(meta["threshold"]))
    elif int(args.debug_val_cases) > 0:
        LOGGER.warning("--debug-val-cases=%d 但这一折没有可用验证病例，跳过整卷推理自检",
                       int(args.debug_val_cases))

    LOGGER.info("=" * 78)
    LOGGER.info("--debug 结束：**没有写盘**（runs/ 下不会出现本折产物）。")
    LOGGER.info("下一步：按上面的显存/单步耗时定 batch_size，然后正式训练")
    LOGGER.info("  python -m src.train --fold %d                # 正式训练（epochs=%d，早停 patience=%d）",
                int(fold), int(epochs), int(train_cfg.get("early_stop_patience", 20)))
    LOGGER.info("  5 折全跑：for f in 0 1 2 3 4; do python -m src.train --fold $f; done")
    return EXIT_OK


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    paths = cfg.get("paths") or {}
    train_cfg = cfg.get("train") or {}
    eval_cfg = cfg.get("eval") or {}
    data_cfg = data_config(cfg)
    fold = int(args.fold)
    seed = int(train_cfg.get("seed", 42))
    epochs = int(train_cfg.get("epochs", 200))
    val_every = max(1, int(train_cfg.get("val_every", 1) or 1))
    early_stop = int(train_cfg.get("early_stop_patience", 20) or 0)
    log_every = int(train_cfg.get("log_every", 50) or 0)
    out_dir = (resolve_path(args.out_dir) if args.out_dir
               else resolve_path(paths.get("runs", "runs")) / f"fold{fold}")

    if not args.debug:
        out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("train", log_file=(None if args.debug else out_dir / "train.log"))
    set_seed(seed)
    device = resolve_device(args.device)

    LOGGER.info("=" * 78)
    LOGGER.info("CT 肝脏肿瘤 2D U-Net 训练：fold %d｜%s｜配置 %s",
                fold, "**--debug 冒烟自检**" if args.debug else "正式训练",
                rel_to_root(resolve_path(args.config)))
    LOGGER.info("=" * 78)
    LOGGER.info("设备：%s%s；随机种子：%d；目标面内尺寸：%s（pad_align=%s）",
                device,
                f"（{torch.cuda.get_device_name(device)}）" if device.type == "cuda" else "",
                seed, format_hw(data_cfg["target_hw"]), data_cfg.get("pad_align"))
    LOGGER.info("训练配置：epochs=%d，batch_size=%d，lr=%g，val_every=%d，早停 patience=%d，"
                "grad_clip_norm=%g，num_workers=%d",
                epochs, int(train_cfg.get("batch_size", 8)), float(train_cfg.get("lr", 1e-3)),
                val_every, early_stop, float(train_cfg.get("grad_clip_norm", 0.0) or 0.0),
                int(data_cfg.get("num_workers", 8)))
    LOGGER.info("产物目录：%s", "（--debug 不落盘）" if args.debug else rel_to_root(out_dir))

    info, problems = check_prerequisites(cfg, fold)
    if problems:
        for item in problems:
            LOGGER.error("  - %s", item)
        LOGGER.error("前置校验失败：发现 %d 个问题，训练未启动（处置见 docs/baseline.md 第 4 节）",
                     len(problems))
        return EXIT_PREREQ
    LOGGER.info("前置校验通过：fold %d 的 train %d 例 %s / val %d 例 %s；cache 清单 %d 例，"
                "预处理指纹与配置一致；splits 指纹 %s",
                fold, len(info["train_cases"]), info["train_cases"],
                len(info["val_cases"]), info["val_cases"],
                len(info["manifest"].get("cases", [])), info.get("splits_digest") or "NA")

    # ---- 数据加载器（--debug 时收窄病例数与 batch_size，但不改配置文件） ----
    run_cfg = cfg
    if args.debug:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.setdefault("train", {})
        if _overrides_batch_size(args.overrides):
            LOGGER.info("--debug：沿用命令行指定的 train.batch_size=%s",
                        run_cfg["train"].get("batch_size"))
        else:
            run_cfg["train"]["batch_size"] = 2
            LOGGER.info("--debug：命令行没有指定 batch_size → 用 2 跑通链路；"
                        "标定显存请显式给 --set train.batch_size=8")
        debug_splits = subset_splits(info["splits"], fold, n_train=int(args.debug_cases),
                                     n_val=int(args.debug_val_cases))
        train_loader = make_train_loader(debug_splits, fold, run_cfg)
        record = next((r for r in debug_splits.get("folds", [])
                       if int(r.get("fold", -1)) == fold), {})
        val_cases = sorted(int(c) for c in record.get("val", []))
        LOGGER.info("--debug 病例：train %s / val %s（原折 %d 例 / %d 例）",
                    sorted(int(c) for c in record.get("train", [])), val_cases,
                    len(info["train_cases"]), len(info["val_cases"]))
    else:
        train_loader = make_train_loader(info["splits"], fold, cfg)
        val_cases = list(info["val_cases"])
        LOGGER.info("训练 loader：%d 例 / %d 层切片 → 一轮 %d 个 batch（batch_size=%d）；"
                    "验证 %d 例（整卷推理）",
                    len(train_loader.dataset.case_ids), len(train_loader.dataset),
                    len(train_loader), int(train_loader.batch_sampler.batch_size),
                    len(val_cases))

    # ---- 模型 / 损失 / 优化器 / AMP ----
    model = build_model(run_cfg)
    model.to(device)
    criterion = build_loss(run_cfg).to(device)
    optimizer, scheduler = build_optimizer_scheduler(model, run_cfg, epochs)
    amp = make_amp_state(run_cfg, device)
    LOGGER.info("可训练参数：%.3f M", count_parameters(model) / 1e6)

    if args.debug:
        try:
            return run_debug(model=model, criterion=criterion, optimizer=optimizer,
                             train_loader=train_loader, val_cases=val_cases,
                             cache_dir=info["cache_dir"], cfg=run_cfg, device=device, amp=amp,
                             args=args, epochs=epochs, fold=fold)
        except NonFiniteLoss as exc:
            LOGGER.error("--debug 出现非有限损失：%s", exc)
            return EXIT_NUMERIC

    # ---- 续跑 ----
    start_epoch = 0
    best = {"dice": float("-inf"), "epoch": 0}
    patience = 0
    if args.resume:
        try:
            checkpoint = load_checkpoint(out_dir / "last.pt", device)
        except FileNotFoundError as exc:
            LOGGER.error("%s", exc)
            return EXIT_RESUME
        except Exception as exc:  # noqa: BLE001 - 文件损坏 / 版本不兼容都归到这里
            LOGGER.error("读 checkpoint 失败（%s: %s）：请删掉 last.pt 从头训练", type(exc).__name__, exc)
            return EXIT_RESUME
        if int(checkpoint.get("fold", -1)) != fold:
            LOGGER.error("checkpoint 属于 fold %s，与 --fold %d 不一致", checkpoint.get("fold"), fold)
            return EXIT_RESUME
        current_hash = config_fingerprint(cfg_for_hash(cfg))
        if checkpoint.get("cfg_hash") and checkpoint["cfg_hash"] != current_hash:
            LOGGER.error("checkpoint 与当前配置不一致（cfg_hash %s vs %s），不能续跑；"
                         "差异如下（最多 8 处）：", checkpoint.get("cfg_hash"), current_hash)
            for line in cfg_diff(checkpoint.get("cfg") or {}, cfg_for_hash(cfg), limit=8):
                LOGGER.error("  - %s", line)
            LOGGER.error("（paths 与 train.epochs 不参与该指纹；要改这些参数请删掉 runs/fold%d/ 重跑）",
                         fold)
            return EXIT_RESUME
        try:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None and checkpoint.get("scheduler_state"):
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            if amp["scaler"].is_enabled() and checkpoint.get("scaler_state"):
                amp["scaler"].load_state_dict(checkpoint["scaler_state"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("恢复 checkpoint 状态失败（%s: %s）：请删掉 last.pt 从头训练",
                         type(exc).__name__, exc)
            return EXIT_RESUME
        start_epoch = int(checkpoint.get("epoch", 0))
        best = {"dice": float((checkpoint.get("best") or {}).get("dice", float("-inf"))),
                "epoch": int((checkpoint.get("best") or {}).get("epoch", 0))}
        patience = int(checkpoint.get("patience", 0))
        sampler = getattr(train_loader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(start_epoch)
        LOGGER.info("续跑：已完成 %d 轮 → 从第 %d 轮继续；当前 best 整卷 Dice=%.4f@ep%d；"
                    "patience=%d/%d；采样顺序已接到 epoch %d",
                    start_epoch, start_epoch + 1, best["dice"], best["epoch"], patience,
                    early_stop, start_epoch + 1)

    if start_epoch >= epochs:
        LOGGER.warning("checkpoint 已经跑满 epochs=%d（已完成 %d 轮），没有可继续的轮次："
                       "要接着跑就把轮数调大，例如 --resume --set train.epochs=%d",
                       epochs, start_epoch, start_epoch + 50)
        return EXIT_OK

    # ---- 产物元信息 ----
    run_info = {
        "stage": "第 3 轮（模型 + 损失 + 训练）",
        "fold": fold,
        "config": rel_to_root(resolve_path(args.config)),
        "args": vars(args),
        "device": str(device),
        "env": environment_summary(device),
        "seed": seed,
        "epochs": epochs,
        "val_every": val_every,
        "early_stop_patience": early_stop,
        "batch_size": int((run_cfg.get("train") or {}).get("batch_size", 8)),
        "target_hw": [int(v) for v in data_cfg["target_hw"]],
        "pad_align": data_cfg.get("pad_align"),
        "cases": {"train": info["train_cases"], "val": info["val_cases"]},
        "n_slices": {"train": int(len(train_loader.dataset))},
        "model": model.describe(),
        "params": int(count_parameters(model)),
        "loss": criterion.describe(),
        "amp": amp["name"],
        "cfg": cfg_for_hash(cfg),
        "cfg_hash": config_fingerprint(cfg_for_hash(cfg)),
        "preprocess_cfg_hash": cache_fingerprint(cfg),
        "splits": info["splits_path"],
        "splits_digest": info.get("splits_digest", ""),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resume_from_epoch": start_epoch,
    }
    save_json(run_info, out_dir / "run.json")
    writer = make_summary_writer(out_dir / "tensorboard",
                                 enabled=bool(train_cfg.get("tensorboard", True)))

    LOGGER.info("开始训练：epoch %d → %d（%d 个 epoch；每 %d 轮验证一次，"
                "按验证集整卷肿瘤 Dice 的 macro 均值选最优/早停）",
                start_epoch + 1, epochs, max(0, epochs - start_epoch), val_every)
    last_epoch = start_epoch
    total_started = time.perf_counter()
    try:
        for epoch in range(start_epoch + 1, epochs + 1):
            epoch_started = time.perf_counter()
            train_stats = train_one_epoch(model, train_loader, criterion, optimizer,
                                          amp["scaler"], device, amp, run_cfg, epoch,
                                          log_every=log_every)
            if epoch == start_epoch + 1 and train_stats.get("first_batch"):
                first = train_stats["first_batch"]
                LOGGER.info("首个 batch 形态：image %s / label %s；病例 %s；z=%s；值域 [%.4f, %.4f]；"
                            "orig_hw %s；补边偏移 %s",
                            tuple(first["image"]), tuple(first["label"]), first["cases"],
                            first["z"][:8], first["image_range"][0], first["image_range"][1],
                            first["orig_hw"][:3], first["pad_offset"][:3])
            if epoch == start_epoch + 1 and train_stats["data_seconds"] > train_stats["compute_seconds"]:
                LOGGER.warning("取数耗时 %.1f s 超过计算耗时 %.1f s：**数据加载是瓶颈**。"
                               "缓存若是 .nii.gz，先跑 python scripts/inflate_cache.py 生成未压缩 .nii"
                               "（nibabel 每读一层会整卷解压，实测 557 ms/层）；已用 .nii 再考虑调大 "
                               "data.num_workers（当前 %d）。",
                               train_stats["data_seconds"], train_stats["compute_seconds"],
                               int(data_cfg.get("num_workers", 8)))
            if scheduler is not None:
                scheduler.step()
            current_lr = float(optimizer.param_groups[0]["lr"])

            val_stats: dict = {}
            if (epoch % val_every == 0) or (epoch == epochs):
                val_stats = validate(model, val_cases, info["cache_dir"], run_cfg, device, amp)
            is_best = False
            if val_stats:
                if float(val_stats["dice_mean"]) > float(best["dice"]):
                    best = {"dice": float(val_stats["dice_mean"]), "epoch": int(epoch)}
                    patience = 0
                    is_best = True
                else:
                    patience += 1

            epoch_seconds = time.perf_counter() - epoch_started
            row = {
                "epoch": epoch,
                "lr": f"{current_lr:.6g}",
                "train_loss": round(train_stats["loss"], 6),
                "train_dice": round(train_stats["dice_loss"], 6),
                "train_ce": round(train_stats["ce_loss"], 6),
                "train_batches": train_stats["batches"],
                "train_pos_ratio": round(train_stats["pos_ratio"], 6),
                "train_seconds": round(train_stats["seconds"], 3),
                "val_dice_mean": round(val_stats["dice_mean"], 6) if val_stats else "",
                "val_dice_min": round(val_stats["dice_min"], 6) if val_stats else "",
                "val_dice_max": round(val_stats["dice_max"], 6) if val_stats else "",
                "val_seconds": round(val_stats["seconds"], 3) if val_stats else "",
                "is_best": int(bool(is_best)),
                "best_dice": round(float(best["dice"]), 6) if math.isfinite(float(best["dice"])) else "",
                "patience": patience,
                "epoch_seconds": round(epoch_seconds, 3),
                "gpu_peak_mb": train_stats["peak_memory_mb"],
            }
            append_metrics_row(out_dir / "metrics.csv", row)

            if writer is not None:
                writer.add_scalar("train/loss", train_stats["loss"], epoch)
                writer.add_scalar("train/dice_loss", train_stats["dice_loss"], epoch)
                writer.add_scalar("train/ce_loss", train_stats["ce_loss"], epoch)
                writer.add_scalar("train/lr", current_lr, epoch)
                writer.add_scalar("train/pos_ratio", train_stats["pos_ratio"], epoch)
                writer.add_scalar("time/epoch_seconds", epoch_seconds, epoch)
                if train_stats["peak_memory_mb"]:
                    writer.add_scalar("train/peak_memory_mb", train_stats["peak_memory_mb"], epoch)
                if val_stats:
                    writer.add_scalar("val/dice_mean", val_stats["dice_mean"], epoch)
                    writer.add_scalar("val/dice_min", val_stats["dice_min"], epoch)
                    writer.add_scalar("val/dice_max", val_stats["dice_max"], epoch)
                    writer.add_scalar("val/best_dice", float(best["dice"]), epoch)
                    for case_id, rec in sorted(val_stats["cases"].items()):
                        writer.add_scalar(f"val/dice_case_{int(case_id)}", rec["dice"], epoch)
                writer.flush()

            if val_stats:
                per_case = " ".join(f"{int(c)}:{float(v['dice']):.3f}"
                                    for c, v in sorted(val_stats["cases"].items()))
                LOGGER.info("epoch %d/%d | lr %.2e | 训练 loss %.4f（dice %.4f + ce %.4f）| "
                            "%d 个 batch / %.1f s［取数 %.1f s + 计算 %.1f s］| "
                            "验证整卷 Dice %.4f［%s；min %.3f max %.3f］/ %.1f s | "
                            "best %.4f@ep%d | patience %d/%d | 峰值显存 %.0f MB",
                            epoch, epochs, current_lr, train_stats["loss"], train_stats["dice_loss"],
                            train_stats["ce_loss"], train_stats["batches"], train_stats["seconds"],
                            train_stats["data_seconds"], train_stats["compute_seconds"],
                            val_stats["dice_mean"], per_case, val_stats["dice_min"],
                            val_stats["dice_max"], val_stats["seconds"], float(best["dice"]),
                            int(best["epoch"]), patience, early_stop, train_stats["peak_memory_mb"])
            else:
                LOGGER.info("epoch %d/%d | lr %.2e | 训练 loss %.4f（dice %.4f + ce %.4f）| "
                            "%d 个 batch / %.1f s［取数 %.1f s + 计算 %.1f s］| "
                            "本轮不验证（val_every=%d）| best %.4f@ep%d | 峰值显存 %.0f MB",
                            epoch, epochs, current_lr, train_stats["loss"], train_stats["dice_loss"],
                            train_stats["ce_loss"], train_stats["batches"], train_stats["seconds"],
                            train_stats["data_seconds"], train_stats["compute_seconds"],
                            val_every, float(best["dice"]), int(best["epoch"]),
                            train_stats["peak_memory_mb"])

            last_epoch = epoch
            save_checkpoint(out_dir / "last.pt", checkpoint_payload(
                model=model, cfg=cfg, fold=fold, epoch=epoch,
                metrics=val_stats or {"dice_mean": float("nan")},
                best=best, patience=patience, info=info, amp_name=amp["name"],
                optimizer=optimizer, scheduler=scheduler, scaler=amp["scaler"],
                include_optimizer=True))
            if is_best:
                save_checkpoint(out_dir / "best.pt", checkpoint_payload(
                    model=model, cfg=cfg, fold=fold, epoch=epoch, metrics=val_stats, best=best,
                    patience=patience, info=info, amp_name=amp["name"], include_optimizer=False))

            if early_stop and val_stats and patience >= early_stop:
                LOGGER.info("早停：验证集整卷 Dice 连续 %d 轮没有提升（patience=%d），停在第 %d 轮；"
                            "best 仍是 %.4f@ep%d", patience, early_stop, epoch,
                            float(best["dice"]), int(best["epoch"]))
                break
    except NonFiniteLoss as exc:
        LOGGER.error("训练中断：%s", exc)
        LOGGER.error("已保存 last.pt（epoch %d）；可用 --resume 续跑，或先用 "
                     "--set train.amp=off / 调小 train.lr 复现", last_epoch)
        return EXIT_NUMERIC
    except KeyboardInterrupt:
        LOGGER.warning("收到中断（Ctrl-C）：已完成 %d 轮，last.pt 已保存；"
                       "续跑：python -m src.train --fold %d --resume", last_epoch, fold)
        return EXIT_INTERRUPTED

    total_seconds = time.perf_counter() - total_started
    if writer is not None:
        writer.close()

    LOGGER.info("=" * 78)
    if math.isfinite(float(best["dice"])):
        LOGGER.info("训练结束：best 整卷 Dice %.4f @ epoch %d；本次跑完 %d 轮（epoch %d → %d），"
                    "用时 %.1f 分钟", float(best["dice"]), int(best["epoch"]),
                    last_epoch - start_epoch, start_epoch + 1, last_epoch, total_seconds / 60.0)
    else:
        LOGGER.warning("训练结束：一次验证都没做（val_every=%d？），没有 best.pt", val_every)
    LOGGER.info("产物：%s（best.pt / last.pt / metrics.csv / run.json / train.log / tensorboard/）",
                rel_to_root(out_dir))
    run_info.update({
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs_run": int(last_epoch - start_epoch),
        "last_epoch": int(last_epoch),
        "best": {"dice": float(best["dice"]), "epoch": int(best["epoch"])},
        "patience": int(patience),
        "total_seconds": round(total_seconds, 1),
    })
    save_json(run_info, out_dir / "run.json")
    reset_open_cache()
    LOGGER.info("下一步：第 4 轮 python -m src.evaluate --fold %d（汇总指标与 5 折报告）", fold)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
