"""整卷推理：逐层前向 → 概率拼回整卷 → 按 ``pad_offset`` 裁回原始面内尺寸。

整体功能（本文件在**第 3 轮**先落地，供训练期每轮验证使用；第 4 轮在其上补后处理与指标）：
    1. ``predict_volume`` —— 对一个病例的全部切片按 z 顺序逐层前向（每层在 dataset 里已经
       ``/65535`` 还原 [0,1] 并**居中补边**到 ``data.target_hw``），把肿瘤通道的概率写进
       ``(H, W, Z)`` 画布，最后按 ``pad_offset`` 裁回补边前的原始面内尺寸；
    2. ``seg_prob_to_label`` —— 概率 → ``{0,1}`` 二值标签（默认阈值 0.5）；
    3. ``load_label_volume`` —— 按**同一轴序**读 GT 体素（训练期验证与第 4 轮评估共用）。

两条必须记住的口径（前两轮都在同类型的地方踩过坑）：

1. **轴序**：训练样本的 image 与 label 都直接取 nibabel 数组的 ``[:, :, z]``；整卷推理
   也是把这些切片按原顺序写回 ``[:, :, z]``。因此 GT 必须保持 nibabel 原始轴序。
   方形切片在错误转置后形状不变，Dice 会静默接近零，不能靠 shape 检查发现。

2. **补边偏移**：dataset 把每个样本**居中补边**到 ``data.target_hw``（342→512 的偏移是 (85,85)），
   内容在画布里的左上角是 ``pad_offset=(top, left)``。裁回必须用 ``[top:top+H, left:left+W]``；
   写死 ``[:H, :W]`` 会把预测整体错位半个补边量（这是第 2 轮就写进文档的坑）。

性能与显存：整卷画布是 ``data.target_hw`` × nz 的 float32（512×512×488 ≈ 512 MB，主机内存），
    前向本身是逐层无梯度（``torch.no_grad``），只受 batch_slices 影响。

依赖：torch / numpy / nibabel / ``src.dataset``（复用它的补边与读取口径，不另写一遍读盘逻辑）。
前后接口：上游是 ``src.train``（每轮验证）与第 4 轮的 ``src.evaluate``；下游是 ``src.metrics``。
用法：``prob, pred, meta = predict_volume(model, case, cache_dir, cfg, device)``。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

try:
    from src.dataset import CTSliceDataset, format_hw, reset_open_cache
    from src.utils import autocast_context, cache_file, resolve_amp, setup_logger, warn_compressed_cache
except ModuleNotFoundError:  # pragma: no cover - 兜底：把仓库根塞进 sys.path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.dataset import CTSliceDataset, format_hw, reset_open_cache  # type: ignore
    from src.utils import (  # type: ignore
        autocast_context,
        cache_file,
        resolve_amp,
        setup_logger,
        warn_compressed_cache,
    )

LOGGER = setup_logger("infer")

#: 肿瘤所在的输出通道：本项目 ``model.out_channels=2``（0=背景，1=肿瘤）。
#: 将来若改三分类（背景/肝脏/肿瘤），把这里换成 2 即可（或调用时传 ``tumor_channel=``）。
TUMOR_CHANNEL = 1

#: 概率 → 标签的默认阈值（与 ``configs/default.yaml`` 的 ``eval.threshold`` 一致）
DEFAULT_THRESHOLD = 0.5


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------

def _module_device(module: torch.nn.Module) -> torch.device:
    """取模块参数所在设备；没有参数时退回 CPU。"""
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _same_device(left: torch.device, right: torch.device) -> bool:
    """比较两个设备是否兼容（``cuda`` 与 ``cuda:0`` 视为同一个，``index=None`` 表示不指定）。"""
    if left.type != right.type:
        return False
    return left.index is None or right.index is None or int(left.index) == int(right.index)


def resolve_threshold(cfg: dict, threshold: float | None = None) -> float:
    """阈值优先取显式入参，其次 ``cfg['eval']['threshold']``，最后 0.5。"""
    if threshold is not None:
        return float(threshold)
    value = ((cfg or {}).get("eval") or {}).get("threshold", DEFAULT_THRESHOLD)
    return float(DEFAULT_THRESHOLD if value is None else value)


def resolve_infer_amp(cfg: dict, amp: str | None = None) -> tuple:
    """推理用的 AMP 口径：显式入参 > ``eval.amp`` > ``train.amp``。

    推理默认与训练同精度（``train.amp``）：这样验证指标反映的就是训练时的数值口径。
    """
    if amp is None:
        eval_cfg = (cfg or {}).get("eval") or {}
        train_cfg = (cfg or {}).get("train") or {}
        amp = eval_cfg.get("amp") or train_cfg.get("amp") or "off"
    return resolve_amp(amp)


def load_label_volume(case, cache_dir, cfg: dict | None = None, binary: bool = True,
                      as_bool: bool = False) -> np.ndarray:
    """读一例的 GT 掩膜，返回与 ``predict_volume`` 同轴序的 ``(H, W, Z)`` 数组。

    dataset 的 image/label 都直接使用 nibabel ``[:, :, z]``，推理画布也逐层写入
    ``[:, :, z]``，所以这里不能再转置前两维。旧版额外转置使方形病例的
    预测和 GT 在形状相同的情况下错位。
    ``binary=True`` 时按 ``>0`` 取前景（口径与 ``src.dataset`` 一致：**不要**写成 ``==2``，
    预处理后的掩膜已经二值化，只有 {0,1}）。

    ``cfg`` 只用于兼容调用签名（路径解析统一走 ``cache_dir``），可以不传。
    """
    del cfg   # 路径一律由 cache_dir 给出，保留参数只是为了调用处一致
    path = cache_file(cache_dir, "label", case)
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}（先跑 python scripts/preprocess.py，"
                                f"或确认 cache 位置与 --cache-dir 一致）")
    warn_compressed_cache(path, LOGGER)
    import nibabel as nib

    image = nib.load(str(path))
    array = np.asanyarray(image.dataobj)
    if array.ndim != 3:
        raise ValueError(f"case {case} 的 label 不是 3D：shape={array.shape}")
    volume = np.ascontiguousarray(array)  # 与 dataset[:, :, z] 和推理画布保持同一轴序
    if not binary:
        return volume
    if as_bool:
        return volume > 0
    return np.ascontiguousarray((volume > 0).astype(np.uint8))


# --------------------------------------------------------------------------------------
# 概率 → 标签
# --------------------------------------------------------------------------------------

def seg_prob_to_label(prob: np.ndarray, thr: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """肿瘤概率 → ``{0,1}`` 的 uint8 标签（``prob >= thr`` 记为前景）。

    阈值取 ``>=``：概率恰好等于 0.5 时算前景（与第 4 轮 ``src.metrics`` 的口径保持一致）。
    输入形状不限（整卷 ``(H,W,Z)`` 或单层 ``(H,W)`` 都行），输出同形状 uint8。
    """
    array = np.asarray(prob)
    if array.size and not np.isfinite(array).all():
        raise ValueError("概率图里出现 NaN/Inf（模型输出异常或 AMP 数值问题），先排查再二值化")
    return (array >= float(thr)).astype(np.uint8)


# --------------------------------------------------------------------------------------
# 整卷推理
# --------------------------------------------------------------------------------------

@torch.no_grad()
def predict_volume(model: torch.nn.Module, case, cache_dir, cfg: dict, device=None,
                   pad_to_multiple: int = 16, batch_slices: int = 8,
                   threshold: float | None = None, amp: str | None = None,
                   tumor_channel: int = TUMOR_CHANNEL) -> tuple:
    """对一个病例做整卷推理，返回 ``(prob, pred, meta)``。

    参数：
        model：``src.unet.UNet2D``（或任何 ``(B,C,H,W) → (B,C',H,W)`` 的网络，C 由
            ``data.z_context`` 决定：1 → 单层 2D，3 → 2.5D 三层窗）。
        case：病例号（int 或字符串数字）。
        cache_dir：``cache/`` 根目录（其下 ``image/`` 与 ``label/``）。
        cfg：总配置（读 ``data.target_hw`` / ``data.pad_align`` / ``data.z_context`` /
             ``eval.threshold`` / amp 口径）。
        device：推理设备；``None`` 时取模型参数所在设备（与模型不一致会直接报错，不静默搬运）。
        pad_to_multiple：与 ``model.pad_to_multiple`` 对齐；``data.target_hw`` 不是它的整数倍时
            只打印一条告警（模型前向内部会补边再裁回，结果仍然正确，只是多一层无谓开销）。
        batch_slices：一次前向几层（无梯度，可比训练 batch 大；显存不够时调小）。
        threshold：二值化阈值；``None`` 时取 ``cfg['eval']['threshold']``（默认 0.5）。
        amp：``'bf16'`` / ``'fp16'`` / ``'off'``；``None`` 时取 ``eval.amp`` 或 ``train.amp``。
        tumor_channel：肿瘤所在的输出通道（2 分类 = 1；三分类背景/肝脏/肿瘤 = 2）。

    返回：
        prob：``(H, W, Z)`` float32，肿瘤通道概率，**已裁回补边前的原始面内尺寸**；
        pred：``(H, W, Z)`` uint8 ∈ {0,1}，``prob >= threshold``；
        meta：dict（case / orig_hw / target_hw / pad_offset / n_slices / threshold / amp /
              device / batch_slices / seconds / tumor_channel / in_channels / prob_peak /
              pred_voxels）。

    实现要点：
        * 逐层读取复用 ``CTSliceDataset``（与训练完全同一套「2.5D 三层窗 → /65535 → 居中补边」
          口径），因此调用方**不需要**知道输入是 1 通道还是 3 通道——dataset 出来什么就喂什么。
          这也是「读盘口径只有一份」的价值：2.5D 落地时本文件的拼卷逻辑一行都不用改。
        * 概率先写进 ``(target_h, target_w, nz)`` 画布，**最后统一按 pad_offset 裁回**；
        * 模型暂时切到 ``eval()``，函数返回前恢复原来的 ``train/eval`` 状态。
    """
    torch_device = _module_device(model) if device is None else torch.device(device)
    parameter_device = _module_device(model)
    if not _same_device(torch_device, parameter_device):
        raise ValueError(f"模型在 {parameter_device} 上，但 predict_volume 收到 device={torch_device}；"
                         f"请先把模型 .to(device) 再推理")
    amp_name, amp_dtype = resolve_infer_amp(cfg, amp)
    use_amp = amp_dtype is not None and torch_device.type == "cuda"
    thr = resolve_threshold(cfg, threshold)
    tumor_channel = int(tumor_channel)
    if tumor_channel < 0:
        raise ValueError(f"tumor_channel 不能为负，收到 {tumor_channel}")
    batch_slices = max(1, int(batch_slices))
    started = time.perf_counter()

    dataset = CTSliceDataset([int(case)], cache_dir, "infer", cfg, augment=False)
    cache_case = int(case)
    if cache_case not in dataset.case_hw:
        raise ValueError(f"case {cache_case} 不在数据集里：{dataset.case_ids}")
    n_slices = int(dataset.case_n_slices[cache_case])
    orig_h, orig_w = (int(v) for v in dataset.case_hw[cache_case])
    top, left = (int(v) for v in dataset.pad_offset_of_case(cache_case))
    target_h, target_w = (int(v) for v in dataset.target_hw)
    if len(dataset) != n_slices:
        raise RuntimeError(f"单病例数据集的切片数 {len(dataset)} 与记录的 nz={n_slices} 不一致")

    mult = max(1, int(pad_to_multiple or 1))
    if target_h % mult or target_w % mult:
        LOGGER.warning("data.target_hw %s 不是 pad_to_multiple=%d 的整数倍：模型前向会内部补边再裁回，"
                       "结果不受影响，但会多一层无谓开销。", format_hw((target_h, target_w)), mult)

    canvas = np.zeros((target_h, target_w, n_slices), dtype=np.float32)
    was_training = bool(model.training)
    model.eval()
    try:
        for start in range(0, n_slices, batch_slices):
            stop = min(start + batch_slices, n_slices)
            samples = [dataset[i] for i in range(start, stop)]
            images = torch.stack([sample["image"] for sample in samples]).to(torch_device)
            with autocast_context(torch_device.type, amp_dtype, enabled=use_amp):
                logits = model(images)
            if logits.ndim != 4:
                raise RuntimeError(f"模型输出 shape={tuple(logits.shape)}，期望 (B,C,H,W)")
            channels = int(logits.shape[1])
            if channels == 1:
                probs = torch.sigmoid(logits.float())[:, 0]        # 单通道（sigmoid 口径）
            else:
                if tumor_channel >= channels:
                    raise ValueError(f"tumor_channel={tumor_channel} 超出输出通道数 {channels}"
                                     f"（2 分类用 1；三分类背景/肝脏/肿瘤用 2）")
                probs = torch.softmax(logits.float(), dim=1)[:, tumor_channel]
            # 张量 → numpy 一律走 np.asarray（**不要用 .numpy()**：本地假 torch 没有这个方法，
            # 而 np.asarray 对真 torch 的 CPU 张量走 __array__，两边都能用）
            block = np.asarray(probs.detach().to("cpu"))           # (b, target_h, target_w)
            for offset, sample in enumerate(samples):
                z = int(sample["z"])
                if block[offset].shape != (target_h, target_w):
                    raise RuntimeError(f"第 z={z} 层的概率图 shape={block[offset].shape}，"
                                       f"期望 {(target_h, target_w)}")
                canvas[:, :, z] = block[offset]
    finally:
        if was_training:
            model.train()

    prob = np.ascontiguousarray(canvas[top:top + orig_h, left:left + orig_w, :])
    if tuple(prob.shape[:2]) != (orig_h, orig_w):
        raise RuntimeError(f"裁回后的面内尺寸 {tuple(prob.shape[:2])} 与记录的原始尺寸 "
                           f"{(orig_h, orig_w)} 不一致（pad_offset={(top, left)}）")
    pred = seg_prob_to_label(prob, thr)
    meta = {
        "case": cache_case,
        "orig_hw": [orig_h, orig_w],
        "target_hw": [target_h, target_w],
        "pad_offset": [top, left],
        "n_slices": n_slices,
        "threshold": float(thr),
        "tumor_channel": tumor_channel,
        "in_channels": int(dataset.in_channels),   # 1 = 单层 2D；3 = 2.5D 三层窗
        "z_context": int(dataset.z_context),
        "amp": amp_name if use_amp else "off",
        "device": str(torch_device),
        "batch_slices": batch_slices,
        "prob_peak": round(float(prob.max()) if prob.size else 0.0, 6),
        "pred_voxels": int(pred.sum()),
        "seconds": round(time.perf_counter() - started, 3),
    }
    reset_open_cache()      # 单例推理结束后释放 nibabel 句柄（训练期每轮会建多次）
    return prob, pred, meta


if __name__ == "__main__":  # pragma: no cover - 直接运行本文件时给出正确入口
    raise SystemExit("本模块是库，不是脚本；训练期验证由 python -m src.train 调用，"
                     "第 4 轮的整卷评估入口是 python -m src.evaluate。")
