"""数据集与采样：按病例切片读 cache、2.5D 三层窗、统一补边到固定尺寸、**平衡采样**，并施加自实现的 2D 增强。

整体功能：
    1. ``CTSliceDataset`` —— 一个样本 = 一个病人的一层切片（标签）。``__init__`` 只读 **label 体素**
       （用于算 ``pos_flags`` 与索引），**不读 image 体素**；``__getitem__`` 用每 worker 一份的
       ``lru_cache`` 持有 nibabel 代理对象（``memmap=True``），取 **``[z-1, z, z+1]`` 三层**叠成
       通道维（``data.z_context``，默认 1 → 3 通道 2.5D 输入；标签仍取中心层 z），归一化后把
       image 与 label **居中补边到 ``data.target_hw``（默认 512×512）**。
    2. ``BalancedBatchSampler`` —— **正负定比的平衡采样器**：训练侧每批固定 ``n_pos`` 个含肿瘤层 +
       ``n_neg`` 个不含肿瘤层（由 ``data.pos_ratio_train`` 决定，默认 0.5 即各半），阳性池与阴性池
       各自轮转（``_CyclicPool``），因此阳性层会被重复采样、阴性层不再要求一轮覆盖。
       因为所有样本补边后都是同一个形状，``torch.stack`` 恒成立，**不需要按面内尺寸分桶**。
    3. ``make_train_loader`` / ``make_val_loader`` —— 训练侧平衡采样 + 增强；**验证侧顺序、无增强、
       保持原始阳性分布**（天然的不平衡分布正是要评估的对象，不能对它做平衡采样）。
    4. ``build_transforms`` —— 仅训练用的 2D 增强：翻转 / 旋转 90° / 仿射 / 随机 gamma / 高斯噪声，
       **全部自己实现，不依赖 MONAI**（原因见 ``make_augment_steps`` 的说明）。
       2.5D 下：几何增强对**所有通道施加同一套参数**（保持层间几何一致），强度增强只动**中心通道**。

为什么改成「统一补边」而不是「按尺寸分桶」：
    分桶要求同一 batch 内所有人的精确面内尺寸逐像素相同，于是采样器得先选桶再在桶内配额，
    小桶会被摊薄、还会出现大量全阴性 batch。改成统一补边到 512×512（512 是 16 的倍数，
    U-Net 4 级下采样不再需要内部 pad）之后，batch 形状恒为 ``(B,C,512,512)``，
    ``pad_multiple`` / ``bucket_key`` 这些概念整条链路都不需要了。
    代价是补边区域的白像素被浪费，且非 512 病例的补边区在增强后不再严格为 0；
    loss 是否忽略补边区域见 ``pad_to_target`` 的说明。

**病人级隔离（不可协商的约束，第 4 轮的平衡采样没有放松它）**：
    样本单位始终是 ``(case, z)``，而 ``case`` 只能来自本折 ``train`` / ``val`` 列表
    （``data/splits.json`` 的 ``split_level: "patient"``，5 例仅肝脏只进训练集）。
    平衡采样器**只对 ``pos_flags`` 的索引做重排与重复，绝不跨越病例边界**：
      * 阳性池 / 阴性池都是「本折 train 病例的切片索引」的子集；
      * ``[z-1, z, z+1]`` 三层窗在**本病例内部**索引，边界用端点复制（z=0 → 三层都是第 0 层），
        不会读到上一个/下一个病人的切片；
      * ``src/selfcheck_data.py`` 有一条「三层窗同属一个病人、且中心通道等于原单层读取」的
        回归自检专门钉这件事。

依赖：numpy / torch / nibabel（+ 可选 scipy 的连通域，不在本文件用）。
前后接口：上游是 ``scripts/preprocess.py`` 产出的 ``cache/image/<case>.nii``（uint16 归一化；
         **未压缩优先**，也兼容旧的 ``.nii.gz``）与 ``cache/label/<case>.nii``（uint8 二值）、
         ``data/splits.json``、``cache/cache_manifest.json``；
        下游是 ``src/selfcheck_data.py``（自检，先跑）与 ``src/train.py``（训练）。
用法：``python -m src.selfcheck_data`` 做数据形态自检；训练侧由 ``src.train`` 调用本模块的工厂函数。
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset

try:
    from src.utils import (
        cache_file,
        load_config,
        load_json,
        rel_to_root,
        resolve_path,
        setup_logger,
        warn_compressed_cache,
    )
except ModuleNotFoundError:  # pragma: no cover - 兜底：把仓库根塞进 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import (  # type: ignore
        cache_file,
        load_config,
        load_json,
        rel_to_root,
        resolve_path,
        setup_logger,
        warn_compressed_cache,
    )

LOGGER = setup_logger("dataset")

# --------------------------------------------------------------------------------------
# 常量与默认值（configs/default.yaml 的 data / train / model 节会覆盖，这里只是兜底）
# --------------------------------------------------------------------------------------

#: cache 影像的量化口径：uint16 存的是「HU 窗线性映射到 [0,1] 后按 1/65535 量化」的值，
#: 因此还原 [0,1] 只需除以 65535（与 scripts/preprocess.py、scripts/check_cache.py 同一口径）。
UINT16_SCALE = 65535.0

#: 统一补边的目标尺寸（面内）。512 是 16 的倍数：U-Net 4 级下采样 = 2^4，前向内部无需再 pad。
DEFAULT_TARGET_HW = (512, 512)

#: 补边时原图放在目标画布里的位置：``center`` = 居中（数据侧默认），``bottom-right`` = 右下补 0。
DEFAULT_PAD_ALIGN = "center"

#: 2.5D 上下文半径的默认值：1 = 取 [z-1, z, z+1] 三层当 3 通道输入；0 = 退回单层 2D。
DEFAULT_Z_CONTEXT = 1

DEFAULT_DATA_CFG: dict = {
    "index_cache_size": 8,        # 每 worker 保留的 nibabel 句柄数（lru_cache maxsize）
    "target_hw": list(DEFAULT_TARGET_HW),   # 面内统一补边到的目标尺寸（H, W）
    "pad_align": DEFAULT_PAD_ALIGN,         # center = 居中补边；bottom-right = 右下补 0
    "z_context": DEFAULT_Z_CONTEXT,         # 2.5D 上下文半径（见 window_index 的端点复制口径）
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": False,
    "val_batch_size": None,       # None = 用 train.batch_size；验证是顺序读，只影响速度
    "pos_ratio_train": 0.5,       # 训练侧平衡采样的一轮阳性槽位占比目标（0.5 = 正负各半）
    "min_pos_per_batch": 2,       # 每批阳性层数下限（batch 很小时防止比例取整成 0）
    "max_pos_repeat": 8,          # 单个阳性层一轮最多重复几次
    "epoch_samples": None,        # 一轮总槽位数预算；None = 用数据集切片数
    "pos_ratio_tolerance": 0.20,  # 自检用：实测比例偏离目标的告警阈值
    "verify_batch": True,         # collate 时校验 batch 不变量（形状/值域/label 取值）
    "augment": {                  # 仅训练；验证侧不做任何几何变换
        "flip_prob": 0.5,
        "rotate90_prob": 0.5,
        "affine_prob": 0.5,
        "rotation_deg": 15.0,
        "scale_range": [0.9, 1.1],
        "shift_frac": 0.1,
        "gamma_prob": 0.2,
        "gamma_range": [0.75, 1.33],
        "noise_prob": 0.2,
        "noise_std": 0.01,
    },
}


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------

def format_hw(hw: Sequence[int]) -> str:
    """把面内尺寸 ``(H, W)`` 渲染成 ``512x512`` 这样的字符串，便于日志与报告阅读。"""
    return "x".join(str(int(v)) for v in hw)


def resolve_target_hw(cfg: dict) -> tuple:
    """从配置里取面内目标尺寸，并对齐到 ``model.pad_to_multiple`` 的整数倍。

    配置项 ``data.target_hw`` 写的是**期望的**目标尺寸（默认 512×512）；这里再按
    ``model.pad_to_multiple`` 向上对齐，保证补边后的边长一定能被 U-Net 的 2^k 下采样整除
    （512 本身已是 16 的倍数，对齐是恒等操作；写小尺寸做调试时才有实际作用）。
    """
    raw = ((cfg or {}).get("data") or {}).get("target_hw") or list(DEFAULT_TARGET_HW)
    if isinstance(raw, (int, float)):
        raw = [int(raw), int(raw)]
    if len(raw) != 2:
        raise ValueError(f"data.target_hw 必须是 (H, W) 两个整数，收到 {raw!r}")
    mult = max(1, int(((cfg or {}).get("model") or {}).get("pad_to_multiple", 1) or 1))
    import math

    return tuple(int(math.ceil(int(v) / mult) * mult) for v in raw)


def pad_to_target(array: np.ndarray, target_hw: Sequence[int],
                  align: str = DEFAULT_PAD_ALIGN) -> tuple:
    """把 ``(H, W)`` 数组补边到 ``target_hw``，返回 ``(补边后的数组, (top, left))``。

    ``align="center"``（默认）时上下、左右各分一半的填充量，奇数余数放在右下；
    ``align="bottom-right"`` 时全部补在右侧与下侧。
    返回的 ``(top, left)`` 就是原图内容在补边画布里的左上角位置——第 4 轮整卷推理要把预测
    裁回原始面内尺寸，必须按它裁（居中补边时内容不在原点，写死 ``[:H, :W]`` 会错位）。

    补边值固定为 0（= 预处理后 HU 窗下界的背景值），label 也同样补 0。
    两个已知代价，写在这里供后续轮次判断：
      * 补边像素被白算，非 512 的 8 例按尺寸推算浪费约 41%、全部 25 例按例加权约 13%；
      * 增强在补边之后施加，所以非 512 病例的补边区在 gamma/噪声/仿射之后**不再严格为 0**。
        第 3 轮的 loss 可以考虑用 ``orig_hw``+``pad_offset`` 生成 ignore mask；第 6 轮再决定是否要改顺序。
    """
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"pad_to_target 只接受 (H, W) 切片，收到 shape={arr.shape}")
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    height, width = (int(arr.shape[0]), int(arr.shape[1]))
    if height > target_h or width > target_w:
        raise ValueError(f"切片尺寸 {(height, width)} 超过目标尺寸 {(target_h, target_w)}："
                         f"请调大 data.target_hw（补边只能放大，不能缩小）")
    if (height, width) == (target_h, target_w):
        return arr, (0, 0)
    top, left = pad_offset_of((height, width), (target_h, target_w), align)
    padded = np.zeros((target_h, target_w), dtype=arr.dtype)
    padded[top:top + height, left:left + width] = arr
    return padded, (int(top), int(left))


def pad_offset_of(orig_hw: Sequence[int], target_hw: Sequence[int],
                  align: str = DEFAULT_PAD_ALIGN) -> tuple:
    """只算「内容在补边画布里的左上角 ``(top, left)``」，不真的补边。

    补边与裁回必须用**同一套偏移**：``__getitem__`` 用它补边，``collate_samples`` 把它写进 batch，
    第 4 轮整卷推理按它把预测裁回原始面内尺寸。这里做成公开函数，避免两处各算一遍写歪
    （居中补边时内容不在原点，裁回写死 ``[:H, :W]`` 会整体错位 ``pad_h//2`` 像素）。
    """
    height, width = (int(orig_hw[0]), int(orig_hw[1]))
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    if height > target_h or width > target_w:
        raise ValueError(f"原始面内尺寸 {(height, width)} 超过目标尺寸 {(target_h, target_w)}")
    if str(align).lower() in ("bottom-right", "right-bottom", "br") or (height, width) == (target_h, target_w):
        return (0, 0)
    return ((target_h - height) // 2, (target_w - width) // 2)


def parse_cases(value: Any, splits: dict, split: str, fold: int | None = None) -> list:
    """把 ``cases`` 参数解析成 case(int) 列表。

    支持三种写法：
      * ``"train"`` / ``"val"`` / ``"all"``：从 ``splits`` 的折记录里取；``all`` = 各折 train∪val；
      * 形如 ``[31, 33, "35"]`` 的序列（字符串元素也会转成 int）；
      * 形如 ``"31,33,35"`` 的逗号串。

    ``val`` 必须在 ``fold`` 非空时使用——验证集是按折划分的，不给折号会静默取错病人分组。
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("train", "val", "all"):
            folds = (splits or {}).get("folds") or []
            if not folds:
                raise ValueError("splits 里没有 folds，无法按 split 取病例；请检查 data/splits.json")
            if key == "all":
                picked: set = set()
                for rec in folds:
                    picked.update(int(c) for c in rec.get("train", []))
                    picked.update(int(c) for c in rec.get("val", []))
                return sorted(picked)
            if fold is None:
                if key == "train":  # 不指定折时的 train = 各折 train 的并集（等价于全部病人）
                    picked = set()
                    for rec in folds:
                        picked.update(int(c) for c in rec.get("train", []))
                    return sorted(picked)
                raise ValueError("cases='val' 必须同时给出 fold（验证集是按折划分的）")
            rec = next((r for r in folds if int(r.get("fold", -1)) == int(fold)), None)
            if rec is None:
                raise ValueError(f"splits 里没有 fold={fold}（可选：{[r.get('fold') for r in folds]}）")
            return sorted(int(c) for c in rec.get(key, []))
        if not key:
            raise ValueError("cases 不能是空字符串")
        return sorted(int(part) for part in key.replace(" ", "").split(",") if part)

    if isinstance(value, (list, tuple, set)):
        return sorted(int(c) for c in value)
    raise TypeError(f"cases 只能是 'train'/'val'/'all'、逗号串或整数序列，收到 {type(value).__name__}")


def data_config(cfg: dict) -> dict:
    """从总配置里取出 data 节并与默认值合并，顺带校验 ``pad_align`` 与归一化 ``target_hw``。

    历史提醒：早期版本这里还会把 ``model.pad_to_multiple`` 回退成 ``data.pad_multiple`` 供**分桶**用；
    分桶已经删除（现在统一补边到 ``target_hw``），**不要再把对齐值当分桶键**。
    """
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_DATA_CFG.items()}
    user = (cfg or {}).get("data") or {}
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    align = str(merged.get("pad_align", DEFAULT_PAD_ALIGN)).lower()
    if align not in ("center", "bottom-right", "right-bottom", "br"):
        raise ValueError(f"data.pad_align 只支持 'center' / 'bottom-right'，收到 {merged.get('pad_align')!r}")
    merged["pad_align"] = "bottom-right" if align in ("bottom-right", "right-bottom", "br") else "center"
    aligned = resolve_target_hw({"data": {"target_hw": merged.get("target_hw")},
                                 "model": (cfg or {}).get("model") or {}})
    merged["target_hw"] = list(aligned)
    merged["z_context"] = resolve_z_context(merged)
    ratio = float(merged.get("pos_ratio_train", 0.5))
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"data.pos_ratio_train 必须落在 [0,1]，收到 {merged.get('pos_ratio_train')!r}")
    if int(merged.get("min_pos_per_batch", 2)) < 1:
        raise ValueError(f"data.min_pos_per_batch 必须 >= 1，收到 {merged.get('min_pos_per_batch')!r}")
    if int(merged.get("max_pos_repeat", 8)) < 1:
        raise ValueError(f"data.max_pos_repeat 必须 >= 1，收到 {merged.get('max_pos_repeat')!r}")
    return merged


def resolve_z_context(data_cfg: dict | None = None) -> int:
    """取 2.5D 上下文半径（``data.z_context``），负数直接报错。

    半径 r 表示一个样本 = 中心层 z 加上左右各 r 层，叠成 ``2r+1`` 个通道；``r=0`` 就是旧的单层 2D。
    这个值**同时决定 dataset 的输出通道数与 ``model.in_channels``**，两处必须一致——
    ``window_channels`` / ``src.train.check_prerequisites`` 会分别校验。
    """
    value = int(((data_cfg or {}).get("z_context", DEFAULT_Z_CONTEXT)) or 0)
    if value < 0:
        raise ValueError(f"data.z_context 不能为负，收到 {value}")
    return value


def window_channels(z_context: int) -> int:
    """2.5D 窗口的通道数 = ``2 × z_context + 1``（r=1 → 3 通道，r=0 → 1 通道）。"""
    return 2 * int(z_context) + 1


def window_index(z: int, n_slices: int, z_context: int) -> list:
    """返回中心层 ``z`` 的 2.5D 窗口在**同一病例内**的层号列表，长度 ``2r+1``。

    口径（**病人级隔离在这里落地**）：
      * 只做**端点复制**：``z=0`` 时左邻取 0（三层都是第 0 层），``z=nz-1`` 时右邻取 ``nz-1``；
      * **绝不跨病人**：越界时复制本病例的端点层，而不是去读上一个/下一个 case 的切片
        （相邻病人之间没有任何空间连续性，跨过去就是把别的病人的解剖当上下文）；
      * 不做 z 方向插值、不做整卷 padding 后索引：``nz`` 各例不同（74–488），端点复制是唯一
        与病例边界无关的确定性口径。

    例：``z=0, nz=135, r=1 → [0, 0, 1]``；``z=134 → [133, 134, 134]``；``z=5 → [4, 5, 6]``。
    """
    z = int(z)
    n_slices = int(n_slices)
    radius = int(z_context)
    if n_slices < 1:
        raise ValueError(f"n_slices 必须 >= 1，收到 {n_slices}")
    if not 0 <= z < n_slices:
        raise ValueError(f"z={z} 超出 [0, {n_slices - 1}]（样本索引必须落在本病例内）")
    if radius <= 0:
        return [z]
    return [min(max(z + delta, 0), n_slices - 1) for delta in range(-radius, radius + 1)]


def stable_seed(*parts: Any) -> int:
    """由若干值派生一个**跨进程稳定**的 31 位种子。

    不要用 Python 内置 ``hash()``：str/bytes 的 hash 受 ``PYTHONHASHSEED`` 影响，
    跨进程会变，用它做种子会让「换一次运行就换一条采样序列」，训练不可复现。
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2 ** 31 - 1)


# --------------------------------------------------------------------------------------
# 每 worker 一份的 nibabel 句柄缓存
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _open_nii(path_str: str):
    """打开一个 NIfTI 文件并返回 nibabel 图像对象（``mmap=True``）。

    用 ``lru_cache`` 保证「同一 worker 内、最近用过的 N 个 case 只打开一次」：
    DataLoader 有 num_workers 个子进程，每个子进程各持一份缓存，
    因此同时打开的文件句柄数 ≈ num_workers × maxsize，而不是「切片数」个。
    ``mmap=True`` 让 ``dataobj`` 直接映射磁盘，取一层切片不会把整卷读进内存。
    """
    import nibabel as nib

    return nib.load(str(path_str), mmap=True)


def reset_open_cache() -> None:
    """清空句柄缓存（换 cache 目录、或自检里想量「冷读」耗时时用）。"""
    _open_nii.cache_clear()


def configure_open_cache(maxsize: int) -> None:
    """按配置设置句柄缓存的容量（在 DataLoader 起来之前调用；运行中改需要先 ``reset_open_cache``）。"""
    global _open_nii
    _open_nii = lru_cache(maxsize=max(1, int(maxsize)))(_open_nii.__wrapped__)  # type: ignore[attr-defined]


def release_mmap(image) -> None:
    """尽力释放 nibabel ``mmap=True`` 打开的磁盘映射。

    索引阶段只需要「每层有没有肿瘤」，读完整卷 label 后应立刻关掉映射：
    否则 25 例的 memmap 会一直挂到 GC，既占虚拟内存也占文件句柄
    （``ArrayProxy`` 不实现上下文管理器，因此这里显式关底层 ``mmap``）。
    """
    dataobj = getattr(image, "dataobj", None)
    mmap_obj = getattr(dataobj, "_mmap", None)
    if mmap_obj is not None and hasattr(mmap_obj, "close"):
        try:
            mmap_obj.close()
        except (ValueError, OSError):  # 已经关了 / 平台不支持，忽略即可
            pass


@contextmanager
def open_volume(path: str | Path):
    """``with open_volume(p) as img:`` —— 打开一个 NIfTI 并在退出时释放 mmap。

    适合「整卷读一次就用完」的场景（如逐层曲线、清单重建），避免句柄长期占用：
    ``__getitem__`` 那种高频小读不要用它，那里应当依赖 ``_open_nii`` 的 lru_cache。
    """
    img = _open_nii(str(path))
    try:
        yield img
    finally:
        release_mmap(img)
        _open_nii.cache_clear()


@lru_cache(maxsize=256)
def _read_header(path_str: str) -> tuple:
    """只读 header，返回 ``(image_shape, image_dtype, spacing)``；不触碰体素数据。

    nibabel 的 ``load`` 是惰性的，``shape`` / ``get_data_dtype()`` / ``header.get_zooms()``
    都不需要解压体素，因此初始化阶段扫 25 例 header 是毫秒级、且不占内存。
    """
    import nibabel as nib

    img = nib.load(str(path_str))
    spacing = tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])
    return tuple(int(s) for s in img.shape), str(img.get_data_dtype()), spacing


# --------------------------------------------------------------------------------------
# 仅训练的增强：2D 仿射（自行实现，避免 MONAI 各版本仿射 API 的差异）
# --------------------------------------------------------------------------------------

class GridAffine2D:
    """在单张 2D 数组上做「绕中心 旋转 + 缩放 + 平移」的仿射重采样。

    为什么不用 ``monai.transforms.RandAffine``：
      * 它的 ``spatial_size`` 必填、``padding_mode`` 的形状在各版本间有差异，写死容易在远程直接报错；
      * 它的 ``mode`` 在老版本只接受 ``str | int``，传 ``{"image": ..., "label": ...}`` 有风险。
    本类只做「一层切片 + 同一套几何同时作用于 image 与 label」，用 ``grid_sample`` 实现，
    行为确定、与 MONAI 版本无关，也天然满足「label 最近邻、image 双线性」。

    坐标约定：
      * ``rotate_deg > 0``：图像内容在屏幕上**逆时针**旋转（与 MONAI 的 ``RandRotate`` / ``affine_grid`` 一致）；
      * ``scale``：**采样**坐标的缩放，因此 ``scale > 1`` 时输出只放大源图的一小部分 → 内容变大；
        ``scale < 1`` 时内容变小（就是 MONAI ``RandAffine.scale_params`` 的口径）。
      * ``shift_xy``：内容在面内的**像素**位移（``(dx, dy)``，内容向右/向下为正，与 MONAI
        ``RandAffine.translate_range`` 的「正 = 图像往坐标轴正方向平移」一致）；
        单位是像素而不是「比例」——角点对齐的归一化坐标会随尺寸变化，用比例很容易写出
        「只有 1/8 像素」这种看不出来的错（本文件第一版就踩过：把像素位移直接塞进归一化坐标）。
      * 采样落在图外的位置按 0 填充（背景 = 0，正是 label 越界处应取的值）。
    """

    def __init__(self, rotate_deg: float = 0.0, scale: float = 1.0,
                 shift_xy: Sequence[float] = (0.0, 0.0)) -> None:
        self.rotate_deg = float(rotate_deg)
        self.scale = float(scale)
        self.shift_xy = (float(shift_xy[0]), float(shift_xy[1]))
        #: 比例形式的位移（相对面内边长）；仅在还不知道切片尺寸时使用，见 RandAffineSlice2D
        self.shift_frac: tuple = (0.0, 0.0)
        self._theta: torch.Tensor | None = None
        self._theta_hw: tuple | None = None

    def theta(self, height: int, width: int) -> torch.Tensor:
        """返回 ``(1, 2, 3)`` 的仿射矩阵，作用在「以图像中心为原点的**像素**坐标」上。

        矩阵整体作用在像素坐标上，最后才由 ``grid()`` 统一归一化到 [-1, 1]，
        所以第 3 列（平移）必须也是**像素**单位：采样点整体右移 dx，内容就整体左移 dx，
        因此要让内容右移 dx 要写 ``-dx``。

        （第一版在这里又除了一次 ``(W-1)/2``，等于把位移除以 31.5 再在 ``grid()`` 里除一次，
        最终的位移只有预期的 1/31.5，肉眼完全看不出来——本地自检里
        「shift_xy=+8 px 让内容水平移动 +8 像素」这条断言就是为了钉死它。）
        """
        height, width = int(height), int(width)
        if self._theta is None or self._theta_hw != (height, width):
            import math

            rad = math.radians(self.rotate_deg)
            cos, sin = math.cos(rad), math.sin(rad)
            sx = sy = self.scale
            self._theta = torch.tensor(
                [[cos * sx, -sin * sy, -self.shift_xy[0]],
                 [sin * sx, cos * sy, -self.shift_xy[1]]],
                dtype=torch.float32,
            ).unsqueeze(0)
            self._theta_hw = (height, width)
        return self._theta

    def grid(self, height: int, width: int) -> torch.Tensor:
        """构造 ``grid_sample`` 需要的采样网格 ``(1, H, W, 2)``，取值范围 [-1, 1]。

        ``grid[..., 0]`` 对应 x（列方向）、``grid[..., 1]`` 对应 y（行方向），与 ``theta`` 的两行一致；
        该网格表示「输出像素 (y, x) 该去输入的哪个位置取值」，即 theta 的**逆变换**（与
        ``affine_grid`` 的约定相同：正角度 = 内容逆时针转）。
        ``theta`` 在像素坐标下计算，这里统一除以 ``(边-1)/2`` 归一化到 [-1, 1]——
        整个矩阵（含平移）都要一起归一化，不能再提前除一次。
        """
        ys = torch.arange(int(height), dtype=torch.float32)
        xs = torch.arange(int(width), dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # 像素坐标 → 以图像中心为原点
        grid_x = grid_x - (width - 1) / 2.0
        grid_y = grid_y - (height - 1) / 2.0
        base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        theta = self.theta(height, width)
        out = torch.einsum("bxyi,boi->bxyo", base, theta[..., :2]) + theta[:, None, None, :, 2]
        # 中心化像素坐标 → grid_sample 的 [-1, 1] 归一化坐标
        out[..., 0] = out[..., 0] / max(1e-6, (width - 1) / 2.0)
        out[..., 1] = out[..., 1] / max(1e-6, (height - 1) / 2.0)
        return out

    def warp(self, array: np.ndarray, mode: str = "bilinear") -> np.ndarray:
        """对 ``(H, W)`` 或 ``(C, H, W)`` 数组做一次仿射采样，返回同形状 float32 数组。

        ``C>1``（2.5D 三层窗）时所有通道共用**同一个**采样网格 ⇒ 层与层之间的几何关系不会被
        各通道独立的随机变换破坏（这一点是 2.5D 的正确性前提：如果三个通道各转一个角度，
        模型看到的就是「三张对不齐的切片」，学到的层间上下文是假的）。
        """
        src = np.asarray(array, dtype=np.float32)
        if src.ndim == 2:
            src = src[None]                       # (H,W) → (1,H,W)
        if src.ndim != 3:
            raise ValueError(f"GridAffine2D 只接受 (H,W) 或 (C,H,W) 数组，收到 shape={src.shape}")
        height, width = int(src.shape[1]), int(src.shape[2])
        tensor = torch.from_numpy(np.ascontiguousarray(src))[None]   # (1, C, H, W)
        grid = self.grid(height, width)                              # (1, H, W, 2)
        out = F.grid_sample(tensor, grid, mode=mode, padding_mode="zeros", align_corners=True)
        squeezed = out[0].numpy().astype(np.float32, copy=False)
        return squeezed[0] if np.asarray(array).ndim == 2 else squeezed


def _is_multi_channel(image: np.ndarray, z_context: int) -> bool:
    """判断一块 image 是不是 2.5D 的 ``(C,H,W)``。

    **必须同时看形状与 ``z_context``**：单层 2D 时 image 的形状恰好是 ``(1,H,W)``，
    只按 ``ndim==3`` 判断会把它误当成「1 个通道的多通道块」，进而把整形操作写歪
    （本地自检就抓到过一次：``z_context=0`` 时把 image 当成 3D 处理，label 被当成通道）。
    """
    array = np.asarray(image)
    return bool(int(z_context) > 0 and array.ndim == 3 and int(array.shape[0]) == 2 * int(z_context) + 1)


def _center_channel(image: np.ndarray) -> np.ndarray:
    """取 ``(C,H,W)`` 的中心通道（``(H,W)``）；单通道输入原样返回。

    ``C`` 一定是奇数（``2×z_context+1``），中心下标就是 ``C//2``：对 3 通道就是第 1 个通道
    （即层 z），这正是标签所在的层。写死 ``[1]`` 会在 ``z_context≠1`` 时静默取错层。
    """
    array = np.asarray(image)
    if array.ndim != 3:
        return array
    return array[int(array.shape[0]) // 2]


def _rand_affine(rng: random.Random, aug: dict) -> GridAffine2D:
    """按配置随机抽一组仿射参数：旋转 ±``rotation_deg``、缩放 ``scale_range``、平移 ±``shift_frac``。

    平移在这里只抽「相对面内边长的比例」（存进 ``shift_frac``），真正换算成像素要等到
    ``RandAffineSlice2D.__call__`` 拿到切片尺寸之后——比例必须乘上边长才是像素位移。
    """
    rot = rng.uniform(-1.0, 1.0) * float(aug.get("rotation_deg", 15.0))
    lo, hi = (float(x) for x in (aug.get("scale_range") or [0.9, 1.1]))
    scale = rng.uniform(lo, hi)
    frac = float(aug.get("shift_frac", 0.1))
    affine = GridAffine2D(rot, scale, (0.0, 0.0))
    affine.shift_frac = (rng.uniform(-frac, frac), rng.uniform(-frac, frac))
    return affine


class RandAffineSlice2D:
    """按概率对 image/label 施加**同一个** ``GridAffine2D``（旋转 + 缩放 + 平移）。

    同一套几何必须同时作用到 image 与 label，否则掩膜会与影像错位——这是本步自己实现而不是
    拼两个 transform 的原因：随机参数只抽一次，然后逐 key 复用。
    image 用双线性、label 用最近邻（后者避免插值出 0/1 之外的值）。
    """

    def __init__(self, prob: float = 0.5, aug: dict | None = None,
                 keys: Sequence[str] = ("image", "label"),
                 rng: random.Random | None = None) -> None:
        self.prob = float(prob)
        self.aug = dict(aug or {})
        self.keys = tuple(keys)
        self.rng = rng or random
        self.affine: GridAffine2D | None = None
        self.last_params: dict | None = None   # 便于自检断言

    def randomize(self, rng: random.Random | None = None) -> None:
        """抽一组参数；返回是否施加变换（``super().randomize()`` 的等价物，便于自检单独调用）。"""
        rng = rng or self.rng
        if rng.random() >= self.prob:
            self.affine = None
            return
        self.affine = _rand_affine(rng, self.aug)
        self.last_params = {"rotate_deg": self.affine.rotate_deg,
                            "scale": self.affine.scale,
                            "shift_frac": self.affine.shift_frac}

    def __call__(self, data: dict) -> dict:
        self.randomize()
        if self.affine is None:
            return data
        out = dict(data)
        for key in self.keys:
            if key not in out:
                continue
            arr = np.asarray(out[key])
            mode = "nearest" if key == "label" else "bilinear"
            height, width = int(arr.shape[0]), int(arr.shape[1])
            # 比例 → 像素（×边长）；image 与 label 尺寸相同，因此两者拿到同一套几何
            frac_x, frac_y = self.affine.shift_frac
            warped = GridAffine2D(self.affine.rotate_deg, self.affine.scale,
                                  (frac_x * width, frac_y * height)).warp(arr, mode=mode)
            if key == "label":
                warped = np.rint(warped).astype(np.uint8)
            out[key] = warped
        return out


class FlipSlice2D:
    """按概率沿面内某个轴翻转（``axis=0`` 翻行、``axis=1`` 翻列）；image 与 label 同步。"""

    def __init__(self, prob: float = 0.5, axis: int = 0, keys: Sequence[str] = ("image", "label"),
                 rng: random.Random | None = None) -> None:
        if int(axis) not in (0, 1):
            raise ValueError(f"FlipSlice2D 只支持 axis=0/1（面内两轴），收到 {axis}")
        self.prob = float(prob)
        self.axis = int(axis)
        self.keys = tuple(keys)
        self.rng = rng or random

    def __call__(self, data: dict) -> dict:
        if self.rng.random() >= self.prob:
            return data
        out = dict(data)
        for key in self.keys:
            if key in out:
                out[key] = np.flip(np.asarray(out[key]), axis=self.axis).copy()
        return out


class Rotate90Slice2D:
    """按概率把切片整 90° 旋转 ``k∈{0,1,2,3}`` 次（无插值，label 不会被糊）。

    ``max_k=1`` 时固定转 90°；``k = 0`` 那种「抽到但等于不变」的情况在 ``max_k>1`` 时才会出现，
    与 MONAI ``RandRotate90d`` 的语义一致。
    """

    def __init__(self, prob: float = 0.5, max_k: int = 3, keys: Sequence[str] = ("image", "label"),
                 rng: random.Random | None = None) -> None:
        if int(max_k) < 1:
            raise ValueError(f"max_k 必须 >= 1，收到 {max_k}")
        self.prob = float(prob)
        self.max_k = int(max_k)
        self.keys = tuple(keys)
        self.rng = rng or random

    def __call__(self, data: dict) -> dict:
        if self.rng.random() >= self.prob:
            return data
        k = self.rng.randint(1, self.max_k) if self.max_k > 1 else 1
        out = dict(data)
        for key in self.keys:
            if key in out:
                out[key] = np.rot90(np.asarray(out[key]), k).copy()
        return out


class GammaSlice2D:
    """按概率做随机 gamma 校正：``img ** gamma``，``gamma < 1`` 提亮、``> 1`` 压暗。

    这是「随机直方图/对比度扰动」的最简形式（MONAI 的 ``RandHistogramShift`` 用控制点做分段线性
    映射，本质也是单调的强度重排）。要求输入已经是 ``[0,1]``，否则幂运算会发散——预处理后的
    cache 正好是 ``[0,1]``，所以这里直接乘幂即可，且**保序**（不会把亮暗关系翻转）。
    只作用于 image；**输入恒为 2D 的 ``(H,W)``**（2.5D 的上下文通道不在本步骤里，
    见 ``CTSliceDataset.__getitem__`` 的口径说明）。
    """

    def __init__(self, prob: float = 0.2, gamma_range: Sequence[float] = (0.75, 1.33),
                 keys: Sequence[str] = ("image",), rng: random.Random | None = None) -> None:
        lo, hi = (float(x) for x in gamma_range)
        if lo <= 0 or hi <= 0 or lo > hi:
            raise ValueError(f"gamma_range 必须是正数区间且 lo<=hi，收到 {gamma_range}")
        self.prob = float(prob)
        self.gamma_range = (lo, hi)
        self.keys = tuple(keys)
        self.rng = rng or random
        self.last_gamma: float | None = None   # 便于自检断言

    def __call__(self, data: dict) -> dict:
        if self.rng.random() >= self.prob:
            return data
        gamma = self.rng.uniform(*self.gamma_range)
        self.last_gamma = gamma
        out = dict(data)
        for key in self.keys:
            if key in out:
                img = np.clip(np.asarray(out[key], dtype=np.float32), 0.0, 1.0)
                out[key] = np.power(img, gamma, dtype=np.float32)
        return out


class GaussianNoiseSlice2D:
    """按概率加零均值高斯噪声。

    ``std`` 是噪声强度的**上界**：每次从 ``U(0, std)`` 抽一个 ``sigma``（与 MONAI 的
    ``sample_std=True`` 默认行为一致），这样噪声强度本身也随机。为了可复现，用的是
    ``random.Random`` 抽 sigma、``np.random.default_rng(seed)`` 抽噪声（种子由同一个 rng 派生）。
    只作用于 image，最后夹回 ``[0,1]``；**输入恒为 2D 的 ``(H,W)``**（同 ``GammaSlice2D``）。
    """

    def __init__(self, prob: float = 0.2, std: float = 0.01, keys: Sequence[str] = ("image",),
                 rng: random.Random | None = None) -> None:
        if float(std) < 0:
            raise ValueError(f"noise std 必须 >= 0，收到 {std}")
        self.prob = float(prob)
        self.std = float(std)
        self.keys = tuple(keys)
        self.rng = rng or random
        self.last_sigma: float | None = None   # 便于自检断言

    def __call__(self, data: dict) -> dict:
        if self.rng.random() >= self.prob:
            return data
        sigma = self.rng.uniform(0.0, self.std)
        self.last_sigma = sigma
        noise_rng = np.random.default_rng(self.rng.randrange(2 ** 31))
        out = dict(data)
        for key in self.keys:
            if key in out:
                img = np.asarray(out[key], dtype=np.float32)
                noise = noise_rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
                out[key] = np.clip(img + noise, 0.0, 1.0)
        return out


class ClampImageToUnit:
    """把 image 夹回 [0,1]：gamma / 高斯噪声 / 双线性插值都可能把值推出值域。"""

    def __call__(self, data: dict) -> dict:
        if "image" not in data:
            return data
        out = dict(data)
        out["image"] = np.clip(np.asarray(out["image"], dtype=np.float32), 0.0, 1.0)
        return out


class BinarizeLabel:
    """把 label 重新二值化：最近邻插值在边界上可能取到非 0/1 的值。

    **只碰 label**：绝不能写成 ``out["image"] = (image > 0.5)`` —— 那会把 image 也二值化
    （历史兜底写法在这一层很容易被复制粘贴进来，所以这里显式分开写）。
    """

    def __call__(self, data: dict) -> dict:
        if "label" not in data:
            return data
        out = dict(data)
        out["label"] = (np.asarray(out["label"]) > 0.5).astype(np.uint8)
        return out


def make_augment_steps(cfg: dict, seed: int | None = None) -> list:
    """返回训练增强的步骤列表（**唯一事实来源**：``build_transforms`` 与自检脚本共用）。

    **全部自实现，不依赖 MONAI**。原因：MONAI 的增强接口在 array 版 / 字典版之间有两套签名
    （``RandFlip`` 只吃数组、``RandFlipd`` 才吃 dict 且 ``keys`` 是必需参数），参数名还跨版本变过
    （``axis`` → ``spatial_axis``、``shift_range`` 在 1.6 已不存在），我们在这一层连炸过两次；
    而这几步增强本身只是十几行 numpy。自实现之后这一整类「接口/版本不匹配」故障消失，
    且随机性由单一 ``random.Random(seed)`` 驱动，可离线逐项验证。

    顺序即施加顺序：
      1. ``FlipSlice2D`` 两个轴各一次（翻行、翻列）；
      2. ``Rotate90Slice2D(max_k=1)``：整 90° 旋转，label 无插值伪影；
      3. ``RandAffineSlice2D``：旋转 ±15°、缩放 0.9–1.1、平移 ±10%（image 双线性 / label 最近邻）；
      4. ``GammaSlice2D``（随机 gamma 校正，等价于单调的强度重排）+ ``GaussianNoiseSlice2D``；
      5. 末尾把 image 夹回 [0,1]、把 label 重新二值化成 {0,1}。

    只做面内变换、不做弹性形变：逐层独立施加形变会破坏 z 方向一致性（同一病人的相邻层被施以
    不同形变，病灶边界会抖），而逐层形变本身对 2D 基线没有收益。

    **谁负责 2.5D 的通道一致性**：增强流水线本身**只吃 2D 的 ``(H,W)``**（``CTSliceDataset.__getitem__``
    先对中心层做增强、再叠上未增强的邻居层）。因此这里没有"多通道分支"：
      * 几何增强天然对所有通道同步（它们共享同一个增强后的中心层 + 真实邻居层）；
      * 强度增强（4）只作用于被监督的中心层，相邻层保持真实灰度，不会被 gamma/噪声改成
        「与中心层不一致的伪影」；
      * label 始终是 ``(H,W)`` 的中心层掩膜，**不带通道维**——``BinarizeLabel`` 只碰 label。

    ``seed`` 为 None 时用 ``train.seed``；同一个 seed 得到同一串增强参数（但每个样本的随机数
    仍按抽样顺序推进，因此各样本的增强互不相同）。
    """
    aug = dict(data_config(cfg).get("augment") or {})
    if seed is None:
        seed = int(((cfg or {}).get("train") or {}).get("seed", 42))
    rng = random.Random(int(seed) + 104729)   # 与采样器的种子错开一个素数
    keys = ("image", "label")                 # 几何增强同步作用于两者（多通道时一次作用到整块）
    img_only = ("image",)

    return [
        FlipSlice2D(prob=float(aug.get("flip_prob", 0.5)), axis=0, keys=keys, rng=rng),
        FlipSlice2D(prob=float(aug.get("flip_prob", 0.5)), axis=1, keys=keys, rng=rng),
        Rotate90Slice2D(prob=float(aug.get("rotate90_prob", 0.5)), max_k=1, keys=keys, rng=rng),
        RandAffineSlice2D(prob=float(aug.get("affine_prob", 0.5)), aug=aug, keys=keys, rng=rng),
        GammaSlice2D(prob=float(aug.get("gamma_prob", 0.2)),
                     gamma_range=aug.get("gamma_range", [0.75, 1.33]), keys=img_only, rng=rng),
        GaussianNoiseSlice2D(prob=float(aug.get("noise_prob", 0.2)),
                             std=float(aug.get("noise_std", 0.01)), keys=img_only, rng=rng),
        ClampImageToUnit(),
        BinarizeLabel(),
    ]


class ComposeSteps:
    """把若干「dict → dict」的增强步骤串起来（等价于 MONAI 的 ``Compose``，但只做这一件事）。"""

    def __init__(self, steps: Sequence) -> None:
        self.transforms = list(steps)

    def __call__(self, data: dict) -> dict:
        for step in self.transforms:
            data = step(data)
        return data

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self) -> str:
        return f"ComposeSteps({[type(s).__name__ for s in self.transforms]})"


def build_transforms(cfg: dict, train: bool, seed: int | None = None):
    """构建训练增强流水线；``train=False`` 时返回 ``None``（验证侧不做任何几何变换）。

    返回一个可调用对象：输入 ``{"image": (H,W) float32, "label": (H,W) uint8}``，返回同结构的 dict。
    步骤见 ``make_augment_steps``。
    """
    if not train:
        return None
    return ComposeSteps(make_augment_steps(cfg, seed=seed))


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

class CTSliceDataset(Dataset):
    """一个样本 = 一个病人（case）的一层切片；``index[i] = (case, z)``。

    读取策略（避免 DataLoader 里开出上千个文件句柄）：
      * ``__init__``：只 ``nib.load`` 读 header + **label 体素**（算 ``pos_flags`` 与索引），
        **不读 image 体素**；25 例 label 合计几十 MB，读完一例即可释放；
      * ``__getitem__``：走每 worker 一份的 ``lru_cache`` 拿 nibabel 代理对象（``mmap=True``），
        只取 ``[:, :, z]`` 一层；image 除以 65535 还原 [0,1]，label 取 >0 得 {0,1}，
        然后**居中补边到 ``data.target_hw``**，最后（仅训练）施加增强。

    参数：
        cases：``"train"`` / ``"val"`` / ``"all"``、逗号串或整数序列（见 ``parse_cases``）。
        cache_dir：``cache/`` 根目录（其下 image/ 与 label/）。
        split：写进样本字典的分组名（``"train"`` / ``"val"`` / ``"all"``），仅用于追踪与日志。
        cfg：总配置字典（``load_config`` 的返回值）。
        augment：是否施加训练增强；``False`` 时不做任何几何变换（仍然补边）。
        debug：True 时打印逐 case 的索引统计（切片数、含肿瘤切片数、原始/补边后尺寸）。

    ``cases="train"`` 时使用 ``splits`` 里该 fold 的训练集；未给 ``fold`` 则取各折 train 的并集
    （等价于全部 25 例，仅用于自检与冒烟测试）。
    """

    def __init__(self, cases: Any, cache_dir: str | Path, split: str, cfg: dict,
                 augment: bool, debug: bool = False, fold: int | None = None) -> None:
        self.cfg = cfg or {}
        self.data_cfg = data_config(self.cfg)
        self.cache_dir = resolve_path(cache_dir)
        self.split = str(split)
        self.augment = bool(augment)
        self.debug = bool(debug)
        self.fold = fold
        self.target_hw = tuple(int(v) for v in self.data_cfg.get("target_hw", DEFAULT_TARGET_HW))
        self.pad_align = str(self.data_cfg.get("pad_align", DEFAULT_PAD_ALIGN))
        self.index_cache_size = int(self.data_cfg.get("index_cache_size", 8))
        # 2.5D：z_context=1 → 每个样本取 [z-1, z, z+1] 三层，输出通道数 = 3
        self.z_context = resolve_z_context(self.data_cfg)
        self.in_channels = window_channels(self.z_context)
        # 注意：下面两个目录只用来「拼路径给人看」；真正取文件一律走 src.utils.cache_file
        # （.nii 优先、兼容 .nii.gz）或本类的 image_path()/label_path()，
        # 不要再写 f"{case}.nii.gz" 这种把扩展名写死的拼接。
        self.image_dir = self.cache_dir / "image"
        self.label_dir = self.cache_dir / "label"

        splits = load_json((self.cfg.get("paths") or {}).get("splits", "data/splits.json"), default={}) or {}
        self.case_ids: list = parse_cases(cases, splits, self.split, fold=fold)

        # 增强的随机源按 (seed, fold, split) 派生：同一折每次构建得到同一串增强参数
        aug_seed = stable_seed(int(((self.cfg.get("train") or {}).get("seed", 42))),
                               -1 if self.fold is None else int(self.fold), self.split)
        self.transforms = build_transforms(self.cfg, train=self.augment, seed=aug_seed)

        # 清单只用于交叉核对（可选；manifest 不入库，重建靠 scripts/fetch_manifest.py）
        manifest_path = (self.cfg.get("paths") or {}).get("cache_manifest", "cache/cache_manifest.json")
        self.manifest: dict = {}
        if manifest_path:
            doc = load_json(manifest_path, default={}) or {}
            self.manifest = {int(r["case"]): r for r in doc.get("cases", []) if "case" in r}

        self.index: list = []            # [(case:int, z:int), ...]
        self.pos_flags = np.zeros(0, dtype=bool)   # 与 index 一一对应：该切片是否含肿瘤
        self.case_hw: dict = {}          # case -> (H, W) 原始面内尺寸（**补边前**）
        self.case_pad_offset: dict = {}  # case -> (top, left) 内容在补边画布里的左上角
        self.case_n_slices: dict = {}    # case -> nz
        self.case_pos_slices: dict = {}  # case -> 含肿瘤切片数
        self.case_paths: dict = {}       # case -> (image 路径, label 路径)，已按 .nii > .nii.gz 解析
        self.image_dtype: str | None = None
        self._build_index()

    # ---------------- 索引构建 ----------------

    def _build_index(self) -> None:
        """逐 case 读 label 体素，建 ``index`` / ``pos_flags``，并做一致性自检。

        读的是 label（几 MB/例）而不是 image（几十 MB/例）：索引只需要「哪些层有肿瘤」。
        文件路径统一走 ``src.utils.cache_file``：**未压缩 ``.nii`` 优先**（可 mmap，逐层读是页缓存读），
        没有才退回 ``.nii.gz``（会打一次「压缩缓存很慢」的告警）。解析出的路径记进 ``self.case_paths``，
        ``__getitem__`` 直接取用，不再每次拼字符串 + ``exists()``。
        """
        problems: list = []
        missing = [c for c in self.case_ids
                   if not cache_file(self.cache_dir, "image", c).exists()
                   or not cache_file(self.cache_dir, "label", c).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} 个病例在 {rel_to_root(self.cache_dir)} 下缺 image/label：{missing}；"
                f"请先跑 python scripts/preprocess.py（或确认 case 名单与 cache 一致）")

        for case in self.case_ids:
            image_path = cache_file(self.cache_dir, "image", case)
            label_path = cache_file(self.cache_dir, "label", case)
            warn_compressed_cache(image_path, LOGGER)
            warn_compressed_cache(label_path, LOGGER)
            img_shape, img_dtype, spacing = _read_header(str(image_path))
            lab_img = _open_nii(str(label_path))
            lab_arr = np.asanyarray(lab_img.dataobj)
            if lab_arr.ndim != 3:
                problems.append(f"case {case} 的 label 不是 3D：shape={lab_arr.shape}")
                release_mmap(lab_img)
                continue

            # 轴序实测（scripts/probe_axis.py）：nibabel 读回 (nx, ny, nz)，切片轴在最后一维
            nx, ny, nz = (int(lab_arr.shape[0]), int(lab_arr.shape[1]), int(lab_arr.shape[2]))
            height, width = ny, nx

            if tuple(img_shape) != (nx, ny, nz):
                problems.append(f"case {case}：image shape {img_shape} != label shape "
                                f"{(nx, ny, nz)}（缓存不成对或被写坏）")
            if tuple(round(float(s), 3) for s in spacing) != (1.0, 1.0, 1.0):
                problems.append(f"case {case}：spacing {spacing} 不是 1mm（cache 口径被破坏）")
            if height > self.target_hw[0] or width > self.target_hw[1]:
                problems.append(f"case {case}：面内尺寸 {(height, width)} 超过 data.target_hw "
                                f"{self.target_hw}（补边只能放大，不能缩小；请调大 target_hw）")
                release_mmap(lab_img)
                continue

            # 掩膜二值口径：cache 里只含 {0,1}，这里用 >0 取前景（不要写成 ==2）
            binary = (lab_arr > 0).astype(np.uint8)
            per_slice = binary.sum(axis=(0, 1))      # 对 (x, y) 求和 → 长度 nz
            pos_flags = per_slice > 0
            n_pos = int(np.count_nonzero(pos_flags))
            if n_pos > nz:
                problems.append(f"case {case}：含肿瘤切片数 {n_pos} 超过总切片数 {nz}，per-slice 轴用错了")
            if int(per_slice.size) != nz:
                problems.append(f"case {case}：per-slice 长度 {int(per_slice.size)} 与切片数 {nz} 不一致")

            rec = self.manifest.get(int(case))
            if rec:
                recorded_shape = [int(s) for s in rec.get("nib_shape_xyz", [])]
                if recorded_shape and recorded_shape != [nx, ny, nz]:
                    problems.append(f"case {case}：shape 与 cache_manifest 不符 "
                                    f"{(nx, ny, nz)} vs {recorded_shape}")
                if int(rec.get("tumor_slices", -1)) != n_pos:
                    problems.append(f"case {case}：含肿瘤切片数 {n_pos} 与清单 "
                                    f"{rec.get('tumor_slices')} 不符（清单可能过期，跑 fetch_manifest.py 刷新）")

            if self.image_dtype is None:
                self.image_dtype = str(img_dtype)
            elif str(img_dtype) != self.image_dtype:
                problems.append(f"case {case}：影像 dtype={img_dtype} 与其它病例的 {self.image_dtype} "
                                f"不一致（cache 口径不统一，请重跑 preprocess.py）")

            # 内容在补边画布里的左上角（居中补边时不是 (0,0)）：第 4 轮裁回原始尺寸必须用它
            offset = pad_offset_of((height, width), self.target_hw, self.pad_align)

            self.index.extend((int(case), int(z)) for z in range(nz))
            self.pos_flags = np.concatenate(
                [self.pos_flags, pos_flags.astype(bool)]) if self.pos_flags.size else pos_flags.astype(bool)
            self.case_hw[int(case)] = (height, width)
            self.case_pad_offset[int(case)] = offset
            self.case_n_slices[int(case)] = nz
            self.case_pos_slices[int(case)] = n_pos
            self.case_paths[int(case)] = (image_path, label_path)

            # 索引只需要「每层有没有肿瘤」，读完这一例就把映射丢掉（不留在 lru_cache 里）：
            # 真正按层读取发生在 __getitem__，届时会重新打开。
            per_slice = binary = lab_arr = None
            release_mmap(lab_img)
            _open_nii.cache_clear()

            if self.debug:
                LOGGER.info("  case %d：nz=%d 含肿瘤层=%d 原始面内=%s → 补边到 %s（偏移 %s）",
                            int(case), nz, n_pos, format_hw((height, width)),
                            format_hw(self.target_hw), offset)

        if problems:
            for item in problems:
                LOGGER.error("  - %s", item)
            raise RuntimeError(f"数据集初始化失败：发现 {len(problems)} 个问题（见上面的 - 行）")
        if not self.index:
            raise ValueError(f"split={self.split} 下没有任何切片，请检查 cases={self.case_ids!r}")
        if int(self.pos_flags.size) != len(self.index):
            raise RuntimeError(f"pos_flags 长度 {int(self.pos_flags.size)} 与索引长度 {len(self.index)} 不一致")

    # ---------------- 统计与描述 ----------------

    @property
    def n_positive(self) -> int:
        """含肿瘤切片总数。"""
        return int(self.pos_flags.sum())

    @property
    def n_negative(self) -> int:
        """不含肿瘤切片总数。"""
        return int(len(self.index) - self.n_positive)

    @property
    def pos_ratio(self) -> float:
        """全部切片里含肿瘤的比例（第 1 轮实测整体约 12.8%）。"""
        return float(self.n_positive / max(1, len(self.index)))

    def inplane_stats(self) -> dict:
        """逐种原始面内尺寸的统计：切片数、含肿瘤切片数、病例数、补边填充占比。

        ``pad_fraction`` = 该尺寸补到 ``target_hw`` 后被浪费的像素占比
        （``1 - H*W/(target_h*target_w)``），用于估算补边代价与后续是否需要忽略补边区域。
        """
        stats: dict = {}
        target_h, target_w = (int(self.target_hw[0]), int(self.target_hw[1]))
        for case in self.case_ids:
            height, width = self.case_hw[int(case)]
            rec = stats.setdefault((height, width), {"n_slices": 0, "n_pos": 0, "cases": []})
            rec["n_slices"] += int(self.case_n_slices[int(case)])
            rec["n_pos"] += int(self.case_pos_slices[int(case)])
            rec["cases"].append(int(case))
        for (height, width), rec in stats.items():
            rec["n_cases"] = len(rec["cases"])
            rec["n_neg"] = int(rec["n_slices"] - rec["n_pos"])
            rec["pad_fraction"] = round(1.0 - (height * width) / float(target_h * target_w), 4)
            rec["cases"] = sorted(rec["cases"])
        return stats

    def describe(self) -> str:
        """返回多行描述：split、病例、切片数、阳性比例、逐种原始面内尺寸的明细。"""
        lines = [
            f"split={self.split}"
            + (f" fold={self.fold}" if self.fold is not None else "")
            + f"：{len(self.case_ids)} 例、{len(self.index)} 层切片、含肿瘤 {self.n_positive} 层"
              f"（{self.pos_ratio:.4f}）、原始面内尺寸 {len(self.inplane_stats())} 种 →"
              f" 统一补边到 {format_hw(self.target_hw)}（{self.pad_align}）、影像 dtype={self.image_dtype}",
        ]
        for (height, width), stat in sorted(self.inplane_stats().items()):
            ratio = stat["n_pos"] / max(1, stat["n_slices"])
            lines.append(f"  原始 {format_hw((height, width)):>9}：切片 {stat['n_slices']:5d}"
                         f"（含肿瘤 {stat['n_pos']:4d} = {ratio:6.2%}）"
                         f"  病例 {stat['n_cases']:2d} 例 {stat['cases']}"
                         f"  补边占比 {stat['pad_fraction']:.2%}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.index)

    def pad_offset_of_case(self, case: int) -> tuple:
        """该病例的内容在补边画布里的左上角 ``(top, left)``（居中补边时不是 (0,0)）。

        第 4 轮整卷推理把预测裁回原始面内尺寸时用它：``pred[top:top+H, left:left+W]``。
        """
        return self.case_pad_offset[int(case)]

    def image_path(self, case: int) -> Path:
        """该病例影像在 cache 里的实际路径（``.nii`` 优先，没有才是 ``.nii.gz``）。"""
        return self.case_paths[int(case)][0]

    def label_path(self, case: int) -> Path:
        """该病例掩膜在 cache 里的实际路径（``.nii`` 优先，没有才是 ``.nii.gz``）。"""
        return self.case_paths[int(case)][1]

    # ---------------- 取样本 ----------------

    def _to_unit_range(self, arr: np.ndarray) -> np.ndarray:
        """按 cache 口径把影像还原到 [0,1]。

        ``uint16``：存的是归一化值 ×65535，除以 65535 即可（1 步长 ≈ 0.0305 HU）；
        其它 dtype（``preprocess.image_out_dtype=float32``）：cache 里是 HU，按 ``preprocess.hu_clip``
        线性映射，保证下游拿到的永远是 [0,1]。
        """
        out = np.asarray(arr, dtype=np.float32)
        if str(self.image_dtype) == "uint16":
            return out / UINT16_SCALE
        hu_lo, hu_hi = (float(x) for x in
                        ((self.cfg.get("preprocess") or {}).get("hu_clip", [-1000.0, 1000.0])))
        out = (out - hu_lo) / max(1e-6, hu_hi - hu_lo)
        return np.clip(out, 0.0, 1.0)

    def sliding_window(self, case: int, i: int) -> dict:
        """返回样本 ``i`` 的 2.5D 窗口信息（层号、是否端点复制），**不读影像**。

        供自检脚本核对「三层窗只在本病例内索引」这条病人级隔离约束：``z_first`` / ``z_last``
        与 ``clamped`` 一起就能证明越界时是复制本病例端点、而不是去读邻居病例的切片。
        """
        case = int(case)
        z = int(self.index[i][1])
        nz = int(self.case_n_slices[case])
        window = window_index(z, nz, self.z_context)
        return {
            "case": case,
            "index": int(i),
            "z": z,
            "nz": nz,
            "window": [int(v) for v in window],
            "z_first": int(window[0]),
            "z_last": int(window[-1]),
            "clamped": bool(window[0] != z - self.z_context or window[-1] != z + self.z_context),
        }

    def window_planes(self, case: int, z: int) -> np.ndarray:
        """读一层 → ``/65535`` 还原 [0,1] → 居中补边，返回 ``(H,W) float32``（未增强）。

        自检脚本用它取「某个邻居层单独读出来的样子」，与 ``__getitem__`` 里窗口通道逐像素比对；
        ``__getitem__`` 叠邻居层时也调用它（口径只有这一处，不会分叉）。
        """
        case = int(case)
        z = int(z)
        image_path = self.case_paths[case][0]
        nz = int(self.case_n_slices[case])
        if not 0 <= z < nz:
            raise ValueError(f"case {case} 的第 {z} 层越界（nz={nz}）")
        image_proxy = np.asanyarray(_open_nii(str(image_path)).dataobj)
        plane = np.ascontiguousarray(self._to_unit_range(image_proxy[:, :, z]))
        padded, _ = pad_to_target(plane, self.target_hw, self.pad_align)
        return np.ascontiguousarray(padded, dtype=np.float32)

    def __getitem__(self, i: int) -> dict:
        """取一个样本：``(C, target_h, target_w)`` 的 2.5D 窗口 + ``(target_h, target_w)`` 的中心层标签。

        ``C = 2×data.z_context+1``（默认 3 = ``[z-1, z, z+1]``；``z_context=0`` 时 C=1 = 旧口径）。

        **执行顺序（这个顺序是有讲究的，别调换）**：
          1. 读中心层 z 的影像与标签 → ``/65535`` 还原 [0,1] / 二值化 → **居中补边**到 target_hw；
          2. 增强流水线只吃**中心层的 2D 平面**（``(H,W)`` image + ``(H,W)`` label）：几何增强
             （翻转/旋转/仿射）与强度增强（gamma/噪声）都按原口径作用在这一层上；
          3. 把增强后的中心层与**未增强的** ``z±r`` 邻居层叠成 ``(C,H,W)`` 通道维，邻居层在本病例
             内索引（越界用端点复制，见 ``window_index``）。

        为什么先把增强做完再叠层（而不是把 ``(C,H,W)`` 整块交给增强）：
          * 强度增强（gamma / 高斯噪声）从定义上只该改**被监督的那一层** —— 相邻层是真实的解剖
            上下文，跟中心层一起提亮/加噪等于手工制造「上下文与中心层不一致」的伪影；
          * 几何增强天然只作用在中心层上，再由步骤 3 的叠层保证**所有通道共享同一套几何**
            （把增强后的中心层铺到每个通道是 2.5D 最简、最不容易写歪的做法：不存在"三通道被
            不同角度旋转"的可能，也不必给每个增强步骤加通道维分支）；
          * 单层 2D（``z_context=0``）时步骤 3 就是加一个长度为 1 的通道维，与旧口径逐位一致
            （``(1,H,W)``），所以切换 ``z_context`` 不需要两套增强代码。
        """
        case, z = self.index[i]
        image_path, label_path = self.case_paths[case]
        image_proxy = np.asanyarray(_open_nii(str(image_path)).dataobj)
        label_proxy = np.asanyarray(_open_nii(str(label_path)).dataobj)

        # 1) 中心层：a[:, :, z] 的形状是 (ny, nx) = (H, W)（未压缩 .nii 走 mmap，是页缓存读）
        plane = np.ascontiguousarray(self._to_unit_range(image_proxy[:, :, z]))
        label = np.ascontiguousarray((np.asarray(label_proxy[:, :, z]) > 0).astype(np.uint8))
        plane, offset = pad_to_target(plane, self.target_hw, self.pad_align)
        label, label_offset = pad_to_target(label, self.target_hw, self.pad_align)
        if tuple(offset) != tuple(label_offset):
            raise RuntimeError(f"case {case} z={z}：image 与 label 的补边偏移不一致 "
                               f"{tuple(offset)} vs {tuple(label_offset)}")

        # 2) 只对中心层做增强（输入恒为 2D，与旧口径一致）
        if self.transforms is not None:
            out = self.transforms({"image": np.asarray(plane, dtype=np.float32), "label": label})
            plane = np.ascontiguousarray(np.asarray(out["image"], dtype=np.float32))
            label = np.ascontiguousarray(np.asarray(out["label"]).astype(np.uint8))
            if label.ndim == 3 and label.shape[0] == 1:   # 兜底：万一某步给 label 加了通道维
                label = label[0]
            if plane.ndim != 2 or tuple(plane.shape) != tuple(label.shape):
                raise RuntimeError(f"增强后 image {tuple(plane.shape)} 与 label "
                                   f"{tuple(label.shape)} 形状不一致（应都是 2D (H,W)）")

        # 3) 叠 2.5D 窗口：中心层用增强后的，邻居层用未增强的真实切片（同一病例内索引）
        nz = int(self.case_n_slices[case])
        window = window_index(z, nz, self.z_context)
        if self.in_channels == 1:
            image = plane[None]                                     # (1, H, W)
        else:
            planes = []
            for zz in window:
                planes.append(plane if int(zz) == int(z) else self.window_planes(case, int(zz)))
            image = np.stack(planes, axis=0)                        # (C, H, W)
            image = np.clip(image, 0.0, 1.0)                        # 邻居是 [0,1]；中心层可能被增强推到界外

        height, width = (int(image.shape[1]), int(image.shape[2]))
        if (height, width) != tuple(self.target_hw):
            raise RuntimeError(f"case {case} z={z}：补边/增强后切片形状 {(height, width)} 不是 "
                               f"data.target_hw {tuple(self.target_hw)}（补边或增强出了错）")
        if int(image.shape[0]) != int(self.in_channels):
            raise RuntimeError(f"case {case} z={z}：窗口通道数 {int(image.shape[0])} != "
                               f"2×z_context+1 = {self.in_channels}")
        return {
            # (C, target_h, target_w) float32 ∈ [0,1]；C = 2×z_context+1（默认 3 = [z-1,z,z+1]）
            "image": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            "label": torch.from_numpy(label.astype(np.int64)),    # (target_h, target_w) int64 ∈ {0,1}
            "case": str(case),
            "z": int(z),
            "window": [int(v) for v in window],                   # 2.5D 三层窗的层号（自检用）
            "orig_hw": (int(self.case_hw[case][0]), int(self.case_hw[case][1])),
            "pad_offset": (int(offset[0]), int(offset[1])),
        }


# --------------------------------------------------------------------------------------
# 组 batch（collate）
# --------------------------------------------------------------------------------------

def collate_samples(samples: Sequence[dict], verify: bool = True,
                    expect_channels: int | None = None) -> dict:
    """把若干样本拼成一个 batch，并定义**明确的 batch 契约**（训练与自检都按它取值）：

        image      : Tensor (B, C, 512, 512) float32；**C = 2×data.z_context+1**（默认 3 = [z-1,z,z+1]）
        label      : Tensor (B, 512, 512)    int64 ∈ {0,1}（**只监督中心层 z**）
        case       : list[str]，长度 B（每个样本来自哪个病人）
        z          : list[int]，长度 B（每个样本的中心层号）
        window     : list[list[int]]，长度 B（2.5D 三层窗的层号，**全部在本病例内**）
        orig_hw    : list[tuple[int,int]]，长度 B（每个样本**补边前**的面内尺寸）
        pad_offset : list[tuple[int,int]]，长度 B（内容在补边画布里的左上角 (top, left)）

    用 ``data.target_hw`` 之外的尺寸时上面两处的 512 相应替换；契约的结构不变。

    为什么不直接用 DataLoader 的默认 collate：PyTorch 默认会把每样本一个 tuple 的 ``orig_hw``
    做**转置**，得到 ``[(h1,h2,...), (w1,w2,...)]``——看上去像 ``(H,W)`` 但其实是两行列表，
    解包就会炸（远程就是这样炸的：``too many values to unpack``）。显式定义契约后，
    ``orig_hw`` / ``pad_offset`` 都是长度 B 的 tuple 列表，含义不再依赖默认行为。

    ``verify=True``（默认）时顺带校验 batch 的不变量：所有样本形状一致、
    ``label`` 取值 ⊂ {0,1}、``image`` 值域 ⊂ [0,1]、张量形状与 ``orig_hw`` 自洽、
    ``image`` 通道数等于 ``expect_channels``（给了才判；由 ``data.z_context`` 推出来）。
    这些校验的开销可以忽略（只是比较几个整数），但能在训练早期抓住「buffered/memmap 复用、
    增强把尺寸改坏、2.5D 窗口取错层、collate 拼错」这类最难查的问题。

    注意（历史教训）：``image`` 是 ``(B,C,H,W)``、``label`` 是 ``(B,H,W)``，
    **比较空间维时要错开一个通道维**（``image.shape[2:]`` vs ``label.shape[1:]``），
    写成 ``label.shape[2:]`` 会得到空 tuple 而恒真/恒假。
    """
    if not samples:
        raise ValueError("collate_samples 收到空 batch")
    shapes = {tuple(int(s) for s in sample["image"].shape) for sample in samples}
    if len(shapes) != 1:
        raise RuntimeError(f"batch 内样本形状不一致 {sorted(shapes)}——数据集本应把每个样本都补边到"
                           f"同一个 data.target_hw，出现这个说明补边或增强改变了尺寸")
    image = torch.stack([s["image"] for s in samples])
    label = torch.stack([s["label"] for s in samples])
    height, width = int(image.shape[2]), int(image.shape[3])
    batch = {
        "image": image,
        "label": label,
        "case": [str(s["case"]) for s in samples],
        "z": [int(s["z"]) for s in samples],
        "window": [[int(v) for v in s.get("window", [s["z"]])] for s in samples],
        "orig_hw": [(int(s["orig_hw"][0]), int(s["orig_hw"][1])) for s in samples],
        # 补边偏移不在样本里携带，而是由 orig_hw + 张量形状**当场推导**：
        # 这样它一定与 batch 的形状自洽（样本里再存一份就有写歪的可能）。
        "pad_offset": [pad_offset_of(s["orig_hw"], (height, width)) for s in samples],
    }
    if verify:
        if tuple(label.shape[1:]) != (height, width):
            raise RuntimeError(f"batch 内 image {tuple(image.shape)} 与 label {tuple(label.shape)} 尺寸不符"
                               f"（注意 image 是 (B,C,H,W)、label 是 (B,H,W)，比较时要错开通道维）")
        if expect_channels is not None and int(image.shape[1]) != int(expect_channels):
            raise RuntimeError(f"batch 内 image 通道数是 {int(image.shape[1])}，与 data.z_context 推出的 "
                               f"{int(expect_channels)} 不一致（2.5D 窗口或 model.in_channels 配错了）")
        if any(hw[0] > height or hw[1] > width for hw in batch["orig_hw"]):
            raise RuntimeError(f"batch 内样本的 orig_hw 超过张量形状 {(height, width)}："
                               f"{batch['orig_hw']}（补边只能放大，不能缩小）")
        if int(label.numel()) and (int(label.min()) < 0 or int(label.max()) > 1):
            raise RuntimeError(f"label 取值超出 {{0,1}}：min={int(label.min())} max={int(label.max())}")
        if not torch.isfinite(image).all():
            raise RuntimeError("image 里出现非有限值（NaN/Inf）")
        if float(image.min()) < -1e-6 or float(image.max()) > 1.0 + 1e-6:
            raise RuntimeError(f"image 值域 [{float(image.min()):.6f}, {float(image.max()):.6f}] 超出 [0,1]")
    return batch


# --------------------------------------------------------------------------------------
# 单一池 + 阳性层均摊的 BatchSampler
# --------------------------------------------------------------------------------------

class _CyclicPool:
    """无放回轮转池：优先吐出**本轮还没抽过**的元素，抽完了就重排再抽。

    用途：在一个池（阳性层 / 阴性层）内「尽量不重复地」抽样本。只要``总抽取次数 >= 池大小``，
    就保证**池内每个元素至少出现一次**（未抽过的先出，抽完才开始重复），这是「一轮覆盖全部切片」
    这条保证的实现方式。
    池内的洗牌只用传入的 ``rng``（由 seed+epoch 派生），**不碰全局 random**，
    这样采样顺序与全局随机状态、worker 数都无关，同一配置下逐轮可复现。
    """

    def __init__(self, items: Sequence[int], rng: random.Random) -> None:
        self._items = [int(v) for v in items]
        self._rng = rng
        # 未被抽过的（初始为全部，顺序随机）；抽走即从 pool 里移除
        self._pool: list = list(self._items)
        self._rng.shuffle(self._pool)
        self._cursor = 0
        # 抽过的元素：供「重复抽取」阶段使用（保证重复也均匀、且不集中在少数几层）
        self._seen: list = []
        self.drawn = 0

    def __len__(self) -> int:
        return len(self._items)

    def _pop_unseen(self) -> int:
        if self._cursor >= len(self._pool):
            self._seen = list(self._items)
            self._rng.shuffle(self._seen)
            self._pool = self._seen
            self._cursor = 0
        value = self._pool[self._cursor]
        self._cursor += 1
        self.drawn += 1
        return value

    def take(self, k: int) -> list:
        """连续取 ``k`` 个样本；池子不够大时按需重复（重复的元素也均匀分布）。"""
        k = int(k)
        if k <= 0 or not self._items:
            return []
        if k > len(self._items):
            LOGGER.error("池内样本不足：需要 %d 个，池里只有 %d 个（本轮会有重复；"
                         "请检查 batch_size 与 data.pos_ratio_train 是否与数据集规模匹配）",
                         k, len(self._items))
        return [self._pop_unseen() for _ in range(k)]


# --------------------------------------------------------------------------------------
# 平衡采样（BalancedBatchSampler）
# --------------------------------------------------------------------------------------

def slice_usage(drawn: Sequence[int]) -> tuple:
    """统计一串样本索引的重复情况，返回 ``(最小出现次数, 最大出现次数)``。

    用途：自检里核对「阳性层重复上限」「阴性层覆盖率」这两条新口径——
    旧的均摊采样器保证的是「每个阳性层恰好一次」，平衡采样器改成了「重复但不超过上限」。
    """
    indices = [int(v) for v in drawn]
    if not indices:
        return 0, 0
    counts: dict = {}
    for value in indices:
        counts[value] = counts.get(value, 0) + 1
    values = list(counts.values())
    return int(min(values)), int(max(values))


def plan_balanced_slots(n_pos_slices: int, n_neg_slices: int, batch_size: int,
                        pos_ratio: float = 0.5, min_pos_per_batch: int = 2,
                        max_pos_repeat: int = 8, epoch_samples: int | None = None) -> dict:
    """算「一轮 epoch 的槽位计划」：每批几个阳性/阴性、一共几个 batch。

    口径（**固定预算、改比例**，第 4 轮拍板）：
      1. 每批阳性数 ``n_pos = clamp(round(batch_size × pos_ratio), min_pos_per_batch, batch_size-1)``，
         每批阴性数 ``n_neg = batch_size - n_pos``。**每批比例是恒定的**（不像旧口径那样只是"上限"）。
      2. 预算 ``S``：``epoch_samples`` 给了就用它，否则用**数据集切片数**（≈ 旧口径的一轮样本量，
         epoch 墙钟基本不变）。
      3. 阳性槽位总数 ``= min(round(S × pos_ratio), n_pos_slices × max_pos_repeat)``。
         第二项是**重复上限**：阳性层只有 ``n_pos_slices`` 个，比例要得越高就得把同一层反复喂进去，
         上限防止过拟合到少数层。上限真的卡住时比例会低于目标（日志会显式打印）。
      4. ``B = max(ceil(阳性槽位 / n_pos), ceil(阴性槽位 / n_neg))``，再取 ``max(1, ...)``；
         总槽位 ``= B × batch_size``。
      5. 兜底：``n_pos_slices == 0``（理论上不该发生，验证侧才可能）时退化成「全部阴性」；
         阳性层太少、连"每批一个"都填不满时把 ``n_pos`` 压到能在 ``B`` 个 batch 里摊开的程度，
         并置 ``pool_limited=True``（自检会按它放宽阳性数断言，不误报）。

    返回的全部是**计划值**（不含随机性），自检可以拿它逐条等号比对。
    """
    n_pos_slices = int(n_pos_slices)
    n_neg_slices = int(n_neg_slices)
    batch_size = int(batch_size)
    if batch_size < 2:
        raise ValueError(f"batch_size 必须 >= 2（要同时放正负样本），收到 {batch_size}")
    if not 0.0 <= float(pos_ratio) <= 1.0:
        raise ValueError(f"pos_ratio 必须落在 [0,1]，收到 {pos_ratio}")
    budget = int(epoch_samples) if epoch_samples else int(n_pos_slices + n_neg_slices)
    budget = max(1, budget)

    def clamp_pos(value: int) -> int:
        return max(1, min(int(value), batch_size - 1))

    n_pos = clamp_pos(int(round(batch_size * float(pos_ratio))))
    n_neg = batch_size - n_pos
    pos_slots = min(int(round(budget * float(pos_ratio))), n_pos_slices * int(max_pos_repeat))
    neg_slots = max(0, int(round(budget * (1.0 - float(pos_ratio)))))
    by_pos = int(math.ceil(pos_slots / n_pos)) if pos_slots > 0 else 0
    by_neg = int(math.ceil(neg_slots / n_neg)) if (neg_slots > 0 and n_neg > 0) else 0
    batches = max(1, by_pos, by_neg)
    # 阳性层的"预算上限"（pos_slots = min(以比例算出的需求, P×max_pos_repeat)）**可能低于批数**：
    # 那就把批数压到阳性能支撑的水平，保证「每个 batch 至少一个阳性层」——这是平衡采样的底线，
    # 宁可轮短一点（一个 epoch 少跑几批），也不要回到"很多批一个阳性都没有"的旧问题。
    # 反过来，正数够多时千万不要动批数：批数是被阴性槽位顶上去的，砍掉它会连阴性一起砍，
    # 比例反而超过目标（本地自检抓到过这个反向错误）。
    if n_pos_slices > 0 and pos_slots < batches:
        batches = max(1, pos_slots)

    pool_limited = False
    if n_pos_slices <= 0:
        n_pos = 0
        n_neg = batch_size
        pool_limited = True
    elif n_pos_slices * n_pos < batches:
        # 连「每个 batch 分到一个阳性层」都做不到：把 n_pos 压到 data 能支撑的值
        n_pos = clamp_pos(max(1, n_pos_slices // batches))
        n_neg = batch_size - n_pos
        pool_limited = True

    slots = batches * batch_size
    # 实际槽位数按「批数 × 每批配比」算，而不是按预算算：阴性槽位可能把 batch 数顶上去，
    # 于是阳性槽位也跟着变多。这里的数就是 __iter__ 真正会产出的量，文档/日志一律以它为准。
    pos_total = batches * n_pos
    neg_total = slots - pos_total
    # 重复上限的最终裁决：批数已经被阴性槽位定死时，"每批 n_pos 个"仍可能超过 max_pos_repeat，
    # 此时把每批阳性数压到上限允许的水平（比例随之下降）——否则 max_pos_repeat 只是打印出来好看。
    repeat_cap = n_pos_slices * int(max_pos_repeat)
    if n_pos_slices > 0 and pos_total > repeat_cap:
        n_pos = int(repeat_cap // batches)
        n_neg = batch_size - n_pos
        pool_limited = pool_limited or n_pos < int(round(batch_size * float(pos_ratio)))
        pos_total = batches * n_pos
        neg_total = slots - pos_total
    return {
        "batch_size": batch_size,
        "n_pos_per_batch": int(n_pos),
        "n_neg_per_batch": int(n_neg),
        "batches": int(batches),
        "slots": int(slots),
        "pos_slots": int(pos_total),
        "neg_slots": int(neg_total),
        "n_pos_slices": n_pos_slices,
        "n_neg_slices": n_neg_slices,
        "budget": int(budget),
        "pos_ratio_target": float(pos_ratio),
        "pos_repeat": (pos_total / n_pos_slices) if n_pos_slices else 0.0,
        "neg_coverage": (min(1.0, neg_total / n_neg_slices) if n_neg_slices else 0.0),
        "actual_pos_ratio": (pos_total / slots) if slots else 0.0,
        "repeat_capped": bool(n_pos_slices > 0 and pos_total >= repeat_cap and n_pos < batch_size - 1
                              and pos_total / max(1, n_pos_slices) >= float(max_pos_repeat) - 1e-9),
        "pool_limited": bool(pool_limited),
    }


class BalancedBatchSampler(BatchSampler):
    """**正负定比的平衡采样器**（第 4 轮取代旧的 ``ProportionalBatchSampler``）。

    为什么换掉旧口径：旧的 `ProportionalBatchSampler` 保证的是「每个阳性层一轮恰好出现一次 +
    阴性层尽量不重复」，于是 batch 内的阳性比例被"阳性层总数 / 一轮 batch 数"死死钉住 ——
    fold 0 上 `bs=16` 实际每批只有 1~2 个阳性（12%），与整体 13.8% 几乎一样，**等于没有过采样**。
    第 3 轮首折正式训练的结果就是这个代价：`dice` 项长期横盘在 0.90（= 肿瘤 soft Dice≈0.1）、
    `ce` 掉到 0.009、验证整卷 Dice 恒 ≈0.000 —— 模型塌缩到全预测背景
    （完整分析见 docs/preprocess_notes.md 7.2 与 8.1）。

    现在的口径（每批比例恒定）：
      * 每批 ``n_pos`` 个含肿瘤层 + ``n_neg`` 个不含肿瘤层（默认 ``pos_ratio_train=0.5`` → 各半）；
      * 阳性层池、阴性层池各自用 ``_CyclicPool`` 轮转：**阳性层会被重复采样**
        （重复次数上限 ``data.max_pos_repeat``），阴性层不再要求一轮覆盖；
      * 一轮的 batch 数由「槽位预算」算出（见 ``plan_balanced_slots``），
        默认预算 = 数据集切片数 ⇒ **epoch 墙钟与旧口径基本一致**，只是每批的阳性从 1~2 个变成 ``n_pos`` 个。

    保证（``src/selfcheck_data.py`` 逐条验证）：
      * 每个 batch 的大小恒为 ``batch_size``，且**阳性层数恰好是 ``n_pos``**（不全阴性 batch）；
      * 一轮内每个阳性层的出现次数 **<= max_pos_repeat**；
      * 采样顺序只由 ``(train.seed, epoch, 病例集合)`` 决定（sha256 派生，不用内置 ``hash()``），
        与 ``num_workers`` 无关；**同一个 epoch 可复现、相邻 epoch 不同**。

    **病人级隔离**（不可协商）：阳性池 / 阴性池都只是 ``dataset`` 的切片索引，而 ``dataset`` 只由
    本折 ``train`` 病例构成；采样器只做「重排 + 重复」，**绝不跨病人、绝不引入 val 病例**。
    验证侧的 ``make_val_loader`` 完全不受本类影响（顺序读、原始分布）。

    参数：
        dataset：``CTSliceDataset``（只用它的 ``pos_flags`` / ``case_ids``）。
        batch_size：每批样本数。
        data_cfg：**合并后的 data 节**（``data_config(cfg)`` 的输出）；缺键时用 ``DEFAULT_DATA_CFG``。
        seed：``train.seed``；与 ``epoch``、病例集合一起派生每轮的采样顺序。
    """

    #: 告警的打印上限，避免刷屏
    MAX_WARNINGS = 5

    def __init__(self, dataset: CTSliceDataset, batch_size: int = 8,
                 data_cfg: dict | None = None, seed: int = 42) -> None:
        merged = dict(DEFAULT_DATA_CFG)
        merged.update({k: v for k, v in (data_cfg or {}).items() if k != "augment"})
        if len(dataset) == 0:
            raise ValueError("数据集里没有任何切片（index 为空？）")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size < 2:
            raise ValueError(f"batch_size 必须 >= 2（平衡采样要同时放正负样本），收到 {batch_size}")
        self.seed = int(seed)
        self.pos_ratio_target = float(merged.get("pos_ratio_train", 0.5))
        self.min_pos_per_batch = int(merged.get("min_pos_per_batch", 2) or 1)
        self.max_pos_repeat = int(merged.get("max_pos_repeat", 8) or 1)
        self.epoch_samples = merged.get("epoch_samples")
        self._epoch = 0
        self._warnings = 0

        flags = np.asarray(dataset.pos_flags)
        self.pos_indices: list = [int(v) for v in np.flatnonzero(flags).tolist()]
        self.neg_indices: list = [int(v) for v in np.flatnonzero(~flags).tolist()]
        self.n_positive = len(self.pos_indices)
        self.n_negative = len(self.neg_indices)

        self._plan = plan_balanced_slots(
            n_pos_slices=self.n_positive, n_neg_slices=self.n_negative,
            batch_size=self.batch_size, pos_ratio=self.pos_ratio_target,
            min_pos_per_batch=self.min_pos_per_batch, max_pos_repeat=self.max_pos_repeat,
            epoch_samples=(int(self.epoch_samples) if self.epoch_samples else None),
        )
        #: 兼容字段：旧代码/日志里读的是 ``_n_pos`` / ``_n_batches``
        self._n_pos = int(self._plan["n_pos_per_batch"])
        self._n_neg = int(self._plan["n_neg_per_batch"])
        self._n_batches = int(self._plan["batches"])

    # ---------------- 计划与长度 ----------------

    def __len__(self) -> int:
        """一轮 epoch 的 batch 数（由槽位预算与每批比例算出，见 ``plan_balanced_slots``）。"""
        return int(self._n_batches)

    def batch_targets(self) -> dict:
        """本轮的计划值（供自检逐条等号比对与日志打印）。

        与旧采样器的同名方法不同，这里**不再是区间而是定值**：每批阳性数就是
        ``pos_per_batch``，不存在"摊薄"。除 ``plan_balanced_slots`` 的原字段外，额外带上
        本采样器的配置项（``max_pos_repeat`` / ``pos_ratio_train`` 等），
        这样调用方不必再去翻 sampler 的属性（自检脚本就踩过一次 KeyError）。
        """
        plan = dict(self._plan)
        plan.update({
            "max_pos_repeat": int(self.max_pos_repeat),
            "min_pos_per_batch": int(self.min_pos_per_batch),
            "pos_ratio_train": float(self.pos_ratio_target),
            "epoch_samples": (int(self.epoch_samples) if self.epoch_samples else None),
        })
        return plan

    # ---------------- 迭代 ----------------

    def __iter__(self) -> Iterator[list]:
        self._epoch += 1
        rng = random.Random(self._seed_for_epoch(self._epoch))
        pos_pool = _CyclicPool(self.pos_indices, rng)
        neg_pool = _CyclicPool(self.neg_indices, rng)
        n_pos, n_neg = int(self._n_pos), int(self._n_neg)

        for _ in range(int(self._n_batches)):
            batch = pos_pool.take(n_pos) + neg_pool.take(n_neg)
            if len(batch) < self.batch_size:
                self._warn_shortage(len(batch))
            rng.shuffle(batch)   # 阳性不总排在 batch 头部；用本轮 rng，保证可复现
            yield batch

    def _warn_shortage(self, got: int) -> None:
        if self._warnings >= self.MAX_WARNINGS:
            return
        self._warnings += 1
        LOGGER.warning("本 batch 只凑到 %d/%d 个样本（阳性或阴性池为空？），比例会偏离目标。",
                       got, self.batch_size)

    def _seed_for_epoch(self, epoch: int) -> int:
        """由 (seed, epoch, 病例集合) 派生本轮种子；用 sha256 派生，跨进程、跨机器稳定。"""
        return stable_seed(self.seed, int(epoch), tuple(self.dataset.case_ids))

    def epoch_seed(self, epoch: int) -> int:
        """公开的种子派生（自检脚本用它复现某个 epoch 的采样顺序）。"""
        return self._seed_for_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        """把内部 epoch 计数器设成 ``epoch``：下一次 ``iter()`` 就从 ``epoch + 1`` 开始。

        用途：``src/train.py --resume`` 续跑时，让采样顺序与「一口气跑完」严格一致。
        注意 ``__iter__`` 是**先自增再派生种子**（``_epoch += 1`` 之后用 ``epoch_seed(_epoch)``），
        所以这里传的应当是「已经完成的轮数」。
        """
        self._epoch = int(epoch)

    # ---------------- 描述 ----------------

    def describe(self) -> str:
        """返回采样器配置、一轮计划与实测比例目标的多行描述（训练启动时打进日志）。"""
        plan = self.batch_targets()
        lines = [
            f"BalancedBatchSampler：batch_size={self.batch_size}，"
            f"data.pos_ratio_train={self.pos_ratio_target:.2f} → 每批 {self._n_pos} 正 + {self._n_neg} 阴"
            f"（实际阳性占比 {plan['actual_pos_ratio']:.3f}）；seed={self.seed}",
            f"  数据集 {len(self.dataset)} 层切片（阳性 {self.n_positive} / 阴性 {self.n_negative}）；"
            f"槽位预算 {plan['budget']}（data.epoch_samples="
            f"{'null→用切片数' if not self.epoch_samples else self.epoch_samples}）"
            f" → 一轮 {self._n_batches} 个 batch / {plan['slots']} 个槽位",
            f"  阳性层一轮重复 {plan['pos_repeat']:.2f} 次（上限 data.max_pos_repeat="
            f"{self.max_pos_repeat}，各层尽量均摊）；阴性层覆盖率 {plan['neg_coverage']:.2%}"
            f"（不再要求一轮全过一遍）",
            f"  病人级隔离：样本索引只在本折 train 病例内重排与重复（不跨病人、不引入 val 病例）；"
            f"病例集合 {list(self.dataset.case_ids)}",
        ]
        if self._n_pos < 1:
            lines.append("  **警告**：本折训练集里没有含肿瘤的切片 —— 只能产出全阴性 batch，"
                         "请检查 data/splits.json 与 cache 是否配对。")
        elif plan["pool_limited"]:
            lines.append(f"  **注**：阳性层只有 {self.n_positive} 层，撑不起每批 "
                         f"{self.min_pos_per_batch} 个的下限 → 已退化为每批 {self._n_pos} 个阳性"
                         f"（不会出现全阴性 batch，比例仍尽力贴近目标）。")
        elif plan.get("repeat_capped"):
            lines.append(f"  **注**：阳性重复上限 max_pos_repeat={self.max_pos_repeat} 卡住了比例"
                         f"（目标 {self.pos_ratio_target:.2f} → 实际 {plan['actual_pos_ratio']:.3f}）；"
                         f"想更接近目标就调大 data.max_pos_repeat 或调小 data.epoch_samples。")
        if self.n_negative < self._n_neg:
            lines.append(f"  **警告**：阴性切片只有 {self.n_negative} 个，少于每批所需，"
                         f"每个 epoch 内会被重复抽到（这不影响阳性比例，只是阴性多样性下降）。")
        return "\n".join(lines)


def make_batch_sampler(ds: CTSliceDataset, cfg: dict, generator=None) -> BatchSampler:
    """构造训练用 ``BalancedBatchSampler``（每批固定 ``n_pos`` 个阳性 + ``n_neg`` 个阴性）。

    ``generator`` 只用来取一个额外的 seed 偏移（``torch.Generator`` 的随机数流跨进程不可靠，
    因此真正的随机源是「seed + epoch」派生出的 ``random.Random``，见 ``BalancedBatchSampler``）。
    """
    train_cfg = (cfg or {}).get("train") or {}
    seed = int(train_cfg.get("seed", 42))
    if generator is not None:
        try:
            seed = seed + int(generator.initial_seed()) % 100000
        except Exception:  # noqa: BLE001 - 自定义 generator 没有 initial_seed 时忽略
            pass
    data_cfg = data_config(cfg)
    if train_cfg.get("pos_ratio_target") is not None:
        LOGGER.info("注意：train.pos_ratio_target=%s 自第 4 轮起**不再生效**（保留只为兼容旧命令/旧 "
                    "checkpoint）；每批阳性数改由 data.pos_ratio_train=%.2f / data.min_pos_per_batch=%d / "
                    "data.max_pos_repeat=%d 决定。",
                    train_cfg.get("pos_ratio_target"), float(data_cfg.get("pos_ratio_train", 0.5)),
                    int(data_cfg.get("min_pos_per_batch", 2)), int(data_cfg.get("max_pos_repeat", 8)))
    sampler = BalancedBatchSampler(
        ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        data_cfg=data_cfg,
        seed=seed,
    )
    plan = sampler.batch_targets()
    LOGGER.info("训练采样器（平衡采样）：batch_size=%d → 每批 %d 正 + %d 阴（阳性占比 %.3f，"
                "整体阳性率 %.4f ⇒ 过采样 %.2f 倍）；一轮 %d 个 batch；"
                "阳性层重复 %.2f 次（上限 %d）、阴性层覆盖率 %.2f%%",
                sampler.batch_size, plan["n_pos_per_batch"], plan["n_neg_per_batch"],
                plan["actual_pos_ratio"], ds.pos_ratio,
                plan["actual_pos_ratio"] / max(1e-9, ds.pos_ratio),
                plan["batches"], plan["pos_repeat"], sampler.max_pos_repeat,
                100.0 * plan["neg_coverage"])
    if plan["pool_limited"]:
        LOGGER.warning("阳性层只有 %d 层，撑不起每批 %d 个的下限：本折每批阳性数退化为 %d。",
                       sampler.n_positive, sampler.min_pos_per_batch, plan["n_pos_per_batch"])
    return sampler


# --------------------------------------------------------------------------------------
# DataLoader 工厂
# --------------------------------------------------------------------------------------

def _loader_kwargs(cfg: dict) -> dict:
    """从 data 节取 DataLoader 的通用参数（含每个 worker 的随机源初始化）。"""
    data_cfg = data_config(cfg)
    seed = int(((cfg or {}).get("train") or {}).get("seed", 42))

    def _worker_init(worker_id: int) -> None:
        """让每个 worker 的 numpy/random 种子互不相同但可复现。"""
        np.random.seed((seed + int(worker_id)) % (2 ** 32 - 1))
        random.seed(seed + int(worker_id))

    kwargs = {
        "num_workers": int(data_cfg.get("num_workers", 8)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "worker_init_fn": _worker_init,
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", False))
    return kwargs


def _load_splits(splits: dict | None, cfg: dict) -> dict:
    """splits 为空时从 ``cfg['paths']['splits']`` 读取。"""
    if splits is not None:
        return splits
    return load_json(((cfg or {}).get("paths") or {}).get("splits", "data/splits.json"), default={}) or {}


def _fold_cases(splits: dict, fold: int, which: str) -> list:
    """从 ``data/splits.json`` 取某折的 train / val 病例列表；缺折或缺键时报错而不是静默用全集。"""
    folds = (splits or {}).get("folds") or []
    if not folds:
        raise ValueError("splits 里没有 folds，无法取某折的病例；请检查 data/splits.json")
    rec = next((r for r in folds if int(r.get("fold", -1)) == int(fold)), None)
    if rec is None:
        raise ValueError(f"splits 里没有 fold={fold}（可选：{[r.get('fold') for r in folds]}）")
    if which not in rec:
        raise ValueError(f"fold {fold} 的记录里没有 {which!r} 键：{sorted(rec)}")
    cases = sorted(int(c) for c in rec[which])
    if not cases:
        raise ValueError(f"fold {fold} 的 {which} 为空")
    return cases


def make_train_loader(splits: dict | None, fold: int, cfg: dict) -> DataLoader:
    """训练加载器：**平衡采样**（每批 ``n_pos`` 正 + ``n_neg`` 阴）+ 训练增强；顺序由
    ``BalancedBatchSampler`` 决定（不额外 shuffle）。

    训练病例直接取自 ``splits`` 里该折的 ``train``（21 例 = 16 含肿瘤 + 5 仅肝脏），
    不依赖 ``cases="train"`` 的并集语义，避免「漏传 fold 时静默用上全部 25 例」——
    这也是病人级隔离的入口：val 病例根本不在这个 dataset 里，采样器再怎么重复也抽不到它们。

    采样顺序只由 ``(train.seed, epoch, 病例集合)`` 决定，与 ``num_workers`` 无关，
    因此换机器、改 worker 数都能复现同一条采样序列。
    """
    from src.utils import make_generator

    splits = _load_splits(splits, cfg)
    seed = int(((cfg or {}).get("train") or {}).get("seed", 42))
    data_cfg = data_config(cfg)
    cache_dir = ((cfg or {}).get("paths") or {}).get("cache", "cache")
    ds = CTSliceDataset(_fold_cases(splits, fold, "train"), cache_dir, "train", cfg,
                        augment=True, fold=int(fold))
    generator = make_generator(seed + int(fold))
    batch_sampler = make_batch_sampler(ds, cfg, generator=generator)
    verify = bool(data_cfg.get("verify_batch", True))
    expect = window_channels(resolve_z_context(data_cfg))
    return DataLoader(ds, batch_sampler=batch_sampler, generator=generator,
                      collate_fn=lambda samples: collate_samples(samples, verify=verify,
                                                                 expect_channels=expect),
                      **_loader_kwargs(cfg))


def make_val_loader(splits: dict | None, fold: int, cfg: dict) -> DataLoader:
    """验证加载器：顺序、无增强（仍然补边到 ``data.target_hw``，形状与训练侧一致）。

    **验证侧的分布必须保持原始**（fold 0 上阳性切片只有 ~10%）：指标要反映真实的类不平衡表现，
    所以这里**不做任何平衡采样**、也不重复抽样——顺序读一遍即可。旧版自检曾把训练侧的采样预算
    套到 val batch 上，误报「阳性切片 0 超出预算」，`check_batch` 里现在明确不判 val 的阳性数。

    验证阶段是「按病人整卷推理」（``src/infer.py`` 的 ``predict_volume``），逐层按 z 顺序读即可；
    这里不 shuffle 是为了让每折的验证顺序固定、指标可复现。
    """
    from src.utils import make_generator

    splits = _load_splits(splits, cfg)
    train_cfg = (cfg or {}).get("train") or {}
    data_cfg = data_config(cfg)
    cache_dir = ((cfg or {}).get("paths") or {}).get("cache", "cache")
    ds = CTSliceDataset(_fold_cases(splits, fold, "val"), cache_dir, "val", cfg,
                        augment=False, fold=int(fold))
    batch_size = data_cfg.get("val_batch_size") or int(train_cfg.get("batch_size", 8))
    generator = make_generator(int(train_cfg.get("seed", 42)) + int(fold))
    verify = bool(data_cfg.get("verify_batch", True))
    expect = window_channels(resolve_z_context(data_cfg))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False,
                      generator=generator,
                      collate_fn=lambda samples: collate_samples(samples, verify=verify,
                                                                 expect_channels=expect),
                      **_loader_kwargs(cfg))


if __name__ == "__main__":  # pragma: no cover - 直接运行本文件时给出正确入口
    raise SystemExit("本模块是库，不是脚本；请运行 python -m src.selfcheck_data 做数据自检。")
