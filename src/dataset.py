"""数据集与采样：按病例切片读 cache、统一补边到固定尺寸、定向过采样肿瘤切片，并施加自实现的 2D 增强。

整体功能：
    1. ``CTSliceDataset`` —— 一个样本 = 一个病人的一层切片。``__init__`` 只读 **label 体素**
       （用于算 ``pos_flags`` 与索引），**不读 image 体素**；``__getitem__`` 用每 worker 一份的
       ``lru_cache`` 持有 nibabel 代理对象（``memmap=True``），只取 ``[:, :, z]`` 这一层，
       归一化后把 image/label **居中补边到 ``data.target_hw``（默认 512×512）**。
    2. ``ProportionalBatchSampler`` —— 单一池 + 定向抽样的 ``BatchSampler``：先把含肿瘤切片
       在整轮 batch 上均摊（每个阳性层一轮恰好出现一次，每批不超过 ``n_pos`` 上限），
       再用不含肿瘤的切片把每个 batch 补满。因为所有样本补边后都是同一个形状，
       ``torch.stack`` 恒成立，**不再需要按面内尺寸分桶**。
    3. ``make_train_loader`` / ``make_val_loader`` —— 训练侧定向采样+增强；验证侧顺序、无增强。
    4. ``build_transforms`` —— 仅训练用的 2D 增强：翻转 / 旋转 90° / 仿射 / 随机 gamma / 高斯噪声，
       **全部自己实现，不依赖 MONAI**（原因见 ``make_augment_steps`` 的说明）。

为什么改成「统一补边」而不是「按尺寸分桶」：
    分桶要求同一 batch 内所有人的精确面内尺寸逐像素相同，于是采样器得先选桶再在桶内配额，
    小桶会被摊薄、还会出现大量全阴性 batch。改成统一补边到 512×512（512 是 16 的倍数，
    U-Net 4 级下采样不再需要内部 pad）之后，batch 形状恒为 ``(B,1,512,512)``，
    ``pad_multiple`` / ``bucket_key`` 这些概念整条链路都不需要了。
    代价是补边区域的白像素被浪费，且非 512 病例的补边区在增强后不再严格为 0；
    第 3 轮的 loss 可以考虑忽略补边区域，见 ``pad_to_target`` 的说明。

依赖：numpy / torch / nibabel（+ 可选 scipy 的连通域，不在本文件用）。
前后接口：上游是 ``scripts/preprocess.py`` 产出的 ``cache/image/<case>.nii.gz``（uint16 归一化）
        与 ``cache/label/<case>.nii.gz``（uint8 二值）、``data/splits.json``、``cache/cache_manifest.json``；
        下游是 ``src/selfcheck_data.py``（自检，先跑）与 ``src/train.py``（训练）。
用法：``python -m src.selfcheck_data`` 做数据形态自检；训练侧由 ``src.train`` 调用本模块的工厂函数。
"""

from __future__ import annotations

import hashlib
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
    from src.utils import load_config, load_json, rel_to_root, resolve_path, setup_logger
except ModuleNotFoundError:  # pragma: no cover - 兜底：把仓库根塞进 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import (  # type: ignore
        load_config,
        load_json,
        rel_to_root,
        resolve_path,
        setup_logger,
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

DEFAULT_DATA_CFG: dict = {
    "index_cache_size": 8,        # 每 worker 保留的 nibabel 句柄数（lru_cache maxsize）
    "target_hw": list(DEFAULT_TARGET_HW),   # 面内统一补边到的目标尺寸（H, W）
    "pad_align": DEFAULT_PAD_ALIGN,         # center = 居中补边；bottom-right = 右下补 0
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": False,
    "val_batch_size": None,       # None = 用 train.batch_size；验证是顺序读，只影响速度
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
    merged["target_hw"] = list(resolve_target_hw({"data": {"target_hw": merged.get("target_hw")},
                                                  "model": (cfg or {}).get("model") or {}}))
    return merged


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
        """对 ``(H, W)`` 数组做一次仿射采样，返回同形状 float32 数组。"""
        src = np.asarray(array, dtype=np.float32)
        if src.ndim != 2:
            raise ValueError(f"GridAffine2D 只接受 (H,W) 切片数组，收到 shape={src.shape}")
        height, width = int(src.shape[0]), int(src.shape[1])
        tensor = torch.from_numpy(np.ascontiguousarray(src))[None, None]  # (1, 1, H, W)
        grid = self.grid(height, width)
        out = F.grid_sample(tensor, grid, mode=mode, padding_mode="zeros", align_corners=True)
        return out[0, 0].numpy().astype(np.float32, copy=False)


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
    只作用于 image。
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
    只作用于 image，最后夹回 ``[0,1]``。
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
    """把 label 重新二值化：最近邻插值在边界上可能取到非 0/1 的值。"""

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

    ``seed`` 为 None 时用 ``train.seed``；同一个 seed 得到同一串增强参数（但每个样本的随机数
    仍按抽样顺序推进，因此各样本的增强互不相同）。
    """
    aug = dict(data_config(cfg).get("augment") or {})
    if seed is None:
        seed = int(((cfg or {}).get("train") or {}).get("seed", 42))
    rng = random.Random(int(seed) + 104729)   # 与采样器的种子错开一个素数
    keys = ("image", "label")                 # 几何增强同步作用于两者
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
        self.image_dtype: str | None = None
        self._build_index()

    # ---------------- 索引构建 ----------------

    def _build_index(self) -> None:
        """逐 case 读 label 体素，建 ``index`` / ``pos_flags``，并做一致性自检。

        读的是 label（几 MB/例）而不是 image（几十 MB/例）：索引只需要「哪些层有肿瘤」。
        """
        problems: list = []
        missing = [c for c in self.case_ids
                   if not (self.image_dir / f"{c}.nii.gz").exists()
                   or not (self.label_dir / f"{c}.nii.gz").exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} 个病例在 {rel_to_root(self.cache_dir)} 下缺 image/label：{missing}；"
                f"请先跑 python scripts/preprocess.py（或确认 case 名单与 cache 一致）")

        for case in self.case_ids:
            img_shape, img_dtype, spacing = _read_header(str(self.image_dir / f"{case}.nii.gz"))
            lab_img = _open_nii(str(self.label_dir / f"{case}.nii.gz"))
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

    def __getitem__(self, i: int) -> dict:
        case, z = self.index[i]
        # 只取第 z 层：a[:, :, z] 的形状是 (ny, nx) = (H, W)，与 case_hw 记录的一致
        image = np.asanyarray(_open_nii(str(self.image_dir / f"{case}.nii.gz")).dataobj)[:, :, z]
        label = np.asanyarray(_open_nii(str(self.label_dir / f"{case}.nii.gz")).dataobj)[:, :, z]

        # 先归一化，再补边（uint16 的 0 就是窗下界，补边补 0 与「背景」同值）
        image = np.ascontiguousarray(self._to_unit_range(image))
        label = np.ascontiguousarray((np.asarray(label) > 0).astype(np.uint8))

        # 统一补边：所有样本出来都是 target_hw，torch.stack 永远合法（不再需要分桶）
        image, offset = pad_to_target(image, self.target_hw, self.pad_align)
        label, label_offset = pad_to_target(label, self.target_hw, self.pad_align)
        if offset != label_offset:
            raise RuntimeError(f"case {case} z={z}：image 与 label 的补边偏移不一致 "
                               f"{offset} vs {label_offset}")

        if self.transforms is not None:
            out = self.transforms({"image": image, "label": label})
            image = np.ascontiguousarray(np.asarray(out["image"], dtype=np.float32))
            label = np.ascontiguousarray(np.asarray(out["label"]).astype(np.uint8))
            if image.ndim == 3 and image.shape[0] == 1:   # 兜底：万一某步加了通道维
                image = image[0]
            if label.ndim == 3 and label.shape[0] == 1:
                label = label[0]
            if image.shape != label.shape:
                raise RuntimeError(f"增强后 image/label 形状不一致：{image.shape} vs {label.shape}")

        height, width = (int(s) for s in image.shape)
        if (height, width) != tuple(self.target_hw):
            raise RuntimeError(f"case {case} z={z}：补边/增强后切片形状 {(height, width)} 不是 "
                               f"data.target_hw {tuple(self.target_hw)}（补边或增强出了错）")
        return {
            "image": torch.from_numpy(image).unsqueeze(0),        # (1, target_h, target_w) float32 ∈ [0,1]
            "label": torch.from_numpy(label.astype(np.int64)),    # (target_h, target_w) int64 ∈ {0,1}
            "case": str(case),
            "z": int(z),
            "orig_hw": (int(self.case_hw[case][0]), int(self.case_hw[case][1])),
        }


# --------------------------------------------------------------------------------------
# 组 batch（collate）
# --------------------------------------------------------------------------------------

def collate_samples(samples: Sequence[dict], verify: bool = True) -> dict:
    """把若干样本拼成一个 batch，并定义**明确的 batch 契约**（训练与自检都按它取值）：

        image      : Tensor (B, 1, 512, 512) float32
        label      : Tensor (B, 512, 512)    int64 ∈ {0,1}
        case       : list[str]，长度 B（每个样本来自哪个病人）
        z          : list[int]，长度 B（每个样本是第几层）
        orig_hw    : list[tuple[int,int]]，长度 B（每个样本**补边前**的面内尺寸）
        pad_offset : list[tuple[int,int]]，长度 B（内容在补边画布里的左上角 (top, left)）

    用 ``data.target_hw`` 之外的尺寸时上面两处的 512 相应替换；契约的结构不变。

    为什么不直接用 DataLoader 的默认 collate：PyTorch 默认会把每样本一个 tuple 的 ``orig_hw``
    做**转置**，得到 ``[(h1,h2,...), (w1,w2,...)]``——看上去像 ``(H,W)`` 但其实是两行列表，
    解包就会炸（远程就是这样炸的：``too many values to unpack``）。显式定义契约后，
    ``orig_hw`` / ``pad_offset`` 都是长度 B 的 tuple 列表，含义不再依赖默认行为。

    ``verify=True``（默认）时顺带校验 batch 的不变量：所有样本形状一致、
    ``label`` 取值 ⊂ {0,1}、``image`` 值域 ⊂ [0,1]、张量形状与 ``orig_hw`` 自洽。
    这些校验的开销可以忽略（只是比较几个整数），但能在训练早期抓住「buffered/memmap 复用、
    增强把尺寸改坏、collate 拼错」这类最难查的问题。

    注意（历史教训）：``image`` 是 ``(B,1,H,W)``、``label`` 是 ``(B,H,W)``，
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
        "orig_hw": [(int(s["orig_hw"][0]), int(s["orig_hw"][1])) for s in samples],
        # 补边偏移不在样本里携带，而是由 orig_hw + 张量形状**当场推导**：
        # 这样它一定与 batch 的形状自洽（样本里再存一份就有写歪的可能）。
        "pad_offset": [pad_offset_of(s["orig_hw"], (height, width)) for s in samples],
    }
    if verify:
        if tuple(label.shape[1:]) != (height, width):
            raise RuntimeError(f"batch 内 image {tuple(image.shape)} 与 label {tuple(label.shape)} 尺寸不符"
                               f"（注意 image 是 (B,1,H,W)、label 是 (B,H,W)，比较时要错开通道维）")
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
                         "请检查 batch_size/pos_ratio_target 与数据集规模是否匹配）", k, len(self._items))
        return [self._pop_unseen() for _ in range(k)]


class ProportionalBatchSampler(BatchSampler):
    """「单一池 + batch 内阳性比例」的采样器（不是 torch 自带的任何一种）。

    每个 batch 的构造：
      1. 先把 ``P`` 个含肿瘤层在 ``B`` 个 batch 上**尽量均摊**，得到每批的阳性数
         ``第 q 批 = ceil(P*q/B) - ceil(P*(q-1)/B)``，每批再按 ``n_pos`` 封顶；
      2. 阴性层把每个 batch 补满到 ``batch_size``。

    为什么均摊而不是「每个 batch 恒取 ``pos_ratio_target``」：那样每批要 2 个阳性，
    而整个训练集只有 12–14% 的阳性层，硬凑就要把 batch 数放大到「阴性层一个都别重复」的程度，
    反而制造出上百个**全阴性** batch（旧的分桶实现在远程跑出过 566 个 batch / 117 个全阴性，
    基线口径见 docs/baseline.md）。改成「阳性总数在整轮均摊 + 阴性补满」之后，
    每个阳性层在一轮里**恰好出现一次**，也不会为了凑比例而丢弃阴性层。

    ``pos_ratio_target`` 只是**每批阳性数的上限**，不是每批的实际比例：一轮的阳性层总数是固定的
    （``P``），摊到 ``B`` 个 batch 上每批就是 ``P/B`` 个。fold 0 实测 ``P=617, B=642``
    → 每批实际 1 个（不是上限 2 个），所以「提高每批阳性数」要靠加大 ``batch_size``。

    保证（自检脚本逐条验证）：
      * batch 大小恒为 ``batch_size``（池子够大时），形状恒为 ``(B,1,512,512)``；
      * 一轮 epoch 内**每一层切片至少出现一次**（阳性层恰好一次、无重复）；
      * 采样顺序只由 ``(train.seed, epoch, 病例集合)`` 决定（sha256 派生，不用内置 ``hash()``），
        与 ``num_workers`` 无关；**同一个 epoch 可复现、相邻 epoch 不同**。

    一轮的 batch 数取三个下界的最大值（``自身 __len__`` 即此值）：
      * ``ceil(切片数 / batch_size)``：把整个数据集过一遍；
      * ``ceil(阳性层数 / n_pos)``：给阳性层留够槽位（阳性多、batch 大时这条会成为上界）；
      * ``ceil(阴性层数 / (batch_size - n_pos))``：让阴性层尽量不要重复抽。
    所以 batch 数**可能大于** ``ceil(切片数 / batch_size)``（当 ``P`` 相对 ``n_pos`` 很大时），
    自检里不能拿「切片数/batch_size」直接等号比对。

    **全阴性 batch 数由 batch 数与阳性层数共同决定**：能覆盖全部阴性层的最小 batch 数可能大于
    ``P``（fold 0 实测：``B=642`` 而 ``P=617``），此时按均摊公式必然有 ``B-P`` 个 batch 一个阳性
    也分不到（fold 0 是 25 个）。本地穷举过 ``B`` 的可行区间，这个数量就是该约束下的下界，
    不是采样 bug；要减少它就得接受阴性层重复（或调 ``batch_size``，见下一段）。
    """

    #: 告警的打印上限，避免刷屏
    MAX_WARNINGS = 5

    def __init__(self, dataset: CTSliceDataset, batch_size: int = 8,
                 pos_ratio_target: float = 0.30, seed: int = 42,
                 verbose: bool = False) -> None:
        if int(batch_size) < 1:
            raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
        if not 0.0 <= float(pos_ratio_target) <= 1.0:
            raise ValueError(f"pos_ratio_target 必须落在 [0,1]，收到 {pos_ratio_target}")
        if len(dataset) == 0:
            raise ValueError("数据集里没有任何切片（index 为空？）")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.pos_ratio_target = float(pos_ratio_target)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self._epoch = 0
        self._warnings = 0
        # 一轮 epoch 的 batch 数：先把阳性槽位留够（ceil(P / n_pos)），再用阴性层把每批补满
        self._n_pos = self._target_pos_per_batch()
        self._n_neg = self.batch_size - self._n_pos
        n_pos_slices = int(dataset.pos_flags.sum())
        by_positive = int(-(-n_pos_slices // self._n_pos)) if n_pos_slices > 0 else 0
        by_samples = int(-(-len(dataset) // self.batch_size))
        if self._n_neg > 0:
            # 让阴性层一轮不重复所需的最少 batch 数（不够时 _CyclicPool 会重复抽取）
            by_negative = int(-(-(len(dataset) - n_pos_slices) // self._n_neg))
            self._n_batches = max(1, by_samples, by_positive, by_negative)
        else:
            self._n_batches = max(1, by_positive)
        self._plan = self._build_plan()
        self.n_positive = n_pos_slices
        self.n_negative = int(len(dataset) - n_pos_slices)

    # ---------------- 计划与长度 ----------------

    def _target_pos_per_batch(self) -> int:
        """目标阳性数/批：``min(batch_size, max(1, round(batch_size * pos_ratio_target)))``。

        ``batch_size=8, pos_ratio_target=0.30`` → ``round(2.4) = 2``：这是每批的**上限**。
        实际每批放几个还要看阳性层总数够不够摊（见类 docstring：fold 0 上 617 个阳性层
        摊到 642 个 batch，于是每批实际只有 1 个，批次平均比例 0.12 而不是 0.25）。
        想提高每批阳性数就把 ``batch_size`` 提上去：按 fold 0 的数字，``batch_size=16``（n_pos=5）
        时一轮 351 个 batch、每批 1~2 个阳性、全阴性 0 个；``batch_size=10`` 时每批 2 个。
        """
        return min(self.batch_size, max(1, int(round(self.batch_size * self.pos_ratio_target))))

    def _build_plan(self) -> list:
        """一轮 epoch 内每批的阳性数清单（长度 = batch 数，合计 = 阳性层总数）。

        ``第 q 批 = ceil(P*q/B) - ceil(P*(q-1)/B)``，再按 ``n_pos`` 封顶：
        每批是 ``floor(P/B)`` 或 ``ceil(P/B)``，且合计恰好 P（覆盖每个阳性层一次）。

        不要用 ``max(q, ceil(P*q/B))`` 那种「保证前几批达标」的写法：它会把阳性前置到前 P 批、
        后面整段全 0。
        """
        import math

        batches = max(1, int(self._n_batches))
        total = int(self.dataset.pos_flags.sum())

        def cum(q: int) -> int:
            return min(total, int(math.ceil(total * q / batches)))

        return [min(self._n_pos, cum(q) - cum(q - 1)) for q in range(1, batches + 1)]

    def __len__(self) -> int:
        """一轮 epoch 的 batch 数 = 「切片覆盖 / 阳性槽位 / 阴性槽位」三个下界的最大值。

        注意它**可能大于** ``ceil(切片数 / batch_size)``：阳性层相对 ``n_pos`` 很多时，
        要给每个阳性层留够槽位就得加 batch（多出来的槽位由阴性层填）。
        """
        return int(self._n_batches)

    def batch_targets(self) -> dict:
        """本采样器的「可达阳性数/批」区间，供自检脚本判阈值与打印。

        ``ideal`` = 每批最多放几个阳性（配置目标）；``observed_max`` / ``floor`` = 均摊计划里
        实际出现的最大/最小值。阳性层稀少时两者会低于 ``ideal``，这是覆盖与比例之间的取舍，
        **自检必须按这个区间判，不能拿全局 0.30 直接当阈值**（旧版就是这样在 val 侧误报的）。
        """
        plan = list(self._plan)
        return {
            "batch_size": int(self.batch_size),
            "batches": int(self._n_batches),
            "ideal": int(self._n_pos),
            "floor": int(min(plan)) if plan else 0,
            "observed_max": int(max(plan)) if plan else 0,
            "pos_slices": int(self.n_positive),
            "neg_slices": int(self.n_negative),
            "plan_head": plan[:12],
        }

    # ---------------- 迭代 ----------------

    def __iter__(self) -> Iterator[list]:
        self._epoch += 1
        rng = random.Random(self._seed_for_epoch(self._epoch))
        pos_pool = _CyclicPool(np.flatnonzero(np.asarray(self.dataset.pos_flags)).tolist(), rng)
        neg_pool = _CyclicPool(np.flatnonzero(~np.asarray(self.dataset.pos_flags)).tolist(), rng)

        for want_pos in self._plan:
            batch = pos_pool.take(want_pos) + neg_pool.take(self.batch_size - want_pos)
            if len(batch) < self.batch_size:
                self._warn_shortage(len(batch))
            rng.shuffle(batch)   # 阴性切片不总排在 batch 尾部；用本轮 rng，保证可复现
            yield batch

        self._warn_coverage()

    def _warn_shortage(self, got: int) -> None:
        if self._warnings >= self.MAX_WARNINGS:
            return
        self._warnings += 1
        LOGGER.warning("本 batch 只凑到 %d/%d 个样本（阳性或阴性层太少），比例可能偏离目标；"
                       "一轮仍会覆盖数据集里每一层切片。", got, self.batch_size)

    def _warn_coverage(self) -> None:
        """抽样槽位装不下全部切片时给出一次告警（这会让「一轮覆盖全部切片」失效）。"""
        slots = self._n_batches * self.batch_size
        if slots < len(self.dataset):
            self._warn_shortage(slots)
            LOGGER.warning("一轮 %d 个 batch × %d = %d 个槽位 < 数据集 %d 层切片："
                           "本轮无法覆盖全部切片，请调大 batch 数或检查 batch_size/pos_ratio_target",
                           self._n_batches, self.batch_size, slots, len(self.dataset))

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

    def describe(self) -> str:
        """返回采样器配置与一轮计划的多行描述。"""
        plan = self._plan
        from collections import Counter

        counts = Counter(plan)
        return "\n".join([
            f"ProportionalBatchSampler：batch_size={self.batch_size} "
            f"pos_ratio_target={self.pos_ratio_target} → n_pos={self._n_pos}/批（上限）"
            f"seed={self.seed}",
            f"  数据集 {len(self.dataset)} 层切片（阳性 {self.n_positive} / 阴性 {self.n_negative}），"
            f"一轮 {self._n_batches} 个 batch；阳性层均摊后每批 "
            + "、".join(f"{k} 个的有 {v} 批" for k, v in sorted(counts.items())),
            f"  每个阳性层一轮出现恰好 1 次；阴性层不足时会在 _CyclicPool 里重复抽取"
            f"（含阳性的 batch 会因此被摊薄，这不是 bug）。",
        ])


def make_batch_sampler(ds: CTSliceDataset, cfg: dict, generator=None) -> BatchSampler:
    """构造训练用 ``BatchSampler``（单一池 + batch 内阳性比例固定）。

    ``generator`` 只用来取一个额外的 seed 偏移（``torch.Generator`` 的随机数流跨进程不可靠，
    因此真正的随机源是「seed + epoch」派生出的 ``random.Random``，见 ``ProportionalBatchSampler``）。
    """
    train_cfg = (cfg or {}).get("train") or {}
    seed = int(train_cfg.get("seed", 42))
    if generator is not None:
        try:
            seed = seed + int(generator.initial_seed()) % 100000
        except Exception:  # noqa: BLE001 - 自定义 generator 没有 initial_seed 时忽略
            pass
    sampler = ProportionalBatchSampler(
        ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        pos_ratio_target=float(train_cfg.get("pos_ratio_target", 0.30)),
        seed=seed,
    )
    targets = sampler.batch_targets()
    # 「实际过采样倍数」不能拿 pos_ratio_target / 整体阳性率 算：那个目标只是每批的**上限**。
    # 正确口径是「一轮里阳性层的出现次数 / 一轮总槽位数」再除以整体阳性率
    # （每个阳性层恰好抽一次，所以出现次数就是 P；B×bs 是一轮槽位数）。fold 0 实测
    # 617/(642×8) = 0.120，整体 0.138 → 0.87 倍，也就是**几乎没有重复采样**。
    slots = max(1, targets["batches"] * sampler.batch_size)
    epoch_ratio = targets["pos_slices"] / slots
    LOGGER.info("训练采样器：batch_size=%d，每批阳性上限 n_pos=%d（pos_ratio_target=%.2f）；"
                "一轮 %d 个 batch，每批实际 %d~%d 个阳性层，阳性层一轮出现占比 %.4f"
                "（整体 %.4f，约 %.2f 倍）",
                sampler.batch_size, targets["ideal"], sampler.pos_ratio_target,
                targets["batches"], targets["floor"], targets["observed_max"],
                epoch_ratio, ds.pos_ratio, epoch_ratio / max(1e-9, ds.pos_ratio))
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
    """训练加载器：阳性层均摊采样 + 训练增强；顺序由 ``ProportionalBatchSampler`` 决定（不额外 shuffle）。

    训练病例直接取自 ``splits`` 里该折的 ``train``（21 例 = 16 含肿瘤 + 5 仅肝脏），
    不依赖 ``cases="train"`` 的并集语义，避免「漏传 fold 时静默用上全部 25 例」。

    采样顺序只由 ``(train.seed, epoch, 病例集合)`` 决定，与 ``num_workers`` 无关，
    因此换机器、改 worker 数都能复现同一条采样序列。
    """
    from src.utils import make_generator

    splits = _load_splits(splits, cfg)
    seed = int(((cfg or {}).get("train") or {}).get("seed", 42))
    cache_dir = ((cfg or {}).get("paths") or {}).get("cache", "cache")
    ds = CTSliceDataset(_fold_cases(splits, fold, "train"), cache_dir, "train", cfg,
                        augment=True, fold=int(fold))
    generator = make_generator(seed + int(fold))
    batch_sampler = make_batch_sampler(ds, cfg, generator=generator)
    verify = bool(data_config(cfg).get("verify_batch", True))
    return DataLoader(ds, batch_sampler=batch_sampler, generator=generator,
                      collate_fn=lambda samples: collate_samples(samples, verify=verify),
                      **_loader_kwargs(cfg))


def make_val_loader(splits: dict | None, fold: int, cfg: dict) -> DataLoader:
    """验证加载器：顺序、无增强（仍然补边到 ``data.target_hw``，形状与训练侧一致）。

    验证阶段是「按病人整卷推理」（第 4 轮 ``src/infer.py``），逐层按 z 顺序读即可；
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
    return DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False,
                      generator=generator, collate_fn=lambda samples: collate_samples(samples, verify=verify),
                      **_loader_kwargs(cfg))


if __name__ == "__main__":  # pragma: no cover - 直接运行本文件时给出正确入口
    raise SystemExit("本模块是库，不是脚本；请运行 python -m src.selfcheck_data 做数据自检。")
