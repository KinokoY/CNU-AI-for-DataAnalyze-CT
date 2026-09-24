"""数据集与采样：按病例切片读 cache、按面内尺寸分桶、定向过采样肿瘤切片，并施加自实现的 2D 增强。

整体功能：
    1. ``CTSliceDataset`` —— 一个样本 = 一个病人的一层切片。``__init__`` 只读 **label 体素**
       （用于算 ``pos_flags`` 与索引），**不读 image 体素**；``__getitem__`` 用每 worker 一份的
       ``lru_cache`` 持有 nibabel 代理对象（``memmap=True``），只取 ``[:, :, z]`` 这一层。
    2. ``make_batch_sampler`` —— 自定义 ``BatchSampler``：先按桶 ``(ceil(H/mult)*mult, ceil(W/mult)*mult)``
       选桶（**同一 batch 内面内尺寸必然一致**），再在桶内定向抽 ``n_pos = round(batch_size*pos_ratio)``
       个含肿瘤切片、其余抽不含肿瘤切片，使 batch 内肿瘤切片比例恒为 ``pos_ratio_target``。
       不用裸 ``WeightedRandomSampler``：它的比例不可控，且不同面内尺寸无法同 batch。
    3. ``make_train_loader`` / ``make_val_loader`` —— 训练侧分桶+增强+定向采样；验证侧顺序、无增强。
    4. ``build_transforms`` —— 仅训练用的 2D 增强：翻转 / 旋转 90° / 仿射 / 随机 gamma / 高斯噪声，
       **全部自己实现，不依赖 MONAI**（原因见 ``make_augment_steps`` 的说明）。

依赖：numpy / torch / nibabel（+ 可选 scipy 的连通域，不在本文件用）。
前后接口：上游是 ``scripts/preprocess.py`` 产出的 ``cache/image/<case>.nii.gz``（uint16 归一化）
        与 ``cache/label/<case>.nii.gz``（uint8 二值）、``data/splits.json``、``cache/cache_manifest.json``；
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

#: 面内尺寸分桶的默认对齐（与 model.pad_to_multiple 一致：U-Net 4 级下采样 = 2^4）
DEFAULT_PAD_MULTIPLE = 16

DEFAULT_DATA_CFG: dict = {
    "index_cache_size": 8,        # 每 worker 保留的 nibabel 句柄数（lru_cache maxsize）
    "pad_multiple": DEFAULT_PAD_MULTIPLE,
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": False,
    "val_batch_size": None,       # None = 用 train.batch_size
    "bucket_balance": True,       # 按桶内样本数分配每轮 batch 配额，避免小桶被过度重复
    "min_pos_per_batch": 1,       # 每个 batch 至少含几个肿瘤切片（batch_size 很小时兜底）
    "pos_ratio_tolerance": 0.10,  # 自检用：实测比例偏离目标的告警阈值
    "augment": {
        "flip_prob": 0.5,
        "rotate90_prob": 0.5,
        "affine_prob": 0.5,
        "rotation_deg": 15.0,
        "scale_range": [0.9, 1.1],
        "shift_frac": 0.1,
        "histogram_shift_prob": 0.2,
        "histogram_num_bins": 10,
        "histogram_shift_range": 0.1,
        "noise_prob": 0.2,
        "noise_std": 0.01,
    },
}


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------

def bucket_key(height: int, width: int, pad_multiple: int = DEFAULT_PAD_MULTIPLE) -> tuple:
    """面内尺寸 → 分桶键 ``(ceil(H/mult)*mult, ceil(W/mult)*mult)``。

    与 ``scripts/preprocess.py`` 的 ``build_aggregate`` 用同一套口径，因此清单里的 ``size_buckets``
    可以直接和本函数的输出比对。键的取整值是「pad 之后的边长」，同键的切片才能同 batch。
    """
    mult = max(1, int(pad_multiple))
    return (int(math.ceil(int(height) / mult) * mult), int(math.ceil(int(width) / mult) * mult))


def format_bucket(key: Sequence[int]) -> str:
    """把桶键渲染成 ``512x512`` 这样的字符串，便于日志与报告阅读。"""
    return "x".join(str(int(v)) for v in key)


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
    """从总配置里取出 data 节并与默认值合并。

    ``pad_multiple`` 缺省时回退到 ``model.pad_to_multiple``（键名不同，别写错——写错会静默用 16，
    和 U-Net 的下采样倍数不一致时会在前向里被 pad 掩盖，很难查）。这里顺带校验它是 2 的幂。
    """
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_DATA_CFG.items()}
    user = (cfg or {}).get("data") or {}
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    if "pad_multiple" not in user:
        merged["pad_multiple"] = int(((cfg or {}).get("model") or {}).get(
            "pad_to_multiple", DEFAULT_PAD_MULTIPLE))
    mult = int(merged["pad_multiple"])
    if mult < 1:
        raise ValueError(f"pad_multiple 必须 >= 1，收到 {mult}")
    if mult & (mult - 1):
        raise ValueError(f"pad_multiple 必须是 2 的幂（U-Net 每级下采样 2 倍），收到 {mult}")
    merged["pad_multiple"] = mult
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
        if getattr(self, "affine", None) is None:
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

    ``k = 0`` 时等价于不变，因此 0.5 的概率下实际约有一半的样本完全没有旋转——这与
    MONAI ``RandRotate90d`` 的语义一致（它也是 ``randint(0, max_k) + 1`` 那种「抽到就转」）。
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
    而这四个增强本身只是十几行 numpy。自实现之后这一整类「接口/版本不匹配」故障消失，
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


def build_transforms(cfg: dict, train: bool, seed: int | None = None):
    """构建训练增强流水线；``train=False`` 时返回 ``None``（验证侧不做任何几何变换）。

    返回一个可调用对象：输入 ``{"image": (H,W) float32, "label": (H,W) uint8}``，返回同结构的 dict。
    步骤见 ``make_augment_steps``。
    """
    if not train:
        return None
    return ComposeSteps(make_augment_steps(cfg, seed=seed))


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
        **不读 image 体素**；25 例 label 合计几十 MB，读完这一例即可释放；
      * ``__getitem__``：走每 worker 一份的 ``lru_cache`` 拿 nibabel 代理对象（``mmap=True``），
        只取 ``[:, :, z]`` 一层；image 除以 65535 还原 [0,1]，label 取 >0 得 {0,1}。

    参数：
        cases：``"train"`` / ``"val"`` / ``"all"``、逗号串或整数序列（见 ``parse_cases``）。
        cache_dir：``cache/`` 根目录（其下 image/ 与 label/）。
        split：写进样本字典的分组名（``"train"`` / ``"val"`` / ``"all"``），仅用于追踪与日志。
        cfg：总配置字典（``load_config`` 的返回值）。
        augment：是否施加训练增强；``False`` 时不做任何几何变换。
        debug：True 时打印逐 case 的索引统计（切片数、含肿瘤切片数、桶键）。

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
        self.pad_multiple = int(self.data_cfg.get("pad_multiple", DEFAULT_PAD_MULTIPLE))
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
        self.buckets: dict = {}          # 桶键 -> [index 位置, ...]
        self.case_hw: dict = {}          # case -> (H, W)
        self.case_n_slices: dict = {}    # case -> nz
        self.case_pos_slices: dict = {}  # case -> 含肿瘤切片数
        self.image_dtype: str | None = None
        self._build_index()

    # ---------------- 索引构建 ----------------

    def _build_index(self) -> None:
        """逐 case 读 label 体素，建 ``index`` / ``pos_flags`` / ``buckets``，并做一致性自检。

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

            key = bucket_key(height, width, self.pad_multiple)
            start = len(self.index)
            self.index.extend((int(case), int(z)) for z in range(nz))
            self.pos_flags = np.concatenate(
                [self.pos_flags, pos_flags.astype(bool)]) if self.pos_flags.size else pos_flags.astype(bool)
            self.buckets.setdefault(key, []).extend(range(start, start + nz))
            self.case_hw[int(case)] = (height, width)
            self.case_n_slices[int(case)] = nz
            self.case_pos_slices[int(case)] = n_pos

            # 索引只需要「每层有没有肿瘤」，读完这一例就把映射丢掉（不留在 lru_cache 里）：
            # 真正按层读取发生在 __getitem__，届时会重新打开。
            per_slice = binary = lab_arr = None
            release_mmap(lab_img)
            _open_nii.cache_clear()

            if self.debug:
                LOGGER.info("  case %d：nz=%d 含肿瘤层=%d 面内=%dx%d 桶=%s",
                            int(case), nz, n_pos, height, width, format_bucket(key))

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

    def bucket_stats(self) -> dict:
        """逐桶统计：切片数、含肿瘤切片数、涉及病例数（自检与报告用）。"""
        stats: dict = {}
        for key, positions in sorted(self.buckets.items()):
            flags = self.pos_flags[positions]
            stats[key] = {
                "n_slices": len(positions),
                "n_pos": int(flags.sum()),
                "n_neg": int(len(positions) - int(flags.sum())),
                "n_cases": len({self.index[i][0] for i in positions}),
            }
        return stats

    def cases_of_bucket(self, key: tuple) -> list:
        """该桶里出现的病例列表（升序）。"""
        return sorted({self.index[i][0] for i in self.buckets[key]})

    def describe(self) -> str:
        """返回多行描述：split、病例、切片数、阳性比例、逐桶明细。"""
        lines = [
            f"split={self.split}"
            + (f" fold={self.fold}" if self.fold is not None else "")
            + f"：{len(self.case_ids)} 例、{len(self.index)} 层切片、含肿瘤 {self.n_positive} 层"
              f"（{self.pos_ratio:.4f}）、{len(self.buckets)} 个尺寸桶、影像 dtype={self.image_dtype}",
        ]
        for key, stat in self.bucket_stats().items():
            ratio = stat["n_pos"] / max(1, stat["n_slices"])
            lines.append(f"  桶 {format_bucket(key):>9}：切片 {stat['n_slices']:5d}"
                         f"（含肿瘤 {stat['n_pos']:4d} = {ratio:6.2%}）"
                         f"  病例 {stat['n_cases']:2d} 例 {self.cases_of_bucket(key)}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.index)

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

        image = np.ascontiguousarray(self._to_unit_range(image))
        label = np.ascontiguousarray((np.asarray(label) > 0).astype(np.uint8))

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
        if (height, width) != self.case_hw[case]:
            raise RuntimeError(f"case {case} z={z}：取出切片形状 {(height, width)} 与初始化记录的 "
                               f"{self.case_hw[case]} 不一致（轴序或增强出了错）")
        return {
            "image": torch.from_numpy(image).unsqueeze(0),        # (1, H, W) float32 ∈ [0,1]
            "label": torch.from_numpy(label.astype(np.int64)),    # (H, W) int64 ∈ {0,1}
            "case": str(case),
            "z": int(z),
            "orig_hw": (height, width),
        }


# --------------------------------------------------------------------------------------
# 分桶 + 定向过采样的 BatchSampler
# --------------------------------------------------------------------------------------

class _CyclicPool:
    """无放回轮转池：洗牌后依次吐出元素，吐完再洗牌。

    用途：在一个桶内「尽量不重复地」抽样本。同一个 batch 内不会重复（池内元素远多于 batch_size）；
    一轮 epoch 内是否重复由调用方控制——``BucketBatchSampler`` 会用 ``drawn`` 记录本轮已抽次数，
    在池子快被抽空时主动降低该桶的正样本配额，从而保证**一轮 epoch 覆盖到池内每个样本**，
    而不是反复抓同样的几层（小桶尤其明显）。

    池内的洗牌只用传入的 ``rng``（由 seed+epoch 派生），**不碰全局 random**，
    这样采样顺序与全局随机状态、worker 数都无关，同一配置下逐轮可复现。
    """

    def __init__(self, items: Sequence[int], rng: random.Random) -> None:
        self._items = list(items)
        self._rng = rng
        self._order: list = []
        self._cursor = 0
        self.drawn = 0

    def __len__(self) -> int:
        return len(self._items)

    def pop(self) -> int:
        if self._cursor >= len(self._order):
            self._order = list(self._items)
            self._rng.shuffle(self._order)
            self._cursor = 0
        value = self._order[self._cursor]
        self._cursor += 1
        self.drawn += 1
        return value


class BucketBatchSampler(BatchSampler):
    """按「尺寸桶 + batch 内阳性比例」迭代 batch 的采样器（不是 torch 自带的任何一种）。

    每个 batch 的构造：
      1. 轮流从尚未用过的桶里挑一个桶（桶顺序用固定 seed 洗牌，一轮内每个桶至少出现一次）；
      2. 在该桶内定向抽取：``n_pos = round(batch_size * pos_ratio_target)`` 个含肿瘤切片，
         ``n_neg = batch_size - n_pos`` 个不含肿瘤切片；
      3. 返回这 ``batch_size`` 个全局索引。

    这样保证：**同一 batch 内面内尺寸完全一致**（同桶才有相同 H/W），且**阳性比例恒为
    ``pos_ratio_target``**；而 ``WeightedRandomSampler`` 只能控制期望、每批实际比例会抖动，
    也无法满足「同 batch 同尺寸」的硬约束。

    一轮 epoch 的 batch 配额按桶内样本数加权分配（``bucket_balance=True``），否则 512×512 那个桶
    会吃掉绝大多数 step，而样本很少的小桶被重复采样几十次。某一边样本不足时 batch 会变小并把
    比例拉偏，此时打印一次告警（不中断）。
    """

    #: 「桶内样本不足」告警的打印上限，避免刷屏
    MAX_WARNINGS = 5

    def __init__(self, dataset: CTSliceDataset, batch_size: int = 8,
                 pos_ratio_target: float = 0.30, seed: int = 42,
                 min_pos_per_batch: int = 1, bucket_balance: bool = True,
                 verbose: bool = False) -> None:
        if int(batch_size) < 1:
            raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
        if not 0.0 <= float(pos_ratio_target) <= 1.0:
            raise ValueError(f"pos_ratio_target 必须落在 [0,1]，收到 {pos_ratio_target}")
        if not dataset.buckets:
            raise ValueError("数据集里没有任何尺寸桶（index 为空？）")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.pos_ratio_target = float(pos_ratio_target)
        self.seed = int(seed)
        self.min_pos_per_batch = int(min_pos_per_batch)
        self.bucket_balance = bool(bucket_balance)
        self.verbose = bool(verbose)
        self._epoch = 0
        self._warnings = 0
        self._quota_map = self._quota()   # 一轮 epoch 内每个桶的 batch 数（预算一次即可）

        # 仅用于 describe() 展示「桶内正/负样本量」；真正的抽样池每轮 epoch 重建（见 __iter__）
        self.bucket_counts: dict = {}
        for key, positions in dataset.buckets.items():
            flags = dataset.pos_flags[positions]
            self.bucket_counts[key] = (int(flags.sum()), int(len(positions) - int(flags.sum())))

    # ---------------- 配额与长度 ----------------

    def _quota(self) -> dict:
        """一轮 epoch 内每个桶的 batch 数配额：**按样本数加权**，并保证「装得下该桶的阳性层」。

        每个桶取三者的最大值：

          1. ``bucket_balance=True`` 时按桶内切片数占比分配（否则各桶等分总 batch 数）；
          2. ``ceil(桶内切片数 / batch_size)``：至少能把这桶的切片过一遍；
          3. ``ceil(桶内阳性层数 / n_pos)``：至少能按目标比例装下桶里的阳性层。

        第 3 条很关键：只按样本数加权时，阳性特别密的桶会出现
        「配额 × n_pos < 阳性层数」，于是无论如何都覆盖不全（本文件第一版就是这样，
        实测阳性比例掉到 0.25、还有 10 层一次没见到）。加上第 3 条后，正常尺寸的桶里
        「每 batch 阳性数」恒等于 ``n_pos``、且一轮覆盖每一层。
        """
        sizes = {key: len(positions) for key, positions in self.dataset.buckets.items()}
        total = max(1, sum(sizes.values()))
        n_pos = max(1, self._target_pos_per_batch())
        quota: dict = {}
        for key, size in sizes.items():
            share = size / total if self.bucket_balance else 1.0 / max(1, len(sizes))
            by_samples = int(round(share * total / self.batch_size))
            by_size = int(math.ceil(size / self.batch_size))
            n_pos_slices = int(self.dataset.pos_flags[self.dataset.buckets[key]].sum())
            by_positive = int(math.ceil(n_pos_slices / n_pos)) if n_pos_slices > 0 else 0
            quota[key] = max(1, by_samples, by_size, by_positive)
        return quota

    def __len__(self) -> int:
        """一轮 epoch 的 batch 数（≈ 切片总数 / batch_size，小桶按配额下限保底 1）。"""
        return int(sum(self._quota_map.values()))

    def __iter__(self) -> Iterator[list]:
        self._epoch += 1
        rng = random.Random(self._seed_for_epoch(self._epoch))
        # 每轮换新池：轮转状态跟着 epoch 走，且同一 seed 下逐轮可复现
        pools: dict = {}
        for key, positions in self.dataset.buckets.items():
            flags = self.dataset.pos_flags[positions].tolist()
            pools[key] = (
                _CyclicPool([int(p) for p, f in zip(positions, flags) if f], rng),
                _CyclicPool([int(p) for p, f in zip(positions, flags) if not f], rng),
            )

        keys = list(self.dataset.buckets)
        remaining = dict(self._quota_map)
        caps: dict = {}
        pending: list = []
        while True:
            if not pending:
                pending = [k for k in keys if remaining.get(k, 0) > 0]
                if not pending:
                    break
                rng.shuffle(pending)
            key = pending.pop()
            batches_left = remaining.get(key, 0)      # 含当前这一个 batch
            remaining[key] = batches_left - 1
            batch = self._draw_batch(key, pools[key][0], pools[key][1], rng, batches_left, caps)
            if batch:
                yield batch

        for key, cap in sorted(caps.items()):
            if cap < self._target_pos_per_batch():
                LOGGER.warning("桶 %s：本轮 %d 个 batch 只分到 %d 个阳性层/批（目标 %d）——桶内病灶层太少，"
                               "为了让一轮覆盖每一层只能降低比例；跑完这一折后如仍如此，"
                               "可考虑调低 train.batch_size 或把该桶的病例并入更大桶。",
                               format_bucket(key), self._quota_map.get(key, 0), cap,
                               self._target_pos_per_batch())

    def _target_pos_per_batch(self) -> int:
        """目标阳性数/批：``min(batch_size, max(min_pos_per_batch, round(batch_size * pos_ratio_target)))``。"""
        n_pos = int(round(self.batch_size * self.pos_ratio_target))
        return min(self.batch_size, max(int(self.min_pos_per_batch), n_pos))

    def _draw_batch(self, key: tuple, pos_pool: _CyclicPool, neg_pool: _CyclicPool,
                    rng: random.Random, batches_left: int, caps: dict | None = None) -> list:
        """在一个桶内抽一个 batch：先抽阳性，再用阴性补足。

        阳性数按「累计配额」算，而不是每个 batch 各自 round：

            target_cum(q) = max(q, ceil(P × q / B))       # 上限受 P 与 n_pos 双重约束

        其中 ``q`` = 已出 batch 数（含当前）、``P`` = 本轮该桶的阳性池大小、``B`` = 本轮该桶的
        batch 数、``n_pos`` = 目标阳性数/批。由于配额已由 ``_quota()`` 保证 ``B ≥ ceil(P / n_pos)``，
        实际抽到的阳性数就是 ``min(n_pos, target_cum 的差分)``：一轮内 P 个阳性层**恰好各出现一次**，
        且每批阳性数恒等于 ``n_pos``（正比于目标比例）；不会出现「前期抽太狠、后期全阴性」
        （早期版本用「每 batch 各自 ceil」就踩了这个坑：比例掉到 0.25，还有若干层一次没见到）。

        ``P < B`` 的极小桶做不到「每批都有阳性且每层只出现一次」，此时选择**覆盖优先**：
        阳性按整数节奏稀疏出现，比例低于目标，并在轮末汇总告警。
        """
        n_pos = self._target_pos_per_batch()
        n_neg = self.batch_size - n_pos
        batches_left = max(1, int(batches_left))
        total_batches = self._quota_map.get(key, max(1, batches_left))
        drawn_batches = max(0, total_batches - batches_left)      # 已出的 batch 数
        pool_size = len(pos_pool)

        def target_cum(q: int) -> int:
            return min(pool_size, max(q, int(math.ceil(pool_size * q / max(1, total_batches)))))

        want_pos = target_cum(drawn_batches + 1) - target_cum(drawn_batches)
        want_pos = max(0, min(want_pos, n_pos, pool_size - pos_pool.drawn, self.batch_size))
        want_neg = min(n_neg, len(neg_pool))
        if want_pos < n_pos or want_neg < n_neg:
            self._warn_shortage(key, n_pos, want_pos, n_neg, want_neg)
        # 某一边不够时用另一边补足，保住 batch 大小（比例会偏离目标）
        deficit = self.batch_size - want_pos - want_neg
        if deficit > 0:
            take_neg = min(deficit, max(0, len(neg_pool) - want_neg))
            want_neg += take_neg
            deficit -= take_neg
            want_pos += min(deficit, max(0, len(pos_pool) - want_pos))
        if caps is not None:
            caps[key] = want_pos if key not in caps else min(caps[key], want_pos)

        batch = [pos_pool.pop() for _ in range(want_pos)]
        batch += [neg_pool.pop() for _ in range(want_neg)]
        rng.shuffle(batch)  # 阴性切片不总排在 batch 尾部；用本轮 rng，保证可复现
        return batch

    def _warn_shortage(self, key: tuple, n_pos: int, got_pos: int, n_neg: int, got_neg: int) -> None:
        if self._warnings >= self.MAX_WARNINGS:
            return
        self._warnings += 1
        LOGGER.warning("桶 %s 样本不足：目标 %d 正 / %d 负，本 batch 抽到 %d 正 / %d 负"
                       "（桶内病灶层或阴性层太少）。比例可能低于目标，但一轮仍会覆盖桶内每一层。",
                       format_bucket(key), n_pos, n_neg, got_pos, got_neg)

    def _seed_for_epoch(self, epoch: int) -> int:
        """由 (seed, epoch, 病例集合) 派生本轮种子；用 sha256 派生，跨进程、跨机器稳定。"""
        return stable_seed(self.seed, int(epoch), tuple(self.dataset.case_ids))

    def epoch_seed(self, epoch: int) -> int:
        """公开的种子派生（自检脚本用它复现某个 epoch 的采样顺序）。"""
        return self._seed_for_epoch(epoch)

    def describe(self) -> str:
        """返回采样器配置与逐桶配额的多行描述。"""
        quota = self._quota()
        lines = [
            f"BucketBatchSampler：batch_size={self.batch_size} "
            f"pos_ratio_target={self.pos_ratio_target} "
            f"n_pos={int(round(self.batch_size * self.pos_ratio_target))} seed={self.seed} "
            f"每轮 batch 数={len(self)}（数据集共 {len(self.dataset)} 层切片）",
        ]
        for key in sorted(self.dataset.buckets):
            n_pos, n_neg = self.bucket_counts[key]
            lines.append(f"  桶 {format_bucket(key):>9}：切片 {len(self.dataset.buckets[key]):5d}"
                         f"（正 {n_pos:4d} / 负 {n_neg:5d}）  每轮配额 {quota[key]:4d} 个 batch")
        return "\n".join(lines)


def make_batch_sampler(ds: CTSliceDataset, cfg: dict, generator=None) -> BatchSampler:
    """构造训练用 ``BatchSampler``（分桶 + batch 内阳性比例固定）。

    ``generator`` 只用来取一个额外的 seed 偏移（``torch.Generator`` 的随机数流跨进程不可靠，
    因此真正的随机源是「seed + epoch」派生出的 ``random.Random``，见 ``BucketBatchSampler``）。
    """
    train_cfg = (cfg or {}).get("train") or {}
    data_cfg = data_config(cfg)
    seed = int(train_cfg.get("seed", 42))
    if generator is not None:
        try:
            seed = seed + int(generator.initial_seed()) % 100000
        except Exception:  # noqa: BLE001 - 自定义 generator 没有 initial_seed 时忽略
            pass
    sampler = BucketBatchSampler(
        ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        pos_ratio_target=float(train_cfg.get("pos_ratio_target", 0.30)),
        seed=seed,
        min_pos_per_batch=int(data_cfg.get("min_pos_per_batch", 1)),
        bucket_balance=bool(data_cfg.get("bucket_balance", True)),
    )
    LOGGER.info("训练采样器：batch_size=%d，batch 内目标阳性比例=%.2f（数据集整体阳性率 %.4f，"
                "约 %.1f 倍过采样）", sampler.batch_size, sampler.pos_ratio_target, ds.pos_ratio,
                sampler.pos_ratio_target / max(1e-9, ds.pos_ratio))
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
    """训练加载器：分桶采样 + 训练增强；顺序由 ``BucketBatchSampler`` 决定（不额外 shuffle）。

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
    return DataLoader(ds, batch_sampler=batch_sampler, generator=generator, **_loader_kwargs(cfg))


def make_val_loader(splits: dict | None, fold: int, cfg: dict) -> DataLoader:
    """验证加载器：顺序、无增强、不做分桶采样。

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
    return DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False,
                      generator=generator, **_loader_kwargs(cfg))


if __name__ == "__main__":  # pragma: no cover - 直接运行本文件时给出正确入口
    raise SystemExit("本模块是库，不是脚本；请运行 python -m src.selfcheck_data 做数据自检。")
