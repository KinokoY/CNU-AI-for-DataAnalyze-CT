"""损失函数：**手搓** Soft Dice + 交叉熵（``DiceCELoss``），不依赖 MONAI 的 ``DiceLoss`` / ``DiceCELoss``。

整体功能：
    1. ``one_hot`` —— 整数标签 ``(B,H,W)`` → one-hot ``(B,C,H,W)``（Dice 必须逐类算）；
    2. ``SoftDiceLoss`` —— soft Dice 损失（可微版 Dice），支持逐类查询与两种聚合口径；
    3. ``DiceCELoss`` —— ``lambda_dice × (1 - Dice) + lambda_ce × CE``，训练用的总损失；
    4. ``build_loss(cfg)`` —— 按 ``cfg['loss']`` 构造 ``DiceCELoss``（训练入口唯一入口）。

为什么手搓（而不是 ``monai.losses.DiceCELoss``）：
    本版刻意不依赖医学影像库的高级 API。MONAI 的损失里 ``include_background`` / ``batch`` /
    ``to_onehot_y`` / ``reduction`` 的交互绕，出错时只能看到一条 shape mismatch；
    这里每个量都自己算，``per_class_dice`` 可以单独取出来对日志与自检负责。

关键口径（与 ``docs/preprocess_notes.md`` 第 3 轮记录一致，改前先读那节）：
    * **输入**：``logits (B, C, H, W)``（**未过 softmax**）与 ``target (B, H, W)``（整数标签，
      本项目 label ⊂ {0,1}，C=2）。CE / softmax 都在 **float32** 上做（autocast 下也不降精度）。
    * ``softmax=True``（本项目默认）：多类互斥，``softmax`` 后取各类概率；``False`` 时按多标签
      ``sigmoid`` 处理，CE 换成 ``binary_cross_entropy_with_logits``。
    * ``batch=True``（默认）：交集/并集在**整个 batch + 空间**上先求和，每个类得到一个 Dice；
      ``batch=False``：逐样本算 Dice 再平均。前者对 batch 内样本数不敏感（本项目 batch 里常只有
      1~2 个含肿瘤的切片，逐样本口径会让"整批都没肿瘤"的样本把梯度带偏）。
    * ``smooth``（默认 1e-5）加在分子分母上，避免"预测与 GT 都为空"时出现 0/0。
    * ``lambda_dice`` / ``lambda_ce`` 默认 1.0 / 1.0（相等权重）。
    * **补边区域本轮不做 ignore mask**：非 512 的 8 例补边占比 27%~55%，但补边值恒为背景 0
      （增强后不再严格为 0），第 6 轮看首折曲线再决定是否按 ``pad_offset`` 屏蔽（见 todo）。

依赖：torch（只用 ``torch.nn`` / ``torch.nn.functional``）。
前后接口：上游是 ``src.unet.UNet2D`` 的 logits 与 ``src.dataset`` 的 label；
    下游是 ``src.train``（``build_loss(cfg)`` 后 ``criterion.to(device)``）。
用法：``criterion = build_loss(cfg); loss = criterion(logits, labels)``；
    每个 step 之后可读 ``criterion.last_parts`` 拿到 dice/ce 两项的数值用于日志。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.utils import setup_logger
except ModuleNotFoundError:  # pragma: no cover - 兜底：把仓库根塞进 sys.path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import setup_logger  # type: ignore

LOGGER = setup_logger("losses")

#: Dice 的平滑项默认值（加在分子与分母上）
DEFAULT_SMOOTH = 1e-5


# --------------------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------------------

def one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """整数标签 → one-hot：``(B,H,W)``（或 ``(B,1,H,W)``）→ ``(B,C,H,W)`` float32。

    Dice 必须逐类算，而 ``src.dataset`` 给的是 ``(B,H,W)`` 的整数标签（0=背景，1=肿瘤），
    因此这里统一转成 one-hot。顺手做两件事：
      * 兼容 ``(B,1,H,W)``（有些流程会带一个通道维）；
      * 校验取值落在 ``[0, C-1]``，越界直接报错——标签口径错时宁可炸在这里，
        也不要静默算出一个看不出问题的 Dice。
    """
    num_classes = int(num_classes)
    if num_classes < 1:
        raise ValueError(f"num_classes 必须 >= 1，收到 {num_classes}")
    tensor = target
    if tensor.ndim == 4 and int(tensor.shape[1]) == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"one_hot 期望 (B,H,W) 或 (B,1,H,W) 的整数标签，"
                         f"收到 shape={tuple(target.shape)}")
    tensor = tensor.long()
    if tensor.numel():
        low, high = int(tensor.min()), int(tensor.max())
        if low < 0 or high >= num_classes:
            raise ValueError(f"标签取值 [{low}, {high}] 超出 [0, {num_classes - 1}]："
                             f"请检查 label 口径与 model.out_channels 是否匹配")
    return F.one_hot(tensor, num_classes).permute(0, 3, 1, 2).to(torch.float32)


def dice_loss_from_per_class(dice_per_class: torch.Tensor, class_indices: Sequence[int]) -> torch.Tensor:
    """把逐类 Dice 折算成损失 ``1 - mean(Dice)``（只对选中的类求平均）。

    ``dice_per_class`` 形状 ``(C,)``（``batch=True``）或 ``(B,C)``（``batch=False``），
    两种口径都按同样的方式取平均，保证「损失」与「日志里的 Dice」是同一批数字推出来的。
    """
    indices = [int(i) for i in class_indices]
    if not indices:
        raise ValueError("没有可用于计算 Dice 的类别（class_indices 为空）")
    selected = dice_per_class[:, indices] if dice_per_class.ndim == 2 else dice_per_class[indices]
    return 1.0 - selected.mean()


# --------------------------------------------------------------------------------------
# Soft Dice
# --------------------------------------------------------------------------------------

class SoftDiceLoss(nn.Module):
    """soft Dice 损失（可微）：``1 - mean_c( (2·Σ p·y + s) / (Σ p + Σ y + s) )``。

    参数：
        include_background：是否把背景类也算进平均。本项目默认 ``True``（与 MONAI 默认、
            nnU-Net 的 soft Dice 口径一致）；设成 ``False`` 就是「只优化肿瘤类」，
            第 6 轮调参时可以只改配置试一遍。
        batch：``True`` = 在 batch+空间上聚合后再按类平均；``False`` = 逐样本算完再平均。
        smooth：平滑项，避免空/空时 0/0。
        softmax：``True`` 时对 logits 做 ``softmax``（多类互斥）；``False`` 时做 ``sigmoid``（多标签）。
        to_onehot_y：仅作**接口说明**用（本项目内部一律 one-hot，无论它写什么）；
            保留这个参数是为了与 ``build_loss`` 的调用签名一致，方便对照旧文档。

    用法：``SoftDiceLoss()(logits, target)`` 返回标量损失；
    ``per_class_dice(logits, target)`` 返回逐类 Dice（日志/自检用）。
    """

    def __init__(self, include_background: bool = True, batch: bool = True,
                 smooth: float = DEFAULT_SMOOTH, softmax: bool = True,
                 to_onehot_y: bool = False) -> None:
        super().__init__()
        if float(smooth) <= 0:
            raise ValueError(f"smooth 必须为正数（加在分母上防 0/0），收到 {smooth}")
        self.include_background = bool(include_background)
        self.batch = bool(batch)
        self.smooth = float(smooth)
        self.softmax = bool(softmax)
        self.to_onehot_y = bool(to_onehot_y)

    # ---------------- 内部 ----------------

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """logits → 概率（**float32**：autocast 下也要保证 Dice 的数值口径稳定）。"""
        logits = logits.float()
        if self.softmax:
            return torch.softmax(logits, dim=1)
        return torch.sigmoid(logits)

    def class_indices(self, num_classes: int) -> list:
        """参与平均的类别下标：``include_background=False`` 时去掉第 0 类（背景）。"""
        num_classes = int(num_classes)
        if num_classes < 1:
            raise ValueError(f"类别数必须 >= 1，收到 {num_classes}")
        if self.include_background:
            return list(range(num_classes))
        if num_classes == 1:
            raise ValueError("include_background=False 需要至少 2 个输出通道"
                             "（单通道没有『背景类』可以排除；请用 include_background=true）")
        return list(range(1, num_classes))

    def per_class_dice(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """逐类 soft Dice。

        返回形状：``batch=True`` → ``(C,)``；``batch=False`` → ``(B,C)``。
        """
        probs = self.probabilities(logits)
        num_classes = int(probs.shape[1])
        onehot = one_hot(target, num_classes).to(probs.dtype)
        dims = (0, 2, 3) if self.batch else (2, 3)     # batch=True：把 batch 维一起求和
        intersection = (probs * onehot).sum(dim=dims)
        denom = probs.sum(dim=dims) + onehot.sum(dim=dims)
        return (2.0 * intersection + self.smooth) / (denom + self.smooth)

    # ---------------- 前向 ----------------

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"SoftDiceLoss 期望 logits (B,C,H,W)，收到 shape={tuple(logits.shape)}")
        dice = self.per_class_dice(logits, target)
        return dice_loss_from_per_class(dice, self.class_indices(int(logits.shape[1])))


# --------------------------------------------------------------------------------------
# Dice + CE
# --------------------------------------------------------------------------------------

class DiceCELoss(nn.Module):
    """``lambda_dice × (1 - Dice) + lambda_ce × CE``：本项目的训练损失。

    参数（前 5 个与 ``docs/todo.md`` 里写死的接口一致）：
        softmax=True：对 logits 做 softmax 后算 Dice，CE 走 ``cross_entropy``（多类互斥）；
        to_onehot_y=False：本项目 label 已是整数标签，Dice 内部一律转 one-hot；
        batch=True：Dice 在整个 batch 上聚合（见 ``SoftDiceLoss``）；
        lambda_dice / lambda_ce：两项的权重（默认 1.0 / 1.0）。
    额外参数：
        include_background=True：Dice 是否包含背景类；
        smooth=1e-5：Dice 的平滑项；
        ce_class_weights=None：CE 的类别权重（如 ``[0.2, 1.0]`` 给肿瘤类加权），``None`` = 不加权。

    ``forward(logits, target)`` 返回总损失（标量张量，可直接 ``backward()``）；
    每个 step 之后可以读 ``self.last_parts``：
        ``{"total", "dice_loss", "ce_loss", "dice_per_class": [...], "class_indices": [...]}``
    数值都来自**当前这次**前向（float），供日志记录使用；不需要时忽略即可。
    """

    def __init__(self, softmax: bool = True, to_onehot_y: bool = False, batch: bool = True,
                 lambda_dice: float = 1.0, lambda_ce: float = 1.0,
                 include_background: bool = True, smooth: float = DEFAULT_SMOOTH,
                 ce_class_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        if float(lambda_dice) == 0.0 and float(lambda_ce) == 0.0:
            raise ValueError("lambda_dice 与 lambda_ce 不能同时为 0（损失恒为 0，训练无意义）")
        self.softmax = bool(softmax)
        self.to_onehot_y = bool(to_onehot_y)
        self.batch = bool(batch)
        self.lambda_dice = float(lambda_dice)
        self.lambda_ce = float(lambda_ce)
        self.include_background = bool(include_background)
        self.smooth = float(smooth)

        self.dice = SoftDiceLoss(include_background=include_background, batch=batch,
                                 smooth=smooth, softmax=softmax, to_onehot_y=to_onehot_y)
        # CE 的类别权重：注册成 buffer（persistent=False）→ criterion.to(device) 时一起搬到 GPU；
        # 「不加权」用 0 长度张量表示，避免依赖 None buffer 的版本差异。
        if ce_class_weights is None or len(list(ce_class_weights)) == 0:
            self.register_buffer("ce_class_weights", torch.zeros(0, dtype=torch.float32),
                                 persistent=False)
        else:
            weights = torch.tensor([float(v) for v in ce_class_weights], dtype=torch.float32)
            if bool((weights < 0).any()):
                raise ValueError(f"ce_class_weights 不能为负：{list(ce_class_weights)}")
            self.register_buffer("ce_class_weights", weights, persistent=False)

        self.last_parts: dict = {}

    # ---------------- 前向 ----------------

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"DiceCELoss 期望 logits (B,C,H,W)，收到 shape={tuple(logits.shape)}")
        batch_size, num_classes, height, width = (int(v) for v in logits.shape)
        if tuple(int(v) for v in target.shape) != (batch_size, height, width):
            raise ValueError(f"target 形状 {tuple(target.shape)} 与 logits {tuple(logits.shape)} 不匹配"
                             f"（期望 (B,H,W)=({batch_size},{height},{width})）")

        class_idx = self.dice.class_indices(num_classes)
        dice_per_class = self.dice.per_class_dice(logits, target)
        dice_loss = dice_loss_from_per_class(dice_per_class, class_idx)

        if self.softmax:
            weight = self.ce_class_weights if int(self.ce_class_weights.numel()) > 0 else None
            if weight is not None and int(weight.numel()) != num_classes:
                raise ValueError(f"ce_class_weights 长度 {int(weight.numel())} 与输出通道数 "
                                 f"{num_classes} 不一致")
            ce_loss = F.cross_entropy(logits.float(), target.long(), weight=weight)
        else:
            # 多标签（sigmoid）口径：BCE 的 input 与 target 必须同精度，
            # autocast 下 logits 可能是 bf16，所以两边都显式转成 float32。
            onehot = one_hot(target, num_classes).to(torch.float32)
            ce_loss = F.binary_cross_entropy_with_logits(logits.float(), onehot)

        total = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        per_class_mean = (dice_per_class.mean(dim=0) if dice_per_class.ndim == 2
                          else dice_per_class)
        self.last_parts = {
            "total": float(total.detach()),
            "dice_loss": float(dice_loss.detach()),
            "ce_loss": float(ce_loss.detach()),
            "dice_per_class": [float(v) for v in per_class_mean.detach().float().cpu().tolist()],
            "class_indices": list(class_idx),
        }
        return total

    # ---------------- 描述 ----------------

    def describe(self) -> str:
        """一行摘要（权重、聚合口径、类别选择），训练启动时打进日志。"""
        weights = (list(self.ce_class_weights.tolist())
                   if int(self.ce_class_weights.numel()) > 0 else None)
        return (f"DiceCELoss（自实现）：λ_dice={self.lambda_dice:g} λ_ce={self.lambda_ce:g}；"
                f"softmax={self.softmax} batch={self.batch} "
                f"include_background={self.include_background} smooth={self.smooth:g}；"
                f"to_onehot_y={self.to_onehot_y}（内部一律 one-hot）；"
                f"ce_class_weights={weights}")


# --------------------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------------------

def build_loss(cfg: dict) -> DiceCELoss:
    """按 ``cfg['loss']`` 构造训练损失（缺键时用与 ``configs/default.yaml`` 一致的默认值）。

    对齐 ``docs/todo.md`` 的接口：``DiceCELoss(softmax=True, to_onehot_y=False, batch=True,
    lambda_dice=1.0, lambda_ce=1.0)``；其余项（include_background / smooth / ce_class_weights）
    由配置提供，都有默认值，所以只写前 5 个参数也能工作。
    """
    loss_cfg = dict((cfg or {}).get("loss") or {})
    criterion = DiceCELoss(
        softmax=bool(loss_cfg.get("softmax", True)),
        to_onehot_y=bool(loss_cfg.get("to_onehot_y", False)),
        batch=bool(loss_cfg.get("batch", True)),
        lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
        lambda_ce=float(loss_cfg.get("lambda_ce", 1.0)),
        include_background=bool(loss_cfg.get("include_background", True)),
        smooth=float(loss_cfg.get("smooth", DEFAULT_SMOOTH)),
        ce_class_weights=loss_cfg.get("ce_class_weights"),
    )
    LOGGER.info("%s", criterion.describe())
    return criterion


if __name__ == "__main__":  # pragma: no cover - 直接运行本文件时给出正确入口
    raise SystemExit("本模块是库，不是脚本；训练请运行 python -m src.train --fold 0 --debug。")
