"""损失函数：**手搓** Soft Dice + 交叉熵（``DiceCELoss``），不依赖 MONAI 的 ``DiceLoss`` / ``DiceCELoss``。

整体功能：
    1. ``one_hot`` —— 整数标签 ``(B,H,W)`` → one-hot ``(B,C,H,W)``（Dice 必须逐类算）；
    2. ``positive_sample_mask`` —— ``(B,)`` 的「该样本是否含前景」掩码（``positive_only`` 用）；
    3. ``SoftDiceLoss`` —— soft Dice 损失（可微版 Dice），支持逐类查询、两种聚合口径与样本筛选；
    4. ``DiceCELoss`` —— ``lambda_dice × (1 - Dice) + lambda_ce × CE``，训练用的总损失；
    5. ``build_loss(cfg)`` —— 按 ``cfg['loss']`` 构造 ``DiceCELoss``（训练入口唯一入口）。

为什么手搓（而不是 ``monai.losses.DiceCELoss``）：
    本版刻意不依赖医学影像库的高级 API。MONAI 的损失里 ``include_background`` / ``batch`` /
    ``to_onehot_y`` / ``reduction`` 的交互绕，出错时只能看到一条 shape mismatch；
    这里每个量都自己算，``per_class_dice`` 可以单独取出来对日志与自检负责。

关键口径（与 ``docs/preprocess_notes.md`` 第 3/4 轮记录一致，改前先读那两节）：
    * **输入**：``logits (B, C, H, W)``（**未过 softmax**）与 ``target (B, H, W)``（整数标签，
      本项目 label ⊂ {0,1}，C=2）。CE / softmax 都在 **float32** 上做（autocast 下也不降精度）。
    * ``softmax=True``（本项目默认）：多类互斥，``softmax`` 后取各类概率；``False`` 时按多标签
      ``sigmoid`` 处理，CE 换成 ``binary_cross_entropy_with_logits``。
    * ``batch=True``（默认）：交集/并集在**整个 batch + 空间**上先求和，每个类得到一个 Dice；
      ``batch=False``：逐样本算 Dice 再平均。
    * **``include_background=False``（必须）**：Dice 只算肿瘤类，全预测背景的解不再是 0.5 的平凡最优。
    * **``positive_only=True``（第 4 轮新增，默认开）**：Dice 只在 ``Y.sum()>0`` 的样本上聚合，
      全背景样本不参与 Dice（它们只会撑大并集分母），**但 CE 仍对整批所有样本计算**
      （背景那一半继续负责压假阳性）。与 ``data.pos_ratio_train`` 的平衡采样配套使用。
    * ``smooth``（默认 1e-5）加在分子分母上，避免"预测与 GT 都为空"时出现 0/0。
    * ``lambda_dice`` / ``lambda_ce`` 默认 1.0 / 1.0（相等权重）。
    * ``ce_class_weights``（默认 null）：CE 的类别权重，**留作下一级杠杆**，不是本轮的解法。
    * **补边区域本轮仍不做 ignore mask**：非 512 的 8 例补边占比 27%~55%，但补边值恒为背景 0
      （增强后不再严格为 0），要看曲线再决定是否按 ``pad_offset`` 屏蔽（见 todo）。

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


def positive_sample_mask(target: torch.Tensor) -> torch.Tensor:
    """返回 ``(B,)`` 的 bool 掩码：该样本的标签**是否含前景**（``Y.sum() > 0``）。

    用于 ``positive_only=True`` 的 Dice 口径（第 4 轮）：一张全背景切片对「肿瘤重叠度」没有任何
    信息量，它的 ``Y`` 全 0，只会把 Dice 的分母（并集）撑大、把真正的阳性样本摊薄。
    平衡采样把 batch 内阳性比例提到 50% 之后，这一项让 Dice 梯度**只花在"有肿瘤可找"的样本上**，
    而 CE 仍然对整批计算（背景那一半负责压假阳性，见 ``DiceCELoss``）。
    """
    tensor = target
    if tensor.ndim == 4 and int(tensor.shape[1]) == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"positive_sample_mask 期望 (B,H,W) 或 (B,1,H,W)，收到 {tuple(target.shape)}")
    return (tensor.reshape(int(tensor.shape[0]), -1) > 0).any(dim=1)


def dice_loss_from_per_class(dice_per_class: torch.Tensor, class_indices: Sequence[int],
                             batch_mask: torch.Tensor | None = None) -> torch.Tensor:
    """把逐类 Dice 折算成损失 ``1 - mean(Dice)``（只对选中的类求平均）。

    ``dice_per_class`` 形状 ``(C,)``（``batch=True``）或 ``(B,C)``（``batch=False``），
    两种口径都按同样的方式取平均，保证「损失」与「日志里的 Dice」是同一批数字推出来的。

    ``batch_mask``（``(B,)`` bool，仅 ``batch=False`` 时有意义）：先按「该样本是否含前景」筛行，
    再在筛出来的样本上对类求平均 —— 这就是 ``positive_only`` 的逐样本口径。
    一个都没筛出来时返回 0（**而不是 1**：没有前景样本时 Dice 项没有定义，
    返回 0 等价于"这一批不参与 Dice"，这比返回 1.0 更中性，也不会把梯度带偏）。
    """
    indices = [int(i) for i in class_indices]
    if not indices:
        raise ValueError("没有可用于计算 Dice 的类别（class_indices 为空）")
    if dice_per_class.ndim == 2:
        selected = dice_per_class[:, indices]          # (B, K)
        if batch_mask is not None:
            mask = batch_mask.to(selected.device).reshape(-1).bool()
            if bool(mask.any()):
                selected = selected[mask]
            else:
                return selected.new_zeros(())
        return 1.0 - selected.mean()
    return 1.0 - dice_per_class[indices].mean()


# --------------------------------------------------------------------------------------
# Soft Dice
# --------------------------------------------------------------------------------------

class SoftDiceLoss(nn.Module):
    """soft Dice 损失（可微）：``1 - mean_c( (2·Σ p·y + s) / (Σ p + Σ y + s) )``。

    参数：
        include_background：是否把背景类也算进平均。**本项目默认 False**（配置里写死；
            早年的 docstring 把它写成了 True，是文档漂移，已改正）。True 时损失存在
            「全预测背景」的平凡最优解（dice 项 = 1-(1+0)/2 = 0.5），第 3 轮首次正式训练就是
            掉进了这个解；False 时 dice 项 = ``1 - 肿瘤 soft Dice``，与上报指标同向。
        batch：``True`` = 在 batch+空间上聚合后再按类平均；``False`` = 逐样本算完再平均。
        smooth：平滑项，避免空/空时 0/0。
        softmax：``True`` 时对 logits 做 ``softmax``（多类互斥）；``False`` 时做 ``sigmoid``（多标签）。
        to_onehot_y：仅作**接口说明**用（本项目内部一律 one-hot，无论它写什么）；
            保留这个参数是为了与 ``build_loss`` 的调用签名一致，方便对照旧文档。
        positive_only：**只在含前景的样本上算 Dice**（第 4 轮）。``batch=True`` 时先按
            ``Y.sum()>0`` 挑出该批里含肿瘤的样本，再在它们上聚合 soft Dice；``batch=False`` 时
            逐样本算完后只对含前景的那些求平均。全背景样本对"肿瘤重叠度"没有信息量，
            它们只会撑大并集分母把梯度摊薄 —— 但它们的 CE 照算（见 ``DiceCELoss``）。

    用法：``SoftDiceLoss()(logits, target)`` 返回标量损失；
    ``per_class_dice(logits, target)`` 返回逐类 Dice（日志/自检用）。
    """

    def __init__(self, include_background: bool = True, batch: bool = True,
                 smooth: float = DEFAULT_SMOOTH, softmax: bool = True,
                 to_onehot_y: bool = False, positive_only: bool = False) -> None:
        super().__init__()
        if float(smooth) <= 0:
            raise ValueError(f"smooth 必须为正数（加在分母上防 0/0），收到 {smooth}")
        self.include_background = bool(include_background)
        self.batch = bool(batch)
        self.smooth = float(smooth)
        self.softmax = bool(softmax)
        self.to_onehot_y = bool(to_onehot_y)
        self.positive_only = bool(positive_only)
        self.last_n_pos_samples: int | None = None   # 上一次前向里含前景的样本数（日志用）

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

    def per_class_dice(self, logits: torch.Tensor, target: torch.Tensor,
                       sample_mask: torch.Tensor | None = None) -> torch.Tensor:
        """逐类 soft Dice。

        返回形状：``batch=True`` → ``(C,)``；``batch=False`` → ``(B,C)``（``sample_mask`` 给了时
        只返回被选中的那些样本行，行数 = 前景样本数）。

        ``sample_mask``（``(B,)`` bool）用于 ``positive_only``：``batch=True`` 时先在样本维上
        筛选后再聚合（等价于「只把含肿瘤的切片放进 Dice 的并集/交集」），``batch=False`` 时直接筛行。
        """
        probs = self.probabilities(logits)
        num_classes = int(probs.shape[1])
        onehot = one_hot(target, num_classes).to(probs.dtype)
        if sample_mask is not None:
            mask = sample_mask.to(probs.device).reshape(-1).bool()
            probs = probs[mask]
            onehot = onehot[mask]
            if int(probs.shape[0]) == 0:
                # 该 batch 一个前景样本都没有：返回「不可用」的标记张量（NaN），由 forward 处理
                shape = (num_classes,) if self.batch else (0, num_classes)
                return torch.full(shape, float("nan"), dtype=probs.dtype, device=probs.device)
        dims = (0, 2, 3) if self.batch else (2, 3)     # batch=True：把 batch 维一起求和
        intersection = (probs * onehot).sum(dim=dims)
        denom = probs.sum(dim=dims) + onehot.sum(dim=dims)
        return (2.0 * intersection + self.smooth) / (denom + self.smooth)

    # ---------------- 前向 ----------------

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"SoftDiceLoss 期望 logits (B,C,H,W)，收到 shape={tuple(logits.shape)}")
        mask = positive_sample_mask(target) if self.positive_only else None
        if mask is not None and not bool(mask.any()):
            # 一个前景样本都没有：Dice 项没有定义 → 记 0（不参与），不要让 NaN 传进总损失
            self.last_n_pos_samples = 0
            return logits.sum() * 0.0
        dice = self.per_class_dice(logits, target, sample_mask=mask)
        self.last_n_pos_samples = int(mask.sum()) if mask is not None else int(logits.shape[0])
        return dice_loss_from_per_class(dice, self.class_indices(int(logits.shape[1])),
                                        batch_mask=(mask if not self.batch else None))


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
        include_background=False：Dice 是否包含背景类（**必须 False**，见 ``SoftDiceLoss``）；
        smooth=1e-5：Dice 的平滑项；
        positive_only=False：**Dice 只在含前景的样本上算**（第 4 轮；CE 仍然对整批算）；
        ce_class_weights=None：CE 的类别权重（如 ``[0.2, 1.0]`` 给肿瘤类加权），``None`` = 不加权，
            这是本版留的**下一级杠杆**（默认关闭，先看平衡采样的曲线再决定是否启用）。

    ``forward(logits, target)`` 返回总损失（标量张量，可直接 ``backward()``）；
    每个 step 之后可以读 ``self.last_parts``：
        ``{"total", "dice_loss", "ce_loss", "dice_per_class": [...], "class_indices": [...],
           "positive_only": bool, "n_pos_samples": int, "n_samples": int}``
    数值都来自**当前这次**前向（float），供日志记录使用；不需要时忽略即可。
    """

    def __init__(self, softmax: bool = True, to_onehot_y: bool = False, batch: bool = True,
                 lambda_dice: float = 1.0, lambda_ce: float = 1.0,
                 include_background: bool = True, smooth: float = DEFAULT_SMOOTH,
                 ce_class_weights: Sequence[float] | None = None,
                 positive_only: bool = False) -> None:
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
        self.positive_only = bool(positive_only)

        self.dice = SoftDiceLoss(include_background=include_background, batch=batch,
                                 smooth=smooth, softmax=softmax, to_onehot_y=to_onehot_y,
                                 positive_only=positive_only)
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
        # Dice 用 `positive_only` 筛选样本；CE **永远对整批算**（背景那一半负责压假阳性，
        # 这正是"平衡采样 + Dice 只算前景"能成立的前提：不能把背景信号一起丢掉）。
        sample_mask = positive_sample_mask(target) if self.positive_only else None
        n_pos_samples = int(sample_mask.sum()) if sample_mask is not None else batch_size
        dice_per_class = self.dice.per_class_dice(logits, target, sample_mask=sample_mask)
        if dice_per_class.numel() == 0 or not bool(torch.isfinite(dice_per_class).all()):
            # 这批一个含前景的样本都没有：Dice 项无定义 → 记 0（与 SoftDiceLoss.forward 同口径）
            dice_loss = logits.sum() * 0.0
            per_class_mean = torch.zeros(num_classes, dtype=torch.float32)
        else:
            dice_loss = dice_loss_from_per_class(dice_per_class, class_idx,
                                                 batch_mask=(sample_mask if not self.batch else None))
            per_class_mean = (dice_per_class.mean(dim=0) if dice_per_class.ndim == 2
                              else dice_per_class)

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

        self.last_parts = {
            "total": float(total.detach()),
            "dice_loss": float(dice_loss.detach()),
            "ce_loss": float(ce_loss.detach()),
            "dice_per_class": [float(v) for v in per_class_mean.detach().float().cpu().tolist()],
            "class_indices": list(class_idx),
            "positive_only": bool(self.positive_only),
            "n_pos_samples": int(n_pos_samples),
            "n_samples": int(batch_size),
        }
        return total

    # ---------------- 描述 ----------------

    def describe(self) -> str:
        """一行摘要（权重、聚合口径、类别选择、Dice 的样本筛选），训练启动时打进日志。"""
        weights = (list(self.ce_class_weights.tolist())
                   if int(self.ce_class_weights.numel()) > 0 else None)
        dice_scope = ("只在含前景的样本上算（CE 仍对整批算）" if self.positive_only
                      else "对整批所有样本算")
        return (f"DiceCELoss（自实现）：λ_dice={self.lambda_dice:g} λ_ce={self.lambda_ce:g}；"
                f"softmax={self.softmax} batch={self.batch} "
                f"include_background={self.include_background} smooth={self.smooth:g}；"
                f"to_onehot_y={self.to_onehot_y}（内部一律 one-hot）；"
                f"dice_positive_only={self.positive_only}（Dice {dice_scope}）；"
                f"ce_class_weights={weights}")


# --------------------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------------------

def build_loss(cfg: dict) -> DiceCELoss:
    """按 ``cfg['loss']`` 构造训练损失（缺键时用与 ``configs/default.yaml`` 一致的默认值）。

    对齐 ``docs/todo.md`` 的接口：``DiceCELoss(softmax=True, to_onehot_y=False, batch=True,
    lambda_dice=1.0, lambda_ce=1.0)``；其余项（include_background / smooth / ce_class_weights /
    positive_only）由配置提供，都有默认值，所以只写前 5 个参数也能工作。
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
        positive_only=bool(loss_cfg.get("dice_positive_only", False)),
    )
    LOGGER.info("%s", criterion.describe())
    return criterion


if __name__ == "__main__":  # pragma: no cover - 直接运行本文件时给出正确入口
    raise SystemExit("本模块是库，不是脚本；训练请运行 python -m src.train --fold 0 --debug。")
