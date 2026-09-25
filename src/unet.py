"""2D U-Net（**从 torch.nn 手搓**，不依赖 MONAI / segmentation-models-pytorch 的高级封装）。

整体功能：
    1. ``DoubleConv2d`` —— ``(Conv3x3 → Norm → ReLU) × 2`` 的基础块；
    2. ``UNet2D`` —— 4 级下采样的 2D U-Net：编码通道 ``(32, 64, 128, 256)``、
       瓶颈 ``512``（= 最后一级 ×2）、解码逐级 ``ConvTranspose2d`` 上采样后与跳跃连接拼接；
       前向内部按 ``pad_to_multiple`` 把输入补到该值的整数倍，输出再裁回原 H/W；
    3. ``load_encoder_pretrained`` —— 预训练编码器的**接口占位**：基础版 ``name=None`` 直接返回，
       进阶版启用 resnet18/34 时才需要实现（见函数 docstring，代码只读本地权重文件、不走网络）；
    4. ``count_parameters`` / ``build_unet`` —— 参数量统计与「配置 → 模型」的工厂。

为什么手搓（而不是 ``monai.networks.nets.UNet``）：
    本版刻意不用医学影像库的高级 API。原因有两条，都在前两轮真实发生过：MONAI 的参数名跨版本
    变过（``axis``→``spatial_axis``、``shift_range`` 在 1.6 已不存在），而且它的网络里有大量
    可选项（``strides``/``res_block``/``norm_name`` 写法各异），出问题时很难一眼看出是哪一层错了。
    这里只用 ``torch.nn`` 的层，通道数、下采样次数、补边位置都能逐行核对。

输入输出契约：
    ``forward(x)``：``x`` 是 ``(B, in_channels, H, W)`` 的 float 张量，
    返回 **logits** ``(B, out_channels, H, W)``（不过 softmax；softmax/CE 在 ``src.losses`` 里做）。
    H/W 可以是任意正整数——内部会补到 ``pad_to_multiple`` 的整数倍再裁回；
    但 ``data.target_hw``（默认 512×512）已是 16 的倍数，所以常规路径不会触发内部补边。

依赖：torch（只用 ``torch.nn`` / ``torch.nn.functional``）。
前后接口：上游是 ``src.dataset`` 产出的 ``(B,1,512,512)`` 影像；下游是 ``src.losses.build_loss``
    与 ``src.train``；推理侧由 ``src.infer.predict_volume`` 调用。
用法：``model = build_unet(cfg)``（等价于 ``UNet2D(**cfg["model"])``，见 ``src.train.build_model``）。
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

LOGGER = setup_logger("unet")

#: 编码器通道数的默认值（与 configs/default.yaml 的 model.encoder_channels 一致）
DEFAULT_ENCODER_CHANNELS = (32, 64, 128, 256)

#: 支持的归一化写法：batch（默认，与配置一致）/ instance / group / none
NORM_CHOICES = ("batch", "instance", "group", "none")


# --------------------------------------------------------------------------------------
# 基础块
# --------------------------------------------------------------------------------------

def _group_count(channels: int, max_groups: int = 8) -> int:
    """给 ``GroupNorm`` 挑一个能整除 ``channels`` 的组数（取不超过 ``max_groups`` 的最大因子）。"""
    channels = int(channels)
    for groups in range(min(int(max_groups), channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def make_norm(norm: str, channels: int) -> nn.Module:
    """按配置名建归一化层：``batch`` / ``instance`` / ``group`` / ``none``。

    默认 ``batch``（``nn.BatchNorm2d``）。之所以把这几档写在一起：batch_size 只有 8，
    第 6 轮若要试 ``instance``/``group``（小 batch 更稳），只改配置即可，不必动网络代码。
    """
    kind = str(norm or "none").strip().lower()
    channels = int(channels)
    if kind in ("batch", "batchnorm", "bn"):
        return nn.BatchNorm2d(channels)
    if kind in ("instance", "instancenorm", "in"):
        return nn.InstanceNorm2d(channels, affine=True)
    if kind in ("group", "groupnorm", "gn"):
        return nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels)
    if kind in ("none", "identity", "no", ""):
        return nn.Identity()
    raise ValueError(f"model.norm 只支持 {NORM_CHOICES}，收到 {norm!r}")


def _conv_bias(norm: str) -> bool:
    """有归一化层时卷积不要 bias（归一化的 beta 已经承担偏置）；``norm=none`` 时保留 bias。"""
    return str(norm or "none").strip().lower() in ("none", "identity", "no", "")


class DoubleConv2d(nn.Module):
    """``(Conv3x3 → Norm → ReLU) × 2``：U-Net 每一级的基本块。

    参数：
        in_channels / out_channels：输入、输出通道数。
        norm：归一化类型（见 ``make_norm``）。
        mid_channels：中间层通道数，默认等于 ``out_channels``（保持「两级同宽」的经典写法）。
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch",
                 mid_channels: int | None = None) -> None:
        super().__init__()
        mid = int(mid_channels if mid_channels is not None else out_channels)
        bias = _conv_bias(norm)
        self.block = nn.Sequential(
            nn.Conv2d(int(in_channels), mid, kernel_size=3, padding=1, bias=bias),
            make_norm(norm, mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, int(out_channels), kernel_size=3, padding=1, bias=bias),
            make_norm(norm, int(out_channels)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down2d(nn.Module):
    """``MaxPool2d(2)`` 下采样 + ``DoubleConv2d``（编码器第 2 级起用它）。"""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch") -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv2d(int(in_channels), int(out_channels), norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up2d(nn.Module):
    """``ConvTranspose2d(2, stride=2)`` 上采样 → 与跳跃连接 ``cat`` → ``DoubleConv2d``。

    ``ConvTranspose2d`` 是经典 U-Net 的写法（可学习上采样）。尺寸本应严格翻倍，
    但为了防御「输入 H/W 是奇数」这类边界情况，拼接前会比一次跳跃连接的空间尺寸，
    不一致时用最近邻插值对齐（``data.target_hw`` 是 16 的倍数，常规路径不会走到这一步）。
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 norm: str = "batch") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(int(in_channels), int(in_channels), kernel_size=2, stride=2)
        self.conv = DoubleConv2d(int(in_channels) + int(skip_channels), int(out_channels), norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if tuple(x.shape[-2:]) != tuple(skip.shape[-2:]):
            x = F.interpolate(x, size=tuple(skip.shape[-2:]), mode="nearest")
        return self.conv(torch.cat([skip, x], dim=1))


# --------------------------------------------------------------------------------------
# U-Net
# --------------------------------------------------------------------------------------

class UNet2D(nn.Module):
    """4 级下采样的 2D U-Net（输入 1 通道 CT 切片 → 每像素分类 logits）。

    通道走向（以默认 ``encoder_channels=(32,64,128,256)`` 为例）：
        ``1 → 32 →(pool) 64 →(pool) 128 →(pool) 256 →(pool) 512(瓶颈)``
        再逐级 ``上采样 + 拼接`` 回到 ``32``，最后 ``1x1 卷积`` 输出 ``out_channels``。
    下采样 4 次 = 2^4 = 16，因此 ``pad_to_multiple`` 默认 16：
    输入边长必须是 16 的整数倍（``data.target_hw=512`` 满足），否则前向内部先补边再裁回。

    参数：
        in_channels：输入通道数（本项目 1）。
        out_channels：输出类别数（本项目 2：0=背景，1=肿瘤）。
        encoder_channels：编码器各级通道数；瓶颈取最后一级 ×2。
        norm：``batch`` / ``instance`` / ``group`` / ``none``（见 ``make_norm``）。
        pad_to_multiple：前向内部把 H/W 补到该值的整数倍再裁回（16 = 4 级下采样）。
        encoder_pretrained / pretrained_weights_path：预训练编码器配置（基础版为 None；
            真正加载由 ``load_encoder_pretrained`` 负责，构造时只是记录下来，不做任何 IO）。

    权重初始化：对所有卷积用 ``kaiming_normal_``（fan_in, relu），偏置置 0，
    归一化的 weight/bias 置 1/0。显式写出来是为了让「换 torch 版本导致默认初始化变化」
    不影响本项目的复现性（随机性由 ``train.seed`` 固定）。
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2,
                 encoder_channels: Sequence[int] = DEFAULT_ENCODER_CHANNELS,
                 norm: str = "batch", pad_to_multiple: int = 16,
                 encoder_pretrained: str | None = None,
                 pretrained_weights_path: str | None = None) -> None:
        super().__init__()
        channels = [int(c) for c in (encoder_channels or DEFAULT_ENCODER_CHANNELS)]
        if not channels:
            raise ValueError("encoder_channels 不能为空")
        if any(c <= 0 for c in channels):
            raise ValueError(f"encoder_channels 必须全为正整数，收到 {channels}")
        if int(out_channels) < 1:
            raise ValueError(f"out_channels 必须 >= 1，收到 {out_channels}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.encoder_channels = tuple(channels)
        self.norm = str(norm)
        self.pad_to_multiple = max(1, int(pad_to_multiple))
        self.encoder_pretrained = encoder_pretrained
        self.pretrained_weights_path = pretrained_weights_path

        # ---- 编码器：第 1 级在全分辨率，其后每级先 MaxPool2d(2) ----
        self.encoders = nn.ModuleList()
        prev = self.in_channels
        for level, width in enumerate(self.encoder_channels):
            self.encoders.append(
                DoubleConv2d(prev, width, norm=self.norm) if level == 0
                else Down2d(prev, width, norm=self.norm))
            prev = width

        # ---- 瓶颈：再下采样一次，通道翻倍 ----
        self.pool_to_bottleneck = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv2d(prev, self.encoder_channels[-1] * 2, norm=self.norm)

        # ---- 解码器：逐级上采样 + 拼接跳跃连接 ----
        self.ups = nn.ModuleList()
        prev = self.encoder_channels[-1] * 2
        for skip_width in reversed(self.encoder_channels):
            self.ups.append(Up2d(prev, skip_width, skip_width, norm=self.norm))
            prev = skip_width

        # ---- 输出头：1x1 卷积把通道压到类别数 ----
        self.head = nn.Conv2d(self.encoder_channels[0], self.out_channels, kernel_size=1)

        self._init_weights()

    # ---------------- 前向 ----------------

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """把 H/W 补到 ``pad_to_multiple`` 的整数倍（补在右下）。

        用 ``replicate`` 而不是补 0：输入的 0 是 HU 窗下界（相当于空气），
        在右下角贴一圈"空气"会给第一层卷积制造一条假边界；复制边缘像素更中性。
        ``data.target_hw=512`` 已是 16 的倍数，所以这一步在正式流程里是空操作。
        """
        mult = self.pad_to_multiple
        if mult <= 1:
            return x
        height, width = int(x.shape[-2]), int(x.shape[-1])
        pad_h = (-height) % mult
        pad_w = (-width) % mult
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"UNet2D 期望输入 (B, C, H, W)，收到 shape={tuple(x.shape)}")
        if int(x.shape[1]) != self.in_channels:
            raise ValueError(f"UNet2D 期望 {self.in_channels} 个输入通道，"
                             f"收到 {int(x.shape[1])}（shape={tuple(x.shape)}）")
        height, width = int(x.shape[-2]), int(x.shape[-1])

        out = self._pad_input(x)
        skips: list = []
        for encoder in self.encoders:
            out = encoder(out)
            skips.append(out)
        out = self.bottleneck(self.pool_to_bottleneck(out))
        for up, skip in zip(self.ups, reversed(skips)):
            out = up(out, skip)
        logits = self.head(out)

        if tuple(logits.shape[-2:]) != (height, width):   # 裁回补边前的 H/W
            logits = logits[..., :height, :width]
        return logits

    # ---------------- 描述 ----------------

    def _init_weights(self) -> None:
        """显式初始化（见类 docstring：避免 torch 默认初始化变更影响复现性）。"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def describe(self) -> str:
        """一行摘要（通道走向、归一化、补边对齐、参数量），训练启动时打进日志。"""
        return (f"UNet2D(in={self.in_channels}, out={self.out_channels}, "
                f"encoder={list(self.encoder_channels)}, bottleneck={self.encoder_channels[-1] * 2}, "
                f"下采样 {len(self.encoder_channels)} 次（对齐 {self.pad_to_multiple}）, "
                f"norm={self.norm}, 可训练参数 {count_parameters(self) / 1e6:.3f} M)")


# --------------------------------------------------------------------------------------
# 工厂 / 统计 / 预训练接口
# --------------------------------------------------------------------------------------

def build_unet(cfg: dict) -> UNet2D:
    """按 ``cfg['model']`` 建一个 ``UNet2D``（不做任何权重加载，也不搬到设备上）。

    这里只认 ``configs/default.yaml`` 的 ``model`` 节；缺键时用默认值，
    因此 ``--set model.encoder_channels=[16,32,64,128]`` 这类临时改动可以直接生效。
    """
    model_cfg = (cfg or {}).get("model") or {}
    channels = model_cfg.get("encoder_channels") or list(DEFAULT_ENCODER_CHANNELS)
    return UNet2D(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 2)),
        encoder_channels=tuple(int(c) for c in channels),
        norm=str(model_cfg.get("norm", "batch")),
        pad_to_multiple=int(model_cfg.get("pad_to_multiple", 16)),
        encoder_pretrained=model_cfg.get("encoder_pretrained"),
        pretrained_weights_path=model_cfg.get("pretrained_weights_path"),
    )


def load_encoder_pretrained(model: nn.Module, name: str | None = None,
                            path: str | None = None) -> dict:
    """预训练编码器加载接口（**基础版只占位，不加载任何东西**）。

    基础版（``model.encoder_pretrained=null``）：记一行日志后直接返回，编码器随机初始化，
    随机性由 ``train.seed`` 固定——这是本版报告的基线口径。

    ``name`` 给成 ``"resnet18"`` / ``"resnet34"`` 时**直接报错**，而不是静默忽略：
    本版 ``UNet2D`` 的编码器是 ``DoubleConv2d``（通道 32/64/128/256），
    与 resnet18/34 的 64/128/256/512 逐级残差块不是同一套参数，形状根本对不上，
    硬加载只会得到一堆 shape mismatch。进阶版要做的是：
      1) 用 ``torchvision.models.resnet18/34(weights=None)`` 建骨架；
      2) 从 ``path``（``model.pretrained_weights_path``，绝对路径，**只读本地文件、绝不走网络**）
         或 ``~/.cache/torch/hub/checkpoints/`` 读 ``state_dict``；
      3) 首层 3 通道卷积核按通道均值压成 1 通道（CT 是单通道灰度）；
      4) 各级 stage 输出接到对应层级的解码器，解码器仍随机初始化。

    返回一个 dict 说明本次是否真的加载了（供日志与报告记录）。
    """
    if not name:
        LOGGER.info("预训练编码器：未启用（model.encoder_pretrained=null）——编码器随机初始化，"
                    "随机性由 train.seed 固定（基础版基线口径）")
        return {"loaded": False, "name": None, "reason": "未配置（基础版）"}
    key = str(name).strip().lower()
    if key not in ("resnet18", "resnet34"):
        raise ValueError(f"model.encoder_pretrained 只支持 null / 'resnet18' / 'resnet34'，收到 {name!r}")
    raise NotImplementedError(
        f"基础版不实现预训练编码器（收到 encoder_pretrained={name!r}，"
        f"pretrained_weights_path={path!r}）：本版 UNet2D 的编码器是 DoubleConv 结构"
        f"（通道 {list(getattr(model, 'encoder_channels', DEFAULT_ENCODER_CHANNELS))}），"
        f"与 resnet18/34 的 64/128/256/512 残差块参数形状不匹配，无法直接加载。"
        f"请把 model.encoder_pretrained 设回 null（基础版口径），"
        f"或按 src/unet.py::load_encoder_pretrained 的 docstring 在进阶版里实现移植。")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """统计参数量；``trainable_only=True`` 只数 ``requires_grad`` 的参数。"""
    return int(sum(p.numel() for p in model.parameters()
                   if (p.requires_grad or not trainable_only)))


if __name__ == "__main__":  # pragma: no cover - 直接运行本文件时给出正确入口
    raise SystemExit("本模块是库，不是脚本；训练请运行 python -m src.train --fold 0 --debug。")
