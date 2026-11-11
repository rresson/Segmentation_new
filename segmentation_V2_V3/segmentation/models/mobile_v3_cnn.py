"""MobileNetV3 风格 CNN，以及 CNN/head 共用的基础模块。"""

from __future__ import annotations

from torch import Tensor, nn


def make_norm(channels: int, norm: str) -> nn.Module:
    """创建归一化层。"""
    if norm == "bn":
        return nn.BatchNorm2d(channels)

    if norm == "gn":
        return nn.GroupNorm(8, channels)

    raise ValueError(f"Unknown normalization: {norm}")


class ConvNormAct(nn.Sequential):
    """卷积、归一化与可选激活函数。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        norm: str = "bn",
        activation: str = "hs",
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            make_norm(out_channels, norm),
        ]

        if activation == "hs":
            layers.append(nn.Hardswish())
        elif activation == "relu":
            layers.append(nn.ReLU())
        elif activation != "linear":
            raise ValueError(
                f"Unknown activation: {activation}"
            )

        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    """通过全局池化生成通道权重。"""

    def __init__(self, channels: int) -> None:
        super().__init__()

        # 通道数约为输入的 1/4，向上取整为 8 的倍数。
        squeeze_channels = max(
            8,
            ((channels + 31) // 32) * 8,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(
            channels,
            squeeze_channels,
            kernel_size=1,
        )
        self.relu = nn.ReLU()
        self.expand = nn.Conv2d(
            squeeze_channels,
            channels,
            kernel_size=1,
        )
        self.gate = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = self.pool(x)
        scale = self.relu(self.reduce(scale))
        scale = self.gate(self.expand(scale))

        return x * scale


class InvertedResidualV3(nn.Module):
    """扩展、DW、SE、线性投影组成的倒残差模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expanded_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = True,
        activation: str = "hs",
        norm: str = "bn",
    ) -> None:
        super().__init__()

        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")

        if kernel_size not in (3, 5):
            raise ValueError("kernel_size must be 3 or 5")

        self.use_residual = (
            stride == 1
            and in_channels == out_channels
        )

        self.expand = (
            ConvNormAct(
                in_channels,
                expanded_channels,
                norm=norm,
                activation=activation,
            )
            if expanded_channels != in_channels
            else nn.Identity()
        )

        self.depthwise = ConvNormAct(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=expanded_channels,
            norm=norm,
            activation=activation,
        )

        self.se = (
            SqueezeExcitation(expanded_channels)
            if use_se
            else nn.Identity()
        )

        # 线性瓶颈：投影后不添加激活。
        self.project = ConvNormAct(
            expanded_channels,
            out_channels,
            norm=norm,
            activation="linear",
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.expand(x)
        output = self.depthwise(output)
        output = self.se(output)
        output = self.project(output)

        if self.use_residual:
            return x + output

        return output


class MobileV3Backbone(nn.Module):
    """三尺度 CNN backbone。

    输入 [B, 3, 256, 256] 时输出：
        C1: [B,  32, 64, 64]
        C2: [B,  64, 32, 32]
        C3: [B, 128, 16, 16]
    """

    out_channels = (32, 64, 128)

    def __init__(self) -> None:
        super().__init__()

        self.stem = ConvNormAct(
            3,
            16,
            kernel_size=3,
            stride=2,
        )

        self.stage1 = nn.Sequential(
            InvertedResidualV3(
                16,
                32,
                48,
                stride=2,
                use_se=False,
                activation="relu",
            ),
            InvertedResidualV3(
                32,
                32,
                64,
                use_se=False,
                activation="relu",
            ),
        )

        self.stage2 = nn.Sequential(
            InvertedResidualV3(
                32,
                64,
                96,
                kernel_size=5,
                stride=2,
            ),
            InvertedResidualV3(
                64,
                64,
                128,
                kernel_size=5,
            ),
        )

        self.stage3 = nn.Sequential(
            InvertedResidualV3(
                64,
                128,
                192,
                kernel_size=5,
                stride=2,
            ),
            InvertedResidualV3(
                128,
                128,
                384,
                kernel_size=5,
            ),
            InvertedResidualV3(
                128,
                128,
                384,
                kernel_size=5,
            ),
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)

        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)

        return [c1, c2, c3]