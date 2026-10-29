"""MobileNetV3 风格分割头：低分辨率上下文增强与轻量 FPN。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from .mobile_v3_cnn import ConvNormAct, InvertedResidualV3


class SeparableConv(nn.Sequential):
    """DW 空间卷积 + PW 通道混合，使用 GroupNorm。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            ConvNormAct(
                in_channels,
                in_channels,
                kernel_size=3,
                groups=in_channels,
                norm="gn",
            ),
            ConvNormAct(
                in_channels,
                out_channels,
                norm="gn",
            ),
        )


class MobileV3FPNHead(nn.Module):
    """低分辨率 V3 上下文模块 + 深度可分离 FPN。

    输入：CNN 输出的 C1、C2、C3。
    输出：与 C1 空间尺寸相同的分割 logits。

    head_width 控制 FPN 通道数；
    context 保持 128→768→128 的结构。
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int = 150,
        width: int = 128,
    ) -> None:
        super().__init__()

        if tuple(in_channels) != (32, 64, 128):
            raise ValueError(
                "Expected backbone channels (32, 64, 128)"
            )

        if width < 8 or width % 8:
            raise ValueError(
                "width must be a positive multiple of 8"
            )

        # 对最深层特征进行低分辨率上下文增强。
        self.context = nn.Sequential(
            InvertedResidualV3(
                128,
                128,
                768,
                kernel_size=5,
                norm="gn",
            ),
            InvertedResidualV3(
                128,
                128,
                768,
                kernel_size=5,
                norm="gn",
            ),
        )

        # 将三个尺度投影到统一通道数。
        self.lateral = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    width,
                    kernel_size=1,
                )
                for channels in in_channels
            ]
        )

        # 各尺度的空间特征处理。
        self.refine = nn.ModuleList(
            [
                SeparableConv(width, width)
                for _ in in_channels
            ]
        )

        # 拼接后的多尺度特征融合。
        self.fuse = nn.Sequential(
            SeparableConv(3 * width, width),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Conv2d(
            width,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        features: Sequence[Tensor],
    ) -> Tensor:
        if len(features) != 3:
            raise ValueError(
                "Expected three backbone feature maps"
            )

        # 保留原始 features，不进行原地修改。
        c1, c2, c3 = features
        values = [
            c1,
            c2,
            self.context(c3),
        ]

        pyramid = [
            layer(x)
            for layer, x in zip(self.lateral, values)
        ]

        # 自顶向下融合：C3 → C2 → C1。
        for index in range(2, 0, -1):
            upsampled = functional.interpolate(
                pyramid[index],
                size=pyramid[index - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            pyramid[index - 1] = (
                pyramid[index - 1] + upsampled
            )

        target_size = pyramid[0].shape[-2:]

        merged = [
            functional.interpolate(
                layer(x),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            for layer, x in zip(self.refine, pyramid)
        ]

        fused = self.fuse(
            torch.cat(merged, dim=1)
        )

        # 返回 logits，由原 engine.py 对齐标签尺寸。
        return self.classifier(fused)