"""分割头：FPNHead 保留上传脚本的结构与默认 width=128；SimpleHead 用于 CNN 预训练。"""

from __future__ import annotations

from typing import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional


class FPNHead(nn.Module):
    """可替换的多尺度融合头示例；不保证比你原有分割头更强。

    输入：[B,C1,H1,W1], ...，按空间分辨率由高到低排列。
    输出：[B,num_classes,H1,W1] 原始 logits，禁止在此添加 softmax。
    """

    def __init__(
        self, in_channels: Sequence[int], num_classes: int = 150, width: int = 128
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in in_channels])
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(width, width, 3, padding=1, bias=False),
                    nn.GroupNorm(8, width),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(width * len(in_channels), width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.classifier = nn.Conv2d(width, num_classes, 1)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != len(self.lateral):
            raise ValueError("特征层数与 in_channels 不一致。")
        pyramid = [layer(x) for layer, x in zip(self.lateral, features)]
        for i in range(len(pyramid) - 1, 0, -1):
            pyramid[i - 1] = pyramid[i - 1] + functional.interpolate(
                pyramid[i],
                size=pyramid[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        target_size = pyramid[0].shape[-2:]
        merged = [
            functional.interpolate(
                layer(x), size=target_size, mode="bilinear", align_corners=False
            )
            for layer, x in zip(self.refine, pyramid)
        ]
        return self.classifier(self.fuse(torch.cat(merged, dim=1)))


class SimpleHead(nn.Module):
    """第一阶段只用 1x1 卷积生成逐像素类别 logits，使 CNN 得到监督信号。"""

    def __init__(self, num_classes: int = 150) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        return self.classifier(features[-1])
