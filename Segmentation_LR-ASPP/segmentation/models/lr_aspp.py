"""LR-ASPP with stride-16 residual refinement, using V2's fusion outputs."""
from collections.abc import Sequence

from torch import Tensor, nn
from torch.nn import functional as F


class LRASPPHead(nn.Module):
    """接收 V2 的 (f1, f3)，返回与 f1 空间尺寸相同的类别 logits。"""

    def __init__(self, in_channels: int, num_classes: int, width: int = 168,
                 refine_channels: int = 552):
        super().__init__()
        if width <= 0 or refine_channels < 0:
            raise ValueError("width must be positive; head_refine_channels must be nonnegative")
        # 两路分类器含 bias；其余卷积无 bias，BN 有 scale 和 offset。
        params = (2 * in_channels + 2 + num_classes) * width + (in_channels + 2) * num_classes
        if refine_channels:
            params += (2 * width + 13) * refine_channels + 2 * width
        if params > 250_000:
            raise ValueError(f"LR-ASPP would have {params:,} parameters, exceeding 250,000")
        # V2 的最终融合 f3 是 stride 8。只在 head 内构造 stride-16 表示。
        self.high_pool = nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=False)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(nn.Conv2d(in_channels, width, 1, bias=False), nn.Sigmoid())
        self.low_classifier = nn.Conv2d(in_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(width, num_classes, 1)
        self.refine = None
        if refine_channels:
            self.refine = nn.Sequential(
                nn.Conv2d(width, refine_channels, 1, bias=False),
                nn.BatchNorm2d(refine_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(refine_channels, refine_channels, 3, padding=1,
                          groups=refine_channels, bias=False),
                nn.BatchNorm2d(refine_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(refine_channels, width, 1, bias=False),
                nn.BatchNorm2d(width),
            )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        low, fused = features
        high = self.high_pool(fused)
        h, w = high.shape[-2:]
        # 图 10 的 49×49 核，在小 crop 上按轴截到实际特征尺寸。
        context = F.avg_pool2d(high, kernel_size=(min(49, h), min(49, w)), stride=(16, 20))
        gate = F.interpolate(self.gate(context), size=(h, w), mode="bilinear", align_corners=False)
        semantic = self.project(high)
        if self.refine is not None:
            semantic = F.relu(semantic + self.refine(semantic))
        gated = semantic * gate
        upsampled = F.interpolate(gated, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.low_classifier(low) + self.high_classifier(upsampled)
