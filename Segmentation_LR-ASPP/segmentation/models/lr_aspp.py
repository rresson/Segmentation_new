"""Figure 10 of Searching for MobileNetV3, adapted to the V2 fusion outputs."""
from collections.abc import Sequence

from torch import Tensor, nn
from torch.nn import functional as F


class LRASPPHead(nn.Module):
    """接收 V2 的 (f1, f3)，返回与 f1 空间尺寸相同的类别 logits。"""

    def __init__(self, in_channels: int, num_classes: int, width: int = 128):
        super().__init__()
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

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        low, fused = features
        high = self.high_pool(fused)
        h, w = high.shape[-2:]
        # 图 10 的 49×49 核，在小 crop 上按轴截到实际特征尺寸。
        context = F.avg_pool2d(high, kernel_size=(min(49, h), min(49, w)), stride=(16, 20))
        gate = F.interpolate(self.gate(context), size=(h, w), mode="bilinear", align_corners=False)
        gated = self.project(high) * gate
        upsampled = F.interpolate(gated, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.low_classifier(low) + self.high_classifier(upsampled)
