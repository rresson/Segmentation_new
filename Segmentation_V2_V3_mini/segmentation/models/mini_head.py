"""单 C3、一个 IR+SE context 的 mini 分割头；默认 128→368→128。"""
from torch import nn
from .mobile_v3_blocks import ConvNormAct, InvertedResidualV3


class SeparableConv(nn.Sequential):
    """与原 MobileV3 C3 refine 一致的 DW + PW + GN。"""
    def __init__(self, channels):
        super().__init__(
            ConvNormAct(channels, channels, kernel_size=3, groups=channels, norm="gn"),
            ConvNormAct(channels, channels, norm="gn"),
        )


class MiniC3Head(nn.Module):
    """C3-only: preserve output stride 16, one true PW-expand/DW/SE/PW-project IR.

    width controls the context, lateral, refine and classifier input channels.
    The backbone always supplies a 64-channel C3 feature.
    """
    def __init__(self, num_classes=150, expanded_channels=368, width=128):
        super().__init__()
        if not isinstance(width, int) or isinstance(width, bool) or width < 8 or width % 8:
            raise ValueError('width must be a positive integer divisible by 8')
        if (not isinstance(expanded_channels, int) or isinstance(expanded_channels, bool)
                or expanded_channels <= width or expanded_channels % 8):
            raise ValueError('expanded_channels must be an integer >width and divisible by 8')
        self.width = width
        self.expanded_channels = expanded_channels
        self.input_projection = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.Conv2d(64, width, 1, bias=False), nn.BatchNorm2d(width),
        )
        self.context = nn.Sequential(
            InvertedResidualV3(width, width, expanded_channels, kernel_size=5,
                               norm='gn', use_se=True),
        )
        self.lateral = nn.Conv2d(width, width, 1)
        self.refine = SeparableConv(width)
        self.classifier = nn.Conv2d(width, num_classes, 1)

    def forward(self, features):
        if not isinstance(features, (list, tuple)) or len(features) != 1:
            raise ValueError('Expected exactly one C3 feature map')
        x = self.input_projection(features[0])
        return self.classifier(self.refine(self.lateral(self.context(x))))
