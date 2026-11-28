"""CNN 结构：保留上传文件的 DemoBackbone，并提供可简单训练的 SmallCNNBackbone。"""

from __future__ import annotations

from torch import Tensor, nn


class DemoBackbone(nn.Module):
    """接口演示：返回由浅到深的三层特征。请替换为你已有的网络。"""

    out_channels = (32, 64, 128)

    def __init__(self) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.stage1 = block(3, 32)
        self.stage2 = block(32, 64)
        self.stage3 = block(64, 128)

    def forward(self, x: Tensor) -> list[Tensor]:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        return [c1, c2, c3]


class ConvBlock(nn.Sequential):
    """两个 3x3 卷积；第一个卷积可以下采样，第二个保持分辨率。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, 3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SmallCNNBackbone(nn.Module):
    """RGB 输入，返回三层二维特征；没有 GAP、全连接层或 Softmax。

    输入 [B,3,256,256] 时，输出依次为：
        [B,32,64,64]、[B,64,32,32]、[B,128,16,16]。
    通道数 (32,64,128) 可直接匹配之前示例 FPNHead 的 in_channels。
    """

    out_channels = (32, 64, 128)

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBlock(3, 16, stride=2)
        self.stage1 = ConvBlock(16, 32, stride=2)
        self.stage2 = ConvBlock(32, 64, stride=2)
        self.stage3 = ConvBlock(64, 128, stride=2)

    def forward(self, images: Tensor) -> list[Tensor]:
        x = self.stem(images)
        feature1 = self.stage1(x)
        feature2 = self.stage2(feature1)
        feature3 = self.stage3(feature2)
        return [feature1, feature2, feature3]
