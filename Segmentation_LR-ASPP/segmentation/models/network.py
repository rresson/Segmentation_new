"""V2 context stages 的 stride-2 下采样，保留原 checkpoint 的层编号。"""
from torch import nn


class context_downscaling(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        expansion_ratio
    ):
        super().__init__()

        hidden_channels = int(
            in_channels * expansion_ratio
        )

        self.block = nn.Sequential(

            # pointwise expansion
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.ReLU6(
                inplace=True
            ),

            # depthwise downsampling
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.ReLU6(
                inplace=True
            ),

            # pointwise projection
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            )
        )

    def forward(self, x):

        return self.block(x)
