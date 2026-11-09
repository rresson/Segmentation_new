import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)


class stem_block(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Load ImageNet-1K pretrained MobileNetV2
        mobilenet = mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT if pretrained else None
        )

        # Original features[7]:
        # 32 -> 192 -> 64
        # depthwise stride = 2
        #
        # Change stride from 2 to 1:
        # keep output spatial resolution at H/8 x W/8
        mobilenet.features[7].conv[1][0].stride = (1, 1)

        # Extract features[0] ~ features[7]
        #
        # Input:
        # [B, 3, H, W]
        #
        # Output:
        # [B, 64, H/8, W/8]
        self.backbone = nn.Sequential(
            *list(mobilenet.features.children())[:8]
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":

    model = stem_block()

    x = torch.randn(1, 3, 512, 512)

    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    print(model.backbone)