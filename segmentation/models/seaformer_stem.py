"""上传 Stem_block.py 的 MobileNetV2 stem，保持模块、通道与步幅不变。"""
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class stem_block(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # 仅在从零建立新实验时下载/读取 ImageNet 权重；完整断点会覆盖它。
        mobilenet = mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        mobilenet.features[7].conv[1][0].stride = (1, 1)
        self.backbone = nn.Sequential(*list(mobilenet.features.children())[:8])

    def forward(self, x):
        return self.backbone(x)
