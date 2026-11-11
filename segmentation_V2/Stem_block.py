"""纯 PyTorch MobileNetV2 stem：无需导入 torchvision，兼容原 stem 权重。"""
import torch
from torch import nn


# 与原 MobileNet_V2_Weights.DEFAULT 对应的 ImageNet1K V2 权重。
IMAGENET_WEIGHTS_URL = (
    "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth"
)


class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=(kernel_size - 1) // 2,
                groups=groups, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        self.out_channels = out_channels


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        hidden_channels = int(round(in_channels * expand_ratio))
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.out_channels = out_channels
        self._is_cn = stride > 1

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_channels, kernel_size=1))
        layers.extend([
            ConvBNReLU6(hidden_channels, hidden_channels, stride=stride, groups=hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        # 保持原 MobileNetV2 的层编号，以兼容 conv.* state_dict 键。
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv(x)
        return x + output if self.use_res_connect else output


class stem_block(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 对应 MobileNetV2 features[0] 至 features[7]。
        self.backbone = nn.Sequential(
            ConvBNReLU6(3, 32, stride=2),
            InvertedResidual(32, 16, stride=1, expand_ratio=1),
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
        )
        # 保留用户原 stem 的步幅调整，最终输出 64 通道、1/8 分辨率。
        self.backbone[7].conv[1][0].stride = (1, 1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        if pretrained:
            self._load_imagenet_weights()

    def _load_imagenet_weights(self):
        # 通过 PyTorch 直接读取/下载官方权重，不导入 torchvision。
        # 复用 torch.hub 缓存；新下载文件校验文件名中的 SHA256 前缀。
        full_state = torch.hub.load_state_dict_from_url(
            IMAGENET_WEIGHTS_URL,
            map_location="cpu",
            progress=True,
            check_hash=True,
            weights_only=True,
        )
        prefixes = tuple(f"features.{index}." for index in range(8))
        stem_state = {
            name.removeprefix("features."): value
            for name, value in full_state.items()
            if name.startswith(prefixes)
        }
        self.backbone.load_state_dict(stem_state, strict=True)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = stem_block(pretrained=False).eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256)
        y = model(x)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
