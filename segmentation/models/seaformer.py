"""固定原始 stem，低分辨率上下文 + 单次最终融合的 SeaFormer backbone。"""
import torch
from torch import nn
from torch.nn import functional as F

from .seaformer_attention import ContextBlock, ConvBN
from .seaformer_stem import stem_block


class ContextDownscaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            ConvBN(64, 128, 1), nn.ReLU6(True),
            ConvBN(128, 128, 3, stride=2, padding=1, groups=128), nn.ReLU6(True),
            ConvBN(128, 64, 1))

    def forward(self, x):
        return self.block(x)


class FinalFusion(nn.Module):
    """保留上传 enhanced fusion 的局部细化和语义注入，仅执行一次。"""
    def __init__(self):
        super().__init__()
        self.spatial_projection = ConvBN(64, 64, 1)
        self.spatial_refine = nn.Sequential(
            ConvBN(64, 64, 3, padding=1, groups=64), nn.ReLU6(True),
            ConvBN(64, 64, 1, bn_weight_init=0))
        self.context_project = ConvBN(64, 64, 1)
        self.context_gate = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.zeros_(self.context_gate.bias)

    def forward(self, spatial, context):
        spatial = self.spatial_projection(spatial)
        spatial = spatial + self.spatial_refine(spatial)
        context = F.interpolate(self.context_project(context), size=spatial.shape[-2:],
                                mode="bilinear", align_corners=False)
        return spatial + torch.sigmoid(self.context_gate(context)) * context


class SeaFormerBackbone(nn.Module):
    # 单一输出就是分割头的 C3 输入；不是三个伪造的重复尺度。
    out_channels = (128,)
    output_stride = 16

    def __init__(self, pretrained_stem=True, attention="multiscale",
                 depths=(1, 2, 2), use_learnable_squeeze=False):
        super().__init__()
        if len(depths) != 3 or any(type(depth) is not int or depth < 1 for depth in depths):
            raise ValueError("SeaFormer depths must contain three positive integers")
        self.stem = stem_block(pretrained=pretrained_stem)
        for index, depth in enumerate(depths, 1):
            setattr(self, f"context{index}", nn.Sequential(*[
                ContextBlock(attention, use_learnable_squeeze) for _ in range(depth)]))
        self.context_down1 = ContextDownscaling()
        self.context_down2 = ContextDownscaling()
        self.fusion3 = FinalFusion()
        # 原 MobileV3FPNHead 的 C3 接口是 128 通道、1/16 分辨率。
        # 适配发生在 backbone 内，stem 本身及 head 的 context 均不改变。
        self.c3_projection = nn.Sequential(
            ConvBN(64, 64, 3, stride=2, padding=1, groups=64), nn.ReLU6(True),
            ConvBN(64, 128, 1))

    def forward(self, x):
        spatial = self.stem(x)  # [B, 64, ceil(H/8), ceil(W/8)]
        c1 = self.context1(spatial)
        c2 = self.context2(self.context_down1(c1))
        c3 = self.context3(self.context_down2(c2))
        # C3 路径在 backbone 内注入 stem 空间细节，head 只收到这一张特征图。
        return [self.c3_projection(self.fusion3(spatial, c3))]
