"""合并上传的三种 SeaAttention；共享轴向计算，保留各变体的参数名。"""
import torch
from torch import nn
from torch.nn import functional as F


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bn_weight_init=1.0):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.zeros_(bn.bias)
        super().__init__(conv, bn)


class MatMul(nn.Module):
    """显式模块让统计器观测每一次 QK^T / AV，而不重复统计子卷积。"""
    macs_kind = "attention_macs"

    def forward(self, left, right):
        return torch.matmul(left, right)

    def count_macs(self, args, output):
        return output.numel() * args[0].shape[-1]


class SqueezeAxialPositionalEmbedding4D(nn.Module):
    def __init__(self, dim, shape=16, axis="row"):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, dim, shape))
        self.axis = axis

    def forward(self, x):
        size = x.shape[2] if self.axis == "row" else x.shape[3]
        pos = F.interpolate(self.pos_embed, size=size, mode="linear", align_corners=False)
        return x + pos.unsqueeze(-1 if self.axis == "row" else 2)


class LocalPositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local_pos = ConvBN(channels, channels, 3, padding=1,
                                groups=channels, bn_weight_init=0.001)

    def forward(self, x):
        return self.local_pos(x)


class AdaptiveMultiScaleLocalEnhancement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local_3x3 = nn.Sequential(
            ConvBN(channels, channels, 3, padding=1, groups=channels), nn.ReLU6(True))
        self.local_dilated = nn.Sequential(
            ConvBN(channels, channels, 3, padding=2, dilation=2, groups=channels),
            nn.ReLU6(True))
        hidden = max(channels // 4, 8)
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU6(True),
            nn.Conv2d(hidden, 2 * channels, 1))
        nn.init.zeros_(self.scale_selector[-1].weight)
        nn.init.zeros_(self.scale_selector[-1].bias)
        self.local_fuse = ConvBN(channels, channels, 1, bn_weight_init=0.001)

    def forward(self, v):
        b, c = v.shape[:2]
        weights = self.scale_selector(v).reshape(b, 2, c, 1, 1).softmax(dim=1)
        local = weights[:, 0] * self.local_3x3(v) + weights[:, 1] * self.local_dilated(v)
        return self.local_fuse(local)


class SeaAttention(nn.Module):
    def __init__(self, dim=64, key_dim=16, num_heads=4, attn_ratio=2,
                 pos_shape=16, use_learnable_squeeze=False, variant="multiscale"):
        super().__init__()
        if variant not in ("original", "multiscale", "lepe"):
            raise ValueError(f"Unknown attention variant: {variant}")
        self.num_heads, self.key_dim = num_heads, key_dim
        self.value_dim = int(attn_ratio * key_dim)
        self.qk_channels = key_dim * num_heads
        self.v_channels = self.value_dim * num_heads
        self.scale = key_dim ** -0.5
        self.use_learnable_squeeze = use_learnable_squeeze
        self.to_q = ConvBN(dim, self.qk_channels, 1)
        self.to_k = ConvBN(dim, self.qk_channels, 1)
        self.to_v = ConvBN(dim, self.v_channels, 1)
        if use_learnable_squeeze:
            self.score_predictor = nn.Conv2d(dim, 6, 1)
            if variant != "original":
                nn.init.zeros_(self.score_predictor.weight)
                nn.init.zeros_(self.score_predictor.bias)
        qkv = 2 * self.qk_channels + self.v_channels
        self.detail_dwconv = ConvBN(qkv, qkv, 3, padding=1, groups=qkv)
        self.detail_pwconv = ConvBN(qkv, dim, 1)
        for axis in ("row", "col"):
            for name in ("q", "k"):
                setattr(self, f"pos_emb_{axis}{name}",
                        SqueezeAxialPositionalEmbedding4D(self.qk_channels, pos_shape, axis))
            setattr(self, f"proj_encode_{axis}", nn.Sequential(
                nn.ReLU6(True), ConvBN(self.v_channels, self.v_channels, 1, bn_weight_init=0)))
        if variant == "multiscale":
            self.local_enhance = AdaptiveMultiScaleLocalEnhancement(self.v_channels)
        elif variant == "lepe":
            self.local_pos_2d = LocalPositionalEncoding2D(self.v_channels)
        self.proj = ConvBN(self.v_channels, dim, 1, bn_weight_init=0)
        self.matmul = MatMul()

    def forward(self, x):
        b, _, h, w = x.shape
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        detail = self.detail_pwconv(F.relu6(
            self.detail_dwconv(torch.cat((q, k, v), dim=1)), inplace=True))
        scores = self.score_predictor(x).chunk(6, dim=1) if self.use_learnable_squeeze else None
        outputs = []
        for index, (axis, reduce_dim, length) in enumerate((("row", 3, h), ("col", 2, w))):
            values = []
            for offset, value in enumerate((q, k, v)):
                if scores is None:
                    values.append(value.mean(dim=reduce_dim, keepdim=True))
                else:
                    weight = scores[index * 3 + offset].softmax(dim=reduce_dim)
                    values.append((value * weight).sum(dim=reduce_dim, keepdim=True))
            qa = getattr(self, f"pos_emb_{axis}q")(values[0])
            ka = getattr(self, f"pos_emb_{axis}k")(values[1])
            qa = qa.reshape(b, self.num_heads, self.key_dim, length).transpose(-1, -2)
            ka = ka.reshape(b, self.num_heads, self.key_dim, length)
            va = values[2].reshape(b, self.num_heads, self.value_dim, length).transpose(-1, -2)
            weights = (self.matmul(qa, ka) * self.scale).softmax(dim=-1)
            output = self.matmul(weights, va).transpose(-1, -2)
            shape = (h, 1) if axis == "row" else (1, w)
            output = output.reshape(b, self.v_channels, *shape)
            outputs.append(getattr(self, f"proj_encode_{axis}")(output))
        context = outputs[0] + outputs[1] + v
        if hasattr(self, "local_enhance"):
            context = context + self.local_enhance(v)
        elif hasattr(self, "local_pos_2d"):
            context = context + self.local_pos_2d(v)
        return (F.relu6(self.proj(context) + 3.0) / 6.0) * detail


class FFNBlock(nn.Module):
    def __init__(self, dim=64, ratio=2):
        super().__init__()
        hidden = int(dim * ratio)
        self.ffn = nn.Sequential(
            ConvBN(dim, hidden, 1), nn.ReLU6(True),
            ConvBN(hidden, hidden, 3, padding=1, groups=hidden), nn.ReLU6(True),
            ConvBN(hidden, dim, 1, bn_weight_init=0))

    def forward(self, x):
        return self.ffn(x)


class ContextBlock(nn.Module):
    def __init__(self, variant="multiscale", use_learnable_squeeze=False):
        super().__init__()
        self.attention = SeaAttention(variant=variant, use_learnable_squeeze=use_learnable_squeeze)
        self.ffn = FFNBlock()

    def forward(self, x):
        x = x + self.attention(x)
        return x + self.ffn(x)
