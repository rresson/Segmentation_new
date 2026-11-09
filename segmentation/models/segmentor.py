"""组合 CNN 和分割头，保持原训练和统计接口。"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn

from ..config import Config
from .cnn import DemoBackbone, SmallCNNBackbone
from .heads import FPNHead, SimpleHead
from .mobile_v3_cnn import MobileV3Backbone
from .mobile_v3_head import MobileV3C3Head, MobileV3FPNHead


class Segmentor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x: Tensor) -> Tensor:
        frozen = not any(
            parameter.requires_grad
            for parameter in self.backbone.parameters()
        )

        # 全部冻结时关闭 backbone 的梯度记录。
        # 部分微调时保留完整梯度路径。
        with torch.no_grad() if frozen else nullcontext():
            features = self.backbone(x)

        return self.decode_head(features)


def build_model(cfg: Config, *, initialize_stem: bool | None = None) -> Segmentor:
    backbone: (
        SmallCNNBackbone
        | DemoBackbone
        | MobileV3Backbone
    )

    if cfg.backbone_name == "small_cnn":
        backbone = SmallCNNBackbone()
    elif cfg.backbone_name == "demo":
        backbone = DemoBackbone()
    elif cfg.backbone_name == "mobile_v3":
        backbone = MobileV3Backbone()
    elif cfg.backbone_name == "seaformer":
        # 延迟导入：旧模型无需安装 torchvision；恢复完整权重不需要联网。
        from .seaformer import SeaFormerBackbone

        if initialize_stem is None:
            initialize_stem = cfg.stem_pretrained and not (cfg.resume or cfg.backbone_checkpoint)
        backbone = SeaFormerBackbone(
            pretrained_stem=bool(initialize_stem),
            attention=cfg.seaformer_attention,
            depths=cfg.seaformer_depths,
            use_learnable_squeeze=cfg.seaformer_learnable_squeeze,
        )
    else:
        raise ValueError(
            f"未知 backbone：{cfg.backbone_name}"
        )

    head: nn.Module

    if cfg.backbone_name == "seaformer" and cfg.head_name != "mobile_c3":
        raise ValueError("SeaFormer supports only the single-input MobileV3C3Head")

    if cfg.head_name == "mobile_c3":
        if len(backbone.out_channels) != 1:
            raise ValueError("MobileV3C3Head requires a single-output backbone")
        head = MobileV3C3Head(backbone.out_channels[0], cfg.num_classes, cfg.head_width)
    elif cfg.head_name == "fpn":
        head = FPNHead(
            backbone.out_channels,
            cfg.num_classes,
            cfg.head_width,
        )
    elif cfg.head_name == "simple":
        head = SimpleHead(cfg.num_classes)
    elif cfg.head_name == "mobile_fpn":
        head = MobileV3FPNHead(
            backbone.out_channels,
            cfg.num_classes,
            cfg.head_width,
        )
    else:
        raise ValueError(
            f"未知分割头：{cfg.head_name}"
        )

    return Segmentor(
        backbone=backbone,
        decode_head=head,
    )
