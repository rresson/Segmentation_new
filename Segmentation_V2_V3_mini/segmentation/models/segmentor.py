"""相同 V2 backbone + C3 单路 mini head，冻结边界与原工程一致。"""
from contextlib import nullcontext

import torch
from torch import nn

from ..config import Config, validate_model_config
from .seaformer import SeaFormerBackbone
from .mini_head import MiniC3Head


class Segmentor(nn.Module):
    def __init__(self, backbone, decode_head):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x):
        frozen = not any(p.requires_grad for p in self.backbone.parameters())
        with torch.no_grad() if frozen else nullcontext():
            features = self.backbone(x)
        return self.decode_head([features[2]])


def build_model(cfg: Config, *, initialize_stem: bool | None = None) -> Segmentor:
    validate_model_config(cfg)
    if initialize_stem is None:
        initialize_stem = cfg.stem_pretrained and not (
            cfg.resume or cfg.backbone_checkpoint or cfg.model_checkpoint
        )
    return Segmentor(
        SeaFormerBackbone(pretrained_stem=bool(initialize_stem), attention=cfg.seaformer_attention,
                          depths=cfg.seaformer_depths, use_learnable_squeeze=cfg.seaformer_learnable_squeeze),
        MiniC3Head(num_classes=cfg.num_classes, expanded_channels=cfg.context_expanded_channels,
                   width=cfg.head_width),
    )
