"""组合 CNN 和分割头，保持原训练和统计接口。"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn

from ..config import Config
from .lr_aspp import LRASPPHead
from .seaformer import SeaFormerBackbone


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
            f1, _, f3, _ = self.backbone(x)
        return self.decode_head((f1, f3))


def build_model(cfg: Config, *, initialize_stem: bool | None = None) -> Segmentor:
    if cfg.backbone_name != "seaformer" or cfg.head_name != "lr_aspp":
        raise ValueError("This project supports only seaformer + lr_aspp")
    if initialize_stem is None:
        initialize_stem = cfg.stem_pretrained and not (
            cfg.resume or cfg.backbone_checkpoint or cfg.model_checkpoint
        )
    backbone = SeaFormerBackbone(
        pretrained_stem=bool(initialize_stem), attention=cfg.seaformer_attention,
        depths=cfg.seaformer_depths, use_learnable_squeeze=cfg.seaformer_learnable_squeeze,
    )
    return Segmentor(backbone, LRASPPHead(backbone.out_channels[2], cfg.num_classes,
                                         cfg.head_width, cfg.head_refine_channels))
