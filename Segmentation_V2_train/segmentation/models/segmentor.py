"""仅拆分上传 main_network 的模块归属，完整保留其计算图。"""
from contextlib import nullcontext
import torch
from torch import nn
from .network import main_network
from ..config import Config, network_kwargs


BACKBONE_MODULES = ("stem", "context1", "context2", "context3",
                    "context_down1", "context_down2", "fusion1", "fusion2", "fusion3")


class UploadedBackbone(nn.Module):
    def __init__(self, source):
        super().__init__()
        for name in BACKBONE_MODULES:
            self.add_module(name, getattr(source, name))
        self.out_channels = (source.dim1, source.dim2, source.dim3)

    def forward(self, x):
        spatial = self.stem(x)
        c1 = self.context1(spatial)
        f1 = self.fusion1(spatial=spatial, context=c1)
        c2 = self.context2(self.context_down1(c1))
        f2 = self.fusion2(spatial=f1, context=c2)
        c3 = self.context3(self.context_down2(c2))
        f3 = self.fusion3(spatial=f2, context=c3)
        return f1, f2, f3, c2  # c2 仅供原始 auxiliary_head 使用


class UploadedHead(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.segment_type = source.segment_type
        self.segment_head = source.segment_head
        self.use_aux = source.use_aux
        if self.use_aux:
            self.auxiliary_head = source.auxiliary_head

    def forward(self, features, output_size, return_aux=False):
        f1, f2, f3, c2 = features
        if self.segment_type == "original":
            logits = self.segment_head(f3, output_size=output_size)
        else:
            logits = self.segment_head(f1, f2, f3, output_size=output_size)
        if return_aux and self.use_aux:
            return logits, self.auxiliary_head(c2, output_size=output_size)
        return logits


class Segmentor(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.backbone = UploadedBackbone(source)
        self.decode_head = UploadedHead(source)

    def _forward(self, x, return_aux):
        frozen = not any(p.requires_grad for p in self.backbone.parameters())
        with torch.no_grad() if frozen else nullcontext():
            features = self.backbone(x)
        return self.decode_head(features, x.shape[-2:], return_aux=return_aux)

    def forward(self, x):
        return self._forward(x, self.training and self.decode_head.use_aux)

    def profile_forward(self, x):
        # 用 eval 状态执行训练用到的全部分支，既计入 aux 又不改变 BN/Dropout。
        return self._forward(x, self.decode_head.use_aux)


def build_model(cfg: Config, *, initialize_stem: bool | None = None) -> Segmentor:
    if initialize_stem is None:
        initialize_stem = cfg.stem_pretrained and not (
            cfg.resume or cfg.backbone_checkpoint or cfg.model_checkpoint)
    return Segmentor(main_network(**network_kwargs(cfg), stem_pretrained=bool(initialize_stem)))
