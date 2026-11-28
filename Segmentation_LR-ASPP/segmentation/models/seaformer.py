"""V2 encoder with the original state_dict names and no classification layers."""
from torch import nn

from .Stem_block import stem_block
from .fusion_block import fusion_block2
from .network import context_downscaling
from .seaformer_attention import context_block as OriginalContext
from .seaformer_attention_multiscale import context_block as MultiScaleContext
from .seaformer_attention_lepe import context_block as LePEContext
from .uploaded_backbone import UploadedBackbone


def backbone_network_config(attention='original', use_learnable_squeeze=True):
    return dict(dim=64, key_dim=16, num_heads=4, attn_ratio=2, pos_shape=16,
                use_learnable_squeeze=use_learnable_squeeze, ffn_ratio=2,
                hidden_channel=128, num_class=150, expansion_ratio1=2,
                expansion_ratio2=2, use_aux=False, attention_type=attention,
                fusion_type='enhanced', segment_type='original')


class SeaFormerBackbone(UploadedBackbone):
    output_stride = 8

    def __init__(self, pretrained_stem=True, attention='original',
                 depths=(2, 2, 2), use_learnable_squeeze=True):
        if tuple(depths) != (2, 2, 2):
            raise ValueError('Matching segmentation_V2 requires depths=(2, 2, 2)')
        super().__init__()
        cfg = backbone_network_config(attention, use_learnable_squeeze)
        context_types = {"original": OriginalContext, "multiscale": MultiScaleContext, "lepe": LePEContext}
        if attention not in context_types:
            raise ValueError(f"Unknown attention type: {attention}")
        context = context_types[attention]
        dim = cfg["dim"]

        def stage(depth):
            return nn.Sequential(*(context(dim, cfg["key_dim"], cfg["num_heads"], cfg["attn_ratio"],
                                           cfg["pos_shape"], use_learnable_squeeze, cfg["ffn_ratio"])
                                   for _ in range(depth)))

        self.stem = stem_block(pretrained=pretrained_stem)
        self.context1 = stage(depths[0])
        self.context2 = stage(depths[1])
        self.context3 = stage(depths[2])
        self.context_down1 = context_downscaling(dim, dim, cfg["expansion_ratio1"])
        self.context_down2 = context_downscaling(dim, dim, cfg["expansion_ratio2"])
        self.fusion1 = fusion_block2(dim, dim, dim)
        self.fusion2 = fusion_block2(dim, dim, dim)
        self.fusion3 = fusion_block2(dim, dim, dim)
        self.out_channels = (dim, dim, dim)
