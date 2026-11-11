"""与 segmentation_V2 完全相同的 backbone；C3 接口适配属于 decode_head。"""
from .network import main_network
from .uploaded_backbone import UploadedBackbone


def backbone_network_config(attention='multiscale', use_learnable_squeeze=False):
    return dict(dim=64, key_dim=16, num_heads=4, attn_ratio=2, pos_shape=16,
                use_learnable_squeeze=use_learnable_squeeze, ffn_ratio=2,
                hidden_channel=128, num_class=150, expansion_ratio1=2,
                expansion_ratio2=2, use_aux=False, attention_type=attention,
                fusion_type='enhanced', segment_type='original')


class SeaFormerBackbone(UploadedBackbone):
    output_stride = 8

    def __init__(self, pretrained_stem=True, attention='multiscale',
                 depths=(2, 2, 2), use_learnable_squeeze=False):
        if tuple(depths) != (2, 2, 2):
            raise ValueError('Matching segmentation_V2 requires depths=(2, 2, 2)')
        source = main_network(**backbone_network_config(attention, use_learnable_squeeze),
                              stem_pretrained=pretrained_stem, verbose=False)
        super().__init__(source)
