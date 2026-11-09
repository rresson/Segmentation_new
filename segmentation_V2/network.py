import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
from Stem_block import stem_block
from fusion_block import fusion_block, segment_block, fusion_block2, segment_block2

# ============================================================
# Import three SeaFormer Attention variants
# ============================================================
# 1. Original SeaFormer Attention
from seaformer_attention import context_block as OriginalContextBlock
# 2. Adaptive Multi-Scale Local Enhancement version
from seaformer_attention_multiscale import context_block as MultiScaleContextBlock
# 3. 2D Local Positional Encoding version
from seaformer_attention_lepe import context_block as LePEContextBlock


# ============================================================
# Auxiliary segmentation head
# ============================================================
class AuxiliaryHead(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_classes,
    ):
        super().__init__()

        self.head = nn.Sequential(

            nn.Conv2d(in_channels,hidden_channels,kernel_size=3,padding=1,bias=False,),

            nn.BatchNorm2d(hidden_channels),

            nn.ReLU6(inplace=True),

            nn.Dropout2d(p=0.1),

            nn.Conv2d(hidden_channels,num_classes,kernel_size=1,),
        )

    def forward(self,x,output_size):

        logits = self.head(x)
        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits


# ============================================================
# Context downsampling block
# ============================================================
class context_downscaling(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        expansion_ratio
    ):
        super().__init__()

        hidden_channels = int(
            in_channels * expansion_ratio
        )

        self.block = nn.Sequential(

            # pointwise expansion
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.ReLU6(
                inplace=True
            ),

            # depthwise downsampling
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.ReLU6(
                inplace=True
            ),

            # pointwise projection
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# Main Network
# ============================================================
class main_network(nn.Module):

    def __init__(self,dim,key_dim,num_heads,attn_ratio,pos_shape,use_learnable_squeeze,ffn_ratio,
        hidden_channel,num_class,expansion_ratio1,expansion_ratio2,use_aux=True,
        attention_type="original",fusion_type="original",segment_type="original",stem_pretrained=True,
    ):
        super().__init__()
        # ====================================================
        # Select attention implementation
        # ====================================================
        self.attention_type = attention_type.lower()
        if self.attention_type == "original":
            ContextBlock = OriginalContextBlock

        elif self.attention_type == "multiscale":
            ContextBlock = MultiScaleContextBlock

        elif self.attention_type == "lepe":
            ContextBlock = LePEContextBlock
        else:
            raise ValueError(
                f"Unknown attention_type: {attention_type}\n"
                f"Supported types are:\n"
                f"    'original'\n"
                f"    'multiscale'\n"
                f"    'lepe'"
            )
        print(f"[Network] Attention type: "f"{self.attention_type}")

        # ===================================================
        # Select fusion implementation
        # ===================================================
        self.fusion_type = fusion_type.lower()
        if self.fusion_type == "original":
            FusionBlock = fusion_block
        elif self.fusion_type == "enhanced":
            FusionBlock = fusion_block2
        else:
            raise ValueError(
                f"Unknown fusion_type: {fusion_type}\n"
                f"Supported types are:\n"
                f"    'original'\n"
                f"    'enhanced'"
            )

        print(
            f"[Network] Fusion type: "
            f"{self.fusion_type}"
        )

        # ============================================================
        # Select segmentation head
        # ============================================================
        self.segment_type = segment_type.lower()

        if self.segment_type == "original":

            SegmentBlock = segment_block

        elif self.segment_type == "multilevel":

            SegmentBlock = segment_block2

        else:

            raise ValueError(
                f"Unknown segment_type: {segment_type}\n"
                f"Supported types are:\n"
                f"    'original'\n"
                f"    'multilevel'"
            )

        print(
            f"[Network] Segment head: "
            f"{self.segment_type}"
        )


        # ====================================================
        # Stem
        # ====================================================
        self.stem = stem_block(pretrained=stem_pretrained)

        # MobileNetV2 modified stem output channels
        self.stem_out_channel = 64


        # ====================================================
        # Stage 1
        # ====================================================
        self.dim1 = dim
        self.key_dim1 = key_dim
        self.num_heads1 = num_heads
        self.attn_ratio1 = attn_ratio
        self.pos_shape1 = pos_shape
        self.use_learnable_squeeze1 = (
            use_learnable_squeeze
        )
        self.ffn_ratio1 = ffn_ratio

        self.context1 = nn.Sequential(

            ContextBlock(
                self.dim1,
                self.key_dim1,
                self.num_heads1,
                self.attn_ratio1,
                self.pos_shape1,
                self.use_learnable_squeeze1,
                self.ffn_ratio1
            ),

            ContextBlock(
                self.dim1,
                self.key_dim1,
                self.num_heads1,
                self.attn_ratio1,
                self.pos_shape1,
                self.use_learnable_squeeze1,
                self.ffn_ratio1
            ),
        )


        # ====================================================
        # Stage 2
        # ====================================================
        self.dim2 = dim
        self.key_dim2 = key_dim
        self.num_heads2 = num_heads
        self.attn_ratio2 = attn_ratio
        self.pos_shape2 = pos_shape
        self.use_learnable_squeeze2 = (
            use_learnable_squeeze
        )
        self.ffn_ratio2 = ffn_ratio

        self.context2 = nn.Sequential(

            ContextBlock(
                self.dim2,
                self.key_dim2,
                self.num_heads2,
                self.attn_ratio2,
                self.pos_shape2,
                self.use_learnable_squeeze2,
                self.ffn_ratio2
            ),

            ContextBlock(
                self.dim2,
                self.key_dim2,
                self.num_heads2,
                self.attn_ratio2,
                self.pos_shape2,
                self.use_learnable_squeeze2,
                self.ffn_ratio2
            ),
        )


        # ====================================================
        # Stage 3
        # ====================================================
        self.dim3 = dim
        self.key_dim3 = key_dim
        self.num_heads3 = num_heads
        self.attn_ratio3 = attn_ratio
        self.pos_shape3 = pos_shape
        self.use_learnable_squeeze3 = (
            use_learnable_squeeze
        )
        self.ffn_ratio3 = ffn_ratio

        self.context3 = nn.Sequential(

            ContextBlock(
                self.dim3,
                self.key_dim3,
                self.num_heads3,
                self.attn_ratio3,
                self.pos_shape3,
                self.use_learnable_squeeze3,
                self.ffn_ratio3
            ),

            ContextBlock(
                self.dim3,
                self.key_dim3,
                self.num_heads3,
                self.attn_ratio3,
                self.pos_shape3,
                self.use_learnable_squeeze3,
                self.ffn_ratio3
            ),
        )


        # ====================================================
        # Context downsampling
        # ====================================================

        # Stage 1 -> Stage 2
        self.expansion_ratio1 = (
            expansion_ratio1
        )

        self.context_down1 = (
            context_downscaling(
                self.dim1,
                self.dim2,
                self.expansion_ratio1
            )
        )


        # Stage 2 -> Stage 3
        self.expansion_ratio2 = (
            expansion_ratio2
        )

        self.context_down2 = (
            context_downscaling(
                self.dim2,
                self.dim3,
                self.expansion_ratio2
            )
        )


        # ====================================================
        # Fusion blocks
        # ====================================================

        self.fusion1 = FusionBlock(
            self.dim1,
            self.stem_out_channel,
            self.dim1
        )

        self.fusion2 = FusionBlock(
            self.dim2,
            self.dim1,
            self.dim2
        )

        self.fusion3 = FusionBlock(
            self.dim3,
            self.dim2,
            self.dim3
        )


        # ====================================================
        # Auxiliary segmentation head
        # ====================================================
        self.hidden_channel = (
            hidden_channel
        )

        self.use_aux = use_aux

        if self.use_aux:

            self.auxiliary_head = AuxiliaryHead(

                in_channels=self.dim2,

                hidden_channels=int(
                    2 * self.dim2
                ),

                num_classes=num_class,
            )

        # ============================================================
        # Main segmentation head
        # ============================================================

        if self.segment_type == "original":
            self.segment_head = segment_block(
                self.dim3,
                self.hidden_channel,
                num_class
            )

        elif self.segment_type == "multilevel":
            self.segment_head = segment_block2(
                in_channels=(
                    self.dim1,
                    self.dim2,
                    self.dim3,
                ),
                hidden_channel=self.hidden_channel,
                num_class=num_class,
                reduce_channel=32,
                num_refine_blocks=2,
            )

    def forward(self, x):

        # ====================================================
        # Save original image size
        # ====================================================
        output_size = x.shape[-2:]


        # ====================================================
        # Stem
        # ====================================================
        spatial = self.stem(x)


        # ====================================================
        # Stage 1
        # ====================================================
        context_1 = self.context1(
            spatial
        )

        fusion_1 = self.fusion1(
            spatial=spatial,
            context=context_1
        )


        # ====================================================
        # Stage 2
        # ====================================================
        context_2 = self.context_down1(
            context_1
        )

        context_2 = self.context2(
            context_2
        )

        fusion_2 = self.fusion2(
            spatial=fusion_1,
            context=context_2
        )


        # ====================================================
        # Stage 3
        # ====================================================
        context_3 = self.context_down2(
            context_2
        )

        context_3 = self.context3(
            context_3
        )

        fusion_3 = self.fusion3(
            spatial=fusion_2,
            context=context_3
        )

        if self.segment_type == "original":

            logits = self.segment_head(
                fusion_3,
                output_size=output_size
            )

        elif self.segment_type == "multilevel":

            logits = self.segment_head(
                fusion_1,
                fusion_2,
                fusion_3,
                output_size=output_size
            )


        # ====================================================
        # Auxiliary prediction during training
        # ====================================================
        if self.training and self.use_aux:

            aux_logits = self.auxiliary_head(
                context_2,
                output_size=output_size
            )

            return (
                logits,
                aux_logits
            )


        return logits


# ============================================================
# Simple test
# ============================================================
if __name__ == "__main__":

    model = main_network(

        dim=64,

        key_dim=16,

        num_heads=4,

        attn_ratio=2,

        pos_shape=16,

        # For the experiments discussed here,
        # keep this False.
        use_learnable_squeeze=False,

        ffn_ratio=2,

        hidden_channel=128,

        num_class=150,

        expansion_ratio1=2,

        expansion_ratio2=2,

        use_aux=True,

        # --------------------------------------------
        # Choose one:
        #
        # "original"
        # "multiscale"
        # "lepe"
        # --------------------------------------------
        attention_type="multiscale",
        fusion_type="enhanced",
    )

    x = torch.randn(
        1,
        3,
        512,
        512
    )

    model.eval()

    with torch.no_grad():

        y = model(x)

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        y.shape
    )