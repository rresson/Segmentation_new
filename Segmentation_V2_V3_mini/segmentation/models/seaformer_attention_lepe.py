import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


"""build the basic convolutional block"""
class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bn_weight_init=1.0
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )

        bn = nn.BatchNorm2d(out_channels)

        nn.init.constant_(
            bn.weight,
            bn_weight_init
        )

        nn.init.constant_(
            bn.bias,
            0.0
        )

        super().__init__(
            conv,
            bn
        )


class Hard_sigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(
            x + 3.0
        ) / 6.0


class SqueezeAxialPositionalEmbedding4D(nn.Module):
    def __init__(
        self,
        dim,
        shape=16,
        axis="row"
    ):
        super().__init__()

        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                dim,
                shape
            )
        )

        self.axis = axis

    def forward(self, x):

        B, C, H, W = x.shape

        if self.axis == "row":

            N = H

            pos = F.interpolate(
                self.pos_embed,
                size=N,
                mode="linear",
                align_corners=False
            )

            pos = pos.unsqueeze(-1)

        elif self.axis == "col":

            N = W

            pos = F.interpolate(
                self.pos_embed,
                size=N,
                mode="linear",
                align_corners=False
            )

            pos = pos.unsqueeze(2)

        else:
            raise ValueError(
                f"Unknown axis: {self.axis}"
            )

        return x + pos


# ============================================================
# NEW:
# 2D Local Positional Encoding
# ============================================================
class LocalPositionalEncoding2D(nn.Module):
    """
    Lightweight 2D local positional encoding.

    Input:
        [B, C, H, W]

    Operation:
        3x3 depthwise convolution

    Output:
        [B, C, H, W]

    The depthwise convolution introduces a 2D local
    inductive bias and preserves the spatial resolution.
    """

    def __init__(
        self,
        channels,
        residual_init=1e-3
    ):
        super().__init__()

        self.local_pos = ConvBN(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=channels,
            bn_weight_init=residual_init
        )

    def forward(self, x):
        return self.local_pos(x)


class SeaAttention(nn.Module):
    def __init__(
        self,
        dim=64,
        key_dim=16,
        num_heads=4,
        attn_ratio=2,
        pos_shape=16,
        use_learnable_squeeze=False
    ):
        super().__init__()

        self.dim = dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.attn_ratio = attn_ratio
        self.use_learnable_squeeze = (
            use_learnable_squeeze
        )

        self.qk_channels = (
            key_dim * num_heads
        )

        self.v_channels = (
            int(attn_ratio * key_dim)
            * num_heads
        )

        self.value_dim = (
            self.v_channels
            // num_heads
        )

        self.scale = (
            key_dim ** -0.5
        )

        # ====================================================
        # QKV projection
        # ====================================================
        self.to_q = ConvBN(
            dim,
            self.qk_channels,
            kernel_size=1
        )

        self.to_k = ConvBN(
            dim,
            self.qk_channels,
            kernel_size=1
        )

        self.to_v = ConvBN(
            dim,
            self.v_channels,
            kernel_size=1
        )

        # ====================================================
        # Optional learnable squeeze
        # ====================================================
        if self.use_learnable_squeeze:

            self.score_predictor = nn.Conv2d(
                self.dim,
                6,
                kernel_size=1,
                bias=True
            )

            nn.init.zeros_(
                self.score_predictor.weight
            )

            nn.init.zeros_(
                self.score_predictor.bias
            )

        # ====================================================
        # Original detail enhancement branch
        # ====================================================
        qkv_channels = (
            self.qk_channels * 2
            +
            self.v_channels
        )

        self.detail_dwconv = ConvBN(
            qkv_channels,
            qkv_channels,
            kernel_size=3,
            padding=1,
            groups=qkv_channels
        )

        self.detail_pwconv = ConvBN(
            qkv_channels,
            dim,
            kernel_size=1
        )

        # ====================================================
        # Original 1D axial positional embedding
        # ====================================================
        self.pos_emb_rowq = (
            SqueezeAxialPositionalEmbedding4D(
                self.qk_channels,
                pos_shape,
                axis="row"
            )
        )

        self.pos_emb_rowk = (
            SqueezeAxialPositionalEmbedding4D(
                self.qk_channels,
                pos_shape,
                axis="row"
            )
        )

        self.pos_emb_colq = (
            SqueezeAxialPositionalEmbedding4D(
                self.qk_channels,
                pos_shape,
                axis="col"
            )
        )

        self.pos_emb_colk = (
            SqueezeAxialPositionalEmbedding4D(
                self.qk_channels,
                pos_shape,
                axis="col"
            )
        )

        # ====================================================
        # Original row / col projection
        # ====================================================
        self.proj_encode_row = nn.Sequential(

            nn.ReLU6(inplace=True),

            ConvBN(
                self.v_channels,
                self.v_channels,
                kernel_size=1,
                bn_weight_init=0.0
            )
        )

        self.proj_encode_col = nn.Sequential(

            nn.ReLU6(inplace=True),

            ConvBN(
                self.v_channels,
                self.v_channels,
                kernel_size=1,
                bn_weight_init=0.0
            )
        )

        # ====================================================
        # NEW:
        # 2D local positional encoding
        #
        # This complements the original 1D row / col
        # positional encoding with local 2D spatial geometry.
        # ====================================================
        self.local_pos_2d = (
            LocalPositionalEncoding2D(
                channels=self.v_channels,
                residual_init=1e-3
            )
        )

        # ====================================================
        # Original final projection
        # ====================================================
        self.proj = ConvBN(
            self.v_channels,
            dim,
            kernel_size=1,
            bn_weight_init=0.0
        )

        self.gate = Hard_sigmoid()

    def forward(self, x):

        B, C, H, W = x.shape

        # ====================================================
        # Q K V
        # ====================================================
        q = self.to_q(x)

        k = self.to_k(x)

        v = self.to_v(x)

        # ====================================================
        # Optional learnable squeeze
        # ====================================================
        if self.use_learnable_squeeze:

            score_maps = (
                self.score_predictor(x)
            )

            (
                score_q_row,
                score_k_row,
                score_v_row,
                score_q_col,
                score_k_col,
                score_v_col
            ) = score_maps.chunk(
                6,
                dim=1
            )

            alpha_q_row = torch.softmax(
                score_q_row,
                dim=3
            )

            alpha_k_row = torch.softmax(
                score_k_row,
                dim=3
            )

            alpha_v_row = torch.softmax(
                score_v_row,
                dim=3
            )

            alpha_q_col = torch.softmax(
                score_q_col,
                dim=2
            )

            alpha_k_col = torch.softmax(
                score_k_col,
                dim=2
            )

            alpha_v_col = torch.softmax(
                score_v_col,
                dim=2
            )

        # ====================================================
        # Original detail enhancement branch
        # ====================================================
        qkv = torch.cat(
            [q, k, v],
            dim=1
        )

        detail = self.detail_dwconv(
            qkv
        )

        detail = F.relu6(
            detail,
            inplace=True
        )

        detail = self.detail_pwconv(
            detail
        )

        # ====================================================
        # Horizontal squeeze attention
        # ====================================================
        if self.use_learnable_squeeze:

            q_row = (
                q * alpha_q_row
            ).sum(
                dim=3,
                keepdim=True
            )

            k_row = (
                k * alpha_k_row
            ).sum(
                dim=3,
                keepdim=True
            )

            v_row = (
                v * alpha_v_row
            ).sum(
                dim=3,
                keepdim=True
            )

        else:

            q_row = q.mean(
                dim=3,
                keepdim=True
            )

            k_row = k.mean(
                dim=3,
                keepdim=True
            )

            v_row = v.mean(
                dim=3,
                keepdim=True
            )

        q_row = self.pos_emb_rowq(
            q_row
        )

        k_row = self.pos_emb_rowk(
            k_row
        )

        q_row = q_row.reshape(
            B,
            self.num_heads,
            self.key_dim,
            H
        )

        q_row = q_row.permute(
            0, 1, 3, 2
        )

        k_row = k_row.reshape(
            B,
            self.num_heads,
            self.key_dim,
            H
        )

        v_row = v_row.reshape(
            B,
            self.num_heads,
            self.value_dim,
            H
        )

        v_row = v_row.permute(
            0, 1, 3, 2
        )

        attn_row = torch.matmul(
            q_row,
            k_row
        ) * self.scale

        attn_row = torch.softmax(
            attn_row,
            dim=-1
        )

        out_row = torch.matmul(
            attn_row,
            v_row
        )

        out_row = out_row.permute(
            0, 1, 3, 2
        )

        out_row = out_row.reshape(
            B,
            self.v_channels,
            H,
            1
        )

        out_row = self.proj_encode_row(
            out_row
        )

        # ====================================================
        # Vertical squeeze attention
        # ====================================================
        if self.use_learnable_squeeze:

            q_col = (
                q * alpha_q_col
            ).sum(
                dim=2,
                keepdim=True
            )

            k_col = (
                k * alpha_k_col
            ).sum(
                dim=2,
                keepdim=True
            )

            v_col = (
                v * alpha_v_col
            ).sum(
                dim=2,
                keepdim=True
            )

        else:

            q_col = q.mean(
                dim=2,
                keepdim=True
            )

            k_col = k.mean(
                dim=2,
                keepdim=True
            )

            v_col = v.mean(
                dim=2,
                keepdim=True
            )

        q_col = self.pos_emb_colq(
            q_col
        )

        k_col = self.pos_emb_colk(
            k_col
        )

        q_col = q_col.reshape(
            B,
            self.num_heads,
            self.key_dim,
            W
        )

        q_col = q_col.permute(
            0, 1, 3, 2
        )

        k_col = k_col.reshape(
            B,
            self.num_heads,
            self.key_dim,
            W
        )

        v_col = v_col.reshape(
            B,
            self.num_heads,
            self.value_dim,
            W
        )

        v_col = v_col.permute(
            0, 1, 3, 2
        )

        attn_col = torch.matmul(
            q_col,
            k_col
        ) * self.scale

        attn_col = torch.softmax(
            attn_col,
            dim=-1
        )

        out_col = torch.matmul(
            attn_col,
            v_col
        )

        out_col = out_col.permute(
            0, 1, 3, 2
        )

        out_col = out_col.reshape(
            B,
            self.v_channels,
            1,
            W
        )

        out_col = self.proj_encode_col(
            out_col
        )

        # ====================================================
        # NEW:
        # 2D local positional encoding
        # ====================================================
        local_pos = self.local_pos_2d(
            v
        )

        # ====================================================
        # Context fusion
        #
        # Original:
        # out_row + out_col + v
        #
        # New:
        # out_row + out_col + v + local_pos
        # ====================================================
        context = (
            out_row
            +
            out_col
            +
            v
            +
            local_pos
        )

        # ====================================================
        # Original gate
        # ====================================================
        gate = self.proj(
            context
        )

        gate = self.gate(
            gate
        )

        out = gate * detail

        return out


class FFN_block(nn.Module):
    def __init__(
        self,
        dim,
        ffn_ratio
    ):
        super().__init__()

        hidden_dim = int(
            ffn_ratio * dim
        )

        self.ffn = nn.Sequential(

            ConvBN(
                dim,
                hidden_dim,
                kernel_size=1
            ),

            nn.ReLU6(inplace=True),

            ConvBN(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim
            ),

            nn.ReLU6(inplace=True),

            ConvBN(
                hidden_dim,
                dim,
                kernel_size=1,
                bn_weight_init=0.0
            )
        )

    def forward(self, x):
        return self.ffn(x)


class context_block(nn.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads,
        attn_ratio,
        pos_shape,
        use_learnable_squeeze=True,
        ffn_ratio=2
    ):
        super().__init__()

        self.attention = SeaAttention(
            dim=dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            pos_shape=pos_shape,
            use_learnable_squeeze=use_learnable_squeeze
        )

        self.ffn = FFN_block(
            dim=dim,
            ffn_ratio=ffn_ratio
        )

    def forward(self, x):

        x = x + self.attention(x)

        x = x + self.ffn(x)

        return x


if __name__ == "__main__":

    model = SeaAttention(
        dim=64,
        key_dim=16,
        num_heads=4,
        attn_ratio=2,
        pos_shape=16,
        use_learnable_squeeze=False
    )

    x = torch.randn(
        1,
        64,
        128,
        128
    )

    y = model(x)

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        y.shape
    )

    summary(
        model,
        input_size=(
            1,
            64,
            128,
            128
        ),
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "kernel_size"
        ],
        depth=4
    )
