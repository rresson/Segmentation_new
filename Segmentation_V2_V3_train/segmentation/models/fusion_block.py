import torch
import torch.nn as nn
import torch.nn.functional as F

"""This code defines two blocks: fusion_block and segment_block
fusion_block: fuse the output of context branch and spatial branch
segment_block: transform the final feature to a semantic map
"""

#相比version1和2还说，fusion block中从先sigmoid再上采样改成了先上采样再sigmoid

class ConvBN(nn.Sequential):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bn_weight_init=1.0):

        conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,
                  stride=stride,padding=padding,dilation=dilation,groups=groups,bias=False)
        bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0.0)

        super().__init__(conv,bn)



class fusion_block(nn.Module):
    def __init__(self,context_channel,spatial_channel,out_channel):
        super().__init__()
        self.spatial_projection = nn.Conv2d(spatial_channel,out_channel,1)
        self.context_project = nn.Conv2d(context_channel,out_channel,1)

    def forward(self,spatial,context):
        spatial = self.spatial_projection(spatial)
        weight = self.context_project(context)
        weight = F.interpolate(weight,spatial.shape[-2:],mode='bilinear',align_corners=False,)
        weight = torch.sigmoid(weight)
        fusion = spatial + spatial * weight
        return fusion



class fusion_block2(nn.Module):
    """
    Lightweight context-spatial fusion.
    Compared with the original fusion block:
        Original:
            F = S + S * sigmoid(C)

        New:
            1. locally refine spatial feature
            2. directly inject semantic context feature
            F = S_refined + gate * C

    Input:
        spatial : [B, spatial_channel, Hs, Ws]
        context : [B, context_channel, Hc, Wc]

    Output:
        [B, out_channel, Hs, Ws]
    """

    def __init__(self,context_channel,spatial_channel,out_channel,):
        super().__init__()

        # ====================================================
        # Spatial projection
        #
        # Keep the same name/role as original fusion block.
        # ====================================================
        self.spatial_projection = ConvBN(spatial_channel,out_channel,kernel_size=1)
        # ====================================================
        # Lightweight spatial refinement
        #
        # DWConv:
        #     local spatial feature extraction
        #
        # PWConv:
        #     channel interaction
        #
        # The last BN uses zero initialization so that
        # the block initially behaves nearly like identity.
        # ====================================================
        self.spatial_refine = nn.Sequential(
            ConvBN(out_channel,out_channel,kernel_size=3,stride=1,padding=1,groups=out_channel),
            nn.ReLU6(inplace=True),
            ConvBN(out_channel,out_channel,kernel_size=1,bn_weight_init=0.0)
        )
        # ====================================================
        # Context projection
        # ====================================================
        self.context_project = ConvBN(context_channel,out_channel,kernel_size=1)
        # ====================================================
        # Lightweight context gate
        #
        # Use depthwise convolution instead of another
        # expensive full 1x1 convolution.
        # ====================================================
        self.context_gate = nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1,groups=out_channel,bias=True)
        # Initialize gate near neutral region.
        nn.init.zeros_(self.context_gate.weight)

        nn.init.zeros_(self.context_gate.bias)

    def forward(self,spatial,context):

        # ====================================================
        # Spatial branch
        # ====================================================
        spatial = self.spatial_projection(spatial)
        spatial_residual = spatial
        spatial_refined = self.spatial_refine(spatial)
        spatial = (spatial_residual+spatial_refined)
        # ====================================================
        # Context branch
        # ====================================================
        context = self.context_project(context)
        context = F.interpolate(context,size=spatial.shape[-2:],mode="bilinear",align_corners=False)
        # ====================================================
        # Context adaptive gate
        # ====================================================
        gate = self.context_gate(context)
        gate = torch.sigmoid(gate)
        # ====================================================
        # Direct semantic injection
        #
        # Unlike the original fusion:
        #
        #     spatial + spatial * gate
        #
        # context itself now contributes to the output.
        # ====================================================
        fusion = (spatial+gate * context)
        return fusion




class DepthwiseSeparableResidual(nn.Module):
    """
    Deployment-friendly local refinement block.

    Input/output shape:
        [B, C, H, W] -> [B, C, H, W]
    """
    def __init__(self, channels):
        super().__init__()

        self.depthwise = nn.Sequential(
            ConvBN(channels,channels,kernel_size=3,stride=1,padding=1,groups=channels,),
            nn.ReLU6(inplace=True),
        )
        self.pointwise = ConvBN(channels,channels,kernel_size=1,bn_weight_init=0.0,)

        self.activation = nn.ReLU6(inplace=True)

    def forward(self, x):
        identity = x

        x = self.depthwise(x)
        x = self.pointwise(x)

        x = x + identity
        x = self.activation(x)

        return x


class segment_block(nn.Module):
    def __init__(self,in_channel,hidden_channel,num_class=150,num_refine_blocks=2,):
        super().__init__()

        # 先把 encoder 输出投影到 segmentation head 的工作通道
        self.input_projection = nn.Sequential(
            ConvBN(in_channel,hidden_channel,kernel_size=1,),
            nn.ReLU6(inplace=True),
        )

        # 局部空间特征细化
        self.refine = nn.Sequential(
            *[
                DepthwiseSeparableResidual(hidden_channel)
                for _ in range(num_refine_blocks)
            ]
        )
        # 分类前再进行一次轻量局部特征提取
        self.pre_classifier = nn.Sequential(
            ConvBN(hidden_channel,hidden_channel,kernel_size=3,padding=1,groups=hidden_channel,),
            nn.ReLU6(inplace=True),

            ConvBN(hidden_channel,hidden_channel,kernel_size=1,),
            nn.ReLU6(inplace=True),
        )

        # 输出原始 logits，不添加激活函数
        self.classifier = nn.Conv2d(hidden_channel,num_class,kernel_size=1,bias=True,)

    def forward(self, x, output_size):
        x = self.input_projection(x)
        x = self.refine(x)
        x = self.pre_classifier(x)

        logits = self.classifier(x)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False,)

        return logits


class segment_block2(nn.Module):
    """
    Multi-level lightweight segmentation head.

    Inputs:
        fusion_1: [B, C1, H, W]
        fusion_2: [B, C2, H, W]
        fusion_3: [B, C3, H, W]

    Pipeline:
        F1 -> 1x1 projection ->\
        F2 -> 1x1 projection ---> concat -> 1x1 fusion
        F3 -> 1x1 projection ->/             |
                                              v
                                    lightweight refinement
                                              |
                                              v
                                          classifier
    """

    def __init__(self,in_channels,hidden_channel,num_class=150,reduce_channel=32,num_refine_blocks=2,):
        super().__init__()

        c1, c2, c3 = in_channels

        # ====================================================
        # Step 1:
        # Compress three fusion features independently
        #
        # 64 -> 32 for your current network
        # ====================================================
        self.proj1 = nn.Sequential(
            ConvBN(c1,reduce_channel,kernel_size=1),
            nn.ReLU6(inplace=True),
        )

        self.proj2 = nn.Sequential(
            ConvBN(c2,reduce_channel,kernel_size=1),
            nn.ReLU6(inplace=True),
        )

        self.proj3 = nn.Sequential(
            ConvBN(c3,reduce_channel,kernel_size=1),
            nn.ReLU6(inplace=True),
        )

        # ====================================================
        # Step 2:
        # Multi-level feature aggregation
        #
        # 32 + 32 + 32 = 96
        # 96 -> hidden_channel
        # ====================================================
        total_channel = 3 * reduce_channel

        self.fusion = nn.Sequential(
            ConvBN(total_channel,hidden_channel,kernel_size=1),
            nn.ReLU6(inplace=True),
        )

        # ====================================================
        # Step 3:
        # Lightweight local refinement
        # ====================================================
        self.refine = nn.Sequential(
            *[
                DepthwiseSeparableResidual(hidden_channel)
                for _ in range(num_refine_blocks)
            ]
        )

        # ====================================================
        # Step 4:
        # Final local enhancement
        # ====================================================
        self.pre_classifier = nn.Sequential(
            ConvBN(hidden_channel,hidden_channel,kernel_size=3,padding=1,groups=hidden_channel),
            nn.ReLU6(inplace=True),
            ConvBN(hidden_channel,hidden_channel,kernel_size=1),
            nn.ReLU6(inplace=True),
        )

        # ====================================================
        # Step 5:
        # Pixel classifier
        # ====================================================
        self.classifier = nn.Conv2d(hidden_channel,num_class,kernel_size=1,bias=True)

    def forward(self,fusion_1,fusion_2,fusion_3,output_size,):

        # ----------------------------------------------------
        # Three-level feature projection
        # ----------------------------------------------------
        f1 = self.proj1(fusion_1)

        f2 = self.proj2(fusion_2)

        f3 = self.proj3(fusion_3)

        # ----------------------------------------------------
        # Multi-level aggregation
        # ----------------------------------------------------
        x = torch.cat([f1, f2, f3],dim=1)

        x = self.fusion(x)

        # ----------------------------------------------------
        # Lightweight refinement
        # ----------------------------------------------------
        x = self.refine(x)

        x = self.pre_classifier(x)

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------
        logits = self.classifier(x)

        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits
