import torch
import torch.nn as nn
import torch.nn.functional as F

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
