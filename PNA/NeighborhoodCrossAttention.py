import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmcv.cnn.bricks.transformer import build_feedforward_network
from PNA.natten_utils.NCAM import NeighborhoodCrossAttentionModule as NCAM
from PNA.natten_utils.NCAM import NeighborhoodCrossAttentionModuleV2 as NCAMV2

class NeighborhoodCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation,
        rel_pos_bias=True,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super(NeighborhoodCrossAttention, self).__init__()

        # Norm Layer
        self.norm_up =nn.InstanceNorm3d(dim)
        self.norm_skip =nn.InstanceNorm3d(dim)

        # NeighborhoodCrossAttention3D Layer
        self.ncam = NCAM(
            dim = dim,
            num_heads = num_heads,
            kernel_size = kernel_size,
            dilation = dilation,
            rel_pos_bias = rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x, skip):
        '''
        input:
            x: torch.size([1, 64, 16, 16, 2])/....
            skip: torch.size([1, 64, 32, 32, 4])/....
        
        x: (B, C, H, W, Z)
        '''

        # Input Norm
        x = self.norm_up(x)
        skip = self.norm_skip(skip)

        out = self.ncam(x, skip)

        return out
    
class NeighborhoodCrossAttentionV2(nn.Module):
    def __init__(
        self,
        dim,
        num_class,
        num_heads,
        kernel_size,
        dilation,
        rel_pos_bias=True,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super(NeighborhoodCrossAttentionV2, self).__init__()

        # Norm Layer
        self.norm_up =nn.InstanceNorm3d(dim)
        self.norm_skip =nn.InstanceNorm3d(dim)

        # NeighborhoodCrossAttention3D Layer
        self.ncam = NCAMV2(
            dim = dim,
            num_class = num_class,
            num_heads = num_heads,
            kernel_size = kernel_size,
            dilation = dilation,
            rel_pos_bias = rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x, skip):
        '''
        input:
            x: torch.size([1, 20, 32, 32, 4])/....
            skip: torch.size([1, 64, 32, 32, 4])/....
        
        x: (B, C, H, W, Z)
        '''

        # Input Norm
        x = self.norm_up(x)
        skip = self.norm_skip(skip)

        out = self.ncam(skip, x)

        return out