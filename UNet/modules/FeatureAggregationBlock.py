import torch
import torch.nn as nn

from UNet.modules.Base import ResConvBlock

class FeatureAggregationBlock_Conv(nn.Module):
    def __init__(
        self, 
        geo_feat_channels
    ):

        super(FeatureAggregationBlock_Conv, self).__init__()
        
        self.convblock = ResConvBlock(geo_feat_channels=geo_feat_channels)
        self.up_scale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, x, skip):
        
        x = self.up_scale(x)
        x = self.convblock(x, skip)
        
        return x

class FeatureAggregationBlock_TEXTPNA(nn.Module):
    def __init__(
        self, 
        geo_feat_channels
    ):

        super(FeatureAggregationBlock_TEXTPNA, self).__init__()
        
        self.convblock = ResConvBlock(geo_feat_channels=geo_feat_channels)

    def forward(self, x_text, x_pna):
        
        x = self.convblock(x_text, x_pna)
        
        return x

from PNA.NeighborhoodCrossAttention import NeighborhoodCrossAttention as NCA

from PNA.SelfAttention import SelfAttention as SA

from mmcv.cnn.bricks.transformer import build_feedforward_network

from einops import rearrange

class FeatureAggregationBlock_PNA(nn.Module):
    def __init__(
        self, 
        geo_feat_channels,
        num_heads,
        ffn_cfg,

        use_residual,
        bias,

        kernel_size,
        dilation,
        rel_pos_bias,
        qkv_bias,
        attn_drop,
        proj_drop,
    ):

        super(FeatureAggregationBlock_PNA, self).__init__()
        
        self.sa = SA(
            dim=geo_feat_channels,
            num_heads=num_heads,
            ffn_cfg=ffn_cfg,
            use_residual=use_residual,
            bias=bias,
        )

        self.nca = NCA(
            dim=geo_feat_channels,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # Upscale x
        self.up_scale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # FFN Layer
        self.ffn = build_feedforward_network(ffn_cfg)

        # Norm Layer
        self.norm =nn.InstanceNorm3d(geo_feat_channels)
        self.norm_out =nn.InstanceNorm3d(geo_feat_channels)

        
    def forward(self, x, skip):
        b, c, h, w, z = skip.shape

        # Upscale to skip dim
        x = self.up_scale(x)  

        feat_s = self.sa(skip)

        feat_c = self.nca(x, skip)
        
        # Output Norm
        out = self.norm(feat_c + feat_s)

        # FFN Rearrange
        out = rearrange(out, 'b c h w z -> b (h w z) c')

        # FFN FeadForward
        out = self.ffn(out)

        # FFN output
        out = rearrange(out, 'b (h w z) c -> b c h w z', h=h, w=w, z=z)

        # Model Output
        out = self.norm_out(out)

        return out


from PNA.NeighborhoodCrossAttention import NeighborhoodCrossAttentionV2 as NCAV2

class FeatureAggregationBlock_PNAV2(nn.Module):
    def __init__(
        self, 
        geo_feat_channels,
        num_class,
        num_heads,
        ffn_cfg,

        use_residual,
        bias,

        kernel_size,
        dilation,
        rel_pos_bias,
        qkv_bias,
        attn_drop,
        proj_drop,
    ):

        super(FeatureAggregationBlock_PNAV2, self).__init__()
        
        self.sa = SA(
            dim=geo_feat_channels,
            num_heads=num_heads,
            ffn_cfg=ffn_cfg,
            use_residual=use_residual,
            bias=bias,
        )

        self.nca = NCAV2(
            dim=geo_feat_channels,
            num_class=num_class,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # Upscale x
        self.up_scale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # FFN Layer
        self.ffn = build_feedforward_network(ffn_cfg)

        # Norm Layer
        self.norm =nn.InstanceNorm3d(geo_feat_channels)
        self.norm_out =nn.InstanceNorm3d(geo_feat_channels)

        
    def forward(self, x, skip):
        b, c, h, w, z = skip.shape

        # Upscale to skip dim
        x = self.up_scale(x)  

        feat_s = self.sa(skip)

        feat_c = self.nca(x, skip)
        
        # Output Norm
        out = self.norm(feat_c + feat_s)

        # FFN Rearrange
        out = rearrange(out, 'b c h w z -> b (h w z) c')

        # FFN FeadForward
        out = self.ffn(out)

        # FFN output
        out = rearrange(out, 'b (h w z) c -> b c h w z', h=h, w=w, z=z)

        # Model Output
        out = self.norm_out(out)

        return out