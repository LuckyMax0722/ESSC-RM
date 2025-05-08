import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

from SSC.loss.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss, Lovasz_Softmax_Loss

#from UNet.modules.FeatureExtractionBlock import FeatureExtractionBlock_Conv as FEB_Conv
from UNet.modules.FeatureExtractionBlock import FeatureExtractionBlock_ConvV2 as FEB_Conv

from UNet.modules.FeatureAggregationBlock import FeatureAggregationBlock_Conv as FAB_Conv
from UNet.modules.FeatureAggregationBlock import FeatureAggregationBlock_PNA as FAB_PNA

from UNet.modules.FeatureAggregationBlock import FeatureAggregationBlock_PNAV2 as FAB_PNAV2


from UNet.modules.PredHeaders import Header

class UNet(nn.Module):
    def __init__(self, 
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
        super().__init__()
        
        self.conv0 = nn.Conv3d(
            geo_feat_channels, 
            geo_feat_channels, 
            kernel_size=(5, 5, 3), 
            stride=(1, 1, 1), 
            padding=(2, 2, 1), 
            bias=True, 
            padding_mode='replicate'
        )

        self.encoder_128 = FEB_Conv(
            geo_feat_channels=geo_feat_channels
        )

        self.encoder_64 = FEB_Conv(
            geo_feat_channels=geo_feat_channels
        )

        self.encoder_32 = FEB_Conv(
            geo_feat_channels=geo_feat_channels
        )

        self.encoder_16 = FEB_Conv(
            geo_feat_channels=geo_feat_channels
        )

        self.bottleneck = FEB_Conv(
            geo_feat_channels=geo_feat_channels,
            z_down=False
        )

        self.decoder_32 = FAB_PNA(
            geo_feat_channels=geo_feat_channels,
            num_heads=num_heads,
            ffn_cfg=ffn_cfg,

            use_residual=use_residual,
            bias=bias,

            kernel_size=kernel_size[0],
            dilation=dilation[0],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.decoder_64 = FAB_PNA(
            geo_feat_channels=geo_feat_channels,
            num_heads=num_heads,
            ffn_cfg=ffn_cfg,

            use_residual=use_residual,
            bias=bias,

            kernel_size=kernel_size[1],
            dilation=dilation[1],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.decoder_128 = FAB_PNA(
            geo_feat_channels=geo_feat_channels,
            num_heads=num_heads,
            ffn_cfg=ffn_cfg,

            use_residual=use_residual,
            bias=bias,

            kernel_size=kernel_size[2],
            dilation=dilation[2],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.decoder_256 = FAB_Conv(
            geo_feat_channels=geo_feat_channels
        )

        
    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]   
        
        x = self.conv0(x)  # x: ([1, 64, 256, 256, 32])
        
        x, skip_256 = self.encoder_128(x) # x: ([1, 64, 128, 128, 16]) / skip1: ([1, 64, 256, 256, 32])
        
        x, skip_128 = self.encoder_64(x) # x: ([1, 64, 64, 64, 8]) / skip2: ([1, 64, 128, 128, 16])
        
        x, skip_64 = self.encoder_32(x) # x: ([1, 64, 32, 32, 4]) / skip3: ([1, 64, 64, 64, 8])
        
        x, skip_32 = self.encoder_16(x) # x: ([1, 64, 16, 16, 2]) / skip4: ([1, 64, 32, 32, 4])
        
        x_16 = self.bottleneck(x) # x: ([1, 64, 16, 16, 2])
        
        x_32 = self.decoder_32(x_16, skip_32)  # x: ([1, 64, 32, 32, 4])
        
        x_64 = self.decoder_64(x_32, skip_64)  # x: ([1, 64, 64, 64, 8]) 
        
        x_128 = self.decoder_128(x_64, skip_128)  # x: ([1, 64, 128, 128, 16])
        
        x_256 = self.decoder_256(x_128, skip_256)  # x: ([1, 64, 256, 256, 32])
        
        return x_32, x_64, x_128, x_256

class RefinementModule(nn.Module):
    def __init__(
        self,
        num_class,
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

        empty_idx=0,
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_frequencies=None,
    ):
        super(RefinementModule, self).__init__()
        
        self.empty_idx = empty_idx
        
        self.embedding = nn.Embedding(num_class, geo_feat_channels)  # [B, D, H, W, C]
        
        self.unet = UNet(
            geo_feat_channels=geo_feat_channels,
            num_heads=num_heads,
            ffn_cfg=ffn_cfg,

            use_residual=use_residual,
            bias=bias,

            kernel_size=kernel_size,
            dilation=dilation,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            )
        
        self.pred_head_256 = Header(geo_feat_channels=geo_feat_channels, num_class=num_class)
        self.pred_head_128 = Header(geo_feat_channels=geo_feat_channels, num_class=num_class)
        self.pred_head_64 = Header(geo_feat_channels=geo_feat_channels, num_class=num_class)
        self.pred_head_32 = Header(geo_feat_channels=geo_feat_channels, num_class=num_class)

        # voxel losses
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
            
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)

        # loss functions
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(np.array(class_frequencies) + 0.001))
        else:
            self.class_weights = torch.ones(17)/17  # FIXME hardcode 17
            
    def forward(self, x):
        x[x == 255] = 0

        x = self.embedding(x)

        x = rearrange(x, 'b h w z c -> b c h w z')  

        x_32, x_64, x_128, x_256 = self.unet(x)
        
        x_256 = self.pred_head_256(x_256)
        x_128 = self.pred_head_128(x_128)
        x_64 = self.pred_head_64(x_64)
        x_32 = self.pred_head_32(x_32)

        return x_32, x_64, x_128, x_256
    
    def loss(self, output_voxels_list, target_voxels_list):
        loss_dict = {}
        
        suffixes = [32, 64, 128, 256]

        for suffix, (output_voxels, target_voxels) in zip(suffixes, zip(output_voxels_list, target_voxels_list)):
            loss_dict[f'loss_voxel_ce_{suffix}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict[f'loss_voxel_sem_scal_{suffix}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict[f'loss_voxel_geo_scal_{suffix}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict

    def loss_V2(self, output_voxels_list, target_voxels_list):
        loss_dict = {}
        
        suffixes = [32, 64, 128, 256]

        for suffix, (output_voxels, target_voxels) in zip(suffixes, zip(output_voxels_list, target_voxels_list)):
            loss_dict[f'loss_voxel_ce_{suffix}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict[f'loss_voxel_sem_scal_{suffix}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict[f'loss_voxel_geo_scal_{suffix}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
            
            loss_dict[f'loss_lovasz_{suffix}'] = Lovasz_Softmax_Loss(output_voxels, target_voxels, ignore_index=255)

        return loss_dict





class UNetV2(nn.Module):
    def __init__(self, 
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
        super().__init__()
        
        self.conv0 = nn.Conv3d(
            geo_feat_channels, 
            geo_feat_channels, 
            kernel_size=(5, 5, 3), 
            stride=(1, 1, 1), 
            padding=(2, 2, 1), 
            bias=True, 
            padding_mode='replicate'
        )

        self.encoder_128 = FEB_Conv(
            geo_feat_channels=geo_feat_channels
        )

        self.encoder_64 = FEB_Conv(
            geo_feat_channels=geo_feat_channels
        )

        self.encoder_32 = FEB_Conv(
            geo_feat_channels=geo_feat_channels
        )

        self.bottleneck = FEB_Conv(
            geo_feat_channels=geo_feat_channels,
            z_down=False
        )

        self.decoder_64 = FAB_PNAV2(
            geo_feat_channels=geo_feat_channels,
            num_class=num_class,
            num_heads=num_heads,
            ffn_cfg=ffn_cfg,

            use_residual=use_residual,
            bias=bias,

            kernel_size=kernel_size[1],
            dilation=dilation[1],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.decoder_128 = FAB_PNAV2(
            geo_feat_channels=geo_feat_channels,
            num_class=num_class,
            num_heads=num_heads,
            ffn_cfg=ffn_cfg,

            use_residual=use_residual,
            bias=bias,

            kernel_size=kernel_size[2],
            dilation=dilation[2],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # self.decoder_256 = FAB_Conv(
        #     geo_feat_channels=geo_feat_channels
        # )

        self.decoder_256 = FAB_PNAV2(
            geo_feat_channels=geo_feat_channels,
            num_class=num_class,
            num_heads=num_heads,
            ffn_cfg=ffn_cfg,

            use_residual=use_residual,
            bias=bias,

            kernel_size=kernel_size[2],
            dilation=dilation[2],
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )


        self.pred_head_256 = Header(geo_feat_channels=geo_feat_channels, num_class=num_class)
        self.pred_head_128 = Header(geo_feat_channels=geo_feat_channels, num_class=num_class)
        self.pred_head_64 = Header(geo_feat_channels=geo_feat_channels, num_class=num_class)
        self.pred_head_32 = Header(geo_feat_channels=geo_feat_channels, num_class=num_class)

        
    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]   
        
        x = self.conv0(x)  # x: ([1, 64, 256, 256, 32])
        
        x, skip_256 = self.encoder_128(x) # x: ([1, 64, 128, 128, 16]) / skip1: ([1, 64, 256, 256, 32])
        
        x, skip_128 = self.encoder_64(x) # x: ([1, 64, 64, 64, 8]) / skip2: ([1, 64, 128, 128, 16])
        
        x, skip_64 = self.encoder_32(x) # x: ([1, 64, 32, 32, 4]) / skip3: ([1, 64, 64, 64, 8])

        x_32 = self.bottleneck(x) # x: ([1, 64, 32, 32, 4])
        
        x_32 = self.pred_head_32(x_32)

        x_64 = self.decoder_64(x_32, skip_64)  # x: ([1, 64, 64, 64, 8]) 
        
        x_64 = self.pred_head_64(x_64)

        x_128 = self.decoder_128(x_64, skip_128)  # x: ([1, 64, 128, 128, 16])
        
        x_128 = self.pred_head_128(x_128)

        x_256 = self.decoder_256(x_128, skip_256)  # x: ([1, 64, 256, 256, 32])

        x_256 = self.pred_head_256(x_256)
        
        return x_32, x_64, x_128, x_256

class RefinementModuleV2(nn.Module):
    def __init__(
        self,
        num_class,
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

        empty_idx=0,
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_frequencies=None,
    ):
        super(RefinementModuleV2, self).__init__()
        
        self.empty_idx = empty_idx
        
        self.embedding = nn.Embedding(num_class, geo_feat_channels)  # [B, D, H, W, C]
        
        self.unet = UNetV2(
            geo_feat_channels=geo_feat_channels,
            num_class=num_class,
            num_heads=num_heads,
            ffn_cfg=ffn_cfg,

            use_residual=use_residual,
            bias=bias,

            kernel_size=kernel_size,
            dilation=dilation,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            )
        
        

        # voxel losses
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
            
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)

        # loss functions
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(np.array(class_frequencies) + 0.001))
        else:
            self.class_weights = torch.ones(17)/17  # FIXME hardcode 17
            
    def forward(self, x):
        x[x == 255] = 0

        x = self.embedding(x)

        x = rearrange(x, 'b h w z c -> b c h w z')  

        x_32, x_64, x_128, x_256 = self.unet(x)
        
        return x_32, x_64, x_128, x_256
    
    def loss(self, output_voxels_list, target_voxels_list):
        loss_dict = {}
        
        suffixes = [32, 64, 128, 256]

        for suffix, (output_voxels, target_voxels) in zip(suffixes, zip(output_voxels_list, target_voxels_list)):
            loss_dict[f'loss_voxel_ce_{suffix}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict[f'loss_voxel_sem_scal_{suffix}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict[f'loss_voxel_geo_scal_{suffix}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict


class RefinementModuleSingle(nn.Module):
    def __init__(
        self,
        num_class,
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

        empty_idx=0,
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_frequencies=None,
    ):
        super(RefinementModuleSingle, self).__init__()
        
        self.empty_idx = empty_idx
        
        self.embedding = nn.Embedding(num_class, geo_feat_channels)  # [B, D, H, W, C]
        
        self.unet = UNet(
            geo_feat_channels=geo_feat_channels,
            num_heads=num_heads,
            ffn_cfg=ffn_cfg,

            use_residual=use_residual,
            bias=bias,

            kernel_size=kernel_size,
            dilation=dilation,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            )
        
        self.pred_head_256 = Header(geo_feat_channels=geo_feat_channels, num_class=num_class)
    
        # voxel losses
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
            
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)

        # loss functions
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(np.array(class_frequencies) + 0.001))
        else:
            self.class_weights = torch.ones(17)/17  # FIXME hardcode 17
            
    def forward(self, x):
        x[x == 255] = 0

        x = self.embedding(x)

        x = rearrange(x, 'b h w z c -> b c h w z')  

        x_32, x_64, x_128, x_256 = self.unet(x)
        
        x_256 = self.pred_head_256(x_256)

        return x_32, x_64, x_128, x_256
    
    def loss_single(self, output_voxels_list, target_voxels_list):
        loss_dict = {}
        
        suffixes = [256]

        for suffix, (output_voxels, target_voxels) in zip(suffixes, zip(output_voxels_list, target_voxels_list)):
            loss_dict[f'loss_voxel_ce_{suffix}'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict[f'loss_voxel_sem_scal_{suffix}'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict[f'loss_voxel_geo_scal_{suffix}'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict
