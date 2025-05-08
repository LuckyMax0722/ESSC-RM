import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

from SSC.loss.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss, Lovasz_Softmax_Loss

#from UNet.modules.FeatureExtractionBlock import FeatureExtractionBlock_Conv as FEB_Conv
from UNet.modules.FeatureExtractionBlock import FeatureExtractionBlock_ConvV2 as FEB_Conv
from UNet.modules.FeatureAggregationBlock import FeatureAggregationBlock_Conv as FAB_Conv
from UNet.modules.PredHeaders import Header

from TEXT.DualCrossAttentionModule import DualCrossAttentionModule as DCAM
from TEXT.SemanticInteractionGuidanceModule import SemanticInteractionGuidanceModule as SIGM

class UNet(nn.Module):
    def __init__(self, 
            geo_feat_channels,
            text_encoder,
            text_encoder_dim,
            num_heads,
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

        self.decoder_32 = FAB_Conv(
            geo_feat_channels=geo_feat_channels
        )

        self.decoder_64 = FAB_Conv(
            geo_feat_channels=geo_feat_channels
        )

        self.decoder_128 = FAB_Conv(
            geo_feat_channels=geo_feat_channels
        )

        self.decoder_256 = FAB_Conv(
            geo_feat_channels=geo_feat_channels
        )

        self.text_encoder = text_encoder
        self.text_encoder_dim = text_encoder_dim

        if text_encoder == 'BLIP2':
            self.dcam_encoder_128 = DCAM(
                geo_feat_channels=geo_feat_channels,
                num_heads=num_heads
            )

            self.dcam_encoder_64 = DCAM(
                geo_feat_channels=geo_feat_channels,
                num_heads=num_heads
            )

            self.dcam_encoder_32 = DCAM(
                geo_feat_channels=geo_feat_channels,
                num_heads=num_heads
            )

            self.dcam_encoder_16 = DCAM(
                geo_feat_channels=geo_feat_channels,
                num_heads=num_heads
            )

            self.dcam_decoder_128 = DCAM(
                geo_feat_channels=geo_feat_channels,
                num_heads=num_heads
            )

            self.dcam_decoder_64 = DCAM(
                geo_feat_channels=geo_feat_channels,
                num_heads=num_heads
            )

            self.dcam_decoder_32 = DCAM(
                geo_feat_channels=geo_feat_channels,
                num_heads=num_heads
            )

            self.dcam_decoder_16 = DCAM(
                geo_feat_channels=geo_feat_channels,
                num_heads=num_heads
            )

        elif text_encoder in text_encoder_dim:

            text_dim = text_encoder_dim[text_encoder]

            self.sigm_encoder_16 = SIGM(
                geo_feat_channels=geo_feat_channels,
                text_dim=text_dim,
            )

            self.sigm_encoder_32 = SIGM(
                geo_feat_channels=geo_feat_channels,
                text_dim=text_dim,
            )

            self.sigm_encoder_64 = SIGM(
                geo_feat_channels=geo_feat_channels,
                text_dim=text_dim,
            )

            self.sigm_encoder_128 = SIGM(
                geo_feat_channels=geo_feat_channels,
                text_dim=text_dim,
            )

            self.sigm_decoder_16 = SIGM(
                geo_feat_channels=geo_feat_channels,
                text_dim=text_dim,
            )

            self.sigm_decoder_32 = SIGM(
                geo_feat_channels=geo_feat_channels,
                text_dim=text_dim,
            )

            self.sigm_decoder_64 = SIGM(
                geo_feat_channels=geo_feat_channels,
                text_dim=text_dim,
            )

            self.sigm_decoder_128 = SIGM(
                geo_feat_channels=geo_feat_channels,
                text_dim=text_dim,
            )
        
        else:
            raise NotImplementedError(f"Text Encoder: '{text_encoder}' is not implemented.")
        
    def forward(self, x, text):  # [b, geo_feat_channels, X, Y, Z]   
        
        if self.text_encoder == 'BLIP2':
            x = self.conv0(x)  # x: ([1, 64, 256, 256, 32])
            
            x, skip_256 = self.encoder_128(x) # x: ([1, 64, 128, 128, 16]) / skip1: ([1, 64, 256, 256, 32])
            x = self.dcam_encoder_128(x, text)

            x, skip_128 = self.encoder_64(x) # x: ([1, 64, 64, 64, 8]) / skip2: ([1, 64, 128, 128, 16])
            x = self.dcam_encoder_64(x, text)

            x, skip_64 = self.encoder_32(x) # x: ([1, 64, 32, 32, 4]) / skip3: ([1, 64, 64, 64, 8])
            x = self.dcam_encoder_32(x, text)

            x, skip_32 = self.encoder_16(x) # x: ([1, 64, 16, 16, 2]) / skip4: ([1, 64, 32, 32, 4])
            x = self.dcam_encoder_16(x, text)

            x_16 = self.bottleneck(x) # x: ([1, 64, 16, 16, 2])
            x_16 = self.dcam_decoder_16(x_16, text)

            x_32 = self.decoder_32(x_16, skip_32)  # x: ([1, 64, 32, 32, 4])
            x_32 = self.dcam_decoder_32(x_32, text)

            x_64 = self.decoder_64(x_32, skip_64)  # x: ([1, 64, 64, 64, 8]) 
            x_64 = self.dcam_decoder_64(x_64, text)

            x_128 = self.decoder_128(x_64, skip_128)  # x: ([1, 64, 128, 128, 16])
            x_128 = self.dcam_decoder_128(x_128, text)

            x_256 = self.decoder_256(x_128, skip_256)  # x: ([1, 64, 256, 256, 32])
        
        elif self.text_encoder in self.text_encoder_dim:
            x = self.conv0(x)  # x: ([1, 64, 256, 256, 32])
            
            x, skip_256 = self.encoder_128(x) # x: ([1, 64, 128, 128, 16]) / skip1: ([1, 64, 256, 256, 32])
            x = self.sigm_encoder_128(x, text)

            x, skip_128 = self.encoder_64(x) # x: ([1, 64, 64, 64, 8]) / skip2: ([1, 64, 128, 128, 16])
            x = self.sigm_encoder_64(x, text)

            x, skip_64 = self.encoder_32(x) # x: ([1, 64, 32, 32, 4]) / skip3: ([1, 64, 64, 64, 8])
            x = self.sigm_encoder_32(x, text)

            x, skip_32 = self.encoder_16(x) # x: ([1, 64, 16, 16, 2]) / skip4: ([1, 64, 32, 32, 4])
            x = self.sigm_encoder_16(x, text)

            x_16 = self.bottleneck(x) # x: ([1, 64, 16, 16, 2])
            x_16 = self.sigm_decoder_16(x_16, text)

            x_32 = self.decoder_32(x_16, skip_32)  # x: ([1, 64, 32, 32, 4])
            x_32 = self.sigm_decoder_32(x_32, text)

            x_64 = self.decoder_64(x_32, skip_64)  # x: ([1, 64, 64, 64, 8]) 
            x_64 = self.sigm_decoder_64(x_64, text)

            x_128 = self.decoder_128(x_64, skip_128)  # x: ([1, 64, 128, 128, 16])
            x_128 = self.sigm_decoder_128(x_128, text)

            x_256 = self.decoder_256(x_128, skip_256)  # x: ([1, 64, 256, 256, 32])

        return x_32, x_64, x_128, x_256

class RefinementModule(nn.Module):
    def __init__(
        self,
        num_class,
        geo_feat_channels,

        text_encoder,
        text_encoder_dim,
        num_heads,

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
            text_encoder=text_encoder,
            text_encoder_dim=text_encoder_dim,
            num_heads=num_heads,
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
            
    def forward(self, x, text):
        x[x == 255] = 0

        x = self.embedding(x)

        x = rearrange(x, 'b h w z c -> b c h w z')  

        x_32, x_64, x_128, x_256 = self.unet(x, text)
        
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



class RefinementModuleSingle(nn.Module):
    def __init__(
        self,
        num_class,
        geo_feat_channels,

        text_encoder,
        text_encoder_dim,
        num_heads,

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
            text_encoder=text_encoder,
            text_encoder_dim=text_encoder_dim,
            num_heads=num_heads,
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
            
    def forward(self, x, text):
        x[x == 255] = 0

        x = self.embedding(x)

        x = rearrange(x, 'b h w z c -> b c h w z')  

        x_32, x_64, x_128, x_256 = self.unet(x, text)
        
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

    