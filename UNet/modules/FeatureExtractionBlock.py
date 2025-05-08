import torch
import torch.nn as nn

from UNet.modules.Base import ConvBlock

class FeatureExtractionBlock_Conv(nn.Module):
    def __init__(
        self, 
        geo_feat_channels, 
        z_down=True
    ):

        super().__init__()
        self.z_down = z_down

        self.convblock = ConvBlock(input_channels=geo_feat_channels, output_channels=geo_feat_channels)

        if z_down :
            self.downsample = nn.MaxPool3d((2, 2, 2))


    
    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]
        residual_feat = x
        x = self.convblock(x)  # [b, geo_feat_channels, X, Y, Z]
        skip = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]
    
        if self.z_down:
            x = self.downsample(skip)  # [b, geo_feat_channels, X//2, Y//2, Z//2]

            return x, skip
        else:
            return skip  # [b, geo_feat_channels, X, Y, Z]

class FeatureExtractionBlock_ConvV2(nn.Module):
    def __init__(
        self, 
        geo_feat_channels, 
        z_down=True
    ):

        super().__init__()
        self.z_down = z_down

        self.convblock1 = ConvBlock(input_channels=geo_feat_channels, output_channels=geo_feat_channels)

        if z_down :
            #self.downsample = nn.MaxPool3d((2, 2, 2))

            self.convblock2 = ConvBlock(input_channels=geo_feat_channels, output_channels=geo_feat_channels)

            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=True, padding_mode='replicate'),
                nn.InstanceNorm3d(geo_feat_channels)
            )

    
    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]
        residual_feat = x
        x = self.convblock1(x)  # [b, geo_feat_channels, X, Y, Z]
        skip = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]
    
        if self.z_down:
            x = self.downsample(skip)  # [b, geo_feat_channels, X//2, Y//2, Z//2]
            x = self.convblock2(x)
            return x, skip
        else:
            return skip  # [b, geo_feat_channels, X, Y, Z]
