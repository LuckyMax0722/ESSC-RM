import torch
import torch.nn as nn

class Header(nn.Module):
    def __init__(
        self,
        geo_feat_channels,
        num_class
    ):
        super(Header, self).__init__()

        self.conv_head = nn.Conv3d(geo_feat_channels, num_class, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [1, 64, 256, 256, 32]

        x = self.conv_head(x)
   
        return x
