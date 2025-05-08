import torch
import torch.nn as nn

from einops import rearrange


class DualCrossAttentionModule(nn.Module):
    def __init__(
        self, 
        geo_feat_channels,
        text_feat_channels=256,
        attn_dropout=0.05,
        relu_dropout=0.1, 
        res_dropout=0.1, 
        num_heads=4
        ):

        super(DualCrossAttentionModule, self).__init__()

        self.text_self_attention = nn.MultiheadAttention(
            embed_dim=text_feat_channels, 
            num_heads=num_heads, 
            dropout=attn_dropout,
            batch_first=True, 
            )

        self.text_cross_attention = nn.MultiheadAttention(
            embed_dim=text_feat_channels, 
            kdim=geo_feat_channels, 
            vdim=geo_feat_channels, 
            num_heads=num_heads, 
            dropout=attn_dropout,
            batch_first=True,
            )

        self.voxel_cross_attention = nn.MultiheadAttention(
            embed_dim=geo_feat_channels,
            kdim=text_feat_channels, 
            vdim=text_feat_channels, 
            num_heads=num_heads, 
            dropout=attn_dropout,
            batch_first=True,
            )


        self.mlp = nn.Sequential(
            nn.Linear(text_feat_channels, 1024),
            nn.ReLU(),
            nn.Dropout(relu_dropout),
            nn.Linear(1024, text_feat_channels),
            nn.Dropout(res_dropout)
        )

        # Layer Normalization layers
        self.text_layer_norm = nn.LayerNorm(text_feat_channels)
        self.voxel_layer_norm = nn.LayerNorm(geo_feat_channels)

    def forward(self, x, text):
        '''
        Input:
            x: torch.size: [1, c, x, y, z]
            text: torch.size: [1, seq, 256]
        '''
        bs, c, h, w, z = x.shape
        x = rearrange(x, 'b c h w z -> b (h w z) c').contiguous()  # torch.Size([1, h * w * z, 32])

        # Self-attention on text representation
        text_self_att, _ = self.text_self_attention(query = text, key = text, value = text)  # (B, S, text_output_dim)

        # Cross-attention: Text queries, Image keys and values
        enhanced_text_feat, _ = self.text_cross_attention(query = text_self_att, key = x, value = x) 
        text_feat = self.text_layer_norm(text_self_att + enhanced_text_feat)

        # MLP on enhanced text representation
        enhanced_text_feat_mlp = self.mlp(text_feat)  # (B, S, cross_attention_hidden_size)

        # Cross-attention: Image queries, enhanced text keys and values
        enhanced_voxel_feat, _ = self.voxel_cross_attention(query = x, key = enhanced_text_feat_mlp, value = enhanced_text_feat_mlp)
        
        x = self.voxel_layer_norm(x + enhanced_voxel_feat)

        x = rearrange(x, 'b (h w z) c -> b c h w z', h=h, w=w, z=z).contiguous()

        return x