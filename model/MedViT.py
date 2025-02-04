import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MONAI')))

from einops import Rearrange

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, MLPBlock
from monai.networks.layers.factories import Pool
from monai.utils import ensure_tuple_rep

from MONAI.generative.networks.nets.diffusion_model_unet import CrossAttention, BasicTransformerBlock, SpatialTransformer, CrossAttnDownBlock, zero_module


class PatchEmbedding3D(nn.Module):
    def __init__(self, in_ch, emb_size, patch_size):
        super(PatchEmbedding3D, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Conv3d(in_ch, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e d h w -> b (d h w) e')  # Updated for 3D data
        )

    def forward(self, x):
        return self.projection(x)


class PositionalEmbedding3D(nn.Module):
    def __init__(self, in_ch, patch_size, img_size, emb_size=1024):
        super(PositionalEmbedding3D, self).__init__()
        self.patch_size = patch_size

        self.projection = nn.Sequential(
            nn.Conv3d(in_ch, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e d h w -> b (d h w) e')
        )

        num_patches_d = img_size[0] // patch_size[0]
        num_patches_h = img_size[1] // patch_size[1]
        num_patches_w = img_size[2] // patch_size[2]
        num_patches = num_patches_d * num_patches_h * num_patches_w

        self.positions = nn.Parameter(torch.randn(num_patches, emb_size))

    def forward(self, patchified_x):
        x = patchified_x + self.positions

        return x


class MedCrossAttention(CrossAttention):
    def __init__(self, emb_size, **kwargs):
        super(MedCrossAttention, self).__init__(**kwargs)

        self.emb_size = emb_size

        self.to_q = nn.Linear(emb_size, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    

class MedViTBlock(BasicTransformerBlock):
        def __init__(
        self,
        emb_size: int,
        num_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        **kwargs
    ) -> None:
        super(MedViTBlock, self).__init__(emb_size, **kwargs)

        self.attn1 = MedCrossAttention(
                                        emb_size=emb_size,
                                        query_dim=num_channels,
                                        cross_attention_dim=cross_attention_dim,
                                        num_attention_heads=num_attention_heads,
                                        num_head_channels=num_head_channels,
                                        dropout=dropout,
                                        upcast_attention=upcast_attention,
                                        use_flash_attention=use_flash_attention,
        )

        self.attn2 = MedCrossAttention(
                                        emb_size=emb_size,
                                        query_dim=num_channels,
                                        num_attention_heads=num_attention_heads,
                                        num_head_channels=num_head_channels,
                                        dropout=dropout,
                                        upcast_attention=upcast_attention,
                                        use_flash_attention=use_flash_attention,
        )

        self.ff = MLPBlock(hidden_size=num_channels, mlp_dim=num_channels * 4, act="GEGLU", dropout_rate=dropout)


    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # 1. Self-Attention
        x = self.attn1(self.norm1(x), context=context) + x

        # 2. Cross-Attention
        x = self.attn2(self.norm2(x), context=context) + x

        # 3. Feed-forward
        x = self.ff(self.norm3(x)) + x

        return x


class MediTransformer(SpatialTransformer):

    def __init__(self, img_size, emb_size=1024, patch_size=(6,9,6)):
        super(MediTransformer, self).__init__()

        b, c, d, h, w = img_size
        assert (d % patch_size[0] == 0), f'Image Depth must be divisible by the patch size. Now {d}, patch_size[0]: {patch_size[0]}'
        assert (h % patch_size[1] == 0), f'Image Depth must be divisible by the patch size. Now {h}, patch_size[0]: {patch_size[1]}'
        assert (w % patch_size[2] == 0), f'Image Depth must be divisible by the patch size. Now {w}, patch_size[0]: {patch_size[2]}'

        self.emb_size = emb_size
        self.patch_size = patch_size
        self.img_size = img_size
        self.dropout = self.dropout

        self.patch_embedding = PatchEmbedding3D(in_ch=1, emb_size=emb_size, patch_size=patch_size, img_size=img_size)
        self.positional_embedding = PositionalEmbedding3D(in_ch=1, emb_size=emb_size, patch_size=patch_size, img_size=img_size)

        # Do Multi-Head Attention
        self.transformer_blocks = nn.ModuleList( 
            [
            MedViTBlock(
                        emb_size=emb_size,
                        num_channels=self.inner_dim,
                        num_attention_heads=self.num_attention_heads,
                        num_head_channels=self.num_head_channels,
                        dropout=self.dropout,
                        cross_attention_dim=self.cross_attention_dim,
                        upcast_attention=self.upcast_attention,
                        use_flash_attention=self.use_flash_attention,
                        )

            for _ in range(self.num_layers)
            ]
        )

        self.proj_out = zero_module( # For zeroing out the parameters of the module
            Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=self.inner_dim,
                    out_channels=self.in_channels,
                    strides=1,
                    kernel_size=1,
                    padding=0,
                    conv_only=True,
                )
            )

    def foward(self, x, context=None):

        batch = channel = height = width = depth = -1

        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape

        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        residual = x

        x = self.patch_embedding(x)
        x = self.positional_embedding(x)

        inner_dim = x.shape[1]

        for block in self.transformer_blocks:
            x = block(x, context=context)

        if self.spatial_dims == 2:
            x = x.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        if self.spatial_dims == 3:
            x = x.reshape(batch, height, width, depth, inner_dim).permute(0, 4, 1, 2, 3).contiguous()

        x = self.proj_out(x)

        return x + residual