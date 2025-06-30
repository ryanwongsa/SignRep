# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# mae: https://github.com/facebookresearch/mae
# slowfast: https://github.com/facebookresearch/SlowFast
# --------------------------------------------------------


from functools import partial
from typing import Tuple, Optional

import math
import torch
import torch.nn as nn

from models.hiera.hiera_clstoken import Hiera, HieraBlock
from models.hiera.hiera_utils import pretrained_model, undo_windowing, conv_nd


def apply_fusion_head(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(head, nn.Identity):
        return x

    B, num_mask_units = x.shape[0:2]
    # Apply head, e.g [B, #MUs, My, Mx, C] -> head([B * #MUs, C, My, Mx])
    permute = [0] + [len(x.shape) - 2] + list(range(1, len(x.shape) - 2))
    x = head(x.reshape(B * num_mask_units, *x.shape[2:]).permute(permute))

    # Restore original layout, e.g. [B * #MUs, C', My', Mx'] -> [B, #MUs, My', Mx', C']
    permute = [0] + list(range(2, len(x.shape))) + [1]
    x = x.permute(permute).reshape(B, num_mask_units, *x.shape[2:], x.shape[1])
    return x


class MaskedAutoencoderHiera(Hiera):
    """Masked Autoencoder with Hiera backbone"""

    def __init__(
        self,
        in_chans: int = 3,
        patch_stride: Tuple[int, ...] = (4, 4),
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        num_cls_tokens: int = 0,
        **kwdargs,
    ):
        super().__init__(
            in_chans=in_chans,
            patch_stride=patch_stride,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            num_cls_tokens=num_cls_tokens,
            **kwdargs,
        )

        # del self.norm, self.head
        self.norm = nn.Identity()
        self.head = nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        # nn.init.trunc_normal_(self.mask_token, std=0.02)
        # nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        # self.apply(self._mae_init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _mae_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if mask is None:
            mask = self.get_random_mask(x, mask_ratio)  # [B, #MUs_all]

        # Get multi-scale representations from encoder
        res, intermediates, intermediate_cls_tokens = super().forward(
            x, mask, return_intermediates=True
        )
        if self.num_cls_tokens > 0:
            cls_x = res[:, 0]
        else:
            cls_x = res.mean(axis=1)

        return cls_x, intermediates, intermediate_cls_tokens

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.6,
        mask: Optional[torch.Tensor] = None,
        eval: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if eval:
            latent = super().forward(x)
            return latent
        else:
            latent, intermediates_feature_map, intermediate_features_cls = (
                self.forward_encoder(x, mask_ratio, mask=mask)
            )

            return latent, intermediates_feature_map, intermediate_features_cls


# Image Models


@pretrained_model(
    {
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
    },
    default="mae_in1k",
)
def mae_hiera_tiny_224(**kwargs):
    return MaskedAutoencoderHiera(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 7, 2),
        q_pool=2,
        **kwargs,
    )


@pretrained_model(
    {
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth",
    },
    default="mae_in1k",
)
def mae_hiera_small_224(**kwargs):
    return MaskedAutoencoderHiera(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 11, 2),
        q_pool=2,
        **kwargs,
    )


@pretrained_model(
    {
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth",
    },
    default="mae_in1k",
)
def mae_hiera_base_224(**kwargs):
    return MaskedAutoencoderHiera(
        embed_dim=96,
        num_heads=1,
        stages=(2, 3, 16, 3),
        q_pool=2,
        **kwargs,
    )


@pretrained_model(
    {
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth",
    },
    default="mae_in1k",
)
def mae_hiera_base_plus_224(**kwargs):
    return MaskedAutoencoderHiera(
        embed_dim=112,
        num_heads=2,
        stages=(2, 3, 16, 3),
        q_pool=2,
        **kwargs,
    )


@pretrained_model(
    {
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth",
    },
    default="mae_in1k",
)
def mae_hiera_large_224(**kwargs):
    return MaskedAutoencoderHiera(
        embed_dim=144,
        num_heads=2,
        stages=(2, 6, 36, 4),
        q_pool=2,
        **kwargs,
    )


@pretrained_model(
    {
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_224.pth",
    },
    default="mae_in1k",
)
def mae_hiera_huge_224(**kwargs):
    return MaskedAutoencoderHiera(
        embed_dim=256,
        num_heads=4,
        stages=(2, 6, 36, 4),
        q_pool=2,
        **kwargs,
    )


# Video Models


@pretrained_model(
    {
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth",
    },
    default="mae_k400",
)
def mae_hiera_base_16x224(num_classes: int = 400, **kwdargs):
    return MaskedAutoencoderHiera(
        num_classes=num_classes,  # K400 has 400 classes
        input_size=(16, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_pos_embed=True,
        q_pool=2,
        **kwdargs,
    )


@pretrained_model(
    {
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_16x224.pth",
    },
    default="mae_k400",
)
@pretrained_model(None)
def mae_hiera_base_plus_16x224(**kwdargs):
    return mae_hiera_base_16x224(
        embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs
    )


@pretrained_model(
    {
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_16x224.pth",
    },
    default="mae_k400",
)
@pretrained_model(None)
def mae_hiera_large_16x224(**kwdargs):
    return mae_hiera_base_16x224(
        embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs
    )


@pretrained_model(
    {
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_16x224.pth",
    },
    default="mae_k400",
)
def mae_hiera_huge_16x224(**kwdargs):
    return mae_hiera_base_16x224(
        embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs
    )
