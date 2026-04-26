# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from timm.models.vision_transformer import DropPath, Mlp, PatchEmbed

from macronav.pretrain.config.train_param import MASK_TOKEN
from macronav.pretrain.models.mae_vit import Block
from macronav.pretrain.utils.pos_embed import get_2d_sincos_pos_embed


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


class ViT(nn.Module):
    """Vision Transformer Encoder that can reuse weights from SSLViT"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        use_cls_token=False,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.mask_token = MASK_TOKEN

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token (optional)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.embed_pred_head = nn.Linear(self.embed_dim, patch_size**2 * in_chans)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize positional embeddings and other parameters"""
        # Initialize positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize class token if used
        if self.cls_token is not None:
            torch.nn.init.normal_(self.cls_token, std=0.02)

        # Initialize linear layers and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_all_tokens=True, return_attentions=False):
        """
        Forward pass through the encoder

        Args:
            x: Input images [B, C, H, W]
            return_all_tokens: If True, return all patch tokens; if False, return only cls token
            return_attentions: If True, return attention maps from all layers

        Returns:
            If return_all_tokens=False: [B, embed_dim] (cls token features)
            If return_all_tokens=True: [B, N, embed_dim] (all tokens)
            If return_attentions=True: features, attention_maps
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed[:, 1:, :]
        if self.use_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Apply transformer blocks
        attentions = []
        for blk in self.blocks:
            if return_attentions:
                x, attn = blk(x, return_attention=True)
                attentions.append(attn)
            else:
                x = blk(x)
        x = self.norm(x)

        x = self.embed_pred_head(x)

        # Return appropriate tokens
        if self.use_cls_token and not return_all_tokens:
            # Return only cls token
            pred = x[:, 0]  # [B, embed_dim]
        else:
            # Return all tokens or no cls token case
            pred = x

        if return_attentions:
            return pred, attentions
        else:
            return pred

    def load_ssl_weights(self, ssl_checkpoint_path, strict=True):
        """
        Load weights from SSLViT checkpoint

        Args:
            ssl_checkpoint_path: Path to the SSLViT checkpoint
            strict: Whether to strictly match parameter names
        """
        checkpoint = torch.load(ssl_checkpoint_path, map_location="cpu", weights_only=False)

        # Extract encoder weights
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Filter encoder weights
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if any(
                prefix in key
                for prefix in ["patch_embed", "pos_embed", "cls_token", "blocks", "norm", "embed_pred_head"]
            ):
                # Remove 'encoder.'  exclude decoder
                if key.startswith("encoder") or not key.startswith("decoder"):
                    clean_key = key.replace("encoder.", "")
                    encoder_state_dict[clean_key] = value
        if "embed_pred_head.1.weight" in state_dict.keys():  # compatibility with older checkpoints
            encoder_state_dict["embed_pred_head.weight"] = state_dict["embed_pred_head.1.weight"]
            encoder_state_dict["embed_pred_head.bias"] = state_dict["embed_pred_head.1.bias"]
            del encoder_state_dict["embed_pred_head.1.weight"]
            del encoder_state_dict["embed_pred_head.1.bias"]
            del encoder_state_dict["embed_pred_head.0.weight"]
            del encoder_state_dict["embed_pred_head.0.bias"]

        # Load weights
        missing_keys, unexpected_keys = self.load_state_dict(encoder_state_dict, strict=strict)

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print(f"Successfully loaded encoder weights from {ssl_checkpoint_path}")

    def get_feature_extractor(self):
        """Return a feature extractor function for easy inference"""

        def extract_features(x):
            self.eval()
            with torch.no_grad():
                return self.forward(x, return_all_tokens=False)

        return extract_features


def vit_tiny_patch8(**kwargs):
    """ViT encoder matching ssl_vit_tiny_patch8_dec256d3b configuration"""
    model = ViT(
        patch_size=8,
        embed_dim=512,
        depth=6,
        num_heads=4,
        mlp_ratio=2,
        in_chans=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_tiny_patch14(**kwargs):
    """ViT encoder matching ssl_vit_tiny_patch8_dec256d3b configuration"""
    model = ViT(
        patch_size=14,
        embed_dim=512,
        depth=6,
        num_heads=4,
        mlp_ratio=2,
        in_chans=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_small_patch8(**kwargs):
    """ViT encoder matching ssl_vit_tiny_patch8_dec256d3b configuration"""
    model = ViT(
        patch_size=8,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=2,
        in_chans=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_small_patch16(**kwargs):
    """ViT encoder matching ssl_vit_tiny_patch8_dec256d3b configuration"""
    model = ViT(
        patch_size=16,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=2,
        in_chans=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base_patch16(**kwargs):
    """ViT encoder matching standard base configuration"""
    model = ViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    """ViT encoder matching standard large configuration"""
    model = ViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create encoder
    encoder = vit_tiny_patch8(img_size=224, use_cls_token=True)
    encoder.to(device)

    # Test forward pass
    x = torch.randn(2, 1, 224, 224).to(device)

    # Get cls token features
    cls_features = encoder(x, return_all_tokens=False)
    print(f"CLS token features shape: {cls_features.shape}")

    # Get all token features
    all_features = encoder(x, return_all_tokens=True)
    print(f"All token features shape: {all_features.shape}")

    # Get features with attention
    features, attentions = encoder(x, return_attentions=True)
    print(f"Features shape: {features.shape}")
    print(f"Number of attention layers: {len(attentions)}")
    print(f"Attention shape: {attentions[0].shape}")

    # Example of loading SSL weights (uncomment when you have a checkpoint)
    # encoder.load_ssl_weights('path/to/ssl_checkpoint.pth')
