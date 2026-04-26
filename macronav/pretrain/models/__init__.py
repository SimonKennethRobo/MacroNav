from .mae_vit import (
    mae_vit_base_patch16_dec512d8b,
    mae_vit_huge_patch14_dec512d8b,
    mae_vit_large_patch16_dec512d8b,
    mae_vit_tiny_patch8_dec256d3b,
    mae_vit_tiny_patch14_dec256d3b,
    mae_vit_tiny_patch16_dec256d3b,
)
from .ssl_vit import ssl_vit_tiny_patch8_dec256d3b, ssl_vit_tiny_patch14_dec256d3b, ssl_vit_tiny_patch16_dec256d3b
from .vit import (
    vit_base_patch16,
    vit_large_patch16,
    vit_small_patch8,
    vit_small_patch16,
    vit_tiny_patch8,
    vit_tiny_patch14,
)

mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_tiny_patch8 = mae_vit_tiny_patch8_dec256d3b  # decoder: 256 dim, 3 blocks
mae_vit_tiny_patch14 = mae_vit_tiny_patch14_dec256d3b  # decoder: 256 dim, 3 blocks
mae_vit_tiny_patch16 = mae_vit_tiny_patch16_dec256d3b  # decoder: 256 dim, 3 blocks

# used for SSL pretraining
ssl_vit_patch8 = ssl_vit_tiny_patch8_dec256d3b
ssl_vit_patch14 = ssl_vit_tiny_patch14_dec256d3b
ssl_vit_patch16 = ssl_vit_tiny_patch16_dec256d3b

# used for pure visual inference
vit_tiny_patch8 = vit_tiny_patch8
vit_tiny_patch14 = vit_tiny_patch14
vit_small_patch8 = vit_small_patch8
vit_small_patch16 = vit_small_patch16
vit_base_patch16 = vit_base_patch16
vit_large_patch16 = vit_large_patch16
