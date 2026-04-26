import argparse
import math
import random
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from macronav.pretrain.config.train_param import MASK_TOKEN, NAV_PATCH_PXL_NUM_TH, NAV_PATCH_PXL_TH
from macronav.pretrain.models.mae_vit import MaskedAutoencoderViT
from macronav.pretrain.utils.misc import pad_tensor_list


def get_fov_mask(patch_num_w, center=None, shape="circle", fov_ratio=0.5, fov_expand_ratio=0.1):
    """Capture FOV and corresponding expansion areas
    Return:
        fov_mask: core visible area
        fov_expand_mask: total area including expansion
    """
    H = W = patch_num_w

    if center is None:
        center = (random.randint(0, W - 1), random.randint(0, H - 1))

    if shape == "circle":
        radius = int(H * fov_ratio / 2)
        expand_add = max(1, int(H * fov_expand_ratio / 2))
        expand_radius = radius + expand_add

        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        dist = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        mask = dist <= radius
        expand_mask = dist <= expand_radius

    elif shape == "square":
        size = int(H * fov_ratio)
        expand_add = max(2, int(H * fov_expand_ratio))
        expand_size = size + expand_add

        mask = torch.zeros(H, W, dtype=torch.bool)
        y1 = max(0, center[1] - size // 2)
        y2 = min(H, center[1] + size // 2)
        x1 = max(0, center[0] - size // 2)
        x2 = min(W, center[0] + size // 2)
        mask[y1:y2, x1:x2] = True

        expand_mask = torch.zeros(H, W, dtype=torch.bool)
        expand_y1 = max(0, center[1] - expand_size // 2)
        expand_y2 = min(H, center[1] + expand_size // 2)
        expand_x1 = max(0, center[0] - expand_size // 2)
        expand_x2 = min(W, center[0] + expand_size // 2)
        expand_mask[expand_y1:expand_y2, expand_x1:expand_x2] = True

    if (expand_mask & ~mask).sum() == 0:
        if shape == "circle":
            boundary_dist = radius + 0.5
            boundary_mask = (dist > radius) & (dist <= boundary_dist + 1)
            expand_mask = expand_mask | boundary_mask
        elif shape == "square":
            new_expand_mask = torch.zeros(H, W, dtype=torch.bool)
            expand_y1 = max(0, y1 - 1)
            expand_y2 = min(H, y2 + 1)
            expand_x1 = max(0, x1 - 1)
            expand_x2 = min(W, x2 + 1)
            new_expand_mask[expand_y1:expand_y2, expand_x1:expand_x2] = True
            expand_mask = new_expand_mask

    return mask.flatten(), expand_mask.flatten()


def get_fov_dist_weights(patch_num_w, expand_mask, fov_mask):
    """Calculate distance-based weights for expansion area"""
    H = W = patch_num_w
    expand_2d = expand_mask.reshape(H, W)
    fov_2d = fov_mask.reshape(H, W)

    # Calculate distance from FOV boundary
    y_indices, x_indices = torch.where(expand_2d)
    fov_y, fov_x = torch.where(fov_2d)

    if len(fov_y) == 0:
        return torch.ones(expand_mask.sum())

    weights = []
    for y, x in zip(y_indices, x_indices):
        # Find minimum distance to any FOV patch
        dist = torch.min(torch.sqrt((fov_y.float() - y) ** 2 + (fov_x.float() - x) ** 2))
        weight = torch.exp(-dist / 3.0)  # Exponential decay
        weights.append(weight)

    return torch.stack(weights) if weights else torch.tensor([1.0])


def get_mim_mask(
    patch_num_w, patch_size=8, num_vertices=None, center=None, mask_ratio=0.5, smoothness=0.6, size=(224, 224)
):
    """
    Generate a random polygon mask for a single patch grid
    Returns:
        mask: boolean tensor of shape [num_patches] indicating which patches to mask
    """
    H = W = patch_num_w
    if 1:  # Brownian motion mask
        mask = torch.zeros((H, W), dtype=torch.bool)
        steps = patch_num_w * patch_num_w * mask_ratio

        # Convert coordinates to patch indices
        x_patch = random.randint(0, W - 1)
        y_patch = random.randint(0, H - 1)

        directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)]
        current_direction = random.choice(directions)

        for _ in range(int(steps)):
            if 0 <= y_patch < H and 0 <= x_patch < W:
                mask[y_patch, x_patch] = True

            if random.random() < smoothness:
                dx, dy = current_direction
            else:
                current_direction = random.choice(directions)
                dx, dy = current_direction

            new_x_patch = x_patch + dx
            new_y_patch = y_patch + dy

            if 0 <= new_x_patch < W and 0 <= new_y_patch < H:
                x_patch, y_patch = new_x_patch, new_y_patch
            else:
                current_direction = random.choice(directions)
    if 0:  # polygon mask
        if num_vertices is None:
            num_vertices = random.randint(3, 8)  # Random polygon with 3-8 vertices

        # Generate random vertices within the patch grid
        vertices = []
        if center is None:
            center_x = random.randint(0, W - 1)
            center_y = random.randint(0, H - 1)
        else:
            center_x, center_y = center
        max_radius = min(W, H) * mask_ratio / 2

        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices + random.uniform(-math.pi / num_vertices, math.pi / num_vertices)
            radius = random.uniform(max_radius * 0.3, max_radius)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            vertices.append((max(0, min(W - 1, x)), max(0, min(H - 1, y))))

        # Create mask using polygon fill
        mask = torch.zeros(H, W, dtype=torch.bool)

        # Simple polygon fill using scanline algorithm
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

        # For each point, check if it's inside the polygon using ray casting
        for y in range(H):
            for x in range(W):
                inside = False
                j = num_vertices - 1
                for i in range(num_vertices):
                    xi, yi = vertices[i]
                    xj, yj = vertices[j]
                    if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                        inside = not inside
                    j = i
                mask[y, x] = inside

    return mask.flatten()


def get_mim_mask_batch(
    batch_size, patch_num_w, centers=None, mask_ratio=0.5, smoothness=0.5, device="cuda", use_brownian=True
):
    """
    Generate a batch of Brownian motion masks for multiple images in parallel
    Args:
        batch_size: number of masks to generate
        patch_num_w: number of patches in width dimension
        centers: optional list of centers for each mask in the batch
        mask_ratio: approximate ratio of patches to mask
        smoothness: probability of maintaining the same direction
        device: device to create tensors on
        use_brownian: if True, use Brownian motion; if False, use pure random masking
    Returns:
        masks: boolean tensor of shape [batch_size, num_patches] indicating which patches to mask
    """
    H = W = patch_num_w

    # Initialize all masks in the batch
    masks = torch.zeros((batch_size, H, W), dtype=torch.bool, device=device)

    if not use_brownian:  # Pure random masking
        num_patches = H * W
        num_masked = int(num_patches * mask_ratio)
        for b in range(batch_size):
            # Randomly select patches to mask
            indices = torch.randperm(num_patches, device=device)[:num_masked]
            y_coords = indices // W
            x_coords = indices % W
            masks[b, y_coords, x_coords] = True
        masks = masks.reshape(batch_size, -1)
        return masks

    # Brownian motion masking (existing implementation)
    steps = int(H * W * mask_ratio)

    x_patches = torch.randint(0, W, (batch_size,), device=device)
    y_patches = torch.randint(0, H, (batch_size,), device=device)

    if centers is not None:
        for b, center in enumerate(centers):
            if center is not None:
                x_patches[b] = center[0]
                y_patches[b] = center[1]

    directions = torch.tensor([(0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)], device=device)

    current_directions = torch.randint(0, len(directions), (batch_size,), device=device)

    # Perform Brownian motion for each mask in parallel
    st = time.time()
    for _ in range(steps):
        # Mark current patches as masked
        valid_indices = (y_patches >= 0) & (y_patches < H) & (x_patches >= 0) & (x_patches < W)
        batch_indices = torch.arange(batch_size, device=device)
        valid_batch = batch_indices[valid_indices]
        valid_y = y_patches[valid_indices]
        valid_x = x_patches[valid_indices]

        if len(valid_batch) > 0:
            masks[valid_batch, valid_y, valid_x] = True

        # Decide whether to maintain direction or change for each mask
        change_dir = torch.rand(batch_size, device=device) >= smoothness
        new_directions = torch.randint(0, len(directions), (batch_size,), device=device)
        current_directions = torch.where(change_dir, new_directions, current_directions)

        dx = directions[current_directions, 0]
        dy = directions[current_directions, 1]

        new_x_patches = x_patches + dx
        new_y_patches = y_patches + dy

        within_bounds = (new_x_patches >= 0) & (new_x_patches < W) & (new_y_patches >= 0) & (new_y_patches < H)
        x_patches = torch.where(within_bounds, new_x_patches, x_patches)
        y_patches = torch.where(within_bounds, new_y_patches, y_patches)

        hit_boundary = ~within_bounds
        new_rand_directions = torch.randint(0, len(directions), (batch_size,), device=device)
        current_directions = torch.where(hit_boundary, new_rand_directions, current_directions)
    masks = masks.reshape(batch_size, -1)
    return masks


def get_input_noise(img_batch: torch.tensor, random_add: bool = True):
    """
    add noise for imgs in MAE pretraining.

    imgs: [B, C, H, W]
        Noise: gaussian noise, gaussian blur, motion blur, solarization, random erasing
    """
    B, C, H, W = img_batch.shape
    device = img_batch.device
    img_batch_noised = img_batch.clone()

    # Apply different types of noise randomly
    for i in range(B):
        img = img_batch_noised[i]  # [C, H, W]

        # Randomly choose which augmentations to apply (can apply multiple)
        if random_add:
            apply_gaussian_noise = random.random() < 0.3
            apply_gaussian_blur = random.random() < 0.3
            apply_motion_blur = random.random() < 0.3
            # apply_solarization = random.random() < 0.3
            apply_solarization = False
            # apply_random_erasing = random.random() < 0.3
            apply_random_erasing = False
        else:
            apply_gaussian_noise = True
            apply_gaussian_blur = True
            apply_motion_blur = True
            apply_solarization = False
            apply_random_erasing = False

        # 1. Gaussian Noise
        if apply_gaussian_noise:
            noise_std = random.uniform(0.01, 0.2)
            noise = torch.randn_like(img) * noise_std
            img = torch.clamp(img + noise, 0.0, 1.0)

        # 2. Gaussian Blur
        if apply_gaussian_blur:
            # Increase kernel size and sigma for stronger blur
            kernel_size = random.choice([5, 7, 9, 11, 15])
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = random.uniform(1.0, 4.0)  # Increased from (0.5, 2.0)
            img = TF.gaussian_blur(img, kernel_size, sigma)

        # 3. Motion Blur (approximated using directional convolution)
        if apply_motion_blur:
            kernel_size = random.randint(5, 11)
            # Ensure kernel size is odd for symmetric padding
            if kernel_size % 2 == 0:
                kernel_size += 1
            angle = random.uniform(0, 2 * math.pi)

            # Create motion blur kernel
            kernel = torch.zeros((kernel_size, kernel_size), device=device)
            center = kernel_size // 2

            # Create line kernel based on angle
            for k in range(kernel_size):
                x = int(center + (k - center) * math.cos(angle))
                y = int(center + (k - center) * math.sin(angle))
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1.0

            # Normalize kernel
            if kernel.sum() > 0:
                kernel = kernel / kernel.sum()
            else:
                kernel[center, center] = 1.0  # Fallback to identity

            kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

            # Apply convolution with proper padding to maintain dimensions
            padding = kernel_size // 2
            img_padded = F.pad(img.unsqueeze(0), (padding, padding, padding, padding), mode="reflect")
            img = F.conv2d(img_padded, kernel, groups=C, padding=0).squeeze(0)

        # 4. Solarization
        if apply_solarization:
            threshold = random.uniform(0.3, 0.7)
            img = torch.where(img > threshold, 1.0 - img, img)

        # 5. Random Erasing
        if apply_random_erasing:
            # Random erasing parameters
            area_ratio = random.uniform(0.02, 0.1)  # Erase 2-10% of the image
            aspect_ratio = random.uniform(0.3, 3.0)

            area = H * W * area_ratio
            target_h = int(round(math.sqrt(area * aspect_ratio)))
            target_w = int(round(math.sqrt(area / aspect_ratio)))

            if target_h < H and target_w < W:
                top = random.randint(0, H - target_h)
                left = random.randint(0, W - target_w)

                # Fill with random values or mean
                if random.random() < 0.5:
                    # Fill with random noise
                    img[:, top : top + target_h, left : left + target_w] = torch.rand(
                        C, target_h, target_w, device=device
                    )
                else:
                    # Fill with mean value
                    mean_val = img.mean()
                    img[:, top : top + target_h, left : left + target_w] = mean_val

        img_batch_noised[i] = img

    return img_batch_noised


def viz_mim_mask(img_size, patch_size, mask_ratios=[0.5]):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    patch_num_w = img_size // patch_size
    fig.suptitle("MIM Polygon Mask Visualization", fontsize=16)

    for i, mask_ratio in enumerate(mask_ratios):
        # Generate mask batch (batch_size=1 for single mask visualization)
        masks = get_mim_mask_batch(
            batch_size=1,
            patch_num_w=patch_num_w,
            mask_ratio=mask_ratio,
            smoothness=0.5,
            device="cpu",
            use_brownian=True,
        )
        mask = masks[0]  # Extract the single mask from the batch
        H = W = patch_num_w

        # Reshape mask to 2D grid
        mask_2d = mask.reshape(H, W)

        # Create white background image
        img = np.ones((img_size, img_size, 3))

        # Overlay mask on image
        img_with_mask = img.copy()
        for y in range(H):
            for x in range(W):
                if mask_2d[y, x]:
                    # Color masked patches red
                    y_start, y_end = y * patch_size, (y + 1) * patch_size
                    x_start, x_end = x * patch_size, (x + 1) * patch_size
                    img_with_mask[y_start:y_end, x_start:x_end] = [1.0, 0.0, 0.0]  # Red

        # Plot original mask pattern
        axes[0, i].imshow(mask_2d, cmap="Reds", alpha=0.8)
        axes[0, i].set_title(f"Mask Pattern (complexity={mask_ratio})")
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_xticks(range(0, W, 2))
        axes[0, i].set_yticks(range(0, H, 2))

        # Plot image with masked patches
        axes[1, i].imshow(img_with_mask)
        axes[1, i].set_title(f"Masked Patches (complexity={mask_ratio})")
        axes[1, i].axis("off")

        # Add patch grid overlay
        for y in range(0, img_size, patch_size):
            axes[1, i].axhline(y, color="gray", alpha=0.3, linewidth=0.5)
        for x in range(0, img_size, patch_size):
            axes[1, i].axvline(x, color="gray", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig("mim_mask_visualization.png", dpi=150, bbox_inches="tight")
    print("MIM mask visualization saved to tmp/mim_mask_visualization.png")


def viz_fov_mask_batch(img: torch.tensor, patch_size, fov_mask, fov_expand_mask):
    img_size = img.shape[-1]
    img_show = torch.zeros((3, img_size, img_size), dtype=torch.float32)
    num_patch_w = img_size // patch_size
    num_patch_h = img_size // patch_size
    # set  fov_mask to blue, fov_expand_mask to red, and others to white
    for i in range(num_patch_h):
        for j in range(num_patch_w):
            if fov_mask[i * num_patch_w + j]:
                img_show[:, i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size] = (
                    torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1).expand(3, patch_size, patch_size)
                )  # blue
            elif fov_expand_mask[i * num_patch_w + j]:
                img_show[:, i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size] = (
                    torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1).expand(3, patch_size, patch_size)
                )  # red
            else:
                img_show[:, i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size] = (
                    torch.tensor([0, 0, 0]).view(3, 1, 1).expand(3, patch_size, patch_size)
                )  # white
    img_show = img_show.permute(1, 2, 0)  # [H, W, C]
    img_show = (img_show * 255).byte()  # convert to uint8
    img_plt = img_show.cpu().numpy()
    img_raw_plt = img.permute(1, 2, 0).cpu().numpy() / 0.5 + 0.5  # [H, W, C], normalize to [0, 1]
    plt.figure(figsize=(8, 8))
    plt.imshow(img_raw_plt, alpha=1, cmap="gray")
    plt.imshow(img_plt, alpha=0.5)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("tmp/fov_mask.png", bbox_inches="tight")
    print("FOV mask visualization saved to tmp/fov_mask.png")


def get_mae_loss(img_patchified, pred: torch.tensor, mask: torch.tensor, norm_pix_loss=False):
    target = img_patchified
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss_ = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss_


def get_mim_loss(img_patchified: torch.tensor, pred_list: list, polygon_masks: list, norm_pix_loss=False):
    """
    Calculate MIM reconstruction loss
    Args:
        img_patchified: original image patches [B, N, patch_dim]
        pred_list: list of predictions for each batch item
        polygon_masks: list of polygon masks for each batch item
    Returns:
        loss: reconstruction loss
    """
    total_patches = 0
    losses = []

    for b, (pred, mask) in enumerate(zip(pred_list, polygon_masks)):
        if pred.shape[0] == 0:  # No masked patches
            continue

        target = img_patchified[b][mask]

        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss_patch = F.mse_loss(pred, target)
        total_patches += pred.shape[0]
        losses.append(loss_patch)

    if total_patches == 0:
        return torch.tensor(0.0, device=img_patchified.device, requires_grad=True)

    return torch.stack(losses).mean()


def get_fov_loss(img_patchified, pred, fov_mask, expand_only_mask, uncertainty_weight=False, norm_pix_loss=False):
    # Get target patches for expansion area only
    target_patches = img_patchified[:, expand_only_mask, :]

    # Get predicted patches for expansion area (after visible patches)
    visible_patch_num = fov_mask.sum().item()
    pred_patches = pred[:, visible_patch_num:, :]

    # Ensure shapes match
    if pred_patches.shape[1] != target_patches.shape[1]:
        min_patches = min(pred_patches.shape[1], target_patches.shape[1])
        pred_patches = pred_patches[:, :min_patches, :]
        target_patches = target_patches[:, :min_patches, :]

    # Apply pixel normalization if enabled
    if norm_pix_loss:
        mean = target_patches.mean(dim=-1, keepdim=True)
        var = target_patches.var(dim=-1, keepdim=True)
        target_patches = (target_patches - mean) / (var + 1.0e-6) ** 0.5

    # Calculate MSE loss
    recon_loss = F.mse_loss(pred_patches, target_patches, reduction="mean")

    # Optional: Add distance-based weighting
    if uncertainty_weight:
        patch_num_w = img_patchified.shape[1] // img_patchified.shape[2]
        weights = get_fov_dist_weights(patch_num_w, expand_only_mask, fov_mask)
        loss_per_patch = F.mse_loss(pred_patches, target_patches, reduction="none").mean(dim=-1)
        recon_loss = (loss_per_patch * weights).mean()

    return recon_loss


class SSLViT(MaskedAutoencoderViT):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_num_w = img_size // patch_size
        self.encoder_embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.curr_img_batch = None  # only for debug
        self.first_forward = True

        self.embed_pred_head = nn.Linear(
            self.encoder_embed_dim, patch_size**2 * in_chans
        )  # embedding to img patch prediction

    def get_mae_loss(self, img_patchified, pred: torch.tensor, mask: torch.tensor):
        target = img_patchified
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss_ = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss_

    def get_mim_loss(self, img_patchified: torch.tensor, pred_list: list, polygon_masks: list):
        """
        Calculate MIM reconstruction loss
        Args:
            img_patchified: original image patches [B, N, patch_dim]
            pred_list: list of predictions for each batch item
            polygon_masks: list of polygon masks for each batch item
        Returns:
            loss: reconstruction loss
        """
        total_patches = 0
        losses = []

        for b, (pred, mask) in enumerate(zip(pred_list, polygon_masks)):
            if pred.shape[0] == 0:  # No masked patches
                continue

            target = img_patchified[b][mask]

            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5

            loss_patch = F.mse_loss(pred, target)
            total_patches += pred.shape[0]
            losses.append(loss_patch)

        if total_patches == 0:
            return torch.tensor(0.0, device=img_patchified.device, requires_grad=True)

        return torch.stack(losses).mean()

    def get_fov_loss(self, img_patchified, pred, fov_mask, expand_only_mask, uncertainty_weight=False):
        # Get target patches for expansion area only
        target_patches = img_patchified[:, expand_only_mask, :]

        # Get predicted patches for expansion area (after visible patches)
        visible_patch_num = fov_mask.sum().item()
        pred_patches = pred[:, visible_patch_num:, :]

        # Ensure shapes match
        if pred_patches.shape[1] != target_patches.shape[1]:
            min_patches = min(pred_patches.shape[1], target_patches.shape[1])
            pred_patches = pred_patches[:, :min_patches, :]
            target_patches = target_patches[:, :min_patches, :]

        # Apply pixel normalization if enabled
        if self.norm_pix_loss:
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var + 1.0e-6) ** 0.5

        # Calculate MSE loss
        recon_loss = F.mse_loss(pred_patches, target_patches, reduction="mean")

        # Optional: Add distance-based weighting
        if uncertainty_weight:
            weights = get_fov_dist_weights(self.patch_num_w, expand_only_mask, fov_mask)
            loss_per_patch = F.mse_loss(pred_patches, target_patches, reduction="none").mean(dim=-1)
            recon_loss = (loss_per_patch * weights).mean()

        return recon_loss

    def forward_mae(
        self,
        embed_patch: torch.tensor,
        mask_ratio: float = 0.75,
        return_attentions: bool = False,
    ):
        embed_patch = embed_patch.clone()

        # encoding --------------------------------
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(embed_patch, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            if return_attentions:
                x, attn = blk(x, return_attention=True)
            else:
                x = blk(x)
        x = self.norm(x)

        # decoding --------------------------------
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            if return_attentions:
                x, attn = blk(x, return_attention=True)
            else:
                x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x, mask

    def forward_fov(
        self, img_batch, embed_patch: torch.tensor, fov_ratio: float = 0.25, fov_expand_ratio=0.2, center: tuple = None
    ):
        embed_patch = embed_patch.clone()
        (B, _, _) = embed_patch.shape

        batch_use_different_centers = False

        if batch_use_different_centers:
            navigable_mask = (img_batch > NAV_PATCH_PXL_TH).sum(dim=2) > NAV_PATCH_PXL_NUM_TH
            centers = []
            for b in range(B):
                navigable_token_idx = torch.where(navigable_mask[b])[0]
                size = navigable_token_idx.shape[0]
                if size == 0:
                    # If no navigable patches, randomly sample a center
                    sampled_y = torch.randint(0, self.patch_num_w, (1,), device=embed_patch.device)
                    sampled_x = torch.randint(0, self.patch_num_w, (1,), device=embed_patch.device)
                    centers.append((sampled_x.item(), sampled_y.item()))
                    continue
                sampled_nav_token_idx = navigable_token_idx[torch.randint(0, size, (1,))]
                sampled_y = sampled_nav_token_idx // self.patch_num_w
                sampled_x = sampled_nav_token_idx % self.patch_num_w
                centers.append((sampled_x.item(), sampled_y.item()))

            input_embed_batch = []
            for b in range(B):
                fov_mask, fov_expand_mask = get_fov_mask(
                    self.patch_num_w, centers[b], "circle", fov_ratio, fov_expand_ratio
                )
                # Move masks to the correct device first
                fov_mask = fov_mask.to(embed_patch.device)
                fov_expand_mask = fov_expand_mask.to(embed_patch.device)

                visible_patch = embed_patch[b, fov_mask, :]
                expand_only_mask = fov_expand_mask & ~fov_mask
                masked_patch = embed_patch[b, expand_only_mask, :]
                mask_tokens = torch.full(
                    (masked_patch.shape[0], self.encoder_embed_dim), MASK_TOKEN, device=embed_patch.device
                )

                pos_embed_visible = self.pos_embed[0, 1:, :][fov_mask.flatten()]
                pos_embed_masked = self.pos_embed[0, 1:, :][expand_only_mask.flatten()]

                visible_patch = visible_patch + pos_embed_visible
                mask_tokens = mask_tokens + pos_embed_masked

                combined = torch.cat((visible_patch, mask_tokens), dim=0)
                input_embed_batch.append(combined)

            input_embed_batch, padding_mask = pad_tensor_list(input_embed_batch)
        else:  # all samples in batch use the same center and fov masks
            margin = int(self.patch_num_w * fov_ratio / 2)
            if center is None:
                center_y = torch.randint(margin, self.patch_num_w - margin, (1,))
                center_x = torch.randint(margin, self.patch_num_w - margin, (1,))
                center = (center_x, center_y)

            fov_mask, fov_expand_mask = get_fov_mask(self.patch_num_w, center, "circle", fov_ratio, fov_expand_ratio)

            visible_patch = embed_patch[:, fov_mask, :]
            expand_only_mask = fov_expand_mask & ~fov_mask
            masked_patch = embed_patch[:, expand_only_mask, :]
            mask_tokens = torch.full(
                (B, masked_patch.shape[1], self.encoder_embed_dim), MASK_TOKEN, device=embed_patch.device
            )

            pos_embed_visible = self.pos_embed[:, 1:, :][:, fov_mask, :]
            pos_embed_masked = self.pos_embed[:, 1:, :][:, expand_only_mask, :]
            visible_patch = visible_patch + pos_embed_visible
            mask_tokens = mask_tokens + pos_embed_masked

            input_embed_batch = torch.cat((visible_patch, mask_tokens), dim=1)
            padding_mask = None

        for blk in self.blocks:
            input_embed_batch = blk(input_embed_batch, key_padding_mask=padding_mask)
        input_embed_batch = self.norm(input_embed_batch)

        pred = self.embed_pred_head(input_embed_batch)
        fov_mask = fov_mask.to(pred.device)
        expand_only_mask = expand_only_mask.to(pred.device)
        return pred, fov_mask, expand_only_mask

    def forward_mim(
        self, embed_patch: torch.tensor, mask_ratio: float = 0.4, center=None, smoothness=0.7, use_brownian=True
    ):
        """
        Masked Image Modeling with polygon masks - fully batch parallelized
        Args:
            embed_patch: embedded patches [B, N, D]
            mask_ratio: approximate ratio of patches to mask
        Returns:
            pred: predictions for masked patches [B, num_masked, patch_dim]
            polygon_masks: list of polygon masks for each batch item
        """
        B, N, D = embed_patch.shape
        device = embed_patch.device

        if center is None:
            margin = int(self.patch_num_w * mask_ratio / 2)
            center_y = torch.randint(margin, self.patch_num_w - margin, (B,), device=device)
            center_x = torch.randint(margin, self.patch_num_w - margin, (B,), device=device)
            centers = [(center_x[i], center_y[i]) for i in range(B)]
        else:
            centers = [center] * B

        mask_batch = get_mim_mask_batch(
            B, self.patch_num_w, centers, mask_ratio=mask_ratio, smoothness=smoothness, use_brownian=use_brownian
        )
        mask_tensor = torch.full_like(embed_patch, MASK_TOKEN)
        # Create a boolean mask to apply the MASK_TOKEN
        bool_mask = mask_batch.unsqueeze(-1).expand(-1, -1, D)
        masked_embed = torch.where(bool_mask, mask_tensor, embed_patch)

        masked_embed = masked_embed + self.pos_embed[:, 1:, :]

        for blk in self.blocks:
            masked_embed = blk(masked_embed)
        masked_embed = self.norm(masked_embed)

        masked_pred_list = []
        polygon_masks = []

        for b in range(B):
            mask = mask_batch[b]
            polygon_masks.append(mask)
            masked_features = masked_embed[b][mask]

            if masked_features.shape[0] > 0:  # Only process if there are masked patches
                pred = self.embed_pred_head(masked_features)
                masked_pred_list.append(pred)
            else:
                # Empty prediction if no patches are masked
                masked_pred_list.append(torch.tensor([], device=device))

        return masked_pred_list, polygon_masks

    def forward(self, img_batch: torch.tensor, args: argparse.Namespace):
        """
        Input:
            SSL training can either randomly sample one task from mae/mim/fov,
            or run all tasks jointly in one forward.
            args.add_noise(bool): whether to add noise to the input images
        """
        if self.first_forward:
            print(f"SSLViT forward called with args: {args}")
            self.first_forward = False
        self.curr_img_batch = img_batch
        img_patchified = self.patchify(img_batch)  # (B,num_patches, patch_size^2)
        if args.add_noise:
            img_batch = get_input_noise(img_batch)
        embedded_patch = self.patch_embed(img_batch)  # (batch, num_patches=(img_width/patch_size)^2, embed_dim)

        # add pos embed w/o cls token
        embedded_patch = embedded_patch + self.pos_embed[:, 1:, :]

        task_mode = getattr(args, "ssl_train_mode", "joint")
        task_mode = str(task_mode).lower()

        if task_mode in ("sample", "random"):
            selected_tasks = [random.choice(("mae", "fov", "mim"))]
        elif task_mode in ("joint", "all"):
            selected_tasks = ["mae", "mim", "fov"]
        else:
            raise ValueError(
                f"Unknown ssl_train_mode: {task_mode}. "
                "Use one of: sample/random or joint/all"
            )

        loss_dict = {}
        pred_dict = {}
        # print(f"SSL task mode: {task_mode}, selected tasks: {selected_tasks}")

        for task in selected_tasks:
            if task == "mae":
                pred, mask = self.forward_mae(embedded_patch, mask_ratio=args.mask_ratio)
                loss_mae = self.get_mae_loss(img_patchified, pred, mask)
                loss_dict["mae"] = loss_mae
                pred_dict["mae"] = pred
            elif task == "mim":
                maksed_pred_list, mim_masks = self.forward_mim(
                    embedded_patch,
                    mask_ratio=args.mim_ratio,
                    smoothness=args.mim_smoothness,
                    use_brownian=args.mim_use_brownian,
                )
                loss_mim = self.get_mim_loss(img_patchified, maksed_pred_list, mim_masks)
                loss_dict["mim"] = loss_mim
                pred_dict["mim"] = maksed_pred_list
            elif task == "fov":
                pred, fov_mask, fov_expand_mask = self.forward_fov(
                    img_patchified, embedded_patch, fov_ratio=args.fov_ratio, fov_expand_ratio=args.fov_expand_ratio
                )
                loss_fov = self.get_fov_loss(img_patchified, pred, fov_mask, fov_expand_mask)
                loss_dict["fov"] = loss_fov
                pred_dict["fov"] = pred
            else:
                raise ValueError(f"Unknown SSL task: {task}")

        return loss_dict, pred_dict


class SSLViT2(MaskedAutoencoderViT):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_num_w = img_size // patch_size
        self.encoder_embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.curr_img_batch = None  # only for debug

        # aux modules
        self.embed_pred_head = nn.Linear(
            self.encoder_embed_dim, patch_size**2 * in_chans
        )  # embedding to img patch prediction

    def get_mae_input(
        self, embed_patch: torch.Tensor, mask_ratio: float = 0.75, external_mask=None, external_ids_restore=None
    ):
        embed_patch = embed_patch.clone()

        if external_mask is not None and external_ids_restore is not None:
            # Use externally provided mask and ids_restore
            x = embed_patch
            # B, L, D = x.shape
            # x = x[~external_mask].reshape(B, -1, D)
            mask = external_mask
            ids_restore = external_ids_restore
        else:
            # Generate mask internally as before
            x, mask, ids_restore = self.random_masking(embed_patch, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x, ids_restore, mask

    def forward_mae(self, x, ids_restore, return_attentions: bool = False):
        for blk in self.blocks:
            if return_attentions:
                x, attn = blk(x, return_attention=True)
            else:
                x = blk(x)
        x = self.norm(x)

        # decoding --------------------------------
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            if return_attentions:
                x, attn = blk(x, return_attention=True)
            else:
                x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def get_fov_input(
        self,
        img_batch: torch.Tensor,
        embed_patch: torch.Tensor,
        fov_ratio: float = 0.25,
        fov_expand_ratio=0.2,
        center: tuple = None,
        external_masks=None,
    ):
        embed_patch = embed_patch.clone()
        (B, _, _) = embed_patch.shape

        if external_masks is not None:
            fov_mask, fov_expand_mask = external_masks
            fov_mask = fov_mask.to(embed_patch.device)
            fov_expand_mask = fov_expand_mask.to(embed_patch.device)

            visible_patch = embed_patch[:, fov_mask, :]
            expand_only_mask = fov_expand_mask & ~fov_mask
            masked_patch = embed_patch[:, expand_only_mask, :]
            mask_tokens = torch.full(
                (B, masked_patch.shape[1], self.encoder_embed_dim), MASK_TOKEN, device=embed_patch.device
            )

            pos_embed_visible = self.pos_embed[:, 1:, :][:, fov_mask, :]
            pos_embed_masked = self.pos_embed[:, 1:, :][:, expand_only_mask, :]
            visible_patch = visible_patch + pos_embed_visible
            mask_tokens = mask_tokens + pos_embed_masked

            input_embed_batch = torch.cat((visible_patch, mask_tokens), dim=1)
            padding_mask = None

        else:
            batch_use_different_centers = False

            if batch_use_different_centers:
                navigable_mask = (img_batch > NAV_PATCH_PXL_TH).sum(dim=2) > NAV_PATCH_PXL_NUM_TH
                centers = []
                for b in range(B):
                    navigable_token_idx = torch.where(navigable_mask[b])[0]
                    size = navigable_token_idx.shape[0]
                    if size == 0:
                        # If no navigable patches, randomly sample a center
                        sampled_y = torch.randint(0, self.patch_num_w, (1,), device=embed_patch.device)
                        sampled_x = torch.randint(0, self.patch_num_w, (1,), device=embed_patch.device)
                        centers.append((sampled_x.item(), sampled_y.item()))
                        continue
                    sampled_nav_token_idx = navigable_token_idx[torch.randint(0, size, (1,))]
                    sampled_y = sampled_nav_token_idx // self.patch_num_w
                    sampled_x = sampled_nav_token_idx % self.patch_num_w
                    centers.append((sampled_x.item(), sampled_y.item()))

                input_embed_batch = []
                for b in range(B):
                    fov_mask, fov_expand_mask = get_fov_mask(
                        self.patch_num_w, centers[b], "circle", fov_ratio, fov_expand_ratio
                    )
                    # Move masks to the correct device first
                    fov_mask = fov_mask.to(embed_patch.device)
                    fov_expand_mask = fov_expand_mask.to(embed_patch.device)

                    visible_patch = embed_patch[b, fov_mask, :]
                    expand_only_mask = fov_expand_mask & ~fov_mask
                    masked_patch = embed_patch[b, expand_only_mask, :]
                    mask_tokens = torch.full(
                        (masked_patch.shape[0], self.encoder_embed_dim), MASK_TOKEN, device=embed_patch.device
                    )

                    pos_embed_visible = self.pos_embed[0, 1:, :][fov_mask.flatten()]
                    pos_embed_masked = self.pos_embed[0, 1:, :][expand_only_mask.flatten()]

                    visible_patch = visible_patch + pos_embed_visible
                    mask_tokens = mask_tokens + pos_embed_masked

                    combined = torch.cat((visible_patch, mask_tokens), dim=0)
                    input_embed_batch.append(combined)

                input_embed_batch, padding_mask = pad_tensor_list(input_embed_batch)
            else:  # all samples in batch use the same center and fov masks
                margin = int(self.patch_num_w * fov_ratio / 2)
                if center is None:
                    center_y = torch.randint(margin, self.patch_num_w - margin, (1,))
                    center_x = torch.randint(margin, self.patch_num_w - margin, (1,))
                    center = (center_x, center_y)

                fov_mask, fov_expand_mask = get_fov_mask(
                    self.patch_num_w, center, "circle", fov_ratio, fov_expand_ratio
                )

                visible_patch = embed_patch[:, fov_mask, :]
                expand_only_mask = fov_expand_mask & ~fov_mask
                masked_patch = embed_patch[:, expand_only_mask, :]
                mask_tokens = torch.full(
                    (B, masked_patch.shape[1], self.encoder_embed_dim), MASK_TOKEN, device=embed_patch.device
                )

                pos_embed_visible = self.pos_embed[:, 1:, :][:, fov_mask, :]
                pos_embed_masked = self.pos_embed[:, 1:, :][:, expand_only_mask, :]
                visible_patch = visible_patch + pos_embed_visible
                mask_tokens = mask_tokens + pos_embed_masked

                input_embed_batch = torch.cat((visible_patch, mask_tokens), dim=1)
                padding_mask = None

        return input_embed_batch, fov_mask, fov_expand_mask, padding_mask

    def get_mim_input(
        self,
        embed_patch: torch.Tensor,
        mask_ratio: float = 0.4,
        center: tuple = None,
        smoothness: float = 0.0,
        external_mask_batch=None,
        use_brownian: bool = True,
    ):
        B, N, D = embed_patch.shape
        device = embed_patch.device

        if external_mask_batch is not None:
            # Use externally provided mask batch
            mask_batch = external_mask_batch.to(device)
            # align batchsize
            if mask_batch.shape[0] != B:
                mask_batch = mask_batch[:B, :]
        else:
            # Original implementation with internal mask generation
            if center is None:
                margin = int(self.patch_num_w * mask_ratio / 2)
                center_y = torch.randint(margin, self.patch_num_w - margin, (B,), device=device)
                center_x = torch.randint(margin, self.patch_num_w - margin, (B,), device=device)
                centers = [(center_x[i], center_y[i]) for i in range(B)]
            else:
                centers = [center] * B

            mask_batch = get_mim_mask_batch(
                B, self.patch_num_w, centers, mask_ratio=mask_ratio, smoothness=smoothness, use_brownian=use_brownian
            )

        mask_tensor = torch.full_like(embed_patch, MASK_TOKEN)
        bool_mask = mask_batch.unsqueeze(-1).expand(-1, -1, D)
        masked_embed = torch.where(bool_mask, mask_tensor, embed_patch)

        masked_embed = masked_embed + self.pos_embed[:, 1:, :]
        return masked_embed, mask_batch

    def forward_fov(self, input_embed_batch, padding_mask=None):
        """
        Process FOV input embeddings through the model
        Args:
            input_embed_batch: Input embeddings with visible and masked patches
            padding_mask: Optional mask for padding in variable-length sequences
        Returns:
            Predicted patch values
        """
        for blk in self.blocks:
            input_embed_batch = blk(input_embed_batch, key_padding_mask=padding_mask)
        input_embed_batch = self.norm(input_embed_batch)

        pred = self.embed_pred_head(input_embed_batch)

        return pred

    def forward_mim(self, masked_embed, mask_batch):
        B, N, D = masked_embed.shape
        device = masked_embed.device

        for blk in self.blocks:
            masked_embed = blk(masked_embed)
        masked_embed = self.norm(masked_embed)

        masked_pred_list = []

        for b in range(B):
            mask = mask_batch[b]
            masked_features = masked_embed[b][mask]

            if masked_features.shape[0] > 0:  # Only process if there are masked patches
                pred = self.embed_pred_head(masked_features)
                masked_pred_list.append(pred)
            else:
                # Empty prediction if no patches are masked
                masked_pred_list.append(torch.tensor([], device=device))

        return masked_pred_list

    def forward(self, images: torch.Tensor, task, args: argparse.Namespace, external_masks=None):
        """
        Forward pass with option to use external masks for consistent evaluation
        Args:
            images: Input images
            task: Task type ('mae', 'fov', or 'mim')
            args: Arguments for the specific task
            external_masks: Dictionary containing pre-generated masks for each task
                - For 'mae': {'mask': mask_tensor, 'ids_restore': ids_restore_tensor}
                - For 'fov': {'fov_mask': fov_mask, 'fov_expand_mask': fov_expand_mask}
                - For 'mim': {'mask_batch': mask_batch}
        Returns:
            loss and prediction based on the task
        """
        img_patchified = self.patchify(images)
        embed_patch = self.patch_embed(images)
        embed_patch = embed_patch + self.pos_embed[:, 1:, :]

        B, N, D = embed_patch.shape

        # Handle external masks based on task
        task_masks = None if external_masks is None else external_masks.get(task, None)

        if task == "mae":
            if task_masks is not None:
                external_mask = task_masks.get("mask")
                external_ids_restore = task_masks.get("ids_restore")
                if external_mask.shape[0] != B:
                    external_mask = external_mask[:B, :]
                    external_ids_restore = external_ids_restore[:B, :]
                x, ids_restore, mask = self.get_mae_input(
                    embed_patch,
                    args.mask_ratio,
                    external_mask=external_mask,
                    external_ids_restore=external_ids_restore,
                )
            else:
                x, ids_restore, mask = self.get_mae_input(embed_patch, args.mae_mask_ratio)

            pred = self.forward_mae(x, ids_restore)
            loss_mae = get_mae_loss(img_patchified, pred, mask, self.norm_pix_loss)
            return loss_mae, pred

        elif task == "fov":
            if task_masks is not None:
                external_fov_masks = (task_masks.get("fov_mask"), task_masks.get("fov_expand_mask"))
                processed_embeds, fov_mask, fov_expand_mask, padding_mask = self.get_fov_input(
                    images, embed_patch, args.fov_ratio, args.fov_expand_ratio, external_masks=external_fov_masks
                )
                fov_mask = external_fov_masks[0]
                expand_only_mask = external_fov_masks[1] & ~external_fov_masks[0]

            else:
                processed_embeds, fov_mask, fov_expand_mask, padding_mask = self.get_fov_input(
                    images, embed_patch, args.fov_ratio, args.fov_expand_ratio
                )

            pred = self.forward_fov(processed_embeds, padding_mask)
            loss_fov = get_fov_loss(img_patchified, pred, fov_mask, expand_only_mask, norm_pix_loss=self.norm_pix_loss)
            return loss_fov, pred

        elif task == "mim":
            if task_masks is not None:
                masked_embed, mask_batch = self.get_mim_input(
                    embed_patch,
                    mask_ratio=args.mim_mask_ratio,
                    center=task_masks["center"],
                    smoothness=args.mim_smoothness,
                    external_mask_batch=task_masks.get("mask_batch"),
                    use_brownian=task_masks.get("mim_use_brownian", True),
                )
            else:
                masked_embed, mask_batch = self.get_mim_input(
                    embed_patch,
                    mask_ratio=args.mim_mask_ratio,
                    center=args.mim_center,
                    smoothness=args.mim_smoothness,
                    use_brownian=args.mim_use_brownian,
                )

            pred = self.forward_mim(masked_embed, mask_batch)
            loss_mim = get_mim_loss(img_patchified, pred, mask_batch, self.norm_pix_loss)
            return loss_mim, pred


def ssl_vit_tiny_patch8_dec256d3b(**kwargs):
    model = SSLViT(
        patch_size=8,
        embed_dim=512,
        depth=6,
        num_heads=4,
        decoder_embed_dim=256,
        decoder_depth=3,
        decoder_num_heads=4,
        mlp_ratio=2,
        in_chans=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def ssl_vit_tiny_patch14_dec256d3b(**kwargs):
    model = SSLViT(
        patch_size=14,
        embed_dim=512,
        depth=6,
        num_heads=4,
        decoder_embed_dim=256,
        decoder_depth=3,
        decoder_num_heads=4,
        mlp_ratio=2,
        in_chans=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def ssl_vit_tiny_patch16_dec256d3b(**kwargs):
    model = SSLViT(
        patch_size=16,
        embed_dim=512,
        depth=6,
        num_heads=4,
        decoder_embed_dim=256,
        decoder_depth=3,
        decoder_num_heads=4,
        mlp_ratio=2,
        in_chans=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def ssl_vit2_tiny_patch8_dec256d3b(**kwargs):
    model = SSLViT2(
        patch_size=8,
        embed_dim=512,
        depth=6,
        num_heads=4,
        decoder_embed_dim=256,
        decoder_depth=3,
        decoder_num_heads=4,
        mlp_ratio=2,
        in_chans=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


if __name__ == "__main__":
    from macronav.pretrain.utils.datasets import build_infer_transform

    if 0:  # test get_fov_mask
        transform = build_infer_transform(224)
        img_path = "dungeon_hard_000542.png"
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        patch_size = 14
        patch_num_w = 224 // patch_size
        fov_mask, expand_mask = get_fov_mask(
            patch_num_w=patch_num_w, center=(5, 5), shape="circle", fov_ratio=0.3, fov_expand_ratio=0.3
        )
        viz_fov_mask_batch(img_tensor, patch_size=patch_size, fov_mask=fov_mask, fov_expand_mask=expand_mask)
    if 1:  # viz mim masking
        viz_mim_mask(224, 8, [0.2, 0.5, 0.7])
