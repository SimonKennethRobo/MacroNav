# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import tqdm
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

from macronav.pretrain.config import train_param


class RandomResizedCrop(transforms.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F._get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.dataset_path, "train" if is_train else "val")
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    # mean = IMAGENET_DEFAULT_MEAN
    # std = IMAGENET_DEFAULT_STD
    """
    Mean: [0.59226177 0.59216558 0.59208681]
    Standard Deviation: [0.12754066 0.12755591 0.12765019]

    Mean: tensor([2.3654e-06, 2.3642e-06, 2.3633e-06])
    Standard Deviation: tensor([0.0012, 0.0012, 0.0012])

    """
    mean = [0.59226177, 0.59216558, 0.59208681]
    std = [0.12754066, 0.12755591, 0.12765019]
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_transform1(args):
    transform_train = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size), interpolation=3),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=train_param.DATASET_MEAN,
                std=train_param.DATASET_STD,
            ),
        ]
    )
    return transform_train


def build_transform2(args):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomResizedCrop(args.input_size, scale=(0.7, 1.3), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.RandomAffine(
                degrees=(-15, 15),
                translate=(0.1, 0.1),
                shear=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=train_param.DATASET_MEAN, std=train_param.DATASET_STD),
        ]
    )


def build_infer_transform(input_size=224, norm=True, backbone="vit_tiny_patch8"):
    if norm:
        if any(x in backbone for x in ["resnet", "tiny_vit_11m_224", "dino", "deit", "in21k"]):
            transform_infer = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size), interpolation=3),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=IMAGENET_DEFAULT_MEAN,
                        std=IMAGENET_DEFAULT_STD,
                    ),
                ]
            )
        elif "vit_tiny_patch8" in backbone:
            transform_infer = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size), interpolation=3),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=train_param.DATASET_MEAN,
                        std=train_param.DATASET_STD,
                    ),
                ]
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    else:
        transform_infer = transforms.Compose(
            [
                transforms.Resize((input_size, input_size), interpolation=3),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
    return transform_infer


def get_dungeon_transform(env_model="vit_tiny_patch8"):
    if "vit_tiny_patch8" in env_model or "maevit" in env_model:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.59226177],
                    std=[0.12754066],
                ),
            ]
        )
    elif "tinyvit" in env_model or "resnet" in env_model:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6, 0.6, 0.6], std=[0.128, 0.128, 0.128]),
            ]
        )
    return transform


def compute_mean_and_std(image_dir, max_workers=4):
    def process_image(img_path):
        """处理单个图像，返回图像的像素总和和像素平方总和"""
        print(f"Processing image: {img_path}")
        img = Image.open(img_path).convert("RGB")  # 将图像转换为 RGB 模式
        img_np = np.array(img) / 255.0  # 将图像转换为 NumPy 数组并归一化到 [0, 1] 范围

        pixel_sum = np.sum(img_np, axis=(0, 1))
        pixel_squared_sum = np.sum(img_np**2, axis=(0, 1))
        num_pixels = img_np.shape[0] * img_np.shape[1]

        return pixel_sum, pixel_squared_sum, num_pixels

    # 初始化变量
    total_pixel_sum = np.zeros(3)  # 用于累计所有图像的像素总和，RGB三通道
    total_pixel_squared_sum = np.zeros(3)  # 累计所有图像的像素平方总和
    total_num_pixels = 0

    # 获取所有图像文件路径
    img_files = [
        os.path.join(image_dir, img_filename)
        for img_filename in os.listdir(image_dir)
        if img_filename.endswith(".png") or img_filename.endswith(".jpg")
    ]

    # 并行处理图像
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, img_path) for img_path in img_files]

        for future in as_completed(futures):
            pixel_sum, pixel_squared_sum, num_pixels = future.result()
            total_pixel_sum += pixel_sum
            total_pixel_squared_sum += pixel_squared_sum
            total_num_pixels += num_pixels

    # 计算均值和方差
    mean = total_pixel_sum / total_num_pixels
    std = np.sqrt(total_pixel_squared_sum / total_num_pixels - mean**2)

    return mean, std


def compute_mean_and_std_gpu(image_dir, batch_size=512, num_workers=4, device="cuda"):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((500, 500)),
        ]
    )
    dataset = ImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # print(f"Dataset prepared with {len(dataset)} images")

    mean = torch.zeros(3).to(device)
    std = torch.zeros(3).to(device)
    num_pixels = 0
    pbar = tqdm.tqdm(dataloader, total=len(dataloader), unit="batch")

    for images in dataloader:
        images = images.to(device)
        num_pixels += images.size(0) * images.size(2) * images.size(3)

        mean += images.mean([0, 2, 3]) * images.size(0)
        std += images.pow(2).mean([0, 2, 3]) * images.size(0)
        pbar.update(1)

    mean /= num_pixels
    std = torch.sqrt(std / num_pixels - mean**2)

    return mean.cpu(), std.cpu()


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class CachedImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, max_workers=4, max_cache_size=None):
        """
        Cached Image Dataset that preloads images into memory for faster access.

        Args:
            image_dir: Directory containing images
            transform: Transform to apply to images
            max_workers: Number of workers for parallel image loading
            max_cache_size: Maximum number of images to cache (None for no limit)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.max_cache_size = max_cache_size

        # Validate image directory
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")

        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in directory: {image_dir}")

        # Determine how many images to cache
        num_to_cache = len(self.image_files)
        if max_cache_size is not None and max_cache_size < len(self.image_files):
            num_to_cache = max_cache_size
            print(f"Caching only {num_to_cache} out of {len(self.image_files)} images due to cache size limit")

        self.cached_images = []
        self.use_cache = True if num_to_cache == len(self.image_files) else False

        print(f"Loading {num_to_cache} images into memory...")
        self._load_images_to_cache(max_workers, num_to_cache)

        if len(self.cached_images) == 0:
            raise ValueError("Failed to load any images into cache")

        print(f"Successfully cached {len(self.cached_images)} images")

    def _load_single_image(self, img_filename):
        """Load a single image and return it."""
        try:
            img_path = os.path.join(self.image_dir, img_filename)
            image = Image.open(img_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading {img_filename}: {e}")
            return None

    def _load_images_to_cache(self, max_workers, num_to_cache):
        """Load images into memory using parallel processing."""
        # Select which images to cache (first num_to_cache images)
        files_to_cache = self.image_files[:num_to_cache]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit image loading tasks
            futures = [executor.submit(self._load_single_image, img_file) for img_file in files_to_cache]

            # Collect results with progress bar
            with tqdm.tqdm(total=len(futures), desc="Loading images") as pbar:
                for future in as_completed(futures):
                    image = future.result()
                    if image is not None:
                        self.cached_images.append(image)
                    pbar.update(1)

        print(f"Loaded {len(self.cached_images)} out of {num_to_cache} images successfully")

    def __len__(self):
        return len(self.image_files)  # Return total number of files, not just cached

    def __getitem__(self, idx):
        """Get image from cache or load from disk."""
        if self.use_cache:
            # All images are cached
            image = self.cached_images[idx]
        elif idx < len(self.cached_images):
            # Image is in cache
            image = self.cached_images[idx]
        else:
            # Load image from disk
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image
