# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import math
import os
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn.utils.rnn import pad_sequence


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}, gpu {}".format(args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type
        )
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ("checkpoint-%s.pth" % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "args": args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        if "optimizer" in checkpoint and "epoch" in checkpoint and not (hasattr(args, "eval") and args.eval):
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def pretrain_add_noise(imgs: torch.Tensor, random_add=True):
    """
    add noise for imgs in MAE pretraining.

    imgs: [B, C, H, W]
        Noise: gaussian noise, gaussian blur, motion blur, solarization, random erasing
    """
    B, C, H, W = imgs.shape
    device = imgs.device

    # Apply different types of noise randomly
    for i in range(B):
        img = imgs[i]  # [C, H, W]

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

        imgs[i] = img

    return imgs


def save_tensor_png(tensor: torch.tensor, path, idx=None):
    """
    Save batch tensor to PNG images.

    Input:
        tensor: [B, C, H, W] - batch of images
        path: str - directory path to save images
        idx: int or None - if provided, save only the idx-th image
    """
    tensor = tensor.clone().detach().cpu()

    tensor = torch.clamp(tensor, 0, 1)

    os.makedirs(path, exist_ok=True)

    if idx is not None:
        # Save only the specified index
        if idx < tensor.shape[0]:
            img_tensor = tensor[idx]  # [C, H, W]
            # Convert to [H, W, C] and to numpy
            if img_tensor.shape[0] == 1:  # Grayscale
                img_array = img_tensor.squeeze(0).numpy()
                img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode="L")
            else:  # RGB
                img_array = img_tensor.permute(1, 2, 0).numpy()
                img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode="RGB")

            img.save(os.path.join(path, f"image_{idx}.png"))
    else:
        # Save all images in the batch
        for i in range(tensor.shape[0]):
            img_tensor = tensor[i]  # [C, H, W]
            # Convert to [H, W, C] and to numpy
            if img_tensor.shape[0] == 1:  # Grayscale
                img_array = img_tensor.squeeze(0).numpy()
                img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode="L")
            else:  # RGB
                img_array = img_tensor.permute(1, 2, 0).numpy()
                img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode="RGB")

            img.save(os.path.join(path, f"image_{i}.png"))


def pad_tensor_list(tensor_list: list):
    """
    Input:
        tensor_list: batch list of (T,D)
    Returns
        padded_batch: [B, T, D]
        padding_mask: [B, T]
    """
    if not tensor_list:
        return torch.empty(0), torch.empty(0, dtype=torch.bool)
    device = tensor_list[0].device

    lengths = torch.tensor([tensor.shape[0] for tensor in tensor_list])
    max_length = lengths.max().item()

    padded_batch = pad_sequence(tensor_list, batch_first=True, padding_value=0)

    # Create padding mask more efficiently using broadcasting
    padding_mask = torch.arange(max_length).unsqueeze(0) >= lengths.unsqueeze(1)
    padding_mask = padding_mask.to(device)

    return padded_batch, padding_mask
