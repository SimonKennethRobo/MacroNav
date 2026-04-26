# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import utils.lr as lr_sched
import utils.misc as misc
from tqdm import tqdm

from macronav.pretrain.models import mae_vit


def train_one_epoch(
    model: mae_vit.MaskedAutoencoderViT | torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    epoch_pbar=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # Create training progress bar for current epoch
    train_pbar = tqdm(data_loader, desc=f"[{args.exp_name}] Epoch {epoch}", unit="batch", position=1, leave=False)

    for data_iter_step, (samples, _) in enumerate(train_pbar):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        # forward
        with torch.amp.autocast(device_type="cuda"):
            loss_dict, pred_dict = model(samples, args)
            loss = sum(loss_dict.values())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        # Update training progress bar
        train_pbar.set_postfix({"loss": f"{loss_value_reduce:.4f}", "lr": f"{lr:.6f}"})

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            if args.log_xaxis == "step":
                x_value = epoch * len(data_loader) + data_iter_step
            else:  # epoch
                # x_value = int((data_iter_step / len(data_loader) + epoch) * 1000)
                x_value = epoch

            log_writer.add_scalar("loss/train", loss_value_reduce, x_value)

            # Log individual losses from loss_dict
            for loss_name, loss_val in loss_dict.items():
                loss_val_reduce = misc.all_reduce_mean(loss_val.item())
                log_writer.add_scalar(f"loss/train/{loss_name}", loss_val_reduce, x_value)

            log_writer.add_scalar("lr", lr, x_value)

    train_pbar.close()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate_one_epoch(model, data_loader, device, epoch, log_writer, args):
    """Validate the model on validation set with acceleration and logging"""
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"Validation Epoch: [{epoch}]"

    total_loss = 0.0
    total_samples = 0

    # Create validation progress bar
    val_pbar = tqdm(data_loader, desc=f"[{args.exp_name}] Validation", unit="batch", position=1, leave=False)

    # Use autocast for acceleration if available
    with torch.no_grad():
        loss_dict_accumulator = {}
        for batch_idx, (images, _) in enumerate(val_pbar):
            images = images.to(device, non_blocking=True)

            # Forward pass
            with torch.amp.autocast(
                device_type="cuda", enabled=args.mixed_precision if hasattr(args, "mixed_precision") else False
            ):
                # loss, _, _ = model(images)
                loss_dict, pred_dict = model(images, args)
                loss = sum(loss_dict.values())

                # Accumulate individual losses
                for loss_name, loss_val in loss_dict.items():
                    if loss_name not in loss_dict_accumulator:
                        loss_dict_accumulator[loss_name] = 0.0
                    loss_dict_accumulator[loss_name] += loss_val.item() * images.shape[0]

            # Aggregate metrics
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            metric_logger.update(loss=loss.item())

            # Update validation progress bar
            val_pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss / total_samples:.4f}"})

    val_pbar.close()

    # Calculate average individual losses
    avg_loss_dict = {}
    if total_samples > 0:
        for loss_name, total_loss_val in loss_dict_accumulator.items():
            avg_loss_dict[loss_name] = total_loss_val / total_samples

    # Store for TensorBoard logging
    validate_one_epoch._last_loss_dict = avg_loss_dict

    # Synchronize across processes if using distributed training
    if args.distributed:
        metric_logger.synchronize_between_processes()
        # Gather total loss and samples across all processes
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        torch.distributed.all_reduce(total_loss_tensor)
        torch.distributed.all_reduce(total_samples_tensor)
        total_loss = total_loss_tensor.item()
        total_samples = total_samples_tensor.item()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    if log_writer is not None and misc.is_main_process():
        # Calculate x-axis value based on log_xaxis argument
        if args.log_xaxis == "step":
            x_value = (epoch + 1) * len(data_loader)
        else:  # epoch
            x_value = epoch

        log_writer.add_scalar("loss/val", avg_loss, x_value)
        # log_writer.add_scalar("loss/val_detailed", metric_logger.loss.global_avg, x_value)

        for loss_name, loss_val in avg_loss_dict.items():
            log_writer.add_scalar(f"loss/val/{loss_name}", loss_val, x_value)  # Return stats for logging

    # Return stats for logging
    val_stats = {"loss": avg_loss, "loss_detailed": metric_logger.loss.global_avg}

    return val_stats
