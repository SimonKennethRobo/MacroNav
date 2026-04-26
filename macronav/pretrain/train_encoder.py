import argparse
import datetime
import json
import os
import time
from pathlib import Path

import models
import numpy as np
from timm.optim import optim_factory
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from engine_pretrain import train_one_epoch, validate_one_epoch
from tqdm import tqdm

from macronav.pretrain.config import train_param
from macronav.nav_policy.utils.misc import MetricLogger
from macronav.pretrain.utils import misc
from macronav.pretrain.utils.datasets import build_infer_transform, build_transform1, build_transform2
from macronav.pretrain.utils.misc import NativeScalerWithGradNormCount as NativeScaler


def main(args: argparse.Namespace):
    misc.init_distributed_mode(args)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("Args: " + "-" * 60)
    print("{}".format(args).replace(", ", ",\n"))
    print("-" * 60)

    with open(os.path.join(args.output_dir, "args.txt"), mode="w", encoding="utf-8") as f:
        f.write(str(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform_train = build_transform1(args) if train_param.DATASET_TRANS == 1 else build_transform2(args)
    dataset_train = datasets.ImageFolder(os.path.join(args.dataset_path, "train"), transform=transform_train)

    transform_val = build_infer_transform(args.input_size)
    dataset_val = datasets.ImageFolder(os.path.join(args.dataset_path, "val"), transform=transform_val)

    # dataset_train = CachedImageDataset(
    #     image_dir=os.path.join(args.dataset_path, "train","map"),
    #     transform=transform_train,
    #     max_cache_size=2000,
    #     max_workers=args.num_workers,
    # )
    print(dataset_train)
    print("-" * 60)
    print(dataset_val)

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    log_writer = None
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        wandb_cfg = {
            "project": getattr(args, "wandb_project", "rlnav"),
            "entity": getattr(args, "wandb_entity", None),
            "name": getattr(args, "wandb_run_name", args.exp_name),
            "config": getattr(args, "wandb_config", vars(args)),
            "log_dir": args.log_dir,
        }
        log_writer = MetricLogger(use_wandb=getattr(args, "use_wandb", False), config=wandb_cfg)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=6 if args.num_workers > 0 else 0,
    )

    # Add validation data loader
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,  # Use same batch size for consistency
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=6 if args.num_workers > 0 else 0,
    )

    # define the model
    model = models.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    # print(torchsummary.summary(model, (1, args.input_size, args.input_size)))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 64

    print("base lr: %.2e" % (args.lr * 64 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    no_weight_decay_list = model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, "no_weight_decay") else ()
    param_groups = optim_factory.param_groups_weight_decay(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=no_weight_decay_list,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    # print(optimizer)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # Create epoch progress bar with experiment name
    epoch_pbar = tqdm(
        range(args.start_epoch, args.epochs), desc=f"[{args.exp_name}] Training", unit="epoch", position=0
    )
    if log_writer is not None:
        init_epoch = args.start_epoch
        log_writer.add_scalar("misc/fov_ratio", args.fov_ratio, init_epoch)
        log_writer.add_scalar("misc/fov_expand_ratio", args.fov_expand_ratio, init_epoch)
        log_writer.add_scalar("misc/mim_ratio", args.mim_ratio, init_epoch)
        log_writer.add_scalar("misc/mim_smoothness", args.mim_smoothness, init_epoch)

    for epoch in epoch_pbar:
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)

        args.log_xaxis = "epoch"
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            epoch_pbar=epoch_pbar,
        )

        # Run validation every epoch or as specified
        if epoch % args.eval_freq == 0 or epoch + 1 == args.epochs:
            val_stats = validate_one_epoch(model, data_loader_val, device, epoch, log_writer, args)
        else:
            val_stats = {}

        # Update epoch progress bar with current loss
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_stats.get('loss', 0):.4f}",
                "val_loss": f"{val_stats.get('loss', 0):.4f}" if val_stats else "N/A",
            }
        )

        if args.output_dir and (epoch % args.ckpt_save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    epoch_pbar.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    if log_writer is not None:
        log_writer.close()


if __name__ == "__main__":
    args = argparse.Namespace(**train_param.CONFIG_DICT)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_param.save_config_artifacts(train_param.CONFIG_DICT, train_param.__file__)
    main(args)
