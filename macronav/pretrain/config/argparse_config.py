import argparse
import sys


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument("--run_on_slurm", action="store_true", help="Run on slurm")
    parser.add_argument("--exp_name", default="train_test", type=str, help="experiment name")
    parser.add_argument("--log_path", default="logs", type=str)
    parser.add_argument("--ckpt_save_freq", default=100, type=int, help="Checkpoint save frequency (epochs)")

    # effective batch size is batch_size * accum_iter * # gpus
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--accum_iter", default=2, type=int, help="Accumulate gradient iterations")
    parser.add_argument("--eval_freq", default=10, type=int, help="Evaluation frequency (epochs)")

    # Model parameters
    parser.add_argument("--model", default="ssl_vit_patch8", type=str, metavar="MODEL", help="model name")
    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    # base learning rate: absolute_lr = base_lr * total_batch_size / 256
    parser.add_argument("--blr", type=float, default=1e-4, metavar="LR", help="base learning rate")  # 1e-3
    # lower lr bound for cyclic schedulers that hit 0
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound")
    parser.add_argument("--warmup_epochs", type=int, default=2, metavar="N", help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--dataset_path", default="./dataset", type=str, help="dataset path")
    parser.add_argument("--output_dir", default=f"{sys.path[0]}/data/pretrain/output")
    parser.add_argument("--log_dir", default="./data/pretrain/logs", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="ssl-rl-nav-pretrain")
    parser.add_argument("--wandb_entity", type=str, default=None)

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=6, type=int)

    # distributed training parameters
    parser.add_argument("--distributed", action="store_true", help="Use DDP or not")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--mask_ratio", default=0.75, type=float, help="Masking ratio (percentage of removed patches).")
    parser.add_argument(
        "--norm_pix_loss", action="store_true", help="Use (per-patch) normalized pixels as targets for computing loss"
    )
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument("--fov_ratio", default=0.4, type=float)
    parser.add_argument("--fov_expand_ratio", default=0.2, type=float)
    parser.add_argument("--fov_dist_weight_en", action="store_true", help="Enable distance weight for FOV loss")

    parser.add_argument("--mim_ratio", default=0.3, type=float)
    parser.add_argument("--mim_smoothness", default=0.6, type=float)
    parser.add_argument("--mim_use_brownian", action="store_true")

    parser.add_argument("--add_noise", action="store_true", help="Add noise to the input images")

    parser.add_argument("--loss_weight_mae", default=1.0, type=float)
    parser.add_argument("--loss_weight_fov", default=1.0, type=float)
    parser.add_argument("--loss_weight_mim", default=1.0, type=float)

    return parser
