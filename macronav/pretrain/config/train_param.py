import json
import os
import shutil
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from macronav.pretrain.config.argparse_config import get_args_parser


RUN_ON_SLURM = False

# Logging parameters ---------------------
EXP_NAME = "pretrain/run1"
LOG_PATH = "exps"
EXP_DIR = f"{LOG_PATH}/{EXP_NAME}"
OUTPUT_DIR = f"{EXP_DIR}/output"
LOG_DIR = f"{EXP_DIR}/logs"
USE_WANDB = True
WANDB_PROJECT = "macronav"
WANDB_ENTITY = None
CKPT_SAVE_FREQ = 100

# Training parameters ---------------------
EPOCHS = 501
BATCH_SIZE = 64
ACCUM_ITER = 2
EVAL_FREQ = 10
WEIGHT_DECAY = 0.05
LR = None
BLR = 1e-4
MIN_LR = 0.0
WARMUP_EPOCHS = 2
SEED = 42
RESUME = ""
START_EPOCH = 0
NUM_WORKERS = 6

# Runtime parameters ---------------------
DEVICE = "cuda"
DISTRIBUTED = False
WORLD_SIZE = 1
LOCAL_RANK = -1
DIST_ON_ITP = False
DIST_URL = "env://"

# Dataset parameters ---------------------
DATASET_PATH = "dataset/GridMapV1"
INPUT_SIZE = 224

# Model / task parameters ---------------------
MODEL = "ssl_vit_patch8"
MASK_RATIO = 0.75
NORM_PIX_LOSS = False
FOV_RATIO = 0.5
FOV_EXPAND_RATIO = 0.2
FOV_DIST_WEIGHT_EN = False
MIM_RATIO = 0.3
MIM_SMOOTHNESS = 0.6
MIM_USE_BROWNIAN = True
ADD_NOISE = True
LOSS_WEIGHT_MAE = 1.0
LOSS_WEIGHT_FOV = 1.0
LOSS_WEIGHT_MIM = 1.0

# DATASET_MEAN = [0.35318832]
# DATASET_STD = [0.4779606]
DATASET_MEAN = [0.5]
DATASET_STD = [0.5]
DATASET_TRANS = 2

MASK_TOKEN = 0
NAV_PATCH_PXL_TH = 0.9
NAV_PATCH_PXL_NUM_TH = 10



def get_param_dict_from_curr_file():
    import types

    config = {}
    for k, v in globals().items():
        if k.isupper() and not k.startswith("__") and not isinstance(v, types.ModuleType) and not callable(v):
            config[k.lower()] = v
    return config


def save_config_artifacts(config_dict, src_file=None):
    exp_dir = Path(config_dict["exp_dir"])
    output_dir = Path(config_dict["output_dir"])
    log_dir = Path(config_dict["log_dir"])

    exp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with (exp_dir / "train_param.json").open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)

    if src_file is not None:
        shutil.copyfile(os.path.abspath(src_file), exp_dir / "train_param.py")


CONFIG_DICT = {}
if not RUN_ON_SLURM:
    CONFIG_DICT = get_param_dict_from_curr_file()
else:
    args = get_args_parser().parse_args()
    CONFIG_DICT = vars(args)
    config_exp_name = CONFIG_DICT["exp_name"]
    config_log_path = CONFIG_DICT["log_path"]
    config_exp_dir = f"{config_log_path}/{config_exp_name}"
    CONFIG_DICT["exp_dir"] = config_exp_dir
    CONFIG_DICT["output_dir"] = f"{config_exp_dir}/output"
    CONFIG_DICT["log_dir"] = f"{config_exp_dir}/logs"
