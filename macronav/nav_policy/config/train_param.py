import json
import os
import shutil
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from macronav.nav_policy.config.argparse_config import parse_train_args


RUN_ON_SLURM = False

# Logging parameters ---------------------
EXP_NAME = "nav_policy/test"
LOG_DIR = f"exps/{EXP_NAME}"
MODEL_PATH = LOG_DIR + "/models"
TB_PATH = LOG_DIR + "/logs"
GIF_PATH = LOG_DIR + "/gifs"
USE_WANDB = True  # Use wandb instead of tensorboard. Set to True to enable wandb logging
WANDB_PROJECT = "macronav"  # wandb project name
WANDB_ENTITY = None  # wandb entity name (username or team name), None to use default
NUM_GPU = 1
NUM_CPU = os.cpu_count() - 2
RAY_LOCAL_MODE = False  # for ray debugging. Set to True to run everything in the main process, which is useful for debugging but will be slower
SUMMARY_WINDOW = 200  # steps
IMG_SAVE_FREQ = 500  # save image every SAVE_IMG_GAP episodes
MODEL_SAVE_FREQ = 100  # save model every MODEL_SAVE_FREQ episodes

# Environment parameters ---------------------
ENV_LEVEL = "medium"
NUM_AGENT = 2 if not RAY_LOCAL_MODE else 1  # decide the number of parallel environments to collect samples
ENV_RANDOM_ST_ED = True  # random start and end
ENV_RANDOM_LEVEL = False
ENV_RANDOM_MAP = True
K_SIZE = 22  # the number of neighboring nodes
NORMALIZE_UTILITY = True  # normalize the utility of nodes
SENSOR_RANGE = 120  # pixels
EPISODE_MAX_STEP = 128
NAV_MAP_PATH = "dataset/NavSimMapV1"

# RL parameters ---------------------
MAX_EPISODE = 15000
MAX_SAMPLE = 500000
REPLAY_BUFFER_SIZE = 10000
REPLAY_BUFFER_MIN_SAMPLE = 2000 if not RAY_LOCAL_MODE else 128  # minimum size of replay buffer before training
REPLAY_SEQUENCE_LENGTH = 1
REWARD_W_ASTAR = 2
REWARD_W_STEP = 2
GAMMA = 0.99  # discount factor
DECAY_STEP = 256  # not use
ENTROPY_WEIGHT = 0.01
TARGET_Q_NET_UPDATE_FREQ = 64  # update target Q network every * times training
TARGET_Q_NET_UPDATE_SOFT = False

# DL parameters ---------------------
LOAD_MODEL = False  # load the model checkpoint
CKPT_PATH = None  # set None or "" when LOAD_MODEL is True will automatically load the latest model
BATCH_SIZE = 48 if not RAY_LOCAL_MODE else 16  # size of samples from replay buffer to train
GRADIENT_STEPS = 8 if not RAY_LOCAL_MODE else 2
LR_POLICY_NET = 5e-5
LR_Q_NET = 1e-4
LR_ALPHA = 1e-4
DATA_USE_GPU = True  # collect interaction samples using GPU
TRAIN_USE_GPU = True  # train the network using GPU
SEED = 42

# Model parameters ---------------------
ENV_ENCODING_MODEL = "vit_tiny_patch8"
ENV_ENCODING_MODEL_CKPT = "exps/pretrain/encoder3.pth"
NODE_SAMPLE_STEP = {"easy": 20, "medium": 25, "hard": 30, "real": 30}
NODE_PADDING_SIZE = {"easy": 300, "medium": 900, "hard": 1024, "local": K_SIZE + 1}
INPUT_DIM = 7
EMBEDDING_DIM = 128
Q_NET_ARGS = dict(
    encoder_layer=6,
    encoder_head=8,
    decoder_layer=1,
    decoder_head=8,
    lstm_layer=2,
    lstm_hidden_size=128,
    env_encoding_model=ENV_ENCODING_MODEL,
    env_encoding_model_use_pretrained=True,
    use_res_conn=True,
    env_encoding_model_ckpt=ENV_ENCODING_MODEL_CKPT,
    env_encoding_freeze=False,
)
POLICY_NET_ARGS = dict(
    encoder_layer=6,
    encoder_head=8,
    decoder_layer=1,
    decoder_head=8,
    lstm_layer=2,
    lstm_hidden_size=128,
    env_encoding_model=ENV_ENCODING_MODEL,
    env_encoding_model_use_pretrained=True,
    use_res_conn=True,
    env_encoding_model_ckpt=ENV_ENCODING_MODEL_CKPT,
    env_encoding_freeze=False,
)

REPLAY_BUFFER_KEYS = {
    "node_inputs",
    "edge_inputs",
    "current_index",
    "node_padding_mask",
    "curr_node_edge_padding_mask",
    "edge_mask",
    "action",
    "reward",
    "done",
    "next_node_inputs",
    "next_edge_inputs",
    "next_current_index",
    "next_node_padding_mask",
    "next_curr_node_edge_padding_mask",
    "next_edge_mask",
    "gridmap_inputs",
    "next_gridmap_inputs",
}

# ! DO NOT MODIFY BELOW


def get_param_dict_from_curr_file():
    import types

    config = {}
    for k, v in globals().items():
        if k.isupper() and not k.startswith("__") and not isinstance(v, types.ModuleType) and not callable(v):
            config[k.lower()] = v
    return config


def serialize_config_dict(config_dict):
    config_json = config_dict.copy()
    for key, value in config_json.items():
        if isinstance(value, set):
            config_json[key] = list(value)
    return config_json


def save_config_artifacts(config_dict, src_file=None):
    log_dir = Path(config_dict["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    with (log_dir / "train_param.json").open("w", encoding="utf-8") as f:
        json.dump(serialize_config_dict(config_dict), f, indent=4)

    if src_file is not None:
        shutil.copyfile(os.path.abspath(src_file), log_dir / "train_param.py")


REPLAY_BUFFER_KEYS.add("q_edge_mask")
REPLAY_BUFFER_KEYS.add("next_q_edge_mask")

CONFIG_DICT = {}
if not RUN_ON_SLURM:
    CONFIG_DICT = get_param_dict_from_curr_file()
else:
    args = parse_train_args()
    CONFIG_DICT = vars(args)
