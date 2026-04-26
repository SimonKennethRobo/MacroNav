from macronav.nav_policy.config.argparse_config import parse_test_args
import os

RUN_ON_SLURM = False

EXP_NAME = "test"
LOG_DIR = f"exps/nav_policy/{EXP_NAME}"
MODEL_PATH = f"{LOG_DIR}/models"
LOAD_PARAM_FROM_JSON = True
RAY_LOCAL_MODE = False

USE_GPU = True
NUM_GPU = 1
NUM_CPU = os.cpu_count() - 2

NUM_AGENT = 8 if not RAY_LOCAL_MODE else 1
NUM_EPISODE = 200  # 200
NUM_RUN = 1
EPISODE_MAX_STEP = 128

SAVE_GIFS = True
SAVE_TRAJ = True

ENV_LEVEL = "easy"  # easy, medium, hard, real
SENSOR_RANGE = 150  # None means using the training/default value
CUSTOM_MAP_IDX = 0  # for valid_iter.py
NAV_MAP_PATH = "dataset/NavSimMapV1"

TEST_RESULT_PATH = f"{LOG_DIR}/test/{ENV_LEVEL}"


# ! DO NOT MODIFY BELOW
def get_param_dict_from_curr_file():
    import types

    config = {}
    for k, v in globals().items():
        if k.isupper() and not k.startswith("__") and not isinstance(v, types.ModuleType) and not callable(v):
            config[k.lower()] = v
    return config


CONFIG_DICT = {}
if RUN_ON_SLURM:
    CONFIG_DICT = vars(parse_test_args())
    for param in CONFIG_DICT:
        if CONFIG_DICT[param] is not None:
            globals()[param] = CONFIG_DICT[param]
else:
    CONFIG_DICT = get_param_dict_from_curr_file()
