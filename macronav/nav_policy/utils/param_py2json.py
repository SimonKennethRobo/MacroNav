import argparse
import json
import sys

from util import import_module_from_path

exclude_keys = [
    "os",
    "sys",
    "argparse",
    "json",
    "import_module_from_path",
    "__annotations__",
    "__builtins__",
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
]


def convert_py_to_json(py_file_path):
    param = import_module_from_path("train_parameter", py_file_path)
    param_dict = vars(param)
    for key in exclude_keys:
        if key in param_dict:
            del param_dict[key]

    DEFAULTS = {
        "EXP_NAME": "local_window_vit_v4.1",
        "RAY_LOCAL_MODE": False,
        "LOAD_MODEL": False,
        "MAX_EPISODE": 20000,
        "REPLAY_BUFFER_SIZE": 30000,
        "MIN_REPLAY_BUFFER_SIZE": 6000,
        "BATCH_SIZE": 32,
        "REPLAY_SEQUENCE_LENGTH": 1,
        "NODE_SAMPLE_DENSITY": 30,
        "DATA_USE_GPU": True,
        "TRAIN_USE_GPU": True,
        "NUM_GPU": 1,
        "NUM_CPU": 8,
        "LR_POLICY_NET": 1e-5,
        "LR_Q_NET": 2e-5,
        "LR_ALPHA": 1e-4,
        "GAMMA": 0.99,
        "DECAY_STEP": 256,
        "EPISODE_MAX_STEP": 128,
        "ENTROPY_WEIGHT": 0.01,
        "TARGET_Q_NET_UPDATE_FREQ": 64,
        "TRAIN_TIMES_PER_EPISODE": 8,
        "INPUT_DIM": 7,
        "EMBEDDING_DIM": 128,
        "K_SIZE": 20,
        "NODE_PADDING_SIZE": 21,
        "SENSOR_RANGE": 80,
        "ENV_ENCODING_MODEL": "vit_tiny_patch8",
        "NUM_AGENT": 25,
    }

    for key, value in DEFAULTS.items():
        if key not in param_dict:
            param_dict[key] = value
            print(f"Warning: {key} not found in {py_file_path}, using default value: {value}")

    param_dict["NODE_PADDING_SIZE"] = param_dict["K_SIZE"] + 1

    param_dict["Q_NET_ARGS"] = {
        "encoder_layer": 6,
        "decoder_layer": 1,
        "encoder_head": 8,
        "decoder_head": 8,
        "lstm_layer": 1,
        "lstm_hidden_size": 128,
        "env_encoding_model": param_dict["ENV_ENCODING_MODEL"],
        "env_encoding_model_use_pretrained": True,
        "use_res_conn": True,
        "env_encoding_freeze": False,
    }

    param_dict["POLICY_NET_ARGS"] = param_dict["Q_NET_ARGS"].copy()

    REPLAY_BUFFER_KEYS = [
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
    ]

    REPLAY_BUFFER_KEYS.extend(["q_edge_mask", "next_q_edge_mask"])

    param_dict["REPLAY_BUFFER_KEYS"] = REPLAY_BUFFER_KEYS

    print(f"param_dict: {param_dict}")
    json_output = json.dumps(param_dict, indent=4)

    return json_output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python param_py2json.py <py_file_path>")
        sys.exit(-1)
    py_file = sys.argv[1]
    json_result = convert_py_to_json(py_file)
    if len(sys.argv) == 3:
        json_path = sys.argv[2]
    else:
        json_path = py_file.replace(".py", ".json")
    with open(json_path, "w") as f:
        f.write(json_result)
    print(json_result)
