import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from macronav.nav_policy.config import valid_param
from macronav.nav_policy.models.nav import PolicyNet
from macronav.nav_policy.utils.infer_runtime import OnnxPolicyRunner, find_available_onnx
from macronav.nav_policy.utils.misc import import_module_from_path
from macronav.nav_policy.utils.worker import ValidWorker

MODEL_INPUT_KEYS = (
    "node_inputs",
    "edge_inputs",
    "current_index",
    "node_padding_mask",
    "curr_node_edge_padding_mask",
    "edge_mask",
    "gridmap_inputs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Torch vs ONNX policy inference speed using real ValidWorker observations."
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations for each backend.")
    parser.add_argument("--iters", type=int, default=200, help="Measured iterations for each backend.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of real observations sampled from ValidWorker and replayed during benchmark.",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Episode index used as the first validation map when collecting benchmark observations.",
    )
    parser.add_argument(
        "--env-level",
        type=str,
        default=None,
        help="Validation env level used for sample collection. Defaults to the training config env_level when set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if valid_param.USE_GPU and torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device used by the Torch policy.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to valid_param.MODEL_PATH/checkpoint_best.pth.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Optional ONNX path. Defaults to checkpoint sibling '*_policy.onnx' or first .onnx in the folder.",
    )
    parser.add_argument(
        "--check-output",
        action="store_true",
        help="Compare Torch and ONNX outputs on the final replayed sample.",
    )
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for output checking.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for output checking.")
    return parser.parse_args()


def load_train_param_dict() -> dict:
    train_param_dict = {}
    if valid_param.LOAD_PARAM_FROM_JSON:
        with open(f"{valid_param.LOG_DIR}/train_param.json", "r") as f:
            train_param_dict = json.load(f)
    else:
        train_param = import_module_from_path("train_parameter", f"{valid_param.LOG_DIR}/train_param.py")
        train_param_dict_tmp = vars(train_param)
        for key, value in train_param_dict_tmp.items():
            train_param_dict[key.lower()] = value
    train_param_dict["eval_mode"] = True
    return train_param_dict


def load_torch_policy(train_param_dict: dict, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_net = PolicyNet(train_param_dict).to(device)
    policy_net.load_state_dict(checkpoint["policy_model"])
    policy_net.eval()
    return policy_net


def resolve_env_value(value, env_level: str):
    if isinstance(value, dict):
        if env_level in value:
            return value[env_level]
        raise KeyError(f"Missing env-specific config for '{env_level}' in {value}")
    return value


def build_worker_config(train_param_dict: dict, device: torch.device, env_level: str) -> dict:
    worker_config = train_param_dict.copy()
    worker_config.update(valid_param.CONFIG_DICT)
    for key in ("node_padding_size", "node_sample_step", "k_size", "sensor_range"):
        if key in worker_config:
            worker_config[key] = resolve_env_value(worker_config[key], env_level)
    worker_config["device"] = device
    worker_config["env_level"] = env_level
    worker_config["save_img"] = False
    worker_config["save_traj"] = False
    worker_config["greedy"] = True
    worker_config["custom_start_target"] = None
    worker_config["test_result_path"] = tempfile.mkdtemp(prefix="onnx_benchmark_")
    return worker_config


def resolve_benchmark_env_level(train_param_dict: dict, requested_env_level: str | None) -> str:
    if requested_env_level is not None:
        return requested_env_level
    return train_param_dict.get("env_level", valid_param.ENV_LEVEL)


def maybe_get_static_input_dim(policy, input_name: str, axis: int):
    if not hasattr(policy, "session"):
        return None

    for meta in policy.session.get_inputs():
        if meta.name != input_name:
            continue
        if axis >= len(meta.shape):
            return None
        dim = meta.shape[axis]
        return dim if isinstance(dim, int) else None
    return None


def align_worker_config_to_onnx(worker_config: dict, onnx_policy) -> dict:
    aligned = worker_config.copy()
    node_padding_size = maybe_get_static_input_dim(onnx_policy, "node_inputs", 1)
    k_size = maybe_get_static_input_dim(onnx_policy, "curr_node_edge_padding_mask", 2)

    if node_padding_size is not None:
        aligned["node_padding_size"] = node_padding_size
    if k_size is not None:
        aligned["k_size"] = k_size
    return aligned


def validate_samples_against_onnx(samples, onnx_policy, env_level: str):
    if not hasattr(onnx_policy, "session"):
        return

    static_dims = (
        ("node_inputs", maybe_get_static_input_dim(onnx_policy, "node_inputs", 1), 1),
        ("node_padding_mask", maybe_get_static_input_dim(onnx_policy, "node_padding_mask", 2), 2),
        ("curr_node_edge_padding_mask", maybe_get_static_input_dim(onnx_policy, "curr_node_edge_padding_mask", 2), 2),
        ("edge_mask", maybe_get_static_input_dim(onnx_policy, "edge_mask", 1), 1),
    )

    for sample_idx, sample in enumerate(samples):
        sample_by_name = dict(zip(MODEL_INPUT_KEYS, sample))
        for name, expected_dim, axis in static_dims:
            if expected_dim is None:
                continue
            tensor = sample_by_name[name]
            actual_dim = tensor.shape[axis]
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Collected sample {sample_idx} is incompatible with ONNX input '{name}': "
                    f"got {tuple(tensor.shape)}, expected static size {expected_dim}. "
                    f"This usually means the benchmark env/sample config does not match the exported ONNX graph "
                    f"(current env_level='{env_level}'). Re-export the ONNX model for this observation shape or "
                    f"rerun the benchmark with a matching env level."
                )


def clone_model_input(model_input):
    cloned = []
    for item in model_input:
        if isinstance(item, torch.Tensor):
            cloned.append(item.detach().clone())
        else:
            cloned.append(item)
    return tuple(cloned)


def obs_to_model_input(observations: dict):
    return clone_model_input(tuple(observations[key] for key in MODEL_INPUT_KEYS))


def reset_policy_state(policy):
    if hasattr(policy, "reset_state"):
        policy.reset_state()
    elif hasattr(policy, "reset_recurrent_state"):
        policy.reset_recurrent_state()


def sync_if_needed(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def collect_model_inputs(policy_net, worker_config: dict, start_episode: int, num_samples: int):
    samples = []
    episode_idx = start_episode

    while len(samples) < num_samples:
        worker = ValidWorker(0, policy_net, episode_idx, worker_config)
        reset_policy_state(worker.local_policy_net)

        for _ in range(worker.episode_max_step):
            observations = worker.get_observations()
            samples.append(obs_to_model_input(observations))
            if len(samples) >= num_samples:
                break

            next_position, _ = worker.get_action(observations)
            _, done, worker.robot_position, worker.travel_dist = worker.env.step(
                worker.robot_position,
                next_position,
                worker.travel_dist,
            )
            worker.robot_trajs.append(worker.robot_position)
            if done:
                break

        episode_idx += 1

    return samples


def percentile(values: Iterable[float], q: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("Cannot compute percentile of an empty sequence.")
    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    alpha = pos - lo
    return values[lo] * (1.0 - alpha) + values[hi] * alpha


def benchmark_policy(policy, samples, warmup: int, iters: int, device: torch.device):
    reset_policy_state(policy)
    with torch.no_grad():
        for idx in range(warmup):
            sample = samples[idx % len(samples)]
            sync_if_needed(device)
            policy(sample)
            sync_if_needed(device)

        reset_policy_state(policy)
        latencies_ms = []
        last_output = None
        for idx in range(iters):
            sample = samples[idx % len(samples)]
            sync_if_needed(device)
            start = time.perf_counter()
            last_output = policy(sample)
            sync_if_needed(device)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    mean_ms = statistics.fmean(latencies_ms)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": percentile(latencies_ms, 95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else math.inf,
        "last_output": last_output,
    }


def print_report(title: str, stats: dict):
    print(f"[{title}]")
    print(f"  mean   : {stats['mean_ms']:.3f} ms")
    print(f"  median : {stats['median_ms']:.3f} ms")
    print(f"  p95    : {stats['p95_ms']:.3f} ms")
    print(f"  min/max: {stats['min_ms']:.3f} / {stats['max_ms']:.3f} ms")
    print(f"  fps    : {stats['fps']:.2f}")


def compare_outputs(torch_output, onnx_output, rtol: float, atol: float):
    torch_np = torch_output.detach().cpu().numpy()
    onnx_np = onnx_output.detach().cpu().numpy()
    if torch_np.shape != onnx_np.shape:
        return False, f"shape mismatch: torch={torch_np.shape}, onnx={onnx_np.shape}"
    if not np.allclose(torch_np, onnx_np, rtol=rtol, atol=atol):
        diff = np.max(np.abs(torch_np - onnx_np))
        return False, f"value mismatch: max_abs_diff={diff:.6e}"
    return True, "allclose passed"


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else Path(valid_param.MODEL_PATH) / "checkpoint_best.pth"
    )
    onnx_path = Path(args.onnx_path) if args.onnx_path is not None else find_available_onnx(checkpoint_path)
    if onnx_path is None:
        raise FileNotFoundError(
            f"Cannot find ONNX policy beside checkpoint: {checkpoint_path}. "
            "Please export one first or pass --onnx-path explicitly."
        )

    train_param_dict = load_train_param_dict()
    env_level = resolve_benchmark_env_level(train_param_dict, args.env_level)
    torch_policy = load_torch_policy(train_param_dict, checkpoint_path, device)
    onnx_policy = OnnxPolicyRunner(onnx_path, device).eval()
    worker_config = build_worker_config(train_param_dict, device, env_level)
    worker_config = align_worker_config_to_onnx(worker_config, onnx_policy)

    print(f"Checkpoint : {checkpoint_path}")
    print(f"ONNX model : {onnx_path}")
    print(f"Device     : {device}")
    print(f"Env level  : {env_level}")
    print(f"Samples    : {args.num_samples}")
    print()

    samples = collect_model_inputs(
        policy_net=torch_policy,
        worker_config=worker_config,
        start_episode=args.start_episode,
        num_samples=args.num_samples,
    )
    validate_samples_against_onnx(samples, onnx_policy, env_level)
    first_sample = samples[0]
    print("Replay sample tensor shapes:")
    for key, value in zip(MODEL_INPUT_KEYS, first_sample):
        if value is None:
            print(f"  {key}: None")
        else:
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
    print()

    torch_stats = benchmark_policy(torch_policy, samples, args.warmup, args.iters, device)
    onnx_stats = benchmark_policy(onnx_policy, samples, args.warmup, args.iters, device)

    print_report("Torch", torch_stats)
    print()
    print_report("ONNX Runtime", onnx_stats)
    print()
    print(f"Speedup (Torch -> ONNX): {torch_stats['mean_ms'] / onnx_stats['mean_ms']:.3f}x")

    if args.check_output:
        ok, message = compare_outputs(torch_stats["last_output"], onnx_stats["last_output"], args.rtol, args.atol)
        print(f"Output check: {'PASSED' if ok else 'FAILED'}")
        print(f"  {message}")


if __name__ == "__main__":
    main()
