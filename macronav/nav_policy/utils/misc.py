import argparse
import importlib
import logging
import os
import random
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.distributed as dist
import torchsummary


class MetricLogger:
    """Wrapper class to support both wandb and tensorboard logging."""

    def __init__(self, use_wandb: bool = False, config: dict = None):
        """
        Initialize the metric logger.

        Args:
            use_wandb: If True, use wandb for logging; otherwise use tensorboard
            config: Dictionary containing wandb configuration including:
                - project: wandb project name
                - entity: wandb entity (username or team name)
                - name: experiment name
                - config: experiment configuration dict
                - dir: directory to save logs
        """
        self.use_wandb = use_wandb
        self.writer = None
        self.log_dir = config.get("log_dir", None) if config is not None else None

        if use_wandb:
            try:
                import wandb

                if config is None:
                    config = {}

                wandb.init(
                    project=config.get("project", "rlnav"),
                    entity=config.get("entity", None),
                    name=config.get("name", None),
                    config=config.get("config", {}),
                    dir=config.get("log_dir", None),
                    resume="allow",
                )
                self.writer = wandb
                print(f"Initialized wandb logger. Project: {config.get('project', 'rlnav')}")
            except ImportError:
                print("Warning: wandb not installed. Falling back to tensorboard.")
                self.use_wandb = False

        if not self.use_wandb:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.log_dir)
            print(f"Initialized tensorboard logger. Log dir: {self.log_dir}")

    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        """Log a scalar value."""
        if self.writer is None:
            return
        if self.use_wandb:
            metric_name = tag
            self.writer.log({metric_name: scalar_value}, step=global_step)
        else:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def flush(self):
        """Flush the underlying writer if supported."""
        if self.use_wandb or self.writer is None:
            return
        if hasattr(self.writer, "flush"):
            self.writer.flush()

    def close(self):
        """Close the logger."""
        if self.writer is None:
            return
        if self.use_wandb:
            self.writer.finish()
        else:
            self.writer.close()


def print_params(logger: logging.Logger, train_param):
    col_width = 30
    logger.info("*" * 50)
    logger.info(f"{'Experiment name:':<{col_width}}{train_param.EXP_NAME}")
    logger.info(f"{'Logs will be saved in:':<{col_width}}{train_param.LOG_DIR}")
    logger.info(f"{'image_save_gap:':<{col_width}}{train_param.IMG_SAVE_FREQ}")
    logger.info(f"{'model_save_freq:':<{col_width}}{train_param.MODEL_SAVE_FREQ}")

    logger.info("env params".center(50, "-"))
    logger.info(f"{'env_level:':<{col_width}}{train_param.ENV_LEVEL}")
    logger.info(f"{'num_agent:':<{col_width}}{train_param.NUM_AGENT}")
    logger.info(f"{'env_random_map:':<{col_width}}{train_param.ENV_RANDOM_MAP}")
    logger.info(f"{'env_random_level:':<{col_width}}{train_param.ENV_RANDOM_LEVEL}")
    logger.info(f"{'reward_w_astar:':<{col_width}}{train_param.REWARD_W_ASTAR}")
    logger.info(f"{'reward_w_step:':<{col_width}}{train_param.REWARD_W_STEP}")

    logger.info("model params".center(50, "-"))
    logger.info(f"{'env_encoding_model:':<{col_width}}{train_param.ENV_ENCODING_MODEL}")
    logger.info(f"{'env_encoding_model_ckpt:':<{col_width}}{train_param.ENV_ENCODING_MODEL_CKPT}")

    logger.info("training params".center(50, "-"))
    logger.info(f"{'ray_local_mode:':<{col_width}}{train_param.RAY_LOCAL_MODE}")
    logger.info(f"{'max_episode:':<{col_width}}{train_param.MAX_EPISODE}")
    logger.info(f"{'replay_buffer_size:':<{col_width}}{train_param.REPLAY_BUFFER_SIZE}")
    logger.info(f"{'min_replay_buffer_size:':<{col_width}}{train_param.REPLAY_BUFFER_MIN_SAMPLE}")
    logger.info(f"{'batch_size:':<{col_width}}{train_param.BATCH_SIZE}")
    logger.info(
        f"{'replay_sequence_length:':<{col_width}}{getattr(train_param, 'REPLAY_SEQUENCE_LENGTH', 1)}"
    )
    logger.info(f"{'node_sample_step:':<{col_width}}{train_param.NODE_SAMPLE_STEP}")
    logger.info(f"{'node_padding_size:':<{col_width}}{train_param.NODE_PADDING_SIZE}")
    logger.info("*" * 50)


def write_to_tb(writer, tensorboardData, curr_step):
    """
    Write metrics to logger (tensorboard or wandb).

    Args:
        writer: MetricLogger instance (supports both tensorboard and wandb)
        tensorboardData: List of metric values
        curr_step: Current training step
    """
    tensorboardData = np.array(tensorboardData)
    tensorboardData = list(np.nanmean(tensorboardData, axis=0))  # mean over the the log window
    (
        reward,
        value,
        policyLoss,
        qValueLoss,
        entropy,
        policyGradNorm,
        qValueGradNorm,
        log_alpha,
        alphaLoss,
        travel_dist,
        success_rate,
        explored_rate,
        curvature_ave,
        curvature_std,
        curvature_max,
        time_plan_ave,
        time_plan_std,
        time_plan_min,
        time_plan_max,
        time_episode_total,
        env_level,
    ) = tensorboardData  # the order is the same as <metric_name> defination in train_nav.py:55

    def add_scalar_if_valid(tag, scalar_value):
        scalar_value = float(scalar_value)
        if np.isnan(scalar_value):
            return
        writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=curr_step)

    add_scalar_if_valid(tag="Losses/Value", scalar_value=value)
    add_scalar_if_valid(tag="Losses/Policy Loss", scalar_value=policyLoss)
    add_scalar_if_valid(tag="Losses/Alpha Loss", scalar_value=alphaLoss)
    add_scalar_if_valid(tag="Losses/Q Value Loss", scalar_value=qValueLoss)
    add_scalar_if_valid(tag="Losses/Entropy", scalar_value=entropy)
    add_scalar_if_valid(tag="Losses/Policy Grad Norm", scalar_value=policyGradNorm)
    add_scalar_if_valid(tag="Losses/Q Value Grad Norm", scalar_value=qValueGradNorm)
    add_scalar_if_valid(tag="Losses/Log Alpha", scalar_value=log_alpha)
    add_scalar_if_valid(tag="Perf/Reward", scalar_value=reward)
    add_scalar_if_valid(tag="Perf/Travel Distance", scalar_value=travel_dist)
    # writer.add_scalar(tag="Perf/Explored Rate", scalar_value=explored_rate, global_step=curr_step)
    add_scalar_if_valid(tag="Perf/Success Rate", scalar_value=success_rate)
    add_scalar_if_valid(tag="Perf/Curvature Ave", scalar_value=curvature_ave)
    add_scalar_if_valid(tag="Perf/Curvature Std", scalar_value=curvature_std)
    add_scalar_if_valid(tag="Perf/Curvature Max", scalar_value=curvature_max)
    add_scalar_if_valid(tag="Perf/Planning Time Ave", scalar_value=time_plan_ave)
    add_scalar_if_valid(tag="Perf/Planning Time Std", scalar_value=time_plan_std)
    # writer.add_scalar(tag="Perf/Planning Time Min", scalar_value=time_plan_min, global_step=curr_step)
    add_scalar_if_valid(tag="Perf/Planning Time Max", scalar_value=time_plan_max)
    # writer.add_scalar(tag="Perf/Episode Time", scalar_value=time_episode_total, global_step=curr_step)
    add_scalar_if_valid(tag="Perf/env_level", scalar_value=env_level)


def get_logger(log_dir="./logs"):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    os.makedirs(log_dir, exist_ok=True)

    logger_name = f"macronav.{os.path.abspath(log_dir)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Rebuild handlers so repeated initialization stays idempotent.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(f"{log_dir}/cmd_log.txt", mode="w")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def print_sys_stat(logger):
    import gpustat

    logger.info("System Status".center(50, "-"))
    gpustat.print_gpustat()
    logger.info(f"CPU: {psutil.cpu_percent()}%")
    logger.info(
        f"Memory: {psutil.virtual_memory().used / 1024**3:.2f}/{psutil.virtual_memory().total / 1024**3:.2f} GB ({psutil.virtual_memory().percent}%)"
    )
    logger.info("-" * 50)


def dict2argparse_ns(dict: dict):
    ns = argparse.Namespace()
    for key, value in dict.items():
        setattr(ns, key, value)
    return ns


def convert_argparse_ns_to_uppercase(ns: argparse.Namespace):
    ns_local = argparse.Namespace()
    for key, value in ns.__dict__.items():
        setattr(ns_local, key.upper(), value)
    return ns_local


def plot_nodes_coord(nodes, suffix=""):
    plt.cla()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.scatter(nodes[:, 0], 480 - nodes[:, 1], s=1)
    plt.savefig(f"tmp/node_coords_{suffix}.png")


def plot_frontiers(front, suffix=""):
    plt.cla()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ar = np.array(front)
    ax.scatter(ar[:, 0], 480 - ar[:, 1], s=1)
    plt.savefig(f"tmp/frontier_{suffix}.png")


def plot_frontiers_new_obs(observed_frontiers, new_frontiers):
    plt.cla()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.scatter(observed_frontiers[:, 0], 480 - observed_frontiers[:, 1], s=1, c="r", label="observed")
    ax.scatter(new_frontiers[:, 0], 480 - new_frontiers[:, 1], s=1, c="b", label="new")
    plt.legend()
    plt.savefig("tmp/frontier_new_obs.png")


def calcu_spl(sr: List, traj_len: List) -> float:
    """calculate the SRL: Success Rate weighted by Path Length
    Input:
        sr(List[bool]): success indicator of each trajectory
    """
    assert len(sr) == len(traj_len)
    spl = 0
    for i in range(len(sr)):
        # spl += sr[i] * traj_len[i]
        if sr[i]:
            spl += traj_len[i]
    spl /= len(sr)
    return spl


def calcu_ave_curvature(trajectory):
    trajectory = np.array(trajectory)
    n_points = len(trajectory)
    curvatures = []

    for i in range(1, n_points - 1):
        x0, y0 = trajectory[i - 1]
        x1, y1 = trajectory[i]
        x2, y2 = trajectory[i + 1]

        # calculate the first order difference
        dx1 = x1 - x0
        dy1 = y1 - y0
        dx2 = x2 - x1
        dy2 = y2 - y1

        # calculate the second order difference
        d2x = x2 - 2 * x1 + x0
        d2y = y2 - 2 * y1 + y0

        # calculate the curvature
        numerator = abs(dx1 * d2y - d2x * dy1)
        denominator = (dx1**2 + dy1**2) ** (3 / 2)
        if denominator != 0:
            curvature = numerator / denominator
            curvatures.append(curvature)

    # 计算平均曲率
    if len(curvatures) > 0:
        average_curvature = np.mean(curvatures)
    else:
        average_curvature = 0

    return average_curvature


def calcu_curvature(traj):
    """Calculate the curvature of the current point based on the last three points."""
    traj = np.array(traj)
    if len(traj) < 3:
        return 0

    x0, y0 = traj[-3]
    x1, y1 = traj[-2]
    x2, y2 = traj[-1]

    # Calculate the first order differences
    dx1 = x1 - x0
    dy1 = y1 - y0
    dx2 = x2 - x1
    dy2 = y2 - y1

    # Calculate the second order differences
    d2x = x2 - 2 * x1 + x0
    d2y = y2 - 2 * y1 + y0

    # Calculate the curvature
    numerator = abs(dx1 * d2y - d2x * dy1)
    denominator = (dx1**2 + dy1**2) ** (3 / 2)
    if denominator != 0:
        curvature = numerator / denominator
    else:
        curvature = 0

    return curvature


def calcu_traj_derivative_int(traj):
    """
    Calculate the integral of the first and second derivatives (squared) of a given trajectory.

    Args:
        traj: List of (x, y) coordinates representing the trajectory.

    Returns:
        Tuple:
            - Integral of |f'(t)|^2 dt: First derivative squared integral
            - Integral of |f''(t)|^2 dt: Second derivative squared integral
    """
    traj = np.array(traj)
    num_points = len(traj)

    if num_points < 3:
        raise ValueError("Trajectory must contain at least 3 points to compute both first and second derivatives.")

    # First derivative (approximated using finite differences)
    first_derivatives = np.diff(traj, axis=0)  # First derivative (difference between consecutive points)

    # Second derivative (difference between consecutive first derivatives)
    second_derivatives = np.diff(first_derivatives, axis=0)

    # Calculate the norm squared of the derivatives
    first_derivative_norm_sq = np.sum(np.linalg.norm(first_derivatives, axis=1) ** 2)
    second_derivative_norm_sq = np.sum(np.linalg.norm(second_derivatives, axis=1) ** 2)

    # Return the accumulated norms as an approximation to the integral
    return first_derivative_norm_sq, second_derivative_norm_sq


def calculate_spl(success, travel_dist, astar_len):
    """
    Calculate Success weighted by Path Length (SPL)
    SPL = (1/N) * sum(Si * Li / max(Pi, Li))
    where:
    - Si: success indicator (1 if successful, 0 otherwise)
    - Li: shortest path length (A* length)
    - Pi: actual path length taken by agent
    """
    if astar_len == 0:
        return 0.0
    if success:
        return astar_len / max(travel_dist, astar_len)
    else:
        return 0.0


def import_module_from_path(module_name, path_to_module):
    """
    Example:
    module = import_module_from_path("test", "/path/to/test.py")
    """
    spec = importlib.util.spec_from_file_location(module_name, path_to_module)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def coord2img_idx(coord, map_size):
    """
    coord: (x, y)
    map_size: (width, height)
    """
    # return (int(coord[1]), int(map_size[0] - coord[0]))
    return (int(coord[1]), int(coord[0]))


def get_model_size(model: torch.nn.Module, use_summary=False):
    """get the size of the model in MB"""
    total_size = 0
    if not use_summary:
        total_size = sum(p.numel() for p in model.parameters())
    else:
        summary = torchsummary.summary(model, ((1, 23, 7), (1, 1, 22), (1, 1, 1)), device="cpu")
        for layer in summary:
            total_size += layer["nb_params"]
    return total_size


def set_seed(seed=10086):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # forbid hash randomization
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if dist.is_initialized():
        rank = dist.get_rank()
        torch.manual_seed(seed + rank)  # set the seed for the current process
        torch.cuda.manual_seed(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)


def gen_random_model_input(train_param, device, episode_lens_prefix_sum, replay_buffer, curr_samples, pbar):
    replay_buffer = {key: [] for key in train_param.REPLAY_BUFFER_KEYS}
    for batch in range(201):
        node_padding_mask = torch.rand(1, 23).to(device)
        current_index = torch.randint(0, 10, size=(1, 1), dtype=torch.int64).to(device)
        next_current_index = torch.randint(0, 10, size=(1, 1), dtype=torch.int64).to(device)
        node_inputs = torch.rand(23, 7).to(device)
        action = torch.randint(0, 10, size=(1, 1), dtype=torch.int64).to(device)
        curr_node_edge_padding_mask = torch.rand(1, 22).to(device)
        done = torch.rand(1, 1).to(device)
        next_curr_node_edge_padding_mask = torch.rand(1, 22).to(device)
        next_node_padding_mask = torch.rand(1, 23).to(device)
        edge_inputs = torch.rand(1, 22).to(device)
        gridmap_inputs = np.zeros((224, 224), dtype=np.uint8)

        next_node_inputs = torch.rand(23, 7).to(device)
        reward = torch.rand(1, 1).to(device)
        next_edge_inputs = torch.rand(1, 22).to(device)
        next_gridmap_inputs = gridmap_inputs
        edge_mask = torch.rand(23, 23).to(device)
        next_edge_mask = torch.rand(23, 23).to(device)

        replay_buffer["node_padding_mask"].append(node_padding_mask)
        replay_buffer["next_current_index"].append(next_current_index)
        replay_buffer["node_inputs"].append(node_inputs)
        replay_buffer["action"].append(action)
        replay_buffer["curr_node_edge_padding_mask"].append(curr_node_edge_padding_mask)
        replay_buffer["done"].append(done)
        replay_buffer["next_curr_node_edge_padding_mask"].append(next_curr_node_edge_padding_mask)
        replay_buffer["next_node_padding_mask"].append(next_node_padding_mask)
        replay_buffer["edge_inputs"].append(edge_inputs)
        # replay_buffer["gridmap_inputs"].append(gridmap_inputs)
        replay_buffer["next_node_inputs"].append(next_node_inputs)
        replay_buffer["reward"].append(reward)
        replay_buffer["next_edge_inputs"].append(next_edge_inputs)
        # replay_buffer["next_gridmap_inputs"].append(next_gridmap_inputs)
        replay_buffer["edge_mask"].append(edge_mask)
        replay_buffer["next_edge_mask"].append(next_edge_mask)
        replay_buffer["current_index"].append(current_index)
        episode_lens_prefix_sum.add(1)
        curr_samples += 1
        pbar.update(1)


def save_models(models, filename):
    checkpoint = {
        "policy_model": models["policy_model"].state_dict(),
        "q_net1_model": models["q_net1_model"].state_dict(),
        "q_net2_model": models["q_net2_model"].state_dict(),
        "log_alpha": models["log_alpha"],
        "policy_optimizer": models["policy_optimizer"].state_dict(),
        "q_net1_optimizer": models["q_net1_optimizer"].state_dict(),
        "q_net2_optimizer": models["q_net2_optimizer"].state_dict(),
        "log_alpha_optimizer": models["log_alpha_optimizer"].state_dict(),
        "curr_episode": models["curr_episode"],
        "policy_lr_decay": models["policy_lr_decay"].state_dict(),
        "q_net1_lr_decay": models["q_net1_lr_decay"].state_dict(),
        "q_net2_lr_decay": models["q_net2_lr_decay"].state_dict(),
        "log_alpha_lr_decay": models["log_alpha_lr_decay"].state_dict(),
        "curr_samples": models["curr_samples"],
    }
    torch.save(checkpoint, filename)


def load_models(ckpt_path, models):
    checkpoint = torch.load(ckpt_path)
    models["policy_model"].load_state_dict(checkpoint["policy_model"])
    if "q_net1_model" in checkpoint and "q_net2_model" in checkpoint:
        models["q_net1_model"].load_state_dict(checkpoint["q_net1_model"])
        models["q_net2_model"].load_state_dict(checkpoint["q_net2_model"])
    models["policy_optimizer"].load_state_dict(checkpoint["policy_optimizer"])
    if "q_net1_optimizer" in checkpoint and "q_net2_optimizer" in checkpoint:
        models["q_net1_optimizer"].load_state_dict(checkpoint["q_net1_optimizer"])
        models["q_net2_optimizer"].load_state_dict(checkpoint["q_net2_optimizer"])
    models["log_alpha_optimizer"].load_state_dict(checkpoint["log_alpha_optimizer"])
    models["policy_lr_decay"].load_state_dict(checkpoint["policy_lr_decay"])
    if "q_net1_lr_decay" and "q_net2_lr_decay" in checkpoint:
        models["q_net1_lr_decay"].load_state_dict(checkpoint["q_net1_lr_decay"])
        models["q_net2_lr_decay"].load_state_dict(checkpoint["q_net2_lr_decay"])
    models["log_alpha_lr_decay"].load_state_dict(checkpoint["log_alpha_lr_decay"])
    if "episode" in checkpoint:
        models["curr_episode"] = checkpoint["episode"]
    elif "curr_episode" in checkpoint:
        models["curr_episode"] = checkpoint["curr_episode"]
    if "curr_samples" in checkpoint:
        models["curr_samples"] = checkpoint["curr_samples"]
    else:
        samples = input("No 'curr_samples' in checkpoint, please input the current samples: ")
        models["curr_samples"] = int(samples)


def coord_pxl2phi(pixel_coords, map_origin, map_resolution, tf, map_size_slam):
    """
    将 map_model 像素坐标转换为物理坐标

    Args:
        pixel_coords (np.array): 形状为 (N, 2) 的数组，包含 N 个像素坐标点 [x, y]
        map_origin (np.array): 地图原点物理坐标 [x, y]
        map_resolution (float): 地图分辨率
        TF_MAP1_TO_MAP2_PXL (np.array): 从 map1 到 map2 的像素坐标转换矩阵
        map_slam_size (tuple): 真实地图尺寸 (height, width)

    Returns:
        np.array: 形状为 (N, 2) 的数组，包含 N 个物理坐标点 [x, y]
    """
    pixel_coords = np.array(pixel_coords)

    if pixel_coords.ndim == 1:
        pixel_coords = pixel_coords.reshape(1, -1)

    ones = np.ones((pixel_coords.shape[0], 1))
    pixel_coords_homo = np.hstack((pixel_coords, ones))
    coord_map1_pxl = np.dot(pixel_coords_homo, np.linalg.inv(tf).T)[:, :2]

    coord_map1_phi = np.zeros_like(coord_map1_pxl)
    coord_map1_phi[:, 0] = coord_map1_pxl[:, 0] * map_resolution
    coord_map1_phi[:, 1] = (map_size_slam[0] - coord_map1_pxl[:, 1]) * map_resolution

    coord_phi = coord_map1_phi + map_origin

    return coord_phi


def coord_phi2pxl(physical_coords, map_origin, map_resolution, tf, map_size_model):
    """
    Convert physical coordinates to model map pixel coordinates

    Args:
        physical_coords (np.array): Array of shape (N, 2) containing physical coordinates [x, y]
        map_origin (np.array): Map origin in physical coordinates [x, y]
        map_resolution (float): Map resolution (meters/pixel)
        tf (np.array): Transform matrix from map1 to map2 pixel coordinates
        map_size_model (tuple): size of the map that fed into the model

    Returns:
        np.array: Array of shape (N, 2) containing pixel coordinates in model map [x, y]
    """
    physical_coords = np.array(physical_coords)
    map_origin = np.array(map_origin).reshape(1, 2)

    # Handle single point input
    if physical_coords.ndim == 1:
        physical_coords = physical_coords.reshape(1, -1)

    # Convert from physical coordinates to map1 coordinates
    coord_map1_phi = physical_coords - map_origin

    # Convert to map1 pixel coordinates
    coord_map1_pxl = np.zeros_like(coord_map1_phi)
    coord_map1_pxl[:, 0] = coord_map1_phi[:, 0] / map_resolution
    coord_map1_pxl[:, 1] = map_size_model[0] - coord_map1_phi[:, 1] / map_resolution

    # Convert to model map (map2) pixel coordinates using transform
    ones = np.ones((coord_map1_pxl.shape[0], 1))
    coord_map1_pxl_homo = np.hstack((coord_map1_pxl, ones))
    coord_map2_pxl = np.dot(tf, coord_map1_pxl_homo.T).T[:, :2]

    # Convert to integer pixel coordinates
    coord_map2_pxl = coord_map2_pxl.astype(np.int32)

    # Return single point if input was single point
    if coord_map2_pxl.shape[0] == 1:
        return coord_map2_pxl[0]
    return coord_map2_pxl


def env_level_schedule(curr_sample, max_samples, alpha=3, beta=1):
    """sample the env level using beta distribution given the current episode"""
    t = min(curr_sample / max_samples, 1.0)

    a = (alpha - beta) * t + beta
    b = (beta - alpha) * t + alpha
    beta_sample = np.random.beta(a, b)

    # Map [0, 1] to discrete levels {0, 1, 2}
    if beta_sample < 0.33:
        level = 0
    elif beta_sample < 0.67:
        level = 1
    else:
        level = 2

    return level


def tensor_to_np(tensor: torch.Tensor, dataset_mean_std=None) -> np.ndarray:
    """Convert tensor to displayable numpy array."""
    img = tensor.squeeze(0).detach().cpu()
    img = img.permute(1, 2, 0)

    if dataset_mean_std is not None:
        img = torch.clip((img / dataset_mean_std[0] + dataset_mean_std[1]) * 255, 0, 255)
    else:
        img = torch.clip(img * 255, 0, 255)

    if img.shape[2] == 1:
        img = img.squeeze(2)

    return img.numpy()
