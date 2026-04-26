import gc
import os
import threading
import time

import numpy as np
import torch
import tqdm
from models.nav import PolicyNet
from utils.misc import env_level_schedule, set_seed
from utils.replay_buffer import PrefixSum, sample_batch
from macronav.nav_policy.utils.worker import TrainWorker

os.environ["RAY_DEBUG"] = "legacy"
import ray


TRAIN_METRIC_NAMES = [
    "travel_dist",
    "success",
    "explored_rate",
    "curvature_ave",
    "curvature_std",
    "curvature_max",
    "time_plan_ave",
    "time_plan_std",
    "time_plan_min",
    "time_plan_max",
    "time_episode_total",
    "env_level",
]


def init_ray(train_param):
    return ray.init(
        local_mode=train_param.RAY_LOCAL_MODE,
        _temp_dir="/tmp/ray/macronav",
        num_cpus=train_param.NUM_CPU,
        num_gpus=train_param.NUM_GPU,
    )


def create_ray_runner_class(num_gpus_per_agent):
    return ray.remote(num_gpus=num_gpus_per_agent, num_cpus=1)(BaseRayRunner)


class TrainingRuntimeState:
    def __init__(self, replay_buffer_keys, replay_buffer_size, metric_names):
        self.replay_buffer_keys = replay_buffer_keys
        self.replay_buffer = {key: [] for key in replay_buffer_keys}
        self.state_lock = threading.RLock()
        self.global_weights_set = []
        self.curr_episode = 0
        self.curr_samples = 0
        self.episode_lens_prefix_sum = PrefixSum(replay_buffer_size)
        self.metric_names = metric_names
        self.perf_metrics = {key: [] for key in metric_names}

    def update_weights(self, weights_set):
        with self.state_lock:
            self.global_weights_set = weights_set

    def append_job_result(self, episode_buffer, metrics):
        with self.state_lock:
            self.curr_samples += metrics["step_num"]
            self.episode_lens_prefix_sum.add(metrics["step_num"])

            for key in self.replay_buffer_keys:
                self.replay_buffer[key] += episode_buffer[key]
            for key in self.metric_names:
                self.perf_metrics[key].append(metrics[key])

    def trim_replay_buffer(self, replay_buffer_size):
        with self.state_lock:
            while len(self.replay_buffer["done"]) > replay_buffer_size and len(self.episode_lens_prefix_sum) > 0:
                trim_steps = self.episode_lens_prefix_sum.pop_left()
                for key in self.replay_buffer_keys:
                    self.replay_buffer[key] = self.replay_buffer[key][trim_steps:]

    def reset_perf_metrics(self):
        with self.state_lock:
            self.perf_metrics = {key: [] for key in self.metric_names}

    def sample_batch(self, train_config, device):
        with self.state_lock:
            return sample_batch(
                self.replay_buffer,
                train_config,
                device,
                self.episode_lens_prefix_sum,
            )

    def get_perf_metrics_snapshot(self):
        with self.state_lock:
            return {key: list(values) for key, values in self.perf_metrics.items()}


class BaseRayRunner:
    """Policy rollout worker kept separate from the training entrypoint."""

    def __init__(self, meta_agent_id, runner_config: dict):
        self.meta_agent_id = meta_agent_id
        worker_seed = runner_config.get("seed", 10086) + meta_agent_id
        set_seed(worker_seed)

        data_use_gpu = runner_config.get("data_use_gpu", False)
        self.local_device = torch.device("cuda") if data_use_gpu else torch.device("cpu")
        self.runner_config = runner_config.copy()
        self.runner_config["device"] = self.local_device
        self.img_save_freq = self.runner_config.get("img_save_freq", 100)

        start_time = time.time()
        self.local_policy_net = PolicyNet(self.runner_config).to(self.local_device)
        self.local_policy_net.eval()
        duration = time.time() - start_time
        print(f"Runner {meta_agent_id} initialized in {duration:.2f} seconds")

    def get_weights(self):
        return self.local_policy_net.state_dict()

    def set_policy_net_weights(self, weights):
        self.local_policy_net.load_state_dict(weights)

    def _build_worker_config(self, episode_idx, curr_samples):
        worker_config = self.runner_config.copy()
        worker_config["greedy"] = False
        worker_config["save_img"] = episode_idx % self.img_save_freq == 0

        if self.runner_config["env_level"] == "mix":
            if self.runner_config["env_random_level"]:
                worker_config["env_level"] = np.random.choice(["medium", "easy", "hard"])
            else:
                level_idx = env_level_schedule(curr_samples, self.runner_config["max_sample"], alpha=3, beta=1)
                worker_config["env_level"] = ["easy", "medium", "hard"][level_idx]

        worker_config["node_padding_size"] = worker_config["node_padding_size"]["local"]

        worker_config["node_sample_step"] = worker_config["node_sample_step"][worker_config["env_level"]]
        return worker_config

    def do_job(self, episode_idx, curr_samples=None):
        worker_config = self._build_worker_config(episode_idx, curr_samples)
        worker = TrainWorker(self.meta_agent_id, self.local_policy_net, episode_idx, worker_config)
        worker.run_episode(episode_idx)

        episode_buffer = worker.episode_buffer
        episode_perf_metrics = worker.perf_metrics

        del worker
        gc.collect()
        return episode_buffer, episode_perf_metrics

    def job(self, weights_set, episode_idx, curr_samples=None):
        self.set_policy_net_weights(weights_set[0])
        episode_buffer, metrics = self.do_job(episode_idx, curr_samples)
        info = {
            "id": self.meta_agent_id,
            "episode_idx": episode_idx,
        }
        return episode_buffer, metrics, info


class JobLauncher(threading.Thread):
    def __init__(self, meta_agents, state, replay_buffer_size, logger=None):
        super().__init__()
        self.meta_agents = meta_agents
        self.state = state
        self.replay_buffer_size = replay_buffer_size
        self._stop_event = threading.Event()
        self.logprint = print if logger is None else logger.info

    def stop(self, timeout=None):
        self._stop_event.set()
        self.join(timeout)

    def stopped(self):
        return self._stop_event.is_set()

    def _launch_next_job(self, agent_id):
        self.state.curr_episode += 1
        return self.meta_agents[agent_id].job.remote(
            self.state.global_weights_set,
            self.state.curr_episode,
            self.state.curr_samples,
        )

    def run(self):
        job_list = [self._launch_next_job(agent_id) for agent_id, _ in enumerate(self.meta_agents)]

        try:
            while not self.stopped():
                done_ids, job_list = ray.wait(job_list, timeout=0.1)
                done_jobs = ray.get(done_ids)
                for episode_buffer, metrics, info in done_jobs:
                    self.state.append_job_result(episode_buffer, metrics)
                    self.state.trim_replay_buffer(self.replay_buffer_size)
                    job_list.append(self._launch_next_job(info["id"]))
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


def monitor_replay_buffer(state, max_sample, exp_name):
    pbar = tqdm.tqdm(total=max_sample, unit="step", desc=f"[{exp_name}]")
    pbar.n = state.curr_samples
    while True:
        pbar.n = state.curr_samples
        pbar.set_postfix({"curr_epi": state.curr_episode})
        pbar.refresh()
        time.sleep(1)
