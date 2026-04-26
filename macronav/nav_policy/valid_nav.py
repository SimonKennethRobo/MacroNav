import json
import os
import shutil
import time

import config.valid_param as valid_param
import numpy as np
import ray
import torch
import tqdm
from models.nav import PolicyNet
from utils.misc import (
    convert_argparse_ns_to_uppercase,
    dict2argparse_ns,
    get_logger,
    import_module_from_path,
    calculate_spl,
)
from macronav.nav_policy.utils.worker import ValidWorker

os.environ["RAY_DEBUG"] = "legacy"
ray.init(local_mode=valid_param.RAY_LOCAL_MODE, _temp_dir="/tmp/ray/macronav", num_gpus=valid_param.NUM_GPU)


@ray.remote(num_cpus=1, num_gpus=valid_param.NUM_GPU / valid_param.NUM_AGENT)
class RayRunner(object):
    def __init__(self, meta_agent_id, policy_net: PolicyNet, runner_config):
        self.runner_config = runner_config
        self.meta_agent_id = meta_agent_id
        self.local_policy_net = policy_net

    def do_job(self, episode_number):
        worker_config = self.runner_config.copy()
        worker_config["greedy"] = True
        if self.runner_config["env_level"] == "mix":
            worker_config["env_level"] = np.random.choice(["medium", "easy", "hard"])
        worker_config["node_sample_step"] = worker_config["node_sample_step"][worker_config["env_level"]]
        worker_config["node_padding_size"] = worker_config["node_padding_size"]["local"]

        worker = ValidWorker(self.meta_agent_id, self.local_policy_net, episode_number, worker_config)
        worker.run_episode()

        perf_metrics = worker.perf_metrics
        traj = worker.robot_trajs
        return perf_metrics, traj

    def job(self, episode_number):
        metrics, traj = self.do_job(episode_number)
        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }
        return metrics, traj, info


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(valid_param.LOG_DIR, "test")
    os.makedirs(valid_param.TEST_RESULT_PATH, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    shutil.copyfile(os.path.join(curr_dir, "config", "valid_param.py"), os.path.join(test_dir, "valid_param.py"))
    logger = get_logger(valid_param.TEST_RESULT_PATH)
    pbar = tqdm.tqdm(total=valid_param.NUM_EPISODE, desc="Test Episodes", unit="epi")

    if valid_param.LOAD_PARAM_FROM_JSON:
        train_param_ = json.load(open(f"{valid_param.LOG_DIR}/train_param.json", "r"))
        train_param = dict2argparse_ns(train_param_)
        train_param = convert_argparse_ns_to_uppercase(train_param)
    else:
        train_param = import_module_from_path("train_parameter", f"{valid_param.LOG_DIR}/train_param.py")
        logger.info(f"Loaded train_parameter from {valid_param.LOG_DIR}/train_param.py")
        logger.info(f"policy_net_args: {train_param.POLICY_NET_ARGS}")
    train_param_dict = {}
    train_param_dict_tmp = vars(train_param)
    for key in train_param_dict_tmp.keys():
        train_param_dict[key.lower()] = train_param_dict_tmp[key]
    del train_param_dict_tmp

    train_param_dict["eval_mode"] = True

    col_width = 30
    logger.info("*" * 50)
    for key in valid_param.CONFIG_DICT.keys():
        logger.info(f"{key:{col_width}}: {valid_param.CONFIG_DICT[key]}")
    logger.info("*" * 50)

    device = torch.device("cuda") if valid_param.USE_GPU else torch.device("cpu")
    logger.info(f"device: {device}")
    logger.info(f"policy net model args: {train_param.POLICY_NET_ARGS}")

    model_type = "best"
    if device == torch.device("cuda"):
        checkpoint = torch.load(f"{valid_param.MODEL_PATH}/checkpoint_{model_type}.pth")
    else:
        checkpoint = torch.load(
            f"{valid_param.MODEL_PATH}/checkpoint_{model_type}.pth", map_location=torch.device("cpu")
        )
    logger.info(f"Loaded model from: {valid_param.MODEL_PATH}/checkpoint_{model_type}.pth")
    global_policy_net = PolicyNet(train_param_dict).to(device)
    global_policy_net.load_state_dict(checkpoint["policy_model"])
    if hasattr(global_policy_net, "curr_node_lstm"):
        global_policy_net.curr_node_lstm.flatten_parameters()
    global_policy_net.eval()

    runner_config = train_param_dict.copy()
    runner_config.update(valid_param.CONFIG_DICT)
    runner_config["device"] = device

    rollout_results = {
        "success": [],
        "travel_dist": [],
        "time_episode_total": [],
        "time_plan_ave": [],
        "time_plan_max": [],
        "time_plan_std": [],
        "time_plan_min": [],
        "curvature_ave": [],
        "curvature_max": [],
        "curvature_std": [],
        "curvature_min": [],
        "spl": [],
        "path_efficiency": [],
    }

    curr_episode = 0
    completed_episodes = 0
    job_list = []
    meta_agents = [
        RayRunner.remote(agent_id, global_policy_net, runner_config) for agent_id in range(valid_param.NUM_AGENT)
    ]
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(curr_episode))
        curr_episode += 1
        pbar.update(1)

    try:
        while completed_episodes < valid_param.NUM_EPISODE - 1:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)
            completed_episodes += len(done_jobs)

            for job in done_jobs:
                metrics, traj, info = job
                for key in rollout_results.keys():
                    if key == "spl":
                        spl_value = calculate_spl(metrics["success"], metrics["travel_dist"], metrics["astar_len"])
                        rollout_results[key].append(spl_value)
                    elif key == "path_efficiency":
                        # Path efficiency: ratio of optimal path to actual path
                        if metrics["travel_dist"] > 0:
                            efficiency = metrics["astar_len"] / metrics["travel_dist"]
                        else:
                            efficiency = 0.0
                        rollout_results[key].append(efficiency)
                    else:
                        rollout_results[key].append(metrics[key])

                # launch new job
                if completed_episodes < valid_param.NUM_EPISODE - 1 and curr_episode < valid_param.NUM_EPISODE - 1:
                    job_list.append(meta_agents[info["id"]].job.remote(curr_episode))
                    pbar.update(1)
                    curr_episode += 1

            time.sleep(0.1)

        logger.info("-" * 50)
        result_str = f"Exp: {valid_param.EXP_NAME} in Env: {valid_param.ENV_LEVEL}\n"
        result_str += f"Total test episodes: {valid_param.NUM_EPISODE}\n"
        result_str += "Results averaged over all rollouts:\n"

        # Calculate and display key metrics
        success_rate = np.array(rollout_results["success"]).mean()
        spl_score = np.array(rollout_results["spl"]).mean()
        avg_path_efficiency = np.array(rollout_results["path_efficiency"]).mean()

        result_str += f"Success Rate: {success_rate:.4f}\n"
        result_str += f"SPL (Success weighted by Path Length): {spl_score:.4f}\n"
        result_str += f"Average Path Efficiency: {avg_path_efficiency:.4f}\n"
        result_str += "-" * 30 + "\n"

        for key in rollout_results.keys():
            result_str += f"{key}: {np.array(rollout_results[key]).mean():.4f}\n"

        logger.info(result_str)

        # Save detailed results
        with open(f"{valid_param.TEST_RESULT_PATH}/summary.txt", "w") as f:
            f.write(result_str)

        # Save detailed metrics as JSON
        detailed_results = {
            "summary": {
                "success_rate": float(success_rate),
                "spl_score": float(spl_score),
                "avg_path_efficiency": float(avg_path_efficiency),
                "total_episodes": valid_param.NUM_EPISODE,
            },
            "all_metrics": {key: [float(x) for x in rollout_results[key]] for key in rollout_results.keys()},
        }

        with open(f"{valid_param.TEST_RESULT_PATH}/detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2)

        logger.info(f"Test for experiment {valid_param.EXP_NAME} finished")

    except KeyboardInterrupt:
        logger.info("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    for i in range(valid_param.NUM_RUN):
        main()
