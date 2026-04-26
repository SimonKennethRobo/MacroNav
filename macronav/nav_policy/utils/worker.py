import csv
import json
import os
import time
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image

from macronav.nav_policy.envs.map_nav_env import MapNavEnv
from macronav.nav_policy.models.nav import PolicyNet
from macronav.nav_policy.utils.misc import calcu_curvature
from macronav.pretrain.utils.datasets import build_infer_transform

curr_dir = os.path.dirname(os.path.abspath(__file__))


def resolve_nav_map_file(worker_config: dict, episode_idx: int, test_mode: bool) -> str:
    def sort_key(name: str):
        stem = Path(name).stem
        try:
            return (0, int(stem))
        except ValueError:
            digits = "".join(ch for ch in stem if ch.isdigit())
            if digits:
                return (0, int(digits))
            return (1, stem)
    env_level = worker_config.get("env_level", "easy")
    map_root = worker_config.get("nav_map_path", "/tmp/DungeonMaps/")
    split = "test" if test_mode else "train"
    map_dir = os.path.join(map_root, env_level, split)
    map_list = sorted(os.listdir(map_dir), key=sort_key)
    if not map_list:
        raise FileNotFoundError(f"No map files found in {map_dir}")

    if worker_config.get("env_random_map", False) and not test_mode:
        map_name = np.random.choice(map_list)
    else:
        map_name = map_list[episode_idx % len(map_list)]
    return os.path.join(map_dir, map_name)


def resolve_sensor_range(worker_config: dict) -> int:
    return int(worker_config.get("sensor_range", 80))


class TrainWorker:
    """an individual agent with specially allocated resources"""

    def __init__(self, meta_agent_id, policy_net: PolicyNet, episode_idx, worker_config: dict):
        self.agent_id = meta_agent_id
        self.episode_idx = episode_idx
        self.device = worker_config.get("device", torch.device("cpu"))
        self.greedy = worker_config.get("greedy", False)
        self.node_padding_size = worker_config.get("node_padding_size", 360)
        self.k_size = worker_config.get("k_size", 20)
        self.save_img = worker_config.get("save_img", True)
        self.replay_buffer_keys = worker_config.get("replay_buffer_keys", [])
        assert len(self.replay_buffer_keys) > 0
        self.episode_max_step = worker_config.get("episode_max_step", 128)
        self.env_level = worker_config.get("env_level", "easy")
        self.gif_path = worker_config.get("gif_path", f"{curr_dir}/gif")
        self.use_env_encoding = True
        self.use_local_nodes = True
        self.use_qnet_gt_graph = True
        self.norm_utility = worker_config.get("norm_utility", True)
        self.data_use_gpu = worker_config.get("data_use_gpu", True)
        self.env_random_st_ed = worker_config.get("env_random_st_ed", False)
        if self.save_img:
            os.makedirs(self.gif_path, exist_ok=True)

        env_config = worker_config.copy()
        env_config["sensor_range"] = resolve_sensor_range(worker_config)
        env_config["plot"] = True
        env_config["curr_episode"] = self.episode_idx
        map_file = resolve_nav_map_file(worker_config, self.episode_idx, test_mode=False)
        self.env = MapNavEnv(map_file, False, env_config)
        self.env.reset()

        self.local_policy_net = policy_net
        self._reset_policy_state()
        self.current_node_index = 0
        self.travel_dist = 0
        self.robot_position = self.env.start_position
        self.robot_trajs = [self.robot_position]
        self.perf_metrics = dict()
        self.episode_buffer = {key: [] for key in self.replay_buffer_keys}

        env_encoding_model = worker_config.get("env_encoding_model", "vit_tiny_patch8")
        self.img_transform = build_infer_transform(backbone=env_encoding_model)

    def _reset_policy_state(self):
        if hasattr(self.local_policy_net, "reset_state"):
            self.local_policy_net.reset_state()
        elif hasattr(self.local_policy_net, "reset_recurrent_state"):
            self.local_policy_net.reset_recurrent_state()

    def _pad_node_inputs(self, node_coords, node_utility, indicator, direction_vector):
        node_coords = node_coords / self.env.map_gt_size[0]
        if self.norm_utility:
            node_utility = node_utility / 50

        n_nodes = node_coords.shape[0]
        node_utility_inputs = node_utility.reshape(n_nodes, 1)
        direction_vector_inputs = direction_vector.reshape(direction_vector.shape[0], 3).copy()
        direction_vector_inputs[:, 2] /= max(float(self.env.sensor_range), 1.0)

        node_inputs = np.concatenate((node_coords, node_utility_inputs, indicator, direction_vector_inputs), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        try:
            assert n_nodes < self.node_padding_size
        except AssertionError:
            print(f"node_coords.shape[0]:{n_nodes}")
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - n_nodes))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_nodes), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - n_nodes), dtype=torch.int64).to(self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)
        return node_inputs, node_padding_mask

    def _get_local_edge_inputs(self, graph, global_node_indices):
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(global_node_indices)}
        edge_inputs = []
        for global_idx in global_node_indices:
            node_edges = graph.get(str(global_idx), {})
            local_edges = [global_to_local[int(node)] for node in node_edges if int(node) in global_to_local]
            edge_inputs.append(local_edges)
        return edge_inputs

    def _pad_edge_mask(self, edge_inputs):
        adjacent_matrix = self.get_edge_mask(edge_inputs)  # 0 is connected, 1 is not connected
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
        valid_node_num = len(edge_inputs)
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - valid_node_num, 0, self.node_padding_size - valid_node_num), 1
        )
        return padding(edge_mask)

    def _build_current_edge_tensor(self, edge_inputs, current_index_local):
        curr_node_edges = list(edge_inputs[current_index_local])
        valid_edge_num = min(len(curr_node_edges), self.k_size)
        curr_node_edges = curr_node_edges[: self.k_size]
        while len(curr_node_edges) < self.k_size:
            curr_node_edges.append(current_index_local)
        curr_node_edges = torch.tensor(curr_node_edges, dtype=torch.long).unsqueeze(0).unsqueeze(0).to(self.device)
        curr_node_edge_padding_mask = torch.ones((1, 1, self.k_size), dtype=torch.int64).to(self.device)
        curr_node_edge_padding_mask[:, :, :valid_edge_num] = 0
        return curr_node_edges, curr_node_edge_padding_mask

    def _build_gt_edge_mask(self, observed_global_indices):
        graph_gt = self.env.graph_generator.graph_gt.edges
        node_coords_gt = self.env.graph_generator.node_coords_set_gt
        gt_indices = [node_coords_gt.get_index(self.env.node_coords_set.coords[idx]) for idx in observed_global_indices]
        gt_to_local = {gt_idx: local_idx for local_idx, gt_idx in enumerate(gt_indices)}

        gt_edge_inputs = []
        for gt_idx in gt_indices:
            if gt_idx is None:
                raise ValueError("Failed to map observed node to a GT topo node for critic inputs")
            node_edges = graph_gt.get(str(gt_idx), {})
            local_edges = [gt_to_local[int(node)] for node in node_edges if int(node) in gt_to_local]
            if not local_edges:
                local_edges = [gt_to_local[gt_idx]]
            gt_edge_inputs.append(local_edges)
        return self._pad_edge_mask(gt_edge_inputs)

    def get_observations(self):
        """
        Returns:
            node_inputs: (1,self.node_padding_size,7) 7=coords(2)+utility(1)+indicator(1)+direction_vector(3)
            edge_inputs: (1,1,k_size) k_size: number of adjacent nodes for the current node
            edge_mask: adjacency matrix of the graph
        """
        full_node_coords = self.env.node_coords_set.coords
        graph = self.env.graph_edges
        node_utility = self.env.node_utility
        indicator = self.env.indicator
        direction_vector = self.env.direction_vector
        # get the node index of the current robot position
        current_node_index = self.env.find_index_from_coords(self.robot_position)
        current_index_local = current_node_index
        visible_global_indices = list(range(len(full_node_coords)))
        nodes_idx_local2global = None
        if self.use_local_nodes:
            visible_global_indices = [int(key) for key in graph[str(current_node_index)]]
            current_index_local = visible_global_indices.index(current_node_index)
            nodes_idx_local2global = dict(enumerate(visible_global_indices))
            # current node index in the local window
        current_index = torch.tensor([current_index_local]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)

        node_coords = full_node_coords[visible_global_indices]
        node_utility = node_utility[visible_global_indices]
        indicator = indicator[visible_global_indices]
        direction_vector = direction_vector[visible_global_indices]
        node_inputs, node_padding_mask = self._pad_node_inputs(node_coords, node_utility, indicator, direction_vector)

        edge_inputs = self._get_local_edge_inputs(graph, visible_global_indices)
        edge_mask = self._pad_edge_mask(edge_inputs)
        curr_node_edges, curr_node_edge_padding_mask = self._build_current_edge_tensor(edge_inputs, current_index_local)

        gridmap_inputs = np.stack([self.env.lidar_scan] * 3, axis=-1).astype(np.uint8)
        gridmap_inputs = Image.fromarray(gridmap_inputs)
        gridmap_inputs = self.img_transform(gridmap_inputs)[None].to(self.device)  # (batch, 3, 224, 224)
        obs = {
            "current_index": current_index,  # (1, 1, 1)
            "node_inputs": node_inputs,  # (1, node_padding_size, 7)
            "node_padding_mask": node_padding_mask,  # (1, 1, node_padding_size)
            "edge_inputs": curr_node_edges,  # (1, 1, k_size) edges linked to the current node
            "curr_node_edge_padding_mask": curr_node_edge_padding_mask,  # (1, 1, k_size)
            "edge_mask": edge_mask,  # (1, node_padding_size, node_padding_size) adjacency matrix
            "nodes_idx_local2global": nodes_idx_local2global if self.use_local_nodes else None,
            "gridmap_inputs": gridmap_inputs,
        }
        if self.use_qnet_gt_graph:
            obs["q_edge_mask"] = self._build_gt_edge_mask(visible_global_indices)
        return obs

    def get_action(self, observations: dict):
        node_inputs = observations["node_inputs"]
        edge_inputs = observations["edge_inputs"]
        current_index = observations["current_index"]
        node_padding_mask = observations["node_padding_mask"]
        curr_node_edge_padding_mask = observations["curr_node_edge_padding_mask"]
        edge_mask = observations["edge_mask"]
        nodes_idx_local2global = observations["nodes_idx_local2global"]
        gridmap_inputs = observations["gridmap_inputs"]
        with torch.no_grad():
            model_input = (
                node_inputs,
                edge_inputs,
                current_index,
                node_padding_mask,
                curr_node_edge_padding_mask,
                edge_mask,
                gridmap_inputs,
            )
            logp_list = self.local_policy_net(model_input)  # probability of neighbor nodes
        # action_index is the local relative index of neighbor node
        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)
        next_node_index = edge_inputs[0, 0, action_index.item()]
        if self.use_local_nodes:
            next_position = self.env.node_coords_set.coords[nodes_idx_local2global[int(next_node_index)]]
        else:
            next_position = self.env.node_coords_set.coords[next_node_index]
        return next_position, action_index

    def save_observations(self, observations: dict, next_obs=False):
        for key in observations.keys():
            if key == "nodes_idx_local2global":
                continue
            if self.data_use_gpu:
                detached_obs = observations[key]
            else:
                detached_obs = observations[key].detach().cpu()

            target_key = ("next_" + key) if next_obs else key
            self.episode_buffer[target_key] += detached_obs

    def save_action(self, action_index):
        action_index = action_index.unsqueeze(0).unsqueeze(0)
        # if not self.data_use_gpu:
        if 1:
            action_index = action_index.detach().cpu()
        self.episode_buffer["action"] += action_index

    def save_reward_done(self, reward, done):
        reward = torch.FloatTensor([[[reward]]])
        done = torch.tensor([[[int(done)]]])
        # if not self.data_use_gpu:
        if 1:
            reward = reward.detach().cpu()
            done = done.detach().cpu()
        self.episode_buffer["reward"] += reward
        self.episode_buffer["done"] += done

    def run_episode(self, curr_episode):
        self._reset_policy_state()
        done = False
        observations = self.get_observations()
        time_step_plan_list = []
        time_step_list = []
        dist_list = []
        curvature_list = []
        time_episode_st = time.time()
        exp_rate_list = []
        step_num = 0
        for i in range(self.episode_max_step):  # steps for each episode
            st = time.time()
            self.save_observations(observations)

            st_pred = time.time()
            next_position, action_index = self.get_action(observations)
            policy_pred_time = time.time() - st_pred
            time_step_plan_list.append(policy_pred_time)
            self.save_action(action_index)

            reward, done, self.robot_position, self.travel_dist = self.env.step(
                self.robot_position, next_position, self.travel_dist
            )
            self.save_reward_done(reward, done)
            observations = self.get_observations()
            self.save_observations(observations, next_obs=True)
            self.robot_trajs.append(self.robot_position)
            curr_curvature = calcu_curvature(self.robot_trajs)
            dist_list.append(self.travel_dist)
            curvature_list.append(curr_curvature)
            exp_rate_list.append(self.env.explored_rate)
            if self.save_img:
                self.env.plot_env(self.episode_idx, self.gif_path, i, self.travel_dist)
            time_step_list.append(time.time() - st)
            step_num += 1
            if done:
                break

        time_episode_total = time.time() - time_episode_st
        self.perf_metrics["travel_dist"] = self.travel_dist
        self.perf_metrics["explored_rate"] = self.env.explored_rate
        self.perf_metrics["success"] = done
        self.perf_metrics["astar_len"] = self.env.astar_len
        self.perf_metrics["time_episode_total"] = time_episode_total
        self.perf_metrics["time_plan_total"] = np.sum(time_step_plan_list)
        self.perf_metrics["time_plan_ave"] = np.mean(time_step_plan_list)
        self.perf_metrics["time_plan_std"] = np.std(time_step_plan_list)
        self.perf_metrics["time_plan_max"] = np.max(time_step_plan_list)
        self.perf_metrics["time_plan_min"] = np.min(time_step_plan_list)
        self.perf_metrics["curvature_ave"] = np.mean(curvature_list)
        self.perf_metrics["curvature_std"] = np.std(curvature_list)
        self.perf_metrics["curvature_max"] = np.max(curvature_list)
        self.perf_metrics["curvature_min"] = np.min(curvature_list)
        self.perf_metrics["step_num"] = step_num
        self.perf_metrics["env_level"] = {"easy": 0, "medium": 1, "hard": 2, "real": 3}[self.env_level]
        if self.save_img:
            path = self.gif_path
            self.make_gif(path, curr_episode)

    def get_edge_mask(self, edge_inputs):
        """
        Input:
            edge_inputs: list of indices of adjacent nodes for each node
        Returns:
            bias_matrix: (size, size) 0 is connected, 1 is not connected
        """
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
        return bias_matrix

    def make_gif(self, path, n):
        with imageio.get_writer(f"{path}/{n}.gif", mode="I", duration=0.5, loop=0) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)


class ValidWorker(TrainWorker):
    def __init__(self, agent_id, policy_net: PolicyNet, episode_idx, worker_config: dict):

        self.agent_id = agent_id
        self.episode_idx = episode_idx  # Use episode_idx as episode_idx
        self.device = worker_config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.greedy = True
        self.node_padding_size = worker_config.get("node_padding_size", 360)
        self.k_size = worker_config.get("k_size", 20)
        self.save_img = worker_config.get("save_img", True)
        self.episode_max_step = worker_config.get("episode_max_step", 128)
        self.env_level = worker_config.get("env_level", "easy")
        self.use_env_encoding = True
        self.use_local_nodes = True
        self.use_qnet_gt_graph = True
        self.norm_utility = worker_config.get("norm_utility", True)
        self.save_traj = worker_config.get("save_traj", True)
        self.result_path = worker_config.get("test_result_path", f"{curr_dir}/results/{episode_idx}")
        self.custom_start_target = worker_config.get("custom_start_target", None)

        self.epi_result_path = f"{self.result_path}/{episode_idx}"
        os.makedirs(self.epi_result_path, exist_ok=True)

        env_config = worker_config.copy()
        env_config["sensor_range"] = resolve_sensor_range(worker_config)
        env_config["plot"] = worker_config.get("plot", self.save_img)
        map_file = resolve_nav_map_file(worker_config, self.episode_idx, test_mode=True)
        self.env = MapNavEnv(map_file, test_mode=True, env_config=env_config)
        if self.custom_start_target is not None:
            self.env.reset(self.custom_start_target[0], self.custom_start_target[1])
        else:
            self.env.reset()

        self.local_policy_net = policy_net
        self.local_policy_net.device = self.device
        self._reset_policy_state()
        self.current_node_index = 0
        self.travel_dist = 0
        self.robot_position = self.env.start_position
        self.robot_trajs = [self.robot_position]
        self.perf_metrics = dict()

        env_encoding_model = worker_config.get("env_encoding_model", "vit_tiny_patch8")
        self.img_transform = build_infer_transform(backbone=env_encoding_model)

    def run_episode(self):
        self._reset_policy_state()
        done = False
        observations = self.get_observations()
        time_step_plan_list = [0]
        dist_list = [0]
        curvature_list = [0]
        time_episode_st = time.time()
        exp_rate_list = [0]
        for i in range(self.episode_max_step):
            st = time.time()
            next_position, action_index = self.get_action(observations)
            duration = time.time() - st
            time_step_plan_list.append(duration)
            reward, done, self.robot_position, self.travel_dist = self.env.step(
                self.robot_position, next_position, self.travel_dist
            )
            observations = self.get_observations()
            self.robot_trajs.append(self.robot_position)
            curr_curvature = calcu_curvature(self.robot_trajs)
            dist_list.append(self.travel_dist)
            curvature_list.append(curr_curvature)
            exp_rate_list.append(self.env.explored_rate)
            if self.save_img:
                self.env.plot_env(self.episode_idx, self.epi_result_path, i, self.travel_dist)
            if done:
                break

        time_episode_total = time.time() - time_episode_st
        self.perf_metrics["travel_dist"] = self.travel_dist
        self.perf_metrics["explored_rate"] = self.env.explored_rate
        self.perf_metrics["success"] = done
        self.perf_metrics["astar_len"] = self.env.astar_len
        self.perf_metrics["time_episode_total"] = time_episode_total
        self.perf_metrics["time_plan_total"] = np.sum(time_step_plan_list)
        self.perf_metrics["time_plan_ave"] = np.mean(time_step_plan_list)
        self.perf_metrics["time_plan_std"] = np.std(time_step_plan_list)
        self.perf_metrics["time_plan_max"] = np.max(time_step_plan_list)
        self.perf_metrics["time_plan_min"] = np.min(time_step_plan_list)
        self.perf_metrics["curvature_ave"] = np.mean(curvature_list)
        self.perf_metrics["curvature_std"] = np.std(curvature_list)
        self.perf_metrics["curvature_max"] = np.max(curvature_list) * 1.0
        self.perf_metrics["curvature_min"] = np.min(curvature_list) * 1.0

        with open(f"{self.epi_result_path}/ours_metrics.json", "w") as f:
            json.dump(self.perf_metrics, f, indent=4)

        if self.save_traj:
            csv_filename = f"{self.epi_result_path}/trajectory.csv"
            col_keys = ["x", "y", "dist", "explored_rate", "curvature", "plan_step_time"]
            with open(csv_filename, "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=col_keys)
                writer.writeheader()
                for i in range(len(self.robot_trajs)):
                    writer.writerow(
                        {
                            "x": self.robot_trajs[i][0],
                            "y": self.robot_trajs[i][1],
                            "dist": dist_list[i],
                            "explored_rate": exp_rate_list[i],
                            "curvature": curvature_list[i],
                            "plan_step_time": time_step_plan_list[i],
                        }
                    )
        if self.save_img:
            self.make_gif(self.epi_result_path, "ours")

    def run_step(self, step_idx):
        """used for external control"""
        observations = self.get_observations()
        st = time.time()
        next_position, action_index = self.get_action(observations)
        action_duration = time.time() - st
        reward, done, self.robot_position, self.travel_dist = self.env.step(
            self.robot_position, next_position, self.travel_dist
        )
        new_observations = self.get_observations()

        self.robot_trajs.append(self.robot_position)
        if self.save_img:
            self.env.plot_env(self.episode_idx, self.epi_result_path, step_idx, self.travel_dist)
        print(
            f"Episode_{self.episode_idx} | Step {step_idx}, Action:{next_position}, Reward: {reward:.2f}, ActDuration: {action_duration:.4f}s"
        )

        new_observations["lidar_scan"] = self.env.lidar_scan
        return {
            "action": next_position,
            "new_observations": new_observations,
            "reward": reward,
            "done": done,
        }
