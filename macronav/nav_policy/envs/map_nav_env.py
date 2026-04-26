import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch
from PIL import Image
from scipy.signal import convolve2d
from skimage.measure import block_reduce

from .graph import *
from .graph_generator import GraphGenerator
from .sensor import sensor_work


class MapNavEnv:
    """
    self.ground_truth: 2d array, indicating the navigable area. 0 means navigable.
    self.lidar_scan: robot's current lidar scan
    self.frontiers: 2d array of frontier points locations (in pixel map)
    self.graph: graph of nodes, actually the edges
    """

    def __init__(self, map_file, test_mode, env_config: dict):
        self.test_mode = test_mode
        self.env_config = env_config
        self.curr_episode = env_config.get("curr_episode")
        self.sensor_range = env_config.get("sensor_range", 80)
        self.episode_max_step = env_config.get("episode_max_step", 128)
        self.reward_w_step = env_config.get("reward_w_step", 1)
        self.reward_w_astar = env_config.get("reward_w_astar", 1)
        self.env_level = env_config.get("env_level", "easy")
        self.node_sample_step = env_config.get("node_sample_step", 20)
        self.k_size = env_config.get("k_size", 20)
        self.use_astar = env_config.get("use_astar", not self.test_mode)
        self.plot = env_config.get("plot", False)
        self.device = env_config.get("device", "cuda")
        self.env_random_st_ed = env_config.get("env_random_st_ed", False)
        self.map_file = os.fspath(map_file)
        self.map_gt = self._import_map_gt(self.map_file)
        self.map_gt_size = np.shape(self.map_gt)
        self.lidar_scan = np.ones(self.map_gt_size, dtype=np.uint16) * 127  # current explored area
        self.lidar_scan_downsampled = None
        self.lidar_scan_last = copy.deepcopy(self.lidar_scan)  # last lidar scan, for frontier detection
        self.ds_block_size = 4  # lidar scan downsample block size
        self.explored_rate = 0
        self.frontiers = None
        self.reward = {
            "reward_done": 0,
            "reward_step": 0,
            "reward_astar": 0,
        }
        self.step_count = 0

        self.node_coords_set = None
        self.graph_edges = None
        self.node_utility = None  # array of utility values for each node
        self.indicator = None
        self.direction_vector = None
        self.coord2idx_map = None
        self.astar_len = None

        self.robot_pos = None
        self.robot_prev_pos = None
        self.traj_pts = None  # [[x0,y0],...,[xn,yn]]
        self.frame_files = []
        if self.plot:
            self.xPoints = None
            self.yPoints = None
            self.xTarget = None
            self.yTarget = None
        self.graph_generator = None
        self.start_position = None
        self.target_position = None

    def reset(self, start_position=None, target_position=None):
        if self.env_level != "real":
            if not self.test_mode:  # train mode
                if self.env_random_st_ed:
                    self.start_position, self.target_position = self._get_random_st_ed()
                else:
                    self.start_position, self.target_position = self._get_st_ed_from_map()
            else:  # test mode
                if start_position is None:
                    self.start_position, self.target_position = self._get_st_ed_from_map()
                else:
                    self.start_position = np.array(start_position)
                    self.target_position = np.array(target_position)
        else:
            if start_position is None:
                self.start_position, self.target_position = self._get_random_st_ed()
            else:
                self.start_position = np.array(start_position)
                self.target_position = np.array(target_position)

        self.robot_pos = self.start_position
        self.robot_prev_pos = self.start_position
        self.traj_pts = [self.start_position]  # [[x0,y0],...,[xn,yn]]
        if self.plot:
            self.xPoints = [self.start_position[0]]
            self.yPoints = [self.start_position[1]]
            self.xTarget = [self.target_position[0]]
            self.yTarget = [self.target_position[1]]

        self.graph_generator = GraphGenerator(
            map_size=self.map_gt_size,
            sensor_range=self.sensor_range,
            k_size=self.k_size,
            target_position=self.target_position,
            plot=self.plot,
            node_sample_step=self.node_sample_step,
        )
        self.graph_generator.visited_nodes_pos.append(self.start_position)

        self.lidar_scan = sensor_work(self.start_position, self.sensor_range, self.lidar_scan, self.map_gt)
        self.lidar_scan_downsampled = block_reduce(
            self.lidar_scan.copy(), block_size=(self.ds_block_size, self.ds_block_size), func=np.min
        )
        self.frontiers = self._find_frontier()
        self.lidar_scan_last = copy.deepcopy(self.lidar_scan)
        (
            self.node_coords_set,
            self.graph_edges,
            self.node_utility,
            self.indicator,
            self.direction_vector,
        ) = self.graph_generator.generate_graph(self.start_position, self.map_gt, self.lidar_scan, self.frontiers)
        if self.use_astar:
            self.astar_len, _ = self.graph_generator.find_shortest_path(self.start_position, self.target_position)
        else:
            self.astar_len = 0
        self.step_count = 0

    def get_observations(self):
        """
        Returns:
            node_inputs: (1,self.node_padding_size,7) 7=coords(2)+utility(1)+indicator(1)+direction_vector(3)
            edge_inputs: (1,1,k_size) k_size: number of adjacent nodes for the current node
            edge_mask: adjacency matrix of the graph
        """
        node_coords = self.env.node_coords_set.coords
        graph = self.env.graph_edges
        graph_gt = self.env.graph_generator.graph_gt
        node_utility = self.env.node_utility
        indicator = self.env.indicator
        direction_vector = self.env.direction_vector
        # get the node index of the current robot position
        current_node_index = self.env.find_index_from_coords(self.robot_position)
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
        if self.use_local_nodes:
            local_nodes_index = [int(key) for key in graph[str(current_node_index)]]
            local_nodes_coord = node_coords[local_nodes_index]
            local_nodes_utility = node_utility[local_nodes_index]
            local_indicator = indicator[local_nodes_index]
            local_direction_vector = direction_vector[local_nodes_index]
            node_coords = local_nodes_coord
            node_utility = local_nodes_utility
            indicator = local_indicator
            direction_vector = local_direction_vector
            local_nodes_index_relative = np.arange(0, len(node_coords))  # index within the local window
            nodes_idx_local2global = dict(zip(local_nodes_index_relative, local_nodes_index))
            # current node index in the local window
            current_index_local = local_nodes_index_relative[local_nodes_index.index(current_node_index)]
            current_index = torch.tensor([current_index_local]).unsqueeze(0).unsqueeze(0).to(self.device)
        # normalize observations
        node_coords = node_coords / self.env.map_gt_size[0]
        if self.norm_utility:
            node_utility = node_utility / 50
        n_nodes = node_coords.shape[0]
        node_utility_inputs = node_utility.reshape(n_nodes, 1)
        direction_nums = direction_vector.shape[0]
        direction_vector_inputs = direction_vector.reshape(direction_nums, 3)
        direction_vector_inputs[:, 2] /= max(float(self.sensor_range), 1.0)

        # concatenate all the inputs
        node_inputs = np.concatenate((node_coords, node_utility_inputs, indicator, direction_vector_inputs), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        try:
            assert node_coords.shape[0] < self.node_padding_size
        except AssertionError:
            print(f"node_coords.shape[0]:{node_coords.shape[0]}")
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
        node_inputs = padding(node_inputs)
        node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
            self.device
        )
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        if not self.use_local_nodes:
            graph_ = list(graph.values())
            edge_inputs = []  # list of indices of adjacent nodes for each node
            for node in graph_:
                node_edges = list(map(int, node))
                edge_inputs.append(node_edges)
            adjacent_matrix = self.get_edge_mask(edge_inputs)  # 0 is connected, 1 is not connected
            edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
            padding = torch.nn.ConstantPad2d(
                (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1
            )
            edge_mask = padding(edge_mask)  # (1, node_padding_size, node_padding_size)
            curr_node_edges = edge_inputs[current_index]
            while len(curr_node_edges) < self.k_size:
                curr_node_edges.append(0)
            curr_node_edges = torch.tensor(curr_node_edges).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
            # get edge padding mask (1 denotes the padded edges)
            curr_node_edge_padding_mask = torch.zeros((1, 1, self.k_size), dtype=torch.int64).to(self.device)
            one = torch.ones_like(curr_node_edge_padding_mask, dtype=torch.int64).to(self.device)
            curr_node_edge_padding_mask = torch.where(curr_node_edges == 0, one, curr_node_edge_padding_mask)
        else:
            local_graph = [graph[str(i)] for i in local_nodes_index]
            edge_inputs = []  # list of indices of adjacent nodes for each node
            for node in local_graph:
                node_edges = list(map(int, node))
                edge_inputs.append(node_edges)
            adjacent_matrix = self.get_edge_mask(edge_inputs)  # 0 is connected, 1 is not connected
            edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
            local_node_num = len(local_nodes_index)
            assert local_node_num < self.node_padding_size
            padding = torch.nn.ConstantPad2d(
                (0, self.node_padding_size - local_node_num, 0, self.node_padding_size - local_node_num), 1
            )
            edge_mask = padding(edge_mask)
            curr_node_edges = local_nodes_index_relative
            while len(curr_node_edges) < self.k_size:
                curr_node_edges = np.append(curr_node_edges, -1)
            curr_node_edges = torch.tensor(curr_node_edges).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
            # get edge padding mask (1 denotes the padded edges)
            curr_node_edge_padding_mask = torch.zeros((1, 1, self.k_size), dtype=torch.int64).to(self.device)
            one = torch.ones_like(curr_node_edge_padding_mask, dtype=torch.int64).to(self.device)
            curr_node_edge_padding_mask = torch.where(curr_node_edges == -1, one, curr_node_edge_padding_mask)
        gridmap_inputs = None
        if self.use_env_encoding:
            # extend 1 channel to 3 channels
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

    def step(self, robot_position, next_position, travel_dist):
        self.robot_pos = next_position
        self.traj_pts.append(next_position)
        self.step_count += 1
        dist = np.linalg.norm(robot_position - next_position)
        dist_to_target = np.linalg.norm(next_position - self.target_position)
        travel_dist += dist
        if self.use_astar:
            astar_dist_cur_to_target, _ = self.graph_generator.find_shortest_path(robot_position, self.target_position)
            astar_dist_next_to_target, _ = self.graph_generator.find_shortest_path(next_position, self.target_position)
        else:
            astar_dist_cur_to_target = None
            astar_dist_next_to_target = None

        self.graph_generator.visited_nodes_pos.append(next_position)
        next_node_index = self.find_index_from_coords(next_position)

        self.lidar_scan = sensor_work(next_position, self.sensor_range, self.lidar_scan, self.map_gt)
        self.lidar_scan_downsampled = block_reduce(
            self.lidar_scan.copy(), block_size=(self.ds_block_size, self.ds_block_size), func=np.min
        )
        self.frontiers = self._find_frontier()
        self.explored_rate = self._evaluate_exploration_rate()
        reward, done = self._get_reward(astar_dist_cur_to_target, astar_dist_next_to_target, dist_to_target)
        (
            self.node_coords_set,
            self.graph_edges,
            self.node_utility,
            self.indicator,
            self.direction_vector,
        ) = self.graph_generator.update_graph(self.lidar_scan, self.lidar_scan_last, self.frontiers)
        self.robot_prev_pos = robot_position

        if self.plot:
            self.xPoints.append(next_position[0])
            self.yPoints.append(next_position[1])
        self.lidar_scan_last = copy.deepcopy(self.lidar_scan)
        return reward, done, next_position, travel_dist

    def _import_map_gt(self, map_file):
        if self.env_level != "real":
            ground_truth = (skimage.io.imread(map_file, 1) * 255).astype(int)
            ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 60))
            ground_truth = ground_truth * 254 + 1
        else:
            with Image.open(map_file) as image:
                ground_truth = np.array(image.convert("L"))
            ground_truth = (ground_truth > 210).astype(np.uint16) * 254 + 1

        return ground_truth

    def _get_st_ed_from_map(self):
        map_raw = (skimage.io.imread(self.map_file, 1) * 255).astype(int)
        start_position = np.nonzero(map_raw == 209)  # a region
        start_position = np.array(
            [np.array(start_position)[1, 127], np.array(start_position)[0, 127]]
        )  # use the center of region as the location
        target_position = np.nonzero(map_raw == 68)
        target_position = np.array([np.array(target_position)[1, 127], np.array(target_position)[0, 127]])
        return start_position, target_position

    def _get_random_st_ed(self):
        """
        Generate random start and end positions within the navigable areas of the map.
        Start position is selected randomly from navigable cells.
        End position is selected to be outside a threshold radius from start position.
        """
        # Get all navigable (free) cells from the map
        free_cells = np.where(self.map_gt == 255)
        free_coords = np.column_stack((free_cells[1], free_cells[0]))  # (x, y) coordinates

        if len(free_coords) == 0:
            raise ValueError("No navigable cells found in the map")

        # Randomly select a start position
        start_idx = np.random.randint(0, len(free_coords))
        start_position = free_coords[start_idx]

        # Define minimum distance threshold between start and end position (20% of map diagonal)
        map_diagonal = np.sqrt(self.map_gt_size[0] ** 2 + self.map_gt_size[1] ** 2)
        min_distance_threshold = 0.2 * map_diagonal

        # Find all potential end positions that are far enough from start
        distances = np.linalg.norm(free_coords - start_position, axis=1)
        valid_end_indices = np.where(distances > min_distance_threshold)[0]

        # If no valid end positions found, relax the constraint
        if len(valid_end_indices) == 0:
            valid_end_indices = np.argsort(distances)[-10:]  # Take 10 farthest points

        # Randomly select an end position from valid candidates
        end_idx = np.random.choice(valid_end_indices)
        target_position = free_coords[end_idx]

        return start_position, target_position

    def find_index_from_coords(self, position):
        index = self.graph_generator.node_coords_set.get_index(position)
        return index

    def _free_cells(self):
        index = np.where(self.map_gt == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def _get_reward(self, astar_dist_cur_to_target, astar_dist_next_to_target, dist_to_target):
        reward = 0
        reward_drive = 0
        done = False
        reward_step = self.reward_w_step * self.step_count / self.episode_max_step
        reward_astar = 0
        if self.use_astar and astar_dist_cur_to_target is not None and astar_dist_next_to_target is not None:
            reward_astar = (
                (astar_dist_cur_to_target - astar_dist_next_to_target)
                / (self.k_size**0.5 * self.node_sample_step * 2**0.5)
                * self.reward_w_astar
            )
        if dist_to_target <= 2**0.5 * self.node_sample_step + 1:
            reward += 20
            done = True
        # else:
        #     reward -= (
        #         dist_to_target / (np.sqrt(self.map_gt_size[0] ** 2 + self.map_gt_size[1] ** 2)) * self.reward_w_astar
        #     )
        reward += -reward_step + reward_astar - reward_drive
        if self.step_count >= self.episode_max_step - 1:
            done = False
            reward -= 20

        self.reward = {
            "sum": reward,
            "reward_done": 20 if done else 0,
            "reward_step": -reward_step,
            "reward_astar": reward_astar,
        }
        return reward, done

    def _evaluate_exploration_rate(self):
        rate = np.sum(self.lidar_scan == 255) / np.sum(self.map_gt == 255)
        return rate

    def _find_frontier(self):
        # Convert lidar scan to binary map: unknown areas (127) = 1, known areas = 0
        mapping = (self.lidar_scan_downsampled == 127).astype(np.uint8)
        belief = self.lidar_scan_downsampled.copy()

        # Define 8-neighbor kernel for convolution
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

        # Use scipy's convolve2d which is more efficient for this case
        fro_map = convolve2d(mapping, kernel, mode="same", boundary="fill", fillvalue=0)

        # Create masks for known and free cells
        known = ~mapping.astype(bool)  # inverse of mapping
        free = belief == 255

        # Detect frontier cells: cells that are known and have unknown neighbors
        frontier_mask = known & (fro_map > 0) & (fro_map < 8) & free

        # Find coordinates of frontier cells and scale by block size
        # Using np.nonzero instead of np.argwhere for better performance
        y_indices, x_indices = np.nonzero(frontier_mask)
        f = np.column_stack((x_indices, y_indices)) * self.ds_block_size

        return f

    def plot_env(self, idx: str, img_path: str, step: str, travel_dist: float):
        os.makedirs(img_path, exist_ok=True)
        # plt.switch_backend("agg")
        # plt.ion()
        plt.cla()
        plt.axis((0, self.map_gt_size[1], self.map_gt_size[0], 0))
        plt.imshow(self.lidar_scan, cmap="gray", zorder=1)

        plt.plot(self.xTarget, self.yTarget, "o", markersize=10, zorder=2)
        plt.plot(self.xPoints[-1], self.yPoints[-1], "mo", markersize=8, zorder=2)  # current position
        plt.plot(self.xPoints[0], self.yPoints[0], "co", markersize=8, zorder=2)  # start position

        plt.plot(self.xPoints, self.yPoints, "b", linewidth=2, zorder=3)  # trajectory
        plt.scatter(
            self.node_coords_set.coords[:, 0],
            self.node_coords_set.coords[:, 1],
            c=self.node_utility,
            s={"easy": 20, "medium": 15, "hard": 6, "real": 20}[self.env_level] / (self.map_gt_size[0] / 500),
            zorder=3,
        )
        plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c="r", s=2, zorder=4)

        plt.suptitle(
            f"Explored ratio: {self.explored_rate:4.3f}, Dist: {int(travel_dist):8}, Reward: {self.reward['sum']:8.3f}\n"
            + f"R_step: {self.reward['reward_step']:7.3f}, R_astar: {self.reward['reward_astar']:7.3f}"
        )
        plt.tight_layout()
        plt.savefig("{}/{}_{}.png".format(img_path, idx, step), dpi=150)
        frame = "{}/{}_{}.png".format(img_path, idx, step)
        self.frame_files.append(frame)
        return frame
