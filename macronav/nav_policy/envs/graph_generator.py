from typing import List

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line
from sklearn.neighbors import KDTree

from .graph import Graph, Node, a_star
from .sensor import collision_check


class CoordSet:
    def __init__(self):
        self.coords = np.empty((0, 2), dtype=np.uint16)
        self.coord_to_index = {}

    def add_point(self, point):
        point_tuple = tuple(point)

        if point_tuple in self.coord_to_index:
            return self.coord_to_index[point_tuple]

        new_index = len(self.coords)
        self.coords = np.vstack([self.coords, point])
        self.coord_to_index[point_tuple] = new_index

        return new_index

    def add_points(self, points):
        indices = []
        for point in points:
            idx = self.add_point(point)
            indices.append(idx)

        return np.array(indices)

    def get_index(self, point):
        point_tuple = tuple(point)
        if point_tuple in self.coord_to_index:
            return self.coord_to_index[point_tuple]

        # If point not found directly, find the closest point
        if len(self.coords) > 0:
            # Calculate distances to all points
            distances = np.sum((self.coords - np.array(point)) ** 2, axis=1)
            # Return index of the point with minimum distance
            closest_idx = np.argmin(distances)
            return closest_idx

        return None

    def get_all_points(self):
        return self.coords.copy()

    def get_points_by_indices(self, indices):
        return self.coords[indices]

    def __len__(self):
        return len(self.coords)


class GraphGenerator:
    def __init__(self, map_size, k_size, sensor_range, target_position, plot=False, node_sample_step=30):
        self.graph = Graph()
        self.graph_gt = Graph()
        self.lidar_scan_gt = None
        self.node_coords_set = CoordSet()
        self.plot = plot
        self.k_size = k_size
        self.node_sample_step = node_sample_step
        self.map_width = map_size[1]
        self.map_height = map_size[0]  # 0 is height
        self.discret_points = self._gen_discret_points()
        self.sensor_range = sensor_range
        self.visited_nodes_pos = []
        self.nodes_list: List[Node] = []
        self.node_utility = None
        self.indicator = None
        self.direction_vector = None
        self.target_position = target_position

        self.kdtree = None

    def generate_node_coords(self, robot_location, lidar_scan):
        free_area = self._free_area(lidar_scan.astype(np.float32))
        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        discret_points_to_check = self.discret_points[:, 0] + self.discret_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, discret_points_to_check, return_indices=True)
        node_coords = self.discret_points[candidate_indices]
        node_coords = np.concatenate((robot_location.reshape(1, 2), self.target_position.reshape(1, 2), node_coords))
        node_coords_set = CoordSet()
        for coords in node_coords:
            node_coords_set.add_point(coords)
        return node_coords_set

    def generate_graph(self, robot_location, lidar_scan_gt, lidar_scan, frontiers):
        self.graph = Graph()
        self.graph_gt = Graph()
        self.lidar_scan_gt = lidar_scan_gt
        self.node_coords_set = self.generate_node_coords(robot_location, lidar_scan)
        self.node_coords_set_gt = self.generate_node_coords(robot_location, lidar_scan_gt)

        self.graph = self._update_knn_graph(self.graph, self.node_coords_set.coords, lidar_scan)
        self.graph_gt = self._update_knn_graph(self.graph_gt, self.node_coords_set_gt.coords, lidar_scan_gt)

        self.node_utility = []
        self.direction_vector = []
        self.nodes_list = []
        self.indicator = np.zeros((self.node_coords_set.coords.shape[0], 1), dtype=np.uint8)
        robot_pos_idx = self.node_coords_set.get_index(robot_location)
        self.indicator[robot_pos_idx] = 1
        for i, coord in enumerate(self.node_coords_set.coords):
            node = Node(coord, frontiers, lidar_scan, self.target_position)
            self.nodes_list.append(node)
            self.direction_vector.append(node.direction_vector)
            self.node_utility.append(node.utility)

        self.node_utility = np.array(self.node_utility)
        self.direction_vector = np.array(self.direction_vector)
        if 0:
            self.plot_graph(
                self.node_coords_set_gt.coords,
                self.graph_gt.edges,
                lidar_scan_gt,
                title="Graph with Nodes and Edges",
            )
        return (
            self.node_coords_set,
            self.graph.edges,
            self.node_utility,
            self.indicator,
            self.direction_vector,
        )

    def update_graph(self, lidar_scan, lidar_scan_last, frontiers):
        """Update the graph by adding new nodes and edges, and updating observable frontiers."""

        # Filter out the new points in free area from the complete points set
        new_free_area = self._free_area((lidar_scan.astype(np.float32) - lidar_scan_last.astype(np.float32) > 0) * 255)
        free_area_to_check = new_free_area[:, 0] + new_free_area[:, 1] * 1j
        discret_points_to_check = self.discret_points[:, 0] + self.discret_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, discret_points_to_check, return_indices=True)
        new_node_coords = self.discret_points[candidate_indices]
        self.node_coords_set.add_points(new_node_coords)

        self.graph = self._update_knn_graph(self.graph, self.node_coords_set.coords, lidar_scan)
        if 0:
            self.plot_graph(
                self.node_coords_set.coords,
                self.graph.edges,
                lidar_scan,
                title="Graph with Nodes and Edges",
            )

        # Update nodes_list
        self.nodes_list = []
        self.direction_vector = []
        self.node_utility = []
        self.indicator = np.zeros((self.node_coords_set.coords.shape[0], 1), dtype=np.uint8)
        for i, pos in enumerate(self.node_coords_set.coords):
            node = Node(pos, frontiers, lidar_scan, self.target_position)
            self.nodes_list.append(node)
            self.node_utility.append(node.utility)
            self.direction_vector.append(node.direction_vector)
            if np.any((pos == self.visited_nodes_pos).all(axis=1)):
                self.indicator[i] = 1
        self.direction_vector = np.array(self.direction_vector)
        self.node_utility = np.array(self.node_utility)

        return (
            self.node_coords_set,
            self.graph.edges,
            self.node_utility,
            self.indicator,
            self.direction_vector,
        )

    def _gen_discret_points(self):
        x = np.arange(0, self.map_width, self.node_sample_step)
        y = np.arange(0, self.map_height, self.node_sample_step)
        if x[-1] != self.map_width - 1:  # fill to edge
            x = np.append(x, self.map_width - 1)
        if y[-1] != self.map_height - 1:
            y = np.append(y, self.map_height - 1)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def _free_area(self, lidar_scan):
        # free area 255
        index = np.where(lidar_scan == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def _update_knn_graph(self, graph, node_coords, lidar_scan):
        """Find k nearest neighbors for all nodes and update the graph accordingly."""
        graph = Graph()
        self.kdtree = KDTree(node_coords)
        k = min(self.k_size, len(node_coords))
        distances, indices = self.kdtree.query(node_coords, k=k)  # indices is a matrix

        for i, p in enumerate(node_coords):
            for j, neighbour in enumerate(node_coords[indices[i][:]]):
                if graph.has_edge(str(i), str(indices[i][j])):
                    continue

                collision = collision_check(p[0], p[1], neighbour[0], neighbour[1], lidar_scan)
                if not collision:
                    graph.add_edge(str(i), str(indices[i][j]), distances[i, j])
        return graph

    def find_shortest_path(self, current, destination):
        """
        current [x,y]
        """
        start_node = self.node_coords_set_gt.get_index(current)
        end_node = self.node_coords_set_gt.get_index(destination)
        try:
            route, dist = a_star(start_node, end_node, self.node_coords_set_gt.coords, self.graph_gt)
        except Exception as e:
            print("Error in A*:", e)
            return 0, None
        if start_node != end_node:
            assert route != []
        route = list(map(str, route))
        return dist, route

    def find_nearest_node(self, position, node_coords):
        diffs = node_coords - position
        sq_dists = np.einsum("ij,ij->i", diffs, diffs)
        idx = np.argmin(sq_dists)
        return node_coords[idx]

    def plot_graph(self, coords, graph_edges, scan, path=None, title=None):
        plt.close("all")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)
        ax.set_title(title)
        # plot scan
        ax.imshow(scan, cmap="gray", alpha=0.5, extent=(0, self.map_width, self.map_height, 0))

        # Plot nodes
        for i, coord in enumerate(coords):
            # if self.indicator[i] == 1:
            #     color = "red"
            # else:
            #     color = "blue"
            color = "blue"
            ax.scatter(coord[0], coord[1], c=color, s=10)
        ax.scatter(self.target_position[0], self.target_position[1], c="red", s=100, label="Target Position")

        # Plot edges
        for node, edges in graph_edges.items():
            for edge in edges.values():
                start = coords[int(node)]
                end = coords[int(edge.to_node)]
                rr, cc = line(int(start[1]), int(start[0]), int(end[1]), int(end[0]))
                ax.plot(cc, rr, c="gray", linewidth=0.5)

        # Plot path if provided
        if path is not None:
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                rr, cc = line(int(start[1]), int(start[0]), int(end[1]), int(end[0]))
                ax.plot(cc, rr, c="green", linewidth=2)

            curr_pos = path[-1]
            ax.scatter(curr_pos[0], curr_pos[1], c="yellow", s=100, label="Current Position")
            ax.legend()

        plt.show()
        # plt.close(fig)
