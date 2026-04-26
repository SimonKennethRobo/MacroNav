import heapq

import numpy as np
from .sensor import collision_check


class Node:
    def __init__(self, pos, frontiers, lidar_scan, target_position, drive_costmap=None, obs_dist=70):
        """
        pos: (x, y)
        drive_costmap: 2D np array, use opencv img coord sys
        """
        self.pos = pos
        self.observable_frontiers = []
        self.obs_dist = obs_dist  # max observable distance
        self.target_position = target_position
        self.direction_vector = self.get_direction_vector()
        self.drive_cost = drive_costmap[int(self.pos[1]), int(self.pos[0])] if drive_costmap is not None else 0
        self.get_observable_frontiers(frontiers, lidar_scan)
        self.utility = self.get_node_utility()

    def get_observable_frontiers(self, frontiers, lidar_scan):
        dist_list = np.linalg.norm(frontiers - self.pos, axis=-1)
        frontiers_in_range = frontiers[dist_list < self.obs_dist]
        for point in frontiers_in_range:
            collision = collision_check(self.pos[0], self.pos[1], point[0], point[1], lidar_scan)
            if not collision:
                self.observable_frontiers.append(point)

    def get_direction_vector(self):
        dx = self.target_position[0] - self.pos[0]
        dy = self.target_position[1] - self.pos[1]
        mag = (dx**2 + dy**2) ** 0.5
        if mag != 0:
            dx = dx / mag
            dy = dy / mag
        if mag > 80:
            mag = 80
        return [dx, dy, mag]

    def get_node_utility(self):
        utility = len(self.observable_frontiers)
        # if utility < 5:
        #     utility = 0
        return utility

    def set_visited(self):
        self.observable_frontiers = []
        self.utility = 0

    def update_drive_cost(self, drive_costmap):
        self.drive_cost = drive_costmap[int(self.pos[1]), int(self.pos[0])]


class Edge:
    def __init__(self, to_node, length):
        self.to_node = to_node
        self.length = length


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, from_node, to_node, length):
        edge = Edge(to_node, length)
        # edge = to_node
        if from_node in self.edges:
            from_node_edges = self.edges[from_node]
        else:
            self.edges[from_node] = dict()
            from_node_edges = self.edges[from_node]

        from_node_edges[to_node] = edge

    def clear_edge(self, from_node):
        if from_node in self.edges:
            self.edges[from_node] = dict()

    def has_edge(self, from_node, to_node):
        if from_node in self.edges:
            if to_node in self.edges[from_node]:
                return True
        return False


def a_star_heuristic(index, destination, node_coords):
    current = node_coords[index]
    end = node_coords[destination]
    # h = abs(end[0] - current[0]) + abs(end[1] - current[1])
    h = (end[0] - current[0]) ** 2 + (end[1] - current[1]) ** 2
    return h


def a_star(start, destination, node_coords, graph):
    """
    start (int): index of start node
    destination (int): index of destination node
    node_coords: list of node coordinates
    """
    if start == destination:
        return [], 0

    start_str = str(start)
    dest_str = str(destination)
    if dest_str in graph.edges.get(start_str, {}):  # if start can reach destination
        cost = graph.edges[start_str][dest_str].length
        return [start, destination], cost

    open_queue = [(0, start)]
    closed_list = set()
    g = {start: 0}
    f = {start: a_star_heuristic(start, destination, node_coords)}
    parents = {start: start}

    while open_queue:
        _, current = heapq.heappop(open_queue)

        if current in closed_list:
            continue

        if current == destination:
            path = []
            while parents[current] != current:
                path.append(current)
                current = parents[current]
            path.append(start)
            path.reverse()
            return path, g[destination]

        current_str = str(current)
        if current_str in graph.edges:
            for edge in graph.edges[current_str].values():
                neighbor = int(edge.to_node)
                tentative_g = g[current] + edge.length

                if neighbor in closed_list and tentative_g >= g.get(neighbor, float("inf")):
                    continue

                if tentative_g < g.get(neighbor, float("inf")):
                    # Found a better path to neighbor
                    parents[neighbor] = current
                    g[neighbor] = tentative_g
                    f_score = tentative_g + a_star_heuristic(neighbor, destination, node_coords)
                    f[neighbor] = f_score
                    heapq.heappush(open_queue, (f_score, neighbor))

        closed_list.add(current)

    raise ValueError("No Path Found")
