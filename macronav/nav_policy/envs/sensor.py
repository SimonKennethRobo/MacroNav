import numpy as np
from numba import njit


@njit(fastmath=True)
def collision_check(x0, y0, x1, y1, gridmap, update_scan=False, lidar_scan=None, edge_pad_size=10):
    """
    Performs collision checking between two points and optionally updates a lidar scan.

    Parameters:
    -----------
    x0, y0: Starting point coordinates
    x1, y1: Ending point coordinates
    gridmap: 2D array representing the environment/ground truth
    update_scan: Boolean flag to determine if lidar_scan should be updated
    lidar_scan: 2D array to update with scan results (optional), 255 for free space, 1 for obstacle
    127 for unknown
    edge_pad_size: Number of cells to continue checking after finding a collision (optional)

    Returns:
    --------
    collision: Boolean indicating if a collision was detected
    """
    collision = False
    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = int(round(x1))
    y1 = int(round(y1))

    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0

    while 0 <= x < gridmap.shape[1] and 0 <= y < gridmap.shape[0]:
        k = gridmap[y, x]

        # Update lidar scan if requested
        if update_scan and lidar_scan is not None:
            lidar_scan[y, x] = k

        # Check for collision
        if k == 1:
            collision = True
            collision_flag += 1
            if not update_scan or collision_flag >= edge_pad_size:
                break
        elif k == 127:
            collision = True
            if not update_scan:
                break
        elif update_scan and collision_flag > 0:
            break

        if x == x1 and y == y1:
            break

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return collision


@njit(fastmath=True)
def sensor_work(robot_position, sensor_range, lidar_scan, ground_truth):
    """
    Input:
        lidar_scan: np.array of shape (height, width) current lidar_scan
        ground_truth: np.array of shape (height, width)
    Output:
        lidar_scan: updated lidar_scan
    """
    sensor_angle_inc = 0.5 / 180 * np.pi
    num_angles = int(2 * np.pi / sensor_angle_inc)
    x0 = robot_position[0]
    y0 = robot_position[1]
    for i in range(num_angles):
        sensor_angle = i * sensor_angle_inc
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        collision_check(x0, y0, x1, y1, ground_truth, True, lidar_scan)
    return lidar_scan


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image

    ground_truth = np.array(Image.open("0.png"))
    ground_truth = (ground_truth > 210).astype(np.uint8) * 254 + 1
    map_gt_size = ground_truth.shape
    plt.imsave("tmp/ground_truth.png", ground_truth, cmap="gray")

    robot_location = np.array([270, 335])
    target_location = np.array([20, 20])
    lidar_scan = np.ones(map_gt_size) * 127
    lidar_scan = sensor_work(robot_location, 100, lidar_scan, ground_truth)

    plt.imsave("tmp/lidar_scan.png", lidar_scan, cmap="gray")
