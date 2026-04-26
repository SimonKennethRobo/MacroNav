import math

import numpy as np
from numba import cuda, njit


@cuda.jit
def collision_check_kernel(x0, y0, x1, y1, gridmap, result):
    """
    CUDA kernel for collision checking between two points.
    """
    # Get thread ID
    idx = cuda.grid(1)
    if idx >= 1:  # Only one thread needed per line check
        return

    # Convert to integers
    x0_int = int(round(x0))
    y0_int = int(round(y0))
    x1_int = int(round(x1))
    y1_int = int(round(y1))

    dx = abs(x1_int - x0_int)
    dy = abs(y1_int - y0_int)
    x = x0_int
    y = y0_int

    error = dx - dy
    x_inc = 1 if x1_int > x0_int else -1
    y_inc = 1 if y1_int > y0_int else -1
    dx *= 2
    dy *= 2

    while x >= 0 and x < gridmap.shape[1] and y >= 0 and y < gridmap.shape[0]:
        k = gridmap[y, x]

        if k == 1 or k == 127:
            result[0] = 1  # Collision detected
            break

        if x == x1_int and y == y1_int:
            break

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx


@cuda.jit
def sensor_work_kernel(robot_positions, sensor_range, angles, lidar_scan, ground_truth):
    """
    CUDA kernel for parallel sensor scanning.
    """
    # Get thread ID (each thread handles one angle)
    idx = cuda.grid(1)
    if idx >= angles.shape[0]:
        return

    x0 = robot_positions[0]
    y0 = robot_positions[1]

    # Calculate endpoint for this angle
    sensor_angle = angles[idx]
    x1 = x0 + math.cos(sensor_angle) * sensor_range
    y1 = y0 + math.sin(sensor_angle) * sensor_range

    # Perform ray tracing
    x0_int = int(round(x0))
    y0_int = int(round(y0))
    x1_int = int(round(x1))
    y1_int = int(round(y1))

    dx = abs(x1_int - x0_int)
    dy = abs(y1_int - y0_int)
    x = x0_int
    y = y0_int

    error = dx - dy
    x_inc = 1 if x1_int > x0_int else -1
    y_inc = 1 if y1_int > y0_int else -1
    dx *= 2
    dy *= 2

    collision_flag = 0
    edge_pad_num = 10

    while x >= 0 and x < ground_truth.shape[1] and y >= 0 and y < ground_truth.shape[0]:
        k = ground_truth[y, x]

        # Update lidar scan
        cuda.atomic.max(lidar_scan, (y, x), k)

        if k == 1:
            collision_flag += 1
            if collision_flag >= edge_pad_num:
                break
        elif k == 127:
            break
        elif collision_flag > 0:
            break

        if x == x1_int and y == y1_int:
            break

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx


def collision_check_cuda(x0, y0, x1, y1, gridmap):
    """
    Host function to launch collision check kernel.
    """
    # Allocate memory on GPU
    result = cuda.device_array((1,), dtype=np.int32)
    d_gridmap = cuda.to_device(gridmap)

    # Launch kernel with 1 thread (single ray)
    collision_check_kernel[1, 1](x0, y0, x1, y1, d_gridmap, result)

    # Return result
    return result.copy_to_host()[0] == 1


def sensor_work_cuda(robot_position, sensor_range, lidar_scan, ground_truth):
    """
    Host function to launch parallel sensor work kernel.
    """
    # Calculate angles
    sensor_angle_inc = 0.5 / 180 * np.pi
    num_angles = int(2 * np.pi / sensor_angle_inc)
    angles = np.array([i * sensor_angle_inc for i in range(num_angles)], dtype=np.float32)

    # Allocate memory on GPU
    d_angles = cuda.to_device(angles)
    d_robot_position = cuda.to_device(np.array(robot_position))
    d_lidar_scan = cuda.to_device(lidar_scan)
    d_ground_truth = cuda.to_device(ground_truth)

    # Calculate grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = (num_angles + threads_per_block - 1) // threads_per_block

    # Launch kernel
    sensor_work_kernel[blocks_per_grid, threads_per_block](
        d_robot_position, sensor_range, d_angles, d_lidar_scan, d_ground_truth
    )

    # Copy result back to host
    lidar_scan = d_lidar_scan.copy_to_host()
    return lidar_scan


# Using multi-resolution grid to accelerate collision checking
def build_multi_resolution_grid(gridmap, levels=3):
    """
    Build a multi-resolution representation of the gridmap.

    Parameters:
    -----------
    gridmap: ndarray, shape (height, width)
        The original grid map
    levels: int
        Number of resolution levels

    Returns:
    --------
    multi_grid: list of ndarrays
        List of increasingly coarse representations of the gridmap
    """
    multi_grid = [gridmap]

    for i in range(1, levels):
        # Downsample by factor of 2
        h, w = multi_grid[i - 1].shape
        h_new, w_new = h // 2, w // 2

        # Create new grid level
        new_grid = np.zeros((h_new, w_new), dtype=gridmap.dtype)

        # If any cell in a 2x2 block is occupied, mark the corresponding cell in the coarse grid
        for y in range(h_new):
            for x in range(w_new):
                y_orig = y * 2
                x_orig = x * 2

                # Check the 2x2 block
                block = multi_grid[i - 1][y_orig : min(y_orig + 2, h), x_orig : min(x_orig + 2, w)]
                if np.any(block == 1) or np.any(block == 127):
                    new_grid[y, x] = 1

        multi_grid.append(new_grid)

    return multi_grid


def multi_resolution_collision_check(x0, y0, x1, y1, multi_grid):
    """
    Perform collision checking using multi-resolution grid.

    Starts with the coarsest grid and refines only when necessary.

    Parameters:
    -----------
    x0, y0: Starting point coordinates
    x1, y1: Ending point coordinates
    multi_grid: List of increasingly coarse representations of the gridmap

    Returns:
    --------
    collision: Boolean indicating if a collision was detected
    """
    # Start with the coarsest grid
    level = len(multi_grid) - 1
    scale_factor = 2**level

    # Adjust coordinates for coarsest grid
    x0_scaled = x0 // scale_factor
    y0_scaled = y0 // scale_factor
    x1_scaled = x1 // scale_factor
    y1_scaled = y1 // scale_factor

    # Check for collisions at coarsest level
    if not collision_check_simplified(x0_scaled, y0_scaled, x1_scaled, y1_scaled, multi_grid[level]):
        return False  # No collision at coarsest level means no collision at all

    # If collision detected, check finer levels along the path
    while level > 0:
        level -= 1
        scale_factor = 2**level

        x0_scaled = x0 // scale_factor
        y0_scaled = y0 // scale_factor
        x1_scaled = x1 // scale_factor
        y1_scaled = y1 // scale_factor

        if not collision_check_simplified(x0_scaled, y0_scaled, x1_scaled, y1_scaled, multi_grid[level]):
            return False

    # Final check at the finest level
    return collision_check_simplified(x0, y0, x1, y1, multi_grid[0])


@njit(cache=True)
def collision_check_simplified(x0, y0, x1, y1, gridmap):
    """
    Simplified collision check without lidar_scan updates.
    Uses numba for CPU acceleration.
    """
    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = int(round(x1))
    y1 = int(round(y1))

    dx, dy = abs(x1 - x0), abs(y1 - y0)

    if dx > dy:
        steps = dx
    else:
        steps = dy

    if steps == 0:
        # Check the single point
        if 0 <= x0 < gridmap.shape[1] and 0 <= y0 < gridmap.shape[0]:
            return gridmap[y0, x0] == 1 or gridmap[y0, x0] == 127
        return False

    x_inc = (x1 - x0) / steps
    y_inc = (y1 - y0) / steps

    x, y = x0, y0

    for i in range(int(steps) + 1):
        ix = int(round(x))
        iy = int(round(y))

        if 0 <= ix < gridmap.shape[1] and 0 <= iy < gridmap.shape[0]:
            if gridmap[iy, ix] == 1 or gridmap[iy, ix] == 127:
                return True

        x += x_inc
        y += y_inc

    return False


@njit(cache=True)
def collision_check(x0, y0, x1, y1, gridmap, update_scan=False, lidar_scan=None, edge_pad_num=10):
    """
    Performs collision checking between two points and optionally updates a lidar scan.

    Parameters:
    -----------
    x0, y0: Starting point coordinates
    x1, y1: Ending point coordinates
    gridmap: 2D array representing the environment/ground truth
    update_scan: Boolean flag to determine if lidar_scan should be updated
    lidar_scan: 2D array to update with scan results (optional)
    edge_pad_num: Number of cells to continue checking after finding a collision (optional)

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
            if not update_scan or collision_flag >= edge_pad_num:
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


@njit
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
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    from numba import cuda
    from PIL import Image

    # Load ground truth map
    ground_truth = np.array(Image.open("0.png"))
    ground_truth = (ground_truth > 210).astype(np.uint8) * 254 + 1
    map_gt_size = ground_truth.shape

    robot_location = np.array([270, 335])
    sensor_range = 100

    # Test collision check functions
    test_points = [
        ((270, 335), (280, 345)),  # Short diagonal
        ((270, 335), (370, 435)),  # Long diagonal
        ((270, 335), (270, 435)),  # Vertical line
        ((270, 335), (370, 335)),  # Horizontal line
    ]

    print("Testing collision check functions...")
    for start, end in test_points:
        # Original implementation
        orig_result = collision_check_simplified(start[0], start[1], end[0], end[1], ground_truth)

        # CUDA implementation
        cuda_result = collision_check_cuda(start[0], start[1], end[0], end[1], ground_truth)

        print(f"\nTest case: {start} -> {end}")
        print(f"Original result: {orig_result}")
        print(f"CUDA result: {cuda_result}")
        print(f"Match: {orig_result == cuda_result}")

    # Benchmark sensor work implementations
    num_trials = 5
    original_times = []
    cuda_times = []

    print("\nBenchmarking sensor work implementations...")
    for i in range(num_trials):
        # Original implementation
        lidar_scan_original = np.ones(map_gt_size) * 127
        start_time = time.time()
        lidar_scan_original = sensor_work(robot_location, sensor_range, lidar_scan_original, ground_truth)
        original_time = time.time() - start_time
        original_times.append(original_time)

        # CUDA implementation
        lidar_scan_cuda = np.ones(map_gt_size) * 127
        # Warm up GPU
        if i == 0:
            _ = sensor_work_cuda(robot_location, sensor_range, lidar_scan_cuda.copy(), ground_truth)
            cuda.synchronize()

        start_time = time.time()
        lidar_scan_cuda = sensor_work_cuda(robot_location, sensor_range, lidar_scan_cuda, ground_truth)
        cuda.synchronize()
        cuda_time = time.time() - start_time
        cuda_times.append(cuda_time)

        print(f"\nTrial {i + 1}:")
        print(f"Original implementation: {original_time:.4f} seconds")
        print(f"CUDA implementation: {cuda_time:.4f} seconds")
        print(f"Speedup: {original_time / cuda_time:.2f}x")

    # Calculate and print average times
    avg_original = np.mean(original_times)
    avg_cuda = np.mean(cuda_times)
    print(f"\nAverage times over {num_trials} trials:")
    print(f"Original implementation: {avg_original:.4f} seconds")
    print(f"CUDA implementation: {avg_cuda:.4f} seconds")
    print(f"Average speedup: {avg_original / avg_cuda:.2f}x")

    # Visualize results
    plt.figure(figsize=(15, 5))

    # Original implementation result
    plt.subplot(131)
    plt.imshow(lidar_scan_original, cmap="gray")
    plt.title("Original Implementation")
    plt.colorbar()

    # CUDA implementation result
    plt.subplot(132)
    plt.imshow(lidar_scan_cuda, cmap="gray")
    plt.title("CUDA Implementation")
    plt.colorbar()

    # Difference map
    difference = np.abs(lidar_scan_original - lidar_scan_cuda)
    plt.subplot(133)
    plt.imshow(difference, cmap="hot")
    plt.title("Difference Map")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("implementation_comparison.png")
    plt.close()

    # Check numerical accuracy
    max_diff = np.max(np.abs(lidar_scan_original - lidar_scan_cuda))
    mean_diff = np.mean(np.abs(lidar_scan_original - lidar_scan_cuda))
    print("\nNumerical accuracy:")
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")

    # Memory usage analysis
    def get_gpu_memory_usage():
        mem_info = cuda.current_context().get_memory_info()
        return mem_info.total - mem_info.free  # Returns used memory in bytes

    initial_gpu_mem = get_gpu_memory_usage()
    _ = sensor_work_cuda(robot_location, sensor_range, lidar_scan_cuda.copy(), ground_truth)
    final_gpu_mem = get_gpu_memory_usage()

    print("\nMemory usage:")
    print(f"GPU memory used: {(final_gpu_mem - initial_gpu_mem) / 1024 / 1024:.2f} MB")
