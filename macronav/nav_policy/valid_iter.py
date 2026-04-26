"""Iteractive validation
Use mouse and keyboard to control the navigation process. Easy for demo and debugging.

Controls:
- Click to select a point on the map
- Press 's' to set the start point (after clicking)
- Press 'e' to set the end point (after clicking)
- Press space to start navigation
- Press '+' or '=' to zoom in, '-' to zoom out
- Press 'r' to reset the system
- Press ESC to exit

"""

import json
import os
import shutil
from pathlib import Path

import config.valid_param as valid_param
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from macronav.nav_policy.envs.sensor import sensor_work
from macronav.nav_policy.utils.misc import get_logger, import_module_from_path
from macronav.nav_policy.utils.infer_runtime import load_policy_with_backend
from macronav.nav_policy.utils.worker import ValidWorker, resolve_nav_map_file


class CVIteractive:
    def __init__(self, gt_map, lidar_scan, robot_location, scale_factor, env_level="real"):
        self.gt_map = self._normalize_map_image(gt_map)
        self.lidar_scan = lidar_scan
        self.robot_location = np.array(robot_location, dtype=np.int32)
        self.user_scale_factor = scale_factor
        self.render_scale = scale_factor
        self.clicked_point = None
        self.end_point = None
        self.preserve_scan = False
        self.start_point = None
        self.end_point = None
        self.env_level = env_level
        self.trajectory = None
        self.map_gt = None
        self.sensor_range = None
        self.current_raycast = None
        self.last_key = -1
        self.header_height = 54
        self.footer_height = 88
        self.panel_gap = 16
        self.outer_padding = 14
        self.layout_top = self.outer_padding + self.header_height
        self.palette = {
            "bg_top": np.array([18, 23, 35], dtype=np.uint8),
            "bg_bottom": np.array([33, 42, 59], dtype=np.uint8),
            "card": (31, 39, 55),
            "card_border": (78, 94, 124),
            "title": (246, 248, 252),
            "muted": (180, 191, 212),
            "accent": (100, 206, 255),
            "known_free": np.array([86, 197, 133], dtype=np.uint8),
            "known_occ": np.array([66, 97, 220], dtype=np.uint8),
            "ray_free": np.array([255, 214, 92], dtype=np.uint8),
            "ray_occ": np.array([255, 135, 78], dtype=np.uint8),
            "clicked": (122, 246, 168),
            "start": (84, 124, 255),
            "goal": (255, 124, 124),
            "robot": (255, 214, 92),
            "traj": (113, 229, 255),
        }
        self.screen_width, self.screen_height = self._get_screen_size()

    def _align_mask_to_map(self, mask):
        if mask is None:
            return None

        mask_arr = np.asarray(mask)
        target_h, target_w = self.gt_map.shape[:2]
        if mask_arr.shape[:2] == (target_h, target_w):
            return mask_arr

        return cv2.resize(mask_arr.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    def _normalize_map_image(self, image):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _get_screen_size(self):
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            return screen_width, screen_height
        except Exception:
            return 1920, 1080

    def _compute_render_scale(self, image):
        height, width = image.shape[:2]
        target_width = self.screen_width * 0.5
        target_height = self.screen_height * 0.5
        full_width = max(self.screen_width - 40, 1)
        full_height = max(self.screen_height - 80, 1)

        fit_half_scale = min(target_width / width, target_height / height)
        fit_screen_scale = min(full_width / width, full_height / height)
        return max(0.05, min(fit_half_scale * self.user_scale_factor, fit_screen_scale))

    def scale_image(self, image, factor):
        if factor == 1.0:
            return image

        height, width = image.shape[:2]
        new_height, new_width = int(height * factor), int(width * factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def mouse_callback(self, event, x, y, flags, param):
        original_x = int(x / self.render_scale)
        original_y = int(y / self.render_scale)
        original_y -= self.layout_top

        height, width = self.gt_map.shape[:2]
        if original_x >= width or original_y < 0 or original_y >= height:
            return
        original_x = max(0, min(original_x, width - 1))
        original_y = max(0, min(original_y, height - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (original_x, original_y)
            print(f"Clicked at: ({original_x}, {original_y})")

        elif event == cv2.EVENT_MOUSEMOVE:
            pass

    def update_runtime_state(
        self, robot_location=None, lidar_scan=None, trajectory=None, map_gt=None, sensor_range=None
    ):
        if robot_location is not None:
            self.robot_location = np.array(robot_location, dtype=np.int32)
        if lidar_scan is not None:
            self.lidar_scan = lidar_scan.copy()
        if trajectory is not None:
            self.trajectory = [np.array(pt, dtype=np.int32) for pt in trajectory]
        if map_gt is not None:
            self.map_gt = map_gt.copy()
            if self.gt_map.shape[:2] != self.map_gt.shape[:2]:
                self.gt_map = self._normalize_map_image(self.map_gt)
        if sensor_range is not None:
            self.sensor_range = sensor_range

        self.current_raycast = self._compute_current_raycast()

    def _compute_current_raycast(self):
        if self.map_gt is None or self.sensor_range is None or self.robot_location is None:
            return None

        raycast_scan = np.ones_like(self.map_gt, dtype=np.uint16) * 127
        return sensor_work(self.robot_location, self.sensor_range, raycast_scan, self.map_gt)

    def _overlay_scan_on_map(self, display_img):
        display_img = (display_img.astype(np.float32) * 0.78).astype(np.uint8)
        if self.lidar_scan is not None:
            aligned_scan = self._align_mask_to_map(self.lidar_scan)
            known_free = aligned_scan == 255
            known_occ = aligned_scan == 1
            display_img[known_free] = (
                display_img[known_free] * 0.35 + self.palette["known_free"] * 0.65
            ).astype(np.uint8)
            display_img[known_occ] = (
                display_img[known_occ] * 0.25 + self.palette["known_occ"] * 0.75
            ).astype(np.uint8)

        if self.current_raycast is not None:
            aligned_raycast = self._align_mask_to_map(self.current_raycast)
            ray_free = aligned_raycast == 255
            ray_occ = aligned_raycast == 1
            display_img[ray_free] = (display_img[ray_free] * 0.18 + self.palette["ray_free"] * 0.82).astype(np.uint8)
            display_img[ray_occ] = (display_img[ray_occ] * 0.16 + self.palette["ray_occ"] * 0.84).astype(np.uint8)

        return display_img

    def _build_scan_panel(self):
        scan_panel = np.full_like(self.gt_map, 24)
        if self.lidar_scan is not None:
            aligned_scan = self._align_mask_to_map(self.lidar_scan)
            known_free = aligned_scan == 255
            known_occ = aligned_scan == 1
            unknown = aligned_scan == 127
            scan_panel[unknown] = (40, 43, 52)
            scan_panel[known_free] = (234, 239, 244)
            scan_panel[known_occ] = tuple(int(v) for v in self.palette["known_occ"])

        if self.current_raycast is not None:
            aligned_raycast = self._align_mask_to_map(self.current_raycast)
            ray_free = aligned_raycast == 255
            ray_occ = aligned_raycast == 1
            scan_panel[ray_free] = tuple(int(v) for v in self.palette["ray_free"])
            scan_panel[ray_occ] = tuple(int(v) for v in self.palette["ray_occ"])

        return scan_panel

    def _draw_marker(self, display_img, point, color, label):
        center = tuple(map(int, point))
        cv2.circle(display_img, center, 12, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(display_img, center, 7, color, -1, cv2.LINE_AA)
        text_origin = (center[0] + 12, max(center[1] - 10, 18))
        cv2.putText(display_img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (14, 18, 24), 3, cv2.LINE_AA)
        cv2.putText(display_img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 250, 252), 1, cv2.LINE_AA)

    def _draw_annotations(self, display_img):
        if self.trajectory is not None and len(self.trajectory) > 1:
            traj_pts = [tuple(map(int, pt)) for pt in self.trajectory]
            for idx in range(len(traj_pts) - 1):
                cv2.line(display_img, traj_pts[idx], traj_pts[idx + 1], (26, 34, 46), 5, cv2.LINE_AA)
                cv2.line(display_img, traj_pts[idx], traj_pts[idx + 1], self.palette["traj"], 2, cv2.LINE_AA)
            for point in traj_pts[:: max(1, len(traj_pts) // 25 or 1)]:
                cv2.circle(display_img, point, 2, (255, 255, 255), -1, cv2.LINE_AA)

        if self.clicked_point:
            self._draw_marker(display_img, self.clicked_point, self.palette["clicked"], "CLICK")
        if self.start_point:
            self._draw_marker(display_img, self.start_point, self.palette["start"], "START")
        if self.end_point:
            self._draw_marker(display_img, self.end_point, self.palette["goal"], "GOAL")
        if self.robot_location is not None:
            robot_point = tuple(self.robot_location.astype(int))
            cv2.circle(display_img, robot_point, 15, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(display_img, robot_point, 8, self.palette["robot"], -1, cv2.LINE_AA)
            cv2.circle(display_img, robot_point, 3, (35, 41, 52), -1, cv2.LINE_AA)

    def _build_canvas(self, width, height):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        top = self.palette["bg_top"].astype(np.float32)
        bottom = self.palette["bg_bottom"].astype(np.float32)
        for row in range(height):
            alpha = row / max(height - 1, 1)
            canvas[row, :, :] = ((1 - alpha) * top + alpha * bottom).astype(np.uint8)
        return canvas

    def _draw_panel_card(self, canvas, panel, origin, title, subtitle):
        x0, y0 = origin
        panel_h, panel_w = panel.shape[:2]
        card_x0 = x0 - 6
        card_y0 = y0 - 36
        card_x1 = x0 + panel_w + 6
        card_y1 = y0 + panel_h + 6
        cv2.rectangle(canvas, (card_x0, card_y0), (card_x1, card_y1), self.palette["card"], -1, cv2.LINE_AA)
        cv2.rectangle(
            canvas,
            (card_x0, card_y0),
            (card_x1, card_y1),
            self.palette["card_border"],
            1,
            cv2.LINE_AA,
        )
        canvas[y0 : y0 + panel_h, x0 : x0 + panel_w] = panel
        cv2.putText(
            canvas, title, (card_x0 + 14, card_y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.palette["title"], 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            subtitle,
            (card_x0 + 14, card_y0 + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            self.palette["muted"],
            1,
            cv2.LINE_AA,
        )

    def _estimate_travel_distance(self):
        if self.trajectory is None or len(self.trajectory) < 2:
            return 0.0
        diffs = np.diff(np.asarray(self.trajectory, dtype=np.float32), axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())

    def _estimate_explored_ratio(self):
        if self.lidar_scan is None or self.map_gt is None:
            return 0.0
        free_total = max(int(np.sum(self.map_gt == 255)), 1)
        aligned_scan = self._align_mask_to_map(self.lidar_scan)
        free_seen = int(np.sum(aligned_scan == 255))
        return free_seen / free_total

    def _draw_footer(self, canvas):
        footer_y = canvas.shape[0] - self.footer_height
        sensor_text = "N/A" if self.sensor_range is None else str(int(self.sensor_range))
        footer_text = [
            f"Explored {self._estimate_explored_ratio() * 100:5.1f}%   Steps {max(len(self.trajectory or []) - 1, 0):3d}   Travel {self._estimate_travel_distance():7.1f}px",
            f"Zoom {self.user_scale_factor:3.1f}x   Env {self.env_level}   Sensor {sensor_text}px   Click map -> s=start, e=goal, space=run, r=reset, esc=exit",
        ]
        cv2.line(
            canvas,
            (self.outer_padding, footer_y - 10),
            (canvas.shape[1] - self.outer_padding, footer_y - 10),
            self.palette["card_border"],
            1,
            cv2.LINE_AA,
        )
        y = footer_y + 18
        for line in footer_text:
            cv2.putText(canvas, line, (self.outer_padding + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, self.palette["title"], 1, cv2.LINE_AA)
            y += 26

    def _compose_frame(self, main_panel, scan_panel):
        panel_h, panel_w = main_panel.shape[:2]
        canvas_w = panel_w * 2 + self.panel_gap + self.outer_padding * 2
        canvas_h = panel_h + self.header_height + self.footer_height + self.outer_padding * 2
        canvas = self._build_canvas(canvas_w, canvas_h)

        cv2.putText(
            canvas,
            "MacroNav Interactive Validation",
            (self.outer_padding, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            self.palette["title"],
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Live exploration overlay, accumulated scan, and trajectory state",
            (self.outer_padding, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            self.palette["muted"],
            1,
            cv2.LINE_AA,
        )

        left_origin = (self.outer_padding, self.layout_top)
        right_origin = (self.outer_padding + panel_w + self.panel_gap, self.layout_top)
        self._draw_panel_card(canvas, main_panel, left_origin, "World View", "GT map with explored area and live ray casting")
        self._draw_panel_card(canvas, scan_panel, right_origin, "Sensor View", "Cumulative lidar map layered with current cast")
        self._draw_footer(canvas)
        return canvas

    def update_display(self):
        main_panel = self._overlay_scan_on_map(self.gt_map.copy())
        self._draw_annotations(main_panel)

        scan_panel = self._build_scan_panel()
        self._draw_annotations(scan_panel)

        combined = self._compose_frame(main_panel, scan_panel)
        self.render_scale = self._compute_render_scale(combined)
        scaled_img = self.scale_image(combined, self.render_scale)
        cv2.imshow("Simulation", scaled_img)
        cv2.moveWindow("Simulation", 20, 20)

    def pump_events(self, delay=1):
        self.last_key = cv2.waitKey(delay) & 0xFF
        return self.last_key

    def reset(self):
        self.clicked_point = None
        self.start_point = None
        self.end_point = None
        self.lidar_scan = np.ones_like(self.lidar_scan) * 127
        self.trajectory = None
        self.current_raycast = None
        print("System reset")

    def set_trajectory(self, traj):
        self.trajectory = traj


def run_nav(policy_net, runner_config, episode_number, window_instance: CVIteractive = None):
    worker_config = runner_config.copy()
    worker_config["greedy"] = True
    worker_config["save_img"] = False
    worker_config["plot"] = False
    worker_config["use_astar"] = False

    worker_config["node_padding_size"] = 23
    worker_config["node_sample_step"] = 20

    if hasattr(policy_net, "reset_state"):
        policy_net.reset_state()
    elif hasattr(policy_net, "reset_recurrent_state"):
        policy_net.reset_recurrent_state()

    worker = ValidWorker(0, policy_net, episode_number, worker_config)
    window_instance.update_runtime_state(
        robot_location=worker.robot_position,
        lidar_scan=worker.env.lidar_scan,
        trajectory=worker.robot_trajs,
        map_gt=worker.env.map_gt,
        sensor_range=worker.env.sensor_range,
    )

    step_idx = 0
    aborted = False
    while step_idx < valid_param.EPISODE_MAX_STEP:
        returns = worker.run_step(step_idx)
        window_instance.update_runtime_state(
            robot_location=worker.robot_position,
            lidar_scan=returns["new_observations"]["lidar_scan"],
            trajectory=worker.robot_trajs,
            map_gt=worker.env.map_gt,
            sensor_range=worker.env.sensor_range,
        )
        window_instance.update_display()
        pressed = window_instance.pump_events(10)
        if pressed == 27:
            aborted = True
            print("Navigation aborted by user.")
            break
        if returns["done"]:
            break
        step_idx += 1
    if not aborted:
        print("Nav finished.")
    perf_metrics = worker.perf_metrics
    traj = worker.robot_trajs

    return perf_metrics, traj


def main():

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = Path(valid_param.LOG_DIR) / "test"
    os.makedirs(valid_param.TEST_RESULT_PATH, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(curr_dir) / "config" / "valid_param.py", test_dir / "valid_param.py")
    logger = get_logger(valid_param.TEST_RESULT_PATH)

    # load params
    train_param_dict = {}
    if valid_param.LOAD_PARAM_FROM_JSON:
        train_param_dict = json.load(open(f"{valid_param.LOG_DIR}/train_param.json", "r"))
    else:  # load from python file
        train_param = import_module_from_path("train_parameter", f"{valid_param.LOG_DIR}/train_param.py")
        logger.info(f"Loaded train_parameter from {valid_param.LOG_DIR}/train_param.py")
        logger.info(f"policy_net_args: {train_param.POLICY_NET_ARGS}")
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

    # load model
    model_type = "best"
    device = torch.device("cuda") if valid_param.USE_GPU else torch.device("cpu")
    logger.info(f"device: {device}")
    logger.info(f"policy net model args: {train_param_dict['policy_net_args']}")
    checkpoint_path = Path(valid_param.MODEL_PATH) / f"checkpoint_{model_type}.pth"
    config_path = Path(valid_param.LOG_DIR) / "train_param.py"

    policy_net, backend = load_policy_with_backend(train_param_dict, checkpoint_path, config_path, device, logger)
    logger.info(f"Using {backend.upper()} policy inference")

    runner_config_dict = train_param_dict.copy()
    runner_config_dict.update(valid_param.CONFIG_DICT)
    runner_config_dict["device"] = device
    runner_config_dict["env_level"] = valid_param.ENV_LEVEL
    sensor_range = runner_config_dict.get("sensor_range")
    if sensor_range is None:
        sensor_range = train_param_dict.get("sensor_range", 80)
    runner_config_dict["sensor_range"] = sensor_range

    episode_number = valid_param.CUSTOM_MAP_IDX
    if isinstance(episode_number, str) and episode_number.isdigit():
        episode_number = int(episode_number)

    map_file = resolve_nav_map_file(runner_config_dict, episode_number, test_mode=True)
    gt_map = plt.imread(map_file)
    if gt_map.max() <= 1.0:
        gt_map = (gt_map * 255).astype(np.uint8)
    else:
        gt_map = gt_map.astype(np.uint8)

    lidar_scan = np.ones(gt_map.shape[:2], dtype=np.uint16) * 127

    # init iteractive displaying
    robot_location = np.array([400, 400])
    window_instance = CVIteractive(
        gt_map, lidar_scan, robot_location, scale_factor=1.0, env_level=valid_param.ENV_LEVEL
    )
    window_instance.update_runtime_state(sensor_range=sensor_range)
    cv2.namedWindow("Simulation")
    cv2.setMouseCallback("Simulation", window_instance.mouse_callback)

    start_point = None
    end_point = None

    print("start main loop")
    try:
        while True:
            window_instance.update_display()
            key = window_instance.pump_events(1)

            if key == 27:  # ESC to exit
                break

            elif key == ord(" "):  # press space to start navigation
                if window_instance.start_point and window_instance.end_point:
                    runner_config_dict["custom_start_target"] = [start_point, end_point]
                    print("Navigation start")
                    metric, traj = run_nav(policy_net, runner_config_dict, episode_number, window_instance)
                else:
                    print("Please set both start and end points first!")

            elif key == ord("s"):
                start_point = window_instance.clicked_point
                window_instance.start_point = start_point
                print(f"Start point set at: {start_point}")

            elif key == ord("e"):
                end_point = window_instance.clicked_point
                window_instance.end_point = end_point
                print(f"End point set at: {end_point}")

            elif key == ord("+") or key == ord("="):  # Zoom in
                window_instance.user_scale_factor = min(4.0, window_instance.user_scale_factor + 0.1)
                print(f"Zoom level: {window_instance.user_scale_factor:.1f}x")

            elif key == ord("-"):  # Zoom out
                window_instance.user_scale_factor = max(0.2, window_instance.user_scale_factor - 0.1)
                print(f"Zoom level: {window_instance.user_scale_factor:.1f}x")

            elif key == ord("r"):
                window_instance.reset()

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
