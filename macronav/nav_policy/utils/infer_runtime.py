import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from macronav.nav_policy.models.nav import PolicyNet


class OnnxPolicyRunner:
    MODEL_INPUT_ORDER = (
        "node_inputs",
        "edge_inputs",
        "current_index",
        "node_padding_mask",
        "curr_node_edge_padding_mask",
        "edge_mask",
        "gridmap_inputs",
        "lstm_h",
        "lstm_c",
    )

    def __init__(self, onnx_path, device):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required for ONNX policy inference") from exc

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.device = device
        self.input_names = [meta.name for meta in self.session.get_inputs()]
        self.input_types = {meta.name: meta.type for meta in self.session.get_inputs()}
        self.output_names = [meta.name for meta in self.session.get_outputs()]
        self.has_lstm_state = "lstm_h" in self.input_names and "lstm_c" in self.input_names
        self.lstm_h = None
        self.lstm_c = None
        self.state_initialized = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def reset_state(self):
        self.lstm_h = None
        self.lstm_c = None
        self.state_initialized = False

    def _cast_numpy(self, array, input_name):
        input_type = self.input_types[input_name]
        if input_type == "tensor(float16)":
            return array.astype(np.float16, copy=False)
        if input_type == "tensor(float)":
            return array.astype(np.float32, copy=False)
        if input_type == "tensor(double)":
            return array.astype(np.float64, copy=False)
        if input_type == "tensor(int32)":
            return array.astype(np.int32, copy=False)
        if input_type == "tensor(int64)":
            return array.astype(np.int64, copy=False)
        return array

    def _infer_state_shape(self, input_name, batch_size):
        meta = next(meta for meta in self.session.get_inputs() if meta.name == input_name)
        shape = []
        for idx, dim in enumerate(meta.shape):
            if isinstance(dim, int):
                shape.append(dim)
            elif idx == 1:
                shape.append(batch_size)
            else:
                shape.append(1)
        return tuple(shape)

    def _ensure_state(self, batch_size):
        if not self.has_lstm_state:
            return
        if self.lstm_h is not None and self.lstm_h.shape[1] == batch_size and self.lstm_c is not None:
            return

        h_shape = self._infer_state_shape("lstm_h", batch_size)
        c_shape = self._infer_state_shape("lstm_c", batch_size)
        self.lstm_h = np.zeros(h_shape, dtype=np.float32)
        self.lstm_c = np.zeros(c_shape, dtype=np.float32)
        self.state_initialized = False

    def _run_session(self, ort_inputs):
        outputs = self.session.run(None, ort_inputs)
        logp = torch.from_numpy(outputs[0]).to(self.device)
        if self.has_lstm_state:
            self.lstm_h = outputs[1]
            self.lstm_c = outputs[2]
            self.state_initialized = True
        return logp

    def __call__(self, model_input):
        ort_inputs = {}
        input_by_name = {
            input_name: tensor for input_name, tensor in zip(self.MODEL_INPUT_ORDER, model_input) if tensor is not None
        }
        for input_name in self.input_names:
            if input_name in ("lstm_h", "lstm_c"):
                continue
            tensor = input_by_name[input_name]
            array = tensor.detach().cpu().numpy()
            ort_inputs[input_name] = self._cast_numpy(array, input_name)

        if not self.has_lstm_state:
            return self._run_session(ort_inputs)

        batch_size = model_input[0].shape[0]
        self._ensure_state(batch_size)
        ort_inputs["lstm_h"] = self._cast_numpy(self.lstm_h, "lstm_h")
        ort_inputs["lstm_c"] = self._cast_numpy(self.lstm_c, "lstm_c")

        if not self.state_initialized:
            self._run_session(ort_inputs)
            ort_inputs["lstm_h"] = self._cast_numpy(self.lstm_h, "lstm_h")
            ort_inputs["lstm_c"] = self._cast_numpy(self.lstm_c, "lstm_c")

        return self._run_session(ort_inputs)


def find_available_onnx(checkpoint_path: Path):
    preferred = checkpoint_path.with_name(f"{checkpoint_path.stem}_policy.onnx")
    if preferred.exists():
        return preferred

    candidates = sorted(checkpoint_path.parent.glob("*.onnx"))
    if candidates:
        return candidates[0]
    return None


def maybe_export_onnx(checkpoint_path: Path, config_path: Path, onnx_path: Path):
    answer = input(f"No ONNX policy found beside {checkpoint_path.name}. Export one now? [Y/n]: ").strip().lower()
    if answer not in ("", "y", "yes"):
        return False

    from macronav.nav_policy.scripts import export_policy
    export_script = export_policy.__file__
    print(f"Using export script: {export_script}")
    cmd = [
        sys.executable,
        str(export_script),
        "--checkpoint",
        str(checkpoint_path),
        "--config",
        str(config_path),
        "--format",
        "onnx",
        "--onnx-output",
        str(onnx_path),
    ]
    print("Exporting ONNX policy...")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0 and onnx_path.exists()


def load_policy_with_backend(train_param_dict, checkpoint_path: Path, config_path: Path, device, logger):
    onnx_path = find_available_onnx(checkpoint_path)
    if onnx_path is None and config_path.exists():
        target_onnx = checkpoint_path.with_name(f"{checkpoint_path.stem}_policy.onnx")
        if maybe_export_onnx(checkpoint_path, config_path, target_onnx):
            onnx_path = target_onnx
        else:
            logger.info("ONNX export skipped or failed. Falling back to Torch policy inference.")

    if onnx_path is not None:
        try:
            policy_runner = OnnxPolicyRunner(onnx_path, device).eval()
            logger.info(f"Loaded ONNX policy from: {onnx_path}")
            return policy_runner, "onnx"
        except Exception as exc:
            logger.warning(f"Failed to initialize ONNX policy runner from {onnx_path}: {exc}")
            logger.warning("Falling back to Torch policy inference.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_net = PolicyNet(train_param_dict).to(device)
    policy_net.load_state_dict(checkpoint["policy_model"])
    policy_net.eval()
    logger.info(f"Loaded Torch policy from: {checkpoint_path}")
    return policy_net, "torch"
