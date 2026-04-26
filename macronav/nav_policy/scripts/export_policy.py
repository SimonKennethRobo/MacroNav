import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch

from macronav.nav_policy.models.nav import PolicyNet


class PolicyExportWrapper(torch.nn.Module):
    def __init__(self, policy_net: PolicyNet):
        super().__init__()
        self.policy_net = policy_net

    def forward(
        self,
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask,
        curr_node_edge_padding_mask,
        edge_mask,
        gridmap_inputs,
    ):
        return self.policy_net(
            (
                node_inputs,
                edge_inputs,
                current_index,
                node_padding_mask,
                curr_node_edge_padding_mask,
                edge_mask,
                gridmap_inputs,
            )
        )


class PolicyOnnxExportWrapper(torch.nn.Module):
    def __init__(self, policy_net: PolicyNet):
        super().__init__()
        self.policy_net = policy_net

    def forward(
        self,
        node_inputs,
        edge_inputs,
        current_index,
        node_padding_mask,
        curr_node_edge_padding_mask,
        edge_mask,
        gridmap_inputs,
        lstm_h=None,
        lstm_c=None,
    ):
        if self.policy_net.use_lstm:
            self.policy_net.set_recurrent_state(lstm_h, lstm_c)

        logp = self.policy_net(
            (
                node_inputs,
                edge_inputs,
                current_index,
                node_padding_mask,
                curr_node_edge_padding_mask,
                edge_mask,
                gridmap_inputs,
            )
        )

        if not self.policy_net.use_lstm:
            return logp

        next_lstm_h, next_lstm_c = self.policy_net.get_recurrent_state()
        return logp, next_lstm_h, next_lstm_c


def reset_policy_state(policy_net: PolicyNet):
    if hasattr(policy_net, "reset_recurrent_state"):
        policy_net.reset_recurrent_state()


def parse_args():
    parser = argparse.ArgumentParser(description="Export MacroNav navigation policy to TensorRT and/or ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the navigation checkpoint, e.g. exps/nav_policy/encoder_run1/models/checkpoint_best.pth",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="macronav/nav_policy/config/train_param.py",
        help="Path to the nav training config used to build PolicyNet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the exported TensorRT TorchScript module. Defaults to checkpoint path with _policy_trt.ts",
    )
    parser.add_argument(
        "--onnx-output",
        type=str,
        default=None,
        help="Path to the exported ONNX model. Defaults to checkpoint path with _policy.onnx",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=("trt", "onnx", "both"),
        default="both",
        help="Export TensorRT, ONNX, or both",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Static batch size used for compilation")
    parser.add_argument("--node-padding-size", type=int, default=None, help="Override node padding size for export")
    parser.add_argument("--k-size", type=int, default=None, help="Override neighbor count for export")
    parser.add_argument("--img-size", type=int, default=224, help="Env-encoding image size")
    parser.add_argument("--onnx-opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--precision",
        type=str,
        choices=("fp32", "fp16"),
        default="fp16",
        help="Export precision for TensorRT and example tensors",
    )
    return parser.parse_args()


def load_config_module(config_path: str):
    spec = importlib.util.spec_from_file_location("nav_train_param", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config module from: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_policy_config(config_module):
    policy_config = dict(config_module.CONFIG_DICT)
    policy_config["eval_mode"] = True
    return policy_config


def build_model(config_module, checkpoint_path: str, device: torch.device, precision: str):
    policy_config = build_policy_config(config_module)
    policy_config["device"] = device
    model = PolicyNet(policy_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["policy_model"], strict=True)
    if precision == "fp16" and device.type == "cuda":
        model = model.half()
    model.eval()
    return model, policy_config


def resolve_export_sizes(args, config_module):
    node_padding_size = args.node_padding_size
    if node_padding_size is None:
        node_padding_size = config_module.NODE_PADDING_SIZE["local"]

    k_size = args.k_size if args.k_size is not None else config_module.K_SIZE
    return node_padding_size, k_size


def resolve_gridmap_spec(args, policy_net):
    gridmap_channels = 3
    gridmap_img_size = args.img_size

    if getattr(policy_net, "use_env_encoding", False):
        encoder = getattr(getattr(policy_net, "explored_env_encoder", None), "encoder", None)
        patch_embed = getattr(encoder, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        if proj is not None and hasattr(proj, "in_channels"):
            gridmap_channels = proj.in_channels
        if patch_embed is not None and hasattr(patch_embed, "img_size"):
            img_size = patch_embed.img_size
            if isinstance(img_size, tuple):
                if img_size[0] != img_size[1]:
                    raise ValueError(f"Only square gridmap inputs are supported for export, got: {img_size}")
                img_size = img_size[0]
            gridmap_img_size = int(img_size)

    return gridmap_channels, gridmap_img_size


def build_example_inputs(args, config_module, device, policy_net=None):
    node_padding_size, k_size = resolve_export_sizes(args, config_module)
    input_dim = config_module.INPUT_DIM
    use_fp16 = args.precision == "fp16" and device.type == "cuda"
    precision_dtype = torch.float16 if use_fp16 else torch.float32
    gridmap_channels, gridmap_img_size = resolve_gridmap_spec(args, policy_net) if policy_net is not None else (3, args.img_size)

    node_inputs = torch.randn(args.batch_size, node_padding_size, input_dim, device=device).to(precision_dtype)
    edge_inputs = torch.zeros(args.batch_size, 1, k_size, dtype=torch.long, device=device)
    current_index = torch.zeros(args.batch_size, 1, 1, dtype=torch.long, device=device)
    node_padding_mask = torch.zeros(args.batch_size, 1, node_padding_size, dtype=torch.int64, device=device)
    curr_node_edge_padding_mask = torch.zeros(args.batch_size, 1, k_size, dtype=torch.int64, device=device)
    edge_mask = torch.zeros(args.batch_size, node_padding_size, node_padding_size, device=device).to(precision_dtype)
    gridmap_inputs = torch.randn(
        args.batch_size,
        gridmap_channels,
        gridmap_img_size,
        gridmap_img_size,
        device=device,
    ).to(precision_dtype)

    return (
        (
            node_inputs,
            edge_inputs,
            current_index,
            node_padding_mask,
            curr_node_edge_padding_mask,
            edge_mask,
            gridmap_inputs,
        ),
        node_padding_size,
        k_size,
        input_dim,
    )


def build_onnx_example_inputs(args, config_module, policy_config, device, policy_net=None):
    base_inputs, node_padding_size, k_size, input_dim = build_example_inputs(args, config_module, device, policy_net)
    return base_inputs, node_padding_size, k_size, input_dim


def build_trt_inputs(args, node_padding_size, k_size, input_dim):
    import torch_tensorrt

    precision_dtype = torch.half if args.precision == "fp16" else torch.float32
    trt_inputs = [
        torch_tensorrt.Input(
            min_shape=(1, node_padding_size, input_dim),
            opt_shape=(args.batch_size, node_padding_size, input_dim),
            max_shape=(args.batch_size, node_padding_size, input_dim),
            dtype=precision_dtype,
        ),
        torch_tensorrt.Input(
            min_shape=(1, 1, k_size),
            opt_shape=(args.batch_size, 1, k_size),
            max_shape=(args.batch_size, 1, k_size),
            dtype=torch.int64,
        ),
        torch_tensorrt.Input(
            min_shape=(1, 1, 1),
            opt_shape=(args.batch_size, 1, 1),
            max_shape=(args.batch_size, 1, 1),
            dtype=torch.int64,
        ),
        torch_tensorrt.Input(
            min_shape=(1, 1, node_padding_size),
            opt_shape=(args.batch_size, 1, node_padding_size),
            max_shape=(args.batch_size, 1, node_padding_size),
            dtype=torch.int64,
        ),
        torch_tensorrt.Input(
            min_shape=(1, 1, k_size),
            opt_shape=(args.batch_size, 1, k_size),
            max_shape=(args.batch_size, 1, k_size),
            dtype=torch.int64,
        ),
        torch_tensorrt.Input(
            min_shape=(1, node_padding_size, node_padding_size),
            opt_shape=(args.batch_size, node_padding_size, node_padding_size),
            max_shape=(args.batch_size, node_padding_size, node_padding_size),
            dtype=precision_dtype,
        ),
        torch_tensorrt.Input(
            min_shape=(1, 3, args.img_size, args.img_size),
            opt_shape=(args.batch_size, 3, args.img_size, args.img_size),
            max_shape=(args.batch_size, 3, args.img_size, args.img_size),
            dtype=precision_dtype,
        ),
    ]

    return trt_inputs


def export_onnx(wrapper, example_inputs, output_path, args):
    input_names = [
        "node_inputs",
        "edge_inputs",
        "current_index",
        "node_padding_mask",
        "curr_node_edge_padding_mask",
        "edge_mask",
        "gridmap_inputs",
    ]
    output_names = ["logp"]
    dynamic_axes = {
        "node_inputs": {0: "batch"},
        "edge_inputs": {0: "batch"},
        "current_index": {0: "batch"},
        "node_padding_mask": {0: "batch"},
        "curr_node_edge_padding_mask": {0: "batch"},
        "edge_mask": {0: "batch"},
        "gridmap_inputs": {0: "batch"},
        "logp": {0: "batch"},
    }

    if len(example_inputs) == 9:
        input_names.extend(["lstm_h", "lstm_c"])
        output_names.extend(["next_lstm_h", "next_lstm_c"])
        dynamic_axes["lstm_h"] = {1: "batch"}
        dynamic_axes["lstm_c"] = {1: "batch"}
        dynamic_axes["next_lstm_h"] = {1: "batch"}
        dynamic_axes["next_lstm_c"] = {1: "batch"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reset_policy_state(wrapper.policy_net)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(output_path),
            export_params=True,
            opset_version=args.onnx_opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    reset_policy_state(wrapper.policy_net)
    print(f"Saved navigation policy ONNX model to: {output_path}")


def export_tensorrt(wrapper, example_inputs, node_padding_size, k_size, input_dim, args):
    try:
        import torch_tensorrt
    except ImportError as exc:
        raise ImportError(
            "torch_tensorrt is not installed. Install Torch-TensorRT first, then rerun this script."
        ) from exc

    reset_policy_state(wrapper.policy_net)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example_inputs, strict=False)
    enabled_precisions = {torch.float16} if args.precision == "fp16" else {torch.float32}
    trt_module = torch_tensorrt.compile(
        traced,
        ir="ts",
        inputs=build_trt_inputs(args, node_padding_size, k_size, input_dim),
        enabled_precisions=enabled_precisions,
        truncate_long_and_double=True,
    )
    reset_policy_state(wrapper.policy_net)

    output_path = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "_policy_trt.ts")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(trt_module, str(output_path))
    print(f"Saved navigation policy TensorRT module to: {output_path}")


def main():
    args = parse_args()
    needs_trt = args.format in ("trt", "both")
    needs_onnx = args.format in ("onnx", "both")

    if needs_trt and not torch.cuda.is_available():
        raise RuntimeError("TensorRT export requires CUDA. No CUDA device is available.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_module = load_config_module(args.config)
    model, policy_config = build_model(config_module, args.checkpoint, device, args.precision)
    trt_wrapper = PolicyExportWrapper(model).to(device).eval()
    onnx_wrapper = PolicyOnnxExportWrapper(model).to(device).eval()

    example_inputs, node_padding_size, k_size, input_dim = build_example_inputs(args, config_module, device, model)
    onnx_example_inputs, _, _, _ = build_onnx_example_inputs(args, config_module, policy_config, device, model)

    if needs_onnx:
        onnx_output = (
            Path(args.onnx_output)
            if args.onnx_output
            else Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "_policy.onnx")
        )
        export_onnx(onnx_wrapper, onnx_example_inputs, onnx_output, args)

    if needs_trt:
        export_tensorrt(
            trt_wrapper,
            example_inputs,
            node_padding_size,
            k_size,
            input_dim,
            args,
        )


if __name__ == "__main__":
    main()
