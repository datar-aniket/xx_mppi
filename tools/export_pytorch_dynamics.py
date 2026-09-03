#!/usr/bin/env python3
"""Export an xxCar body-derivative PyTorch model to the TensorRT ONNX contract."""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Optional

import torch


class ExportWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        model_input_mode: str,
        normalization: Optional[dict[str, Any]],
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.model_input_mode = model_input_mode
        values = normalization or {}
        self.register_buffer(
            "input_mean", torch.tensor(values.get("input_mean", [0.0] * 6), dtype=torch.float32)
        )
        self.register_buffer(
            "input_std", torch.tensor(values.get("input_std", [1.0] * 6), dtype=torch.float32)
        )
        self.register_buffer(
            "output_mean", torch.tensor(values.get("output_mean", [0.0] * 4), dtype=torch.float32)
        )
        self.register_buffer(
            "output_std", torch.tensor(values.get("output_std", [1.0] * 4), dtype=torch.float32)
        )

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        normalized = (model_input - self.input_mean) / self.input_std
        if self.model_input_mode == "split":
            output = self.model(normalized[:, :4], normalized[:, 4:])
        else:
            output = self.model(normalized)
        return output * self.output_std + self.output_mean


def load_model(checkpoint: Path, factory_spec: Optional[str]) -> torch.nn.Module:
    if factory_spec is None:
        try:
            return torch.jit.load(str(checkpoint), map_location="cpu")
        except RuntimeError:
            loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            if isinstance(loaded, torch.nn.Module):
                return loaded
            raise TypeError("checkpoint is a state_dict; supply --factory package.module:function")

    module_name, separator, function_name = factory_spec.partition(":")
    if not separator:
        raise ValueError("--factory must use package.module:function syntax")
    factory = getattr(importlib.import_module(module_name), function_name)
    model = factory()
    checkpoint_value = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state_dict = checkpoint_value.get("state_dict", checkpoint_value)
    model.load_state_dict(state_dict)
    return model


def vector(values: dict[str, Any], key: str, width: int) -> None:
    if key in values and (len(values[key]) != width or any(float(v) == 0.0 for v in values[key])
                          and key.endswith("std")):
        raise ValueError(f"{key} must contain {width} values and standard deviations must be nonzero")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--factory", help="factory for state_dict checkpoints: package.module:function")
    parser.add_argument("--model-input", choices=("concatenated", "split"), default="concatenated")
    parser.add_argument("--normalization-json", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    normalization = None
    if args.normalization_json:
        normalization = json.loads(args.normalization_json.read_text(encoding="utf-8"))
        for key, width in (("input_mean", 6), ("input_std", 6),
                           ("output_mean", 4), ("output_std", 4)):
            vector(normalization, key, width)

    wrapper = ExportWrapper(load_model(args.checkpoint, args.factory), args.model_input, normalization)
    sample = torch.zeros((8, 6), dtype=torch.float32)
    with torch.inference_mode():
        result = wrapper(sample)
    if tuple(result.shape) != (8, 4):
        raise ValueError(f"model output must be [batch,4], got {tuple(result.shape)}")

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (sample,),
        str(args.onnx),
        input_names=["model_input"],
        output_names=["state_derivative"],
        dynamic_axes={"model_input": {0: "batch"}, "state_derivative": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    contract = {
        "format": "xxcar_body_derivative_v1",
        "input": {"name": "model_input", "shape": ["batch", 6],
                  "fields": ["yaw_rate_radps", "speed_mps", "sideslip_rad",
                             "driven_wheel_speed_mps", "steering_rad", "wheel_torque_nm"]},
        "output": {"name": "state_derivative", "shape": ["batch", 4],
                   "fields": ["yaw_acceleration_radps2", "speed_acceleration_mps2",
                              "sideslip_rate_radps", "driven_wheel_acceleration_mps2"]},
        "normalization_baked_into_graph": normalization is not None,
    }
    args.onnx.with_suffix(args.onnx.suffix + ".json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.onnx} and {args.onnx}.json")


if __name__ == "__main__":
    main()
