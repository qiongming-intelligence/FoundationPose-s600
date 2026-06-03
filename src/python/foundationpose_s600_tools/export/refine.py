#!/usr/bin/env python3
"""Export FoundationPose RefineNet to ONNX for S600 HBM compilation.

Runs on the **x86 export host** (needs torch / omegaconf; onnxruntime for
verification). Builds ``RefineNet`` from the vendored upstream source + the
authorized checkpoint, then exports the contract-defined ONNX graph.

Example:
    PYTHONPATH=src/python python -m foundationpose_s600_tools.export.refine \\
        --contract build/foundationpose_export/contracts/refine_net.json \\
        --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

from foundationpose_s600_tools.export import wrappers

# Upstream run_name shipping the RefineNet checkpoint (predict_pose_refine.py).
REFINE_RUN_NAME = "2023-10-28-18-33-37"


def build_refine_module(torch, root: Path, run_name: str, contract: dict, allow_random_init: bool = False):
    """Construct RefineNet on CPU, load weights, set eval()."""
    wrappers.add_upstream_to_path(root)
    wrappers.install_utils_shim()
    from learning.models.refine_network import RefineNet  # type: ignore

    # rot_rep must match the contract's R dim so the head width lines up.
    rot_dim = int(contract["dimensions"]["R"])
    rot_rep = "axis_angle" if rot_dim == 3 else "6d"
    c_in = int(contract["dimensions"]["Cin"])

    cfg = wrappers.load_config(root, run_name, overrides={"rot_rep": rot_rep, "c_in": c_in})
    model = RefineNet(cfg=cfg, c_in=c_in)
    ckpt_path = root / "weights" / run_name / "model_best.pth"
    if ckpt_path.is_file():
        state = wrappers.load_checkpoint_state(root, run_name)
        model.load_state_dict(state)
    elif allow_random_init:
        print(f"WARNING: {ckpt_path} missing; exporting randomly initialized RefineNet for ABI/export smoke only")
    else:
        raise SystemExit(f"missing upstream checkpoint: {ckpt_path} (download authorized weights first)")
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Export FoundationPose RefineNet to ONNX.")
    parser.add_argument("--contract", type=Path, default=Path("build/foundationpose_export/contracts/refine_net.json"))
    parser.add_argument("--upstream-root", type=str, default=None, help="path to vendored FoundationPose (default: third_party/FoundationPose)")
    parser.add_argument("--run-name", default=REFINE_RUN_NAME, help="weights/{run_name}/ subdir with config.yml + model_best.pth")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamo-exporter", action="store_true", help="use PyTorch's new dynamo ONNX exporter; default is legacy exporter for opset-17 HBM compatibility")
    parser.add_argument("--allow-random-init", action="store_true", help="export with random weights if model_best.pth is missing; smoke/ABI only, not deployable")
    parser.add_argument("--verify", action="store_true", help="compare ONNX Runtime vs PyTorch outputs")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise SystemExit("RefineNet export needs PyTorch; run on the x86 export host") from error

    root = wrappers.upstream_root(args.upstream_root)
    contract = wrappers.read_contract(args.contract)

    model = build_refine_module(torch, root, args.run_name, contract, allow_random_init=args.allow_random_init)

    # RefineNet.forward(A, B) -> {'trans','rot'}. Adapt to the contract's named tuple.
    output_keys = [t["name"] for t in contract["outputs"]]  # ['trans','rot']
    wrapper = wrappers.NamedTupleOutput(
        torch, model, output_keys, call_adapter=lambda m, A, B: m(A, B)
    )

    inputs = wrappers.dummy_inputs(torch, contract)
    onnx_path = wrappers.export_onnx(torch, wrapper, inputs, contract, opset=args.opset, dynamo=args.dynamo_exporter)
    print(f"exported refine_net: {onnx_path}")

    if args.verify:
        wrappers.verify_against_torch(torch, wrapper, inputs, onnx_path, output_keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
