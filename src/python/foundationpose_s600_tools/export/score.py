#!/usr/bin/env python3
"""Export FoundationPose ScoreNetMultiPair to ONNX for S600 HBM compilation.

Runs on the **x86 export host**. ``ScoreNetMultiPair.forward(A, B, L)`` takes the
pairs-per-group ``L`` as a runtime arg; for a fixed-shape BPU graph we bake ``L``
in as a constant (read from the contract name ``score_net_L<N>``) so the exported
graph has no dynamic control flow.

Example:
    PYTHONPATH=src/python python -m foundationpose_s600_tools.export.score \\
        --contract build/foundationpose_export/contracts/score_net_L20.json \\
        --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

from foundationpose_s600_tools.export import wrappers

# Upstream run_name shipping the ScoreNet checkpoint (predict_score.py).
SCORE_RUN_NAME = "2024-01-11-20-02-45"


def build_score_module(torch, root: Path, run_name: str, contract: dict, allow_random_init: bool = False):
    wrappers.add_upstream_to_path(root)
    wrappers.install_utils_shim()
    from learning.models.score_network import ScoreNetMultiPair  # type: ignore

    c_in = int(contract["dimensions"]["Cin"])
    cfg = wrappers.load_config(root, run_name, overrides={"c_in": c_in})
    model = ScoreNetMultiPair(cfg=cfg, c_in=c_in)
    ckpt_path = root / "weights" / run_name / "model_best.pth"
    if ckpt_path.is_file():
        state = wrappers.load_checkpoint_state(root, run_name)
        model.load_state_dict(state)
    elif allow_random_init:
        print(f"WARNING: {ckpt_path} missing; exporting randomly initialized ScoreNet for ABI/export smoke only")
    else:
        raise SystemExit(f"missing upstream checkpoint: {ckpt_path} (download authorized weights first)")
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Export FoundationPose ScoreNetMultiPair to ONNX.")
    parser.add_argument("--contract", type=Path, default=Path("build/foundationpose_export/contracts/score_net_L20.json"))
    parser.add_argument("--upstream-root", type=str, default=None)
    parser.add_argument("--run-name", default=SCORE_RUN_NAME)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamo-exporter", action="store_true", help="use PyTorch's new dynamo ONNX exporter; default is legacy exporter for opset-17 HBM compatibility")
    parser.add_argument("--allow-random-init", action="store_true", help="export with random weights if model_best.pth is missing; smoke/ABI only, not deployable")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise SystemExit("ScoreNet export needs PyTorch; run on the x86 export host") from error

    root = wrappers.upstream_root(args.upstream_root)
    contract = wrappers.read_contract(args.contract)

    # L baked in from the contract dimensions (pinned in contract.partition_dims).
    pairs = int(contract["dimensions"]["L"])

    model = build_score_module(torch, root, args.run_name, contract, allow_random_init=args.allow_random_init)

    output_keys = [t["name"] for t in contract["outputs"]]  # ['score_logit']
    wrapper = wrappers.NamedTupleOutput(
        torch, model, output_keys, call_adapter=lambda m, A, B: m(A, B, L=pairs)
    )

    inputs = wrappers.dummy_inputs(torch, contract)
    onnx_path = wrappers.export_onnx(torch, wrapper, inputs, contract, opset=args.opset, dynamo=args.dynamo_exporter)
    print(f"exported {contract['name']}: {onnx_path}")

    if args.verify:
        wrappers.verify_against_torch(torch, wrapper, inputs, onnx_path, output_keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
