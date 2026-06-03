#!/usr/bin/env python3
"""Cast selected ONNX graph outputs to fp16/bf16 without changing graph internals.

Keeps the internal compute graph FP32 (avoids HMCT failures in softmax/clip
paths during HBM compile) and only changes the dtype of named graph outputs to
reduce output bandwidth.

Note for FoundationPose: unlike SAM3's large mask logits, the RefineNet/ScoreNet
outputs (``trans`` 3, ``rot`` 3/6, ``score_logit`` L) are tiny, so output-FP16
buys almost nothing here. This tool is kept for parity and experimentation; the
meaningful precision levers for these subgraphs are whole-graph FP16/BF16
(:mod:`foundationpose_s600_tools.convert.precision`) and INT8 PTQ via the
calibration set -- both of which must pass the accuracy gate in docs/verification.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

TARGET_TYPES = {
    "fp16": TensorProto.FLOAT16,
    "bf16": TensorProto.BFLOAT16,
    "fp32": TensorProto.FLOAT,
}


def _replace_node_output(model: onnx.ModelProto, old: str, new: str) -> bool:
    replaced = False
    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = new
                replaced = True
    return replaced


def _set_output_elem_type(model: onnx.ModelProto, name: str, elem_type: int) -> None:
    for output in model.graph.output:
        if output.name == name:
            output.type.tensor_type.elem_type = elem_type
            return
    raise ValueError(f"graph output not found: {name}")


def cast_outputs(input_path: Path, output_path: Path, outputs: list[str], precision: str) -> None:
    target = TARGET_TYPES[precision]
    model = onnx.load(input_path, load_external_data=True)
    graph_output_names = {output.name for output in model.graph.output}

    missing = [name for name in outputs if name not in graph_output_names]
    if missing:
        raise SystemExit(f"outputs not found in graph: {', '.join(missing)}")

    existing_node_names = {node.name for node in model.graph.node}
    for name in outputs:
        pre_name = f"{name}__pre_{precision}"
        if pre_name in graph_output_names:
            raise SystemExit(f"intermediate name already exists as graph output: {pre_name}")
        if not _replace_node_output(model, name, pre_name):
            raise SystemExit(f"no producer node found for graph output: {name}")

        cast_name = f"Cast_{name}_to_{precision}"
        suffix = 0
        unique_cast_name = cast_name
        while unique_cast_name in existing_node_names:
            suffix += 1
            unique_cast_name = f"{cast_name}_{suffix}"
        existing_node_names.add(unique_cast_name)

        model.graph.node.append(
            helper.make_node(
                "Cast",
                inputs=[pre_name],
                outputs=[name],
                name=unique_cast_name,
                to=target,
            )
        )
        _set_output_elem_type(model, name, target)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.checker.check_model(output_path)
    print(f"wrote output-cast ONNX: {output_path} ({precision}: {', '.join(outputs)})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--precision", choices=sorted(TARGET_TYPES), default="fp16")
    parser.add_argument(
        "--outputs",
        required=True,
        help="comma-separated graph outputs to cast (e.g. 'score_logit' or 'trans,rot')",
    )
    args = parser.parse_args()
    outputs = [name.strip() for name in args.outputs.split(",") if name.strip()]
    if not outputs:
        raise SystemExit("--outputs must name at least one graph output")
    cast_outputs(args.input, args.output, outputs, args.precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
