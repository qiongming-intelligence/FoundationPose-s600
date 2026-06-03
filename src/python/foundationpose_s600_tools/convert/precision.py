#!/usr/bin/env python3
"""Whole-graph ONNX precision conversion (FP32 -> FP16) for FoundationPose subgraphs.

Wraps ``onnxconverter_common.float16.convert_float_to_float16`` to produce an
all-FP16 graph candidate. This is one of the precision levers worth trying for
RefineNet/ScoreNet on S600 (the outputs are tiny, so output-only FP16 is not
useful; the win, if any, comes from halving the conv/attention compute).

Always validate the result against the FP32 baseline with the accuracy gate in
docs/verification.md before shipping -- RefineNet is a continuous pose regressor
and is sensitive to small numerical drift across refine iterations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def convert_fp16(
    input_path: Path,
    output_path: Path,
    keep_io_types: bool,
    op_block_list: list[str] | None,
) -> None:
    try:
        from onnxconverter_common import float16  # type: ignore
    except ImportError as error:
        raise SystemExit(
            "whole-graph FP16 conversion needs onnxconverter_common "
            "(pip install onnxconverter-common); run on the export host"
        ) from error

    model = onnx.load(input_path, load_external_data=True)
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        op_block_list=op_block_list or None,
        disable_shape_infer=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        converted,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.checker.check_model(output_path)
    print(f"wrote whole-graph FP16 ONNX: {output_path} (keep_io_types={keep_io_types})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--keep-io-types",
        action="store_true",
        help="keep graph inputs/outputs FP32 (cast at the boundary); recommended for stable C++ ABI",
    )
    parser.add_argument(
        "--op-block-list",
        default="",
        help="comma-separated op types to leave in FP32 (e.g. 'Softmax,LayerNormalization')",
    )
    args = parser.parse_args()
    block = [op.strip() for op in args.op_block_list.split(",") if op.strip()]
    convert_fp16(args.input, args.output, args.keep_io_types, block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
