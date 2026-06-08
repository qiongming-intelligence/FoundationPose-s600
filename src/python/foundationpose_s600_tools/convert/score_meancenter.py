#!/usr/bin/env python3
"""Append a rank-preserving mean-centering tail to a ScoreNet ONNX graph.

FoundationPose ScoreNet emits per-candidate ``score_logit`` whose *absolute*
values sit far from zero (≈ -136 on our checkpoint) while the values that
actually drive top-k / argsort differ by only ~0.04. When the whole graph is
quantized to int16 for the S600 BPU, the output quant step (max/32767 ≈ 4e-3 on
a ±136 range) is coarse relative to that 0.04 spread, so ranking collapses even
though every internal cosine is ~1.0.

Subtracting the per-call mean of ``score_logit`` from every element is a no-op
for argmax / top-k / Spearman (a constant shift cancels in every pairwise
comparison), but it recentres the output around 0 so the int16 output scale
drops by ~3 orders of magnitude. This is useful as a diagnostic transform for
separating output-scale collapse from internal attention/head quantization error.
It does **not** by itself prove that a full-BPU ScoreNet is deployable; validate
the final HBM with top-k/rank gates.

The transform is purely structural (Sub of a ReduceMean), inserted at the ONNX
level so we touch neither ``third_party/FoundationPose`` nor the exporter. The
output tensor keeps its name ``score_logit`` so the contract / HBM / C++ ABI is
unchanged.

Usage:
    python -m foundationpose_s600_tools.convert.score_meancenter \
        build/foundationpose_export/onnx/foundationpose_score_net_L20.onnx \
        build/foundationpose_export/onnx/foundationpose_score_net_L20_mc.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path


def mean_center_scorenet(in_path: Path, out_path: Path, output_name: str = "score_logit") -> Path:
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(in_path))
    graph = model.graph

    outs = [o for o in graph.output if o.name == output_name]
    if not outs:
        names = [o.name for o in graph.output]
        raise SystemExit(f"output {output_name!r} not found in {in_path} (have {names})")
    out_vi = outs[0]

    # Reduce over the last axis (per-call candidate axis); score_logit is [1, L].
    rank = len(out_vi.type.tensor_type.shape.dim)
    axis = rank - 1
    pre_name = f"{output_name}_preshift"

    # Rename the existing producer's output to pre_name, then Sub its row-mean.
    renamed = False
    for node in graph.node:
        for i, o in enumerate(node.output):
            if o == output_name:
                node.output[i] = pre_name
                renamed = True
    if not renamed:
        raise SystemExit(f"no node produces {output_name!r} in {in_path}")

    mean_node = helper.make_node(
        "ReduceMean",
        inputs=[pre_name],
        outputs=[f"{output_name}_mean"],
        name=f"{output_name}_ReduceMean_center",
        axes=[axis],
        keepdims=1,
    )
    sub_node = helper.make_node(
        "Sub",
        inputs=[pre_name, f"{output_name}_mean"],
        outputs=[output_name],
        name=f"{output_name}_Sub_center",
    )
    graph.node.extend([mean_node, sub_node])

    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return out_path


def _verify(in_path: Path, out_path: Path, output_name: str = "score_logit") -> None:
    import numpy as np
    import onnxruntime as ort

    s0 = ort.InferenceSession(str(in_path), providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feed = {}
    rng = np.random.default_rng(0)
    for inp in s0.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        feed[inp.name] = rng.standard_normal(shape).astype(np.float32) * 0.25
    a = s0.run([output_name], feed)[0].reshape(-1)
    b = s1.run([output_name], feed)[0].reshape(-1)
    oa, ob = np.argsort(-a), np.argsort(-b)
    print("orig range", float(a.min()), float(a.max()))
    print("centered range", float(b.min()), float(b.max()))
    print("argsort identical:", bool(np.array_equal(oa, ob)))
    print("top1 same:", bool(oa[0] == ob[0]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Append rank-preserving mean-centering tail to a ScoreNet ONNX.")
    ap.add_argument("in_onnx", type=Path)
    ap.add_argument("out_onnx", type=Path)
    ap.add_argument("--output-name", default="score_logit")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    out = mean_center_scorenet(args.in_onnx, args.out_onnx, args.output_name)
    print(f"wrote {out}")
    if args.verify:
        _verify(args.in_onnx, out, args.output_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
