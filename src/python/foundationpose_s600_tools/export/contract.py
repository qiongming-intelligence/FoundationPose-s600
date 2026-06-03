#!/usr/bin/env python3
"""FoundationPose → S600 export contract generation.

A contract is the single source of truth that keeps the
PyTorch -> ONNX -> HBM -> C++ runtime chain consistent: stable tensor names,
concrete shapes, and dtypes. It is pure Python (no torch), so it runs anywhere
-- including the aarch64 S600 box -- while the torch-dependent ONNX export and
the x86-only ``hb_compile`` step consume the contract downstream.

The two FoundationPose learning subgraphs that are BPU candidates:

* ``refine_net``   -- ``RefineNet.forward(A, B)``        (learning/models/refine_network.py)
* ``score_net_L*`` -- ``ScoreNetMultiPair.forward(A, B, L)`` (learning/models/score_network.py)

Both consume a 6-channel crop pair (RGB 3 + XYZ map 3) at ``input_resize`` =
160x160 for the shipped checkpoints. ``c_in``/``rot_rep``/``input_resize`` are
read from ``weights/{run_name}/config.yml`` at export time and can be overridden
here so the contract matches whatever checkpoint is actually being exported.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TensorSpec:
    """One named tensor on a partition boundary.

    ``shape`` entries are either integers or symbolic tokens (``"N"``, ``"C"``,
    ``"H"``, ...) resolved by :func:`concrete_shape` against the dimension map.
    """

    name: str
    dtype: str
    shape: list[object]
    source: str


@dataclass(frozen=True)
class PartitionSpec:
    """A single exportable subgraph and its ONNX/HBM/runtime binding metadata."""

    name: str
    upstream_module: str
    upstream_callable: str
    onnx_name: str
    hbm_name: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    notes: list[str] = field(default_factory=list)


# Symbolic dimensions. ``L`` (pairs-per-group) is partition-specific and injected
# per build; everything else comes from the upstream config / CLI flags.
#   N  : batch of pose candidates fed to the network in one forward
#   Cin: network input channels (rgb 3 + xyz 3 = 6 for shipped ckpts; 4 = legacy)
#   H,W: crop size (input_resize), 160x160 default
#   R  : rotation representation width (axis_angle=3, 6d=6)
#   B  : score groups (outer batch); for the exported fixed-L graph B=1
#   L  : pose pairs scored together per group
def _refine_partition(rot_dim: int) -> PartitionSpec:
    return PartitionSpec(
        name="refine_net",
        upstream_module="learning.models.refine_network",
        upstream_callable="RefineNet.forward",
        onnx_name="foundationpose_refine_net.onnx",
        hbm_name="foundationpose_refine_net.hbm",
        inputs=[
            TensorSpec("A", "float32", ["N", "Cin", "H", "W"], "rendered_crop(rgb+xyz)"),
            TensorSpec("B", "float32", ["N", "Cin", "H", "W"], "observed_crop(rgb+xyz)"),
        ],
        outputs=[
            # Upstream returns a dict {'trans','rot'}; the ONNX wrapper emits these
            # as an ordered tuple with these exact names.
            TensorSpec("trans", "float32", ["N", 3], "refine_net"),
            TensorSpec("rot", "float32", ["N", "R"], "refine_net"),
        ],
        notes=[
            "trans decoded upstream via trans_rep (default 'tracknet': tanh*trans_normalizer).",
            "rot decoded via rot_rep: 'axis_angle' -> R=3 (so3_exp_map), '6d' -> R=6.",
            "Pose delta decode + SE(3) composition stay on CPU/GPU, NOT on BPU.",
        ],
    )


def _score_partition(pairs_per_group: int) -> PartitionSpec:
    return PartitionSpec(
        name=f"score_net_L{pairs_per_group}",
        upstream_module="learning.models.score_network",
        upstream_callable="ScoreNetMultiPair.forward",
        onnx_name=f"foundationpose_score_net_L{pairs_per_group}.onnx",
        hbm_name=f"foundationpose_score_net_L{pairs_per_group}.hbm",
        inputs=[
            # N == B*L. For the exported fixed-shape graph we pin B=1, so N==L.
            TensorSpec("A", "float32", ["N", "Cin", "H", "W"], "rendered_crop(rgb+xyz)"),
            TensorSpec("B", "float32", ["N", "Cin", "H", "W"], "observed_crop(rgb+xyz)"),
        ],
        outputs=[
            TensorSpec("score_logit", "float32", ["Bg", "L"], "score_net"),
        ],
        notes=[
            f"L (pairs per group) baked into the graph as L={pairs_per_group}; "
            "forward's L arg is constant-folded at export.",
            "Outer batch Bg pinned to 1 for export, so N = Bg*L = L.",
            "argmax / iterative tournament selection stay on CPU/GPU, NOT on BPU.",
        ],
    )


def build_partitions(dims: dict[str, int], score_pairs: Iterable[int]) -> dict[str, PartitionSpec]:
    parts: dict[str, PartitionSpec] = {}
    refine = _refine_partition(dims["R"])
    parts[refine.name] = refine
    for n in score_pairs:
        spec = _score_partition(n)
        parts[spec.name] = spec
    return parts


def concrete_shape(shape: list[object], dims: dict[str, int]) -> list[int]:
    """Resolve a possibly-symbolic shape to concrete integers."""
    values: list[int] = []
    for item in shape:
        if isinstance(item, int):
            values.append(item)
            continue
        if item not in dims:
            raise SystemExit(f"shape token is not a known dimension: {item!r} (known: {sorted(dims)})")
        values.append(int(dims[item]))
    return values


def tensor_contract(tensor: TensorSpec, dims: dict[str, int]) -> dict[str, object]:
    data = asdict(tensor)
    data["concrete_shape"] = concrete_shape(tensor.shape, dims)
    return data


def partition_dims(base_dims: dict[str, int], spec: PartitionSpec) -> dict[str, int]:
    """Per-partition dimension map: pins N for score_net so N == Bg*L."""
    dims = dict(base_dims)
    if spec.name.startswith("score_net_L"):
        pairs = int(spec.name.rsplit("L", 1)[1])
        dims["L"] = pairs
        dims["Bg"] = 1
        dims["N"] = dims["Bg"] * pairs
    return dims


def write_contracts(partitions: list[PartitionSpec], out_dir: Path, base_dims: dict[str, int]) -> dict[str, object]:
    contracts_dir = out_dir / "contracts"
    onnx_dir = out_dir / "onnx"
    hbm_dir = Path("models/hbm")
    contracts_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, object] = {
        "format": "foundationpose_s600_export_contract/v1",
        "status": "contract_only",
        "dimensions": base_dims,
        "partitions": [],
    }

    for spec in partitions:
        dims = partition_dims(base_dims, spec)
        contract = asdict(spec)
        contract["inputs"] = [tensor_contract(t, dims) for t in spec.inputs]
        contract["outputs"] = [tensor_contract(t, dims) for t in spec.outputs]
        contract.update(
            {
                "format": "foundationpose_s600_partition_contract/v1",
                "status": "contract_only",
                "dimensions": dims,
                "onnx_path": str(onnx_dir / spec.onnx_name),
                "hbm_path": str(hbm_dir / spec.hbm_name),
                "export_entrypoint": export_entrypoint(spec.name),
                "binding_notes": [
                    "Bind C++ runtime stages by exact tensor names above.",
                    "Re-run export after changing the checkpoint: c_in / rot_rep / "
                    "input_resize must match the loaded config.yml.",
                    "Do not commit upstream checkpoints, ONNX, or HBM files.",
                ],
            }
        )
        path = contracts_dir / f"{spec.name}.json"
        path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        index["partitions"].append(
            {
                "name": spec.name,
                "contract": str(path),
                "onnx": contract["onnx_path"],
                "hbm": contract["hbm_path"],
            }
        )

    (out_dir / "export_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def export_entrypoint(partition_name: str) -> str:
    module = "refine" if partition_name == "refine_net" else "score"
    return f"python -m foundationpose_s600_tools.export.{module}"


def print_export_plan(partitions: list[PartitionSpec], out_dir: Path, base_dims: dict[str, int]) -> None:
    for spec in partitions:
        dims = partition_dims(base_dims, spec)
        onnx_path = out_dir / "onnx" / spec.onnx_name
        hbm_path = Path("models/hbm") / spec.hbm_name
        print(f"[{spec.name}]  ({spec.upstream_callable})")
        print(f"  contract: {out_dir / 'contracts' / f'{spec.name}.json'}")
        print(f"  onnx:     {onnx_path}")
        print(f"  hbm:      {hbm_path}")
        for t in spec.inputs:
            print(f"    in  {t.name:<12} {t.dtype:<8} {concrete_shape(t.shape, dims)}")
        for t in spec.outputs:
            print(f"    out {t.name:<12} {t.dtype:<8} {concrete_shape(t.shape, dims)}")
        print(f"  export:   {spec.export_entrypoint if hasattr(spec, 'export_entrypoint') else export_entrypoint(spec.name)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FoundationPose S600 partition export contracts.")
    parser.add_argument("--out-dir", type=Path, default=Path("build/foundationpose_export"), help="export work directory")
    parser.add_argument("--partition", action="append", help="partition name to include (default: all). 'refine_net' or 'score_net_L<N>'")
    parser.add_argument("--image-size", type=int, default=160, help="crop H=W (upstream input_resize, default 160)")
    parser.add_argument("--c-in", type=int, default=6, help="network input channels: 6=rgb+xyz (shipped), 4=legacy rgb+depth")
    parser.add_argument("--rot-dim", type=int, default=3, choices=[3, 6], help="rot_rep width: 3=axis_angle (default), 6=6d")
    parser.add_argument("--score-pairs", type=int, action="append", help="L value(s) for score_net partitions (repeatable; default 16 and 64)")
    parser.add_argument("--print-plan", action="store_true", help="print the ONNX/HBM export plan with concrete shapes")
    return parser


def export_dims(args: argparse.Namespace) -> dict[str, int]:
    return {
        "Cin": args.c_in,
        "H": args.image_size,
        "W": args.image_size,
        "R": args.rot_dim,
        # N/L/Bg are partition-specific; refine_net N defaults to 1 here and is
        # re-pinned per checkpoint/batch at export time.
        "N": 1,
        "Bg": 1,
        "L": 1,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    score_pairs = args.score_pairs or [16, 64]
    base_dims = export_dims(args)
    all_parts = build_partitions(base_dims, score_pairs)

    if args.partition:
        unknown = [p for p in args.partition if p not in all_parts]
        if unknown:
            raise SystemExit(f"unknown partition(s): {', '.join(unknown)}; known: {', '.join(sorted(all_parts))}")
        selected = [all_parts[p] for p in args.partition]
    else:
        selected = list(all_parts.values())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_contracts(selected, args.out_dir, base_dims)
    if args.print_plan:
        print_export_plan(selected, args.out_dir, base_dims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
