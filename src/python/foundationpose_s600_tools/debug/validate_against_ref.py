#!/usr/bin/env python3
"""Validate S600 HBM output against precomputed ONNX reference bins.

This helper is intended for the S600 board where ONNX Runtime may not be
installed. Generate reference inputs/outputs on the x86 export host, sync them to
S600, then run this script locally with ``hrt_model_exec``. It compares HBM dumps
against ``build/foundationpose_export/accuracy_ref/<partition>/*.onnx.bin``.

Use :mod:`foundationpose_s600_tools.debug.validate_hbm_accuracy` when ONNX
Runtime is available and you want to generate ONNX references on the fly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def rank_metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, Any]:
    ref = ref.reshape(-1).astype(np.float64)
    got = got.reshape(-1).astype(np.float64)
    order_ref = [int(x) for x in np.argsort(-ref)]
    order_got = [int(x) for x in np.argsort(-got)]
    rr = np.empty(len(ref), dtype=np.float64)
    gg = np.empty(len(got), dtype=np.float64)
    for rank, idx in enumerate(np.argsort(ref)):
        rr[idx] = rank
    for rank, idx in enumerate(np.argsort(got)):
        gg[idx] = rank
    return {
        "top1": bool(order_ref[0] == order_got[0]),
        "top5": int(len(set(order_ref[: min(5, len(ref))]) & set(order_got[: min(5, len(ref))]))),
        "top10": int(len(set(order_ref[: min(10, len(ref))]) & set(order_got[: min(10, len(ref))]))),
        "rho": float(np.corrcoef(ref, got)[0, 1]) if np.std(ref) > 0 and np.std(got) > 0 else float("nan"),
        "spearman": float(np.corrcoef(rr, gg)[0, 1]) if np.std(ref) > 0 and np.std(got) > 0 else float("nan"),
        "ref_top10": order_ref[:10],
        "got_top10": order_got[:10],
        "std_ref": float(np.std(ref)),
        "std_got": float(np.std(got)),
        "max_abs": float(np.max(np.abs(got - ref))),
        "mean_abs": float(np.mean(np.abs(got - ref))),
    }


def read_float_bin(path: Path, shape: list[int]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise RuntimeError(f"{path} has {arr.size} float32 values, expected {expected} for shape {shape}")
    return arr.reshape(shape)


def load_contract(root: Path, partition: str) -> dict[str, Any]:
    return json.loads((root / "build/foundationpose_export/contracts" / f"{partition}.json").read_text(encoding="utf-8"))


def run_hbm(
    root: Path,
    hbm: Path,
    partition: str,
    contract: dict[str, Any],
    input_root: Path,
    dump_root: Path,
    core_id: str,
) -> list[np.ndarray]:
    in_dir = input_root / partition
    out_dir = dump_root / partition / hbm.stem
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = []
    for spec in contract["inputs"]:
        p = in_dir / f"{spec['name']}.bin"
        if not p.exists():
            raise RuntimeError(f"missing input file {p}")
        input_files.append(str(p))

    cmd = [
        "/usr/hobot/bin/hrt_model_exec",
        "infer",
        "--model_file",
        str(hbm),
        "--core_id",
        core_id,
        "--frame_count",
        "1",
        "--input_file",
        ",".join(input_files),
        "--enable_dump",
        "true",
        "--dump_format",
        "bin",
        "--dequantize_process",
        "true",
        "--remove_padding_process",
        "true",
        "--dump_path",
        str(out_dir),
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    lines = [
        x
        for x in result.stdout.splitlines()
        if any(s in x.lower() for s in ["infer time", "failed", "error", "load hbm", "model file"])
    ]
    if result.returncode != 0 or "failed" in result.stdout.lower() or "error" in result.stdout.lower():
        raise RuntimeError(f"hrt_model_exec failed for {hbm}:\n" + "\n".join(lines[-80:]))

    outs = []
    for idx, spec in enumerate(contract["outputs"]):
        files = sorted(out_dir.glob(f"model_infer_output_{idx}_*.bin"))
        if not files:
            raise RuntimeError(f"missing dumped output {idx} in {out_dir}")
        shape = [int(x) for x in spec["concrete_shape"]]
        outs.append(read_float_bin(files[0], shape))
    return outs


def validate_one(root: Path, partition: str, contract: dict[str, Any], refs: list[np.ndarray], hbm: Path, args: argparse.Namespace) -> dict[str, Any]:
    gots = run_hbm(root, hbm, partition, contract, args.input_root, args.dump_root, args.core_id)
    rec: dict[str, Any] = {"partition": partition, "hbm": _display_path(hbm, root)}
    if partition == "refine_net":
        for spec, ref, got in zip(contract["outputs"], refs, gots):
            diff = got.reshape(-1) - ref.reshape(-1)
            rec[spec["name"]] = {
                "max_abs": float(np.max(np.abs(diff))),
                "mean_abs": float(np.mean(np.abs(diff))),
                "l2": float(np.linalg.norm(diff)),
                "ref": ref.reshape(-1).tolist(),
                "got": got.reshape(-1).tolist(),
            }
    else:
        rec["score_logit"] = rank_metrics(refs[0], gots[0])
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate S600 HBM outputs against existing ONNX reference bins.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--partition", required=True, choices=["refine_net", "score_net_L16", "score_net_L64"])
    parser.add_argument("--hbm", type=Path, action="append", required=True)
    parser.add_argument("--core-id", default="1")
    parser.add_argument("--input-root", type=Path, default=Path("build/foundationpose_export/accuracy_inputs"))
    parser.add_argument("--ref-root", type=Path, default=Path("build/foundationpose_export/accuracy_ref"))
    parser.add_argument("--dump-root", type=Path, default=Path("build/foundationpose_export/accuracy_quick_against_ref"))
    args = parser.parse_args()

    root = args.root.resolve()
    args.input_root = args.input_root if args.input_root.is_absolute() else root / args.input_root
    args.ref_root = args.ref_root if args.ref_root.is_absolute() else root / args.ref_root
    args.dump_root = args.dump_root if args.dump_root.is_absolute() else root / args.dump_root

    contract = load_contract(root, args.partition)
    refs = []
    for spec in contract["outputs"]:
        ref_file = args.ref_root / args.partition / f"{spec['name']}.onnx.bin"
        refs.append(read_float_bin(ref_file, [int(x) for x in spec["concrete_shape"]]))

    records = []
    for hbm_arg in args.hbm:
        hbm = hbm_arg if hbm_arg.is_absolute() else root / hbm_arg
        rec = validate_one(root, args.partition, contract, refs, hbm, args)
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False, sort_keys=True))

    out_json = args.dump_root / f"{args.partition}_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
