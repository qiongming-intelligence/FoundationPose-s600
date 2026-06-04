#!/usr/bin/env python3
"""Validate FoundationPose S600 HBM outputs against ONNX references.

This is intentionally a host-side debug helper for the S600 board. It generates
multiple deterministic float32 A/B tensor pairs, runs ONNX Runtime for the
reference, runs ``hrt_model_exec infer`` for the HBM, and reports regression or
ranking metrics.

The generated tensors are *not* a substitute for real FoundationPose captured
calibration/evaluation tensors, but they prevent us from accepting a candidate
that only passes a single smoke sample.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def make_input(shape: tuple[int, ...], input_index: int, sample_index: int) -> np.ndarray:
    n = int(np.prod(shape))
    mod = 997 + sample_index % 17
    base = (np.arange(n, dtype=np.int64) * (37 + sample_index * 2) + 17 + input_index * 101 + sample_index * 9973) % mod
    arr = (base.astype(np.float32) / np.float32(mod / 2.0) - np.float32(1.0)).reshape(shape)
    arr *= np.float32(0.12 + 0.02 * (sample_index % 9))
    if len(shape) == 4:
        _, c, h, w = shape
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32).reshape(1, 1, h, 1)
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32).reshape(1, 1, 1, w)
        sx = np.float32(0.08 + 0.025 * ((sample_index * 3) % 11))
        sy = np.float32(0.08 + 0.025 * ((sample_index * 5) % 11))
        z = np.float32(0.25 + 0.06 * (sample_index % 13))
        rgb_shift = np.float32(((sample_index % 7) - 3) * 0.03)
        if c >= 1:
            arr[:, 0:1] += sx * xx + rgb_shift
        if c >= 2:
            arr[:, 1:2] += sy * yy - rgb_shift
        if c >= 3:
            arr[:, 2:3] += np.float32(((sample_index % 5) - 2) * 0.04)
        if c >= 4:
            arr[:, 3:4] += np.float32(0.20 + 0.03 * (sample_index % 5)) * xx
        if c >= 5:
            arr[:, 4:5] += np.float32(0.20 + 0.03 * ((sample_index + 2) % 5)) * yy
        if c >= 6:
            arr[:, 5:6] += z
        if shape[0] > 1:
            cand = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1)
            arr[:, 3:6] += np.float32(0.05 + 0.01 * (sample_index % 5)) * cand
            arr[:, 0:3] += np.float32(0.015) * cand
    return arr.astype(np.float32, copy=False)


def load_contract(root: Path, partition: str) -> dict[str, Any]:
    return json.loads((root / "build/foundationpose_export/contracts" / f"{partition}.json").read_text(encoding="utf-8"))


def run_onnx(root: Path, contract: dict[str, Any], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(root / contract["onnx_path"]), providers=["CPUExecutionProvider"])
    names = [o["name"] for o in contract["outputs"]]
    return [np.asarray(x, dtype=np.float32) for x in sess.run(names, feeds)]


def run_hbm(root: Path, hbm: Path, partition: str, feeds: dict[str, np.ndarray], sample_index: int, dump_root: Path, core_id: str) -> list[np.ndarray]:
    in_dir = dump_root / partition / f"sample_{sample_index:04d}" / "inputs"
    out_dir = dump_root / partition / f"sample_{sample_index:04d}" / "hbm"
    shutil.rmtree(out_dir, ignore_errors=True)
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_files = []
    for name, arr in feeds.items():
        p = in_dir / f"{name}.bin"
        arr.astype(np.float32, copy=False).tofile(p)
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
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    if result.returncode != 0 or "failed" in result.stdout.lower() or "error" in result.stdout.lower():
        lines = [x for x in result.stdout.splitlines() if any(s in x.lower() for s in ["infer time", "failed", "error", "load hbm"])]
        raise RuntimeError(f"hrt_model_exec failed for {hbm}:\n" + "\n".join(lines))
    outputs = []
    for idx, _ in enumerate(load_contract(root, partition)["outputs"]):
        files = sorted(out_dir.glob(f"model_infer_output_{idx}_*.bin"))
        if not files:
            raise RuntimeError(f"missing output {idx} in {out_dir}")
        outputs.append(np.fromfile(files[0], dtype=np.float32))
    return outputs


def rank_metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, Any]:
    ref = ref.reshape(-1)
    got = got.reshape(-1)
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
        "spearman": float(np.corrcoef(rr, gg)[0, 1]) if np.std(got) > 0 else float("nan"),
        "ref_top10": order_ref[:10],
        "got_top10": order_got[:10],
        "std_ref": float(np.std(ref)),
        "std_got": float(np.std(got)),
        "max_abs": float(np.max(np.abs(got - ref))),
        "mean_abs": float(np.mean(np.abs(got - ref))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--partition", required=True, choices=["refine_net", "score_net_L16", "score_net_L64"])
    ap.add_argument("--hbm", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--core-id", default="1")
    ap.add_argument("--dump-root", type=Path, default=Path("build/foundationpose_export/accuracy_multi"))
    args = ap.parse_args()

    root = args.root.resolve()
    contract = load_contract(root, args.partition)
    hbm = args.hbm if args.hbm.is_absolute() else root / args.hbm
    dump_root = args.dump_root if args.dump_root.is_absolute() else root / args.dump_root
    records = []
    for sample in range(args.samples):
        feeds = {}
        for input_index, spec in enumerate(contract["inputs"]):
            shape = tuple(int(x) for x in spec["concrete_shape"])
            feeds[spec["name"]] = make_input(shape, input_index, sample)
        refs = run_onnx(root, contract, feeds)
        gots = run_hbm(root, hbm, args.partition, feeds, sample, dump_root, args.core_id)
        if args.partition == "refine_net":
            rec = {"sample": sample}
            for spec, ref, got in zip(contract["outputs"], refs, gots):
                ref = ref.reshape(-1)
                got = got.reshape(-1)
                diff = got - ref
                rec[spec["name"]] = {
                    "max_abs": float(np.max(np.abs(diff))),
                    "l2": float(np.linalg.norm(diff)),
                    "ref": ref.tolist(),
                    "got": got.tolist(),
                }
        else:
            rec = {"sample": sample, "score_logit": rank_metrics(refs[0], gots[0])}
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False, sort_keys=True))
    # Aggregate compact summary.
    if args.partition == "refine_net":
        for name in [o["name"] for o in contract["outputs"]]:
            max_abs = [r[name]["max_abs"] for r in records]
            l2 = [r[name]["l2"] for r in records]
            print(f"SUMMARY {name} max_abs_max={max(max_abs):.6g} max_abs_mean={np.mean(max_abs):.6g} l2_max={max(l2):.6g} l2_mean={np.mean(l2):.6g}")
    else:
        ms = [r["score_logit"] for r in records]
        print(
            "SUMMARY score_logit "
            f"top1={sum(m['top1'] for m in ms)}/{len(ms)} "
            f"top5_mean={np.mean([m['top5'] for m in ms]):.3f} "
            f"top10_mean={np.mean([m['top10'] for m in ms]):.3f} "
            f"spearman_mean={np.mean([m['spearman'] for m in ms]):.6g} "
            f"rho_mean={np.mean([m['rho'] for m in ms]):.6g} "
            f"max_abs_max={max(m['max_abs'] for m in ms):.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
