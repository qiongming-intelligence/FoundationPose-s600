#!/usr/bin/env python3
"""Two-host real-tensor accuracy gate for FoundationPose S600 HBM.

Unlike :mod:`validate_hbm_accuracy` (synthetic deterministic tensors, single
host), this validates an HBM against the float ONNX golden over the *real*
captured FoundationPose A/B tensors -- the same distribution used for
calibration. It is split into two subcommands because the two capabilities live
on different machines:

  * ``golden`` runs on the x86 capture/compile host (has onnxruntime + ONNX).
    It reads real A/B .npy pairs, runs the float ONNX, and writes a portable
    "eval pack": per-sample input .bin files (NCHW float32, ready for
    hrt_model_exec) plus the float golden outputs as .npy.

  * ``compare`` runs on the S600 board (has /usr/hobot/bin/hrt_model_exec, no
    onnxruntime). It reads the eval pack, runs the HBM per sample, and reports
    the same regression/ranking metrics as validate_hbm_accuracy, but against
    the real-tensor golden.

Layout of an eval pack (one dir per partition)::

    <pack>/meta.json                      # partition, output specs, sample count
    <pack>/sample_000000/A.bin            # float32 NCHW, one file per input
    <pack>/sample_000000/B.bin
    <pack>/sample_000000/golden_trans.npy # float ONNX reference, one per output
    <pack>/sample_000000/golden_rot.npy
    ...

Gates (see docs/verification.md):
  refine_net  -> per-output trans/rot max_abs + l2 vs float ONNX
  score_net_* -> top1/top5/top10 + spearman + rho of score_logit ranking
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from foundationpose_s600_tools.runtime.persistent_bpu import PersistentBpuModelRunner


def load_contract(root: Path, partition: str) -> dict[str, Any]:
    path = root / "build/foundationpose_export/contracts" / f"{partition}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def real_pairs(source_root: Path, partition: str, inputs: list[dict]) -> list[dict[str, Path]]:
    """Collect aligned real .npy tensors per input, indexed by sample.

    Expects <source_root>/<partition>/<input_name>/NNNNNN.npy with the same
    file stems across all inputs (the dump_intermediates / prepare_calibration
    layout). Samples are the intersection of stems present for every input.
    """
    per_input: dict[str, dict[str, Path]] = {}
    for spec in inputs:
        name = spec["name"]
        d = source_root / partition / name
        files = sorted(d.glob("*.npy"))
        if not files:
            raise SystemExit(f"no real tensors for input {name}: {d}")
        per_input[name] = {f.stem: f for f in files}
    common = set.intersection(*[set(m.keys()) for m in per_input.values()])
    if not common:
        raise SystemExit(f"no common sample stems across inputs for {partition}")
    return [{name: per_input[name][stem] for name in per_input} for stem in sorted(common)]


def cmd_golden(args: argparse.Namespace) -> int:
    import onnxruntime as ort

    root = args.root.resolve()
    contract = load_contract(root, args.partition)
    inputs = contract["inputs"]
    out_names = [o["name"] for o in contract["outputs"]]
    pairs = real_pairs(args.source_root.resolve(), args.partition, inputs)
    if args.limit:
        pairs = pairs[: args.limit]

    sess = ort.InferenceSession(str(root / contract["onnx_path"]), providers=["CPUExecutionProvider"])
    pack = (args.pack if args.pack.is_absolute() else root / args.pack) / args.partition
    shutil.rmtree(pack, ignore_errors=True)
    pack.mkdir(parents=True, exist_ok=True)

    for i, pair in enumerate(pairs):
        feeds = {}
        for spec in inputs:
            shape = tuple(int(x) for x in spec["concrete_shape"])
            arr = np.load(pair[spec["name"]]).astype(np.float32, copy=False)
            if arr.shape != shape:
                raise SystemExit(f"sample {i} input {spec['name']} shape {arr.shape} != contract {shape}")
            feeds[spec["name"]] = arr
        refs = sess.run(out_names, feeds)
        sdir = pack / f"sample_{i:06d}"
        sdir.mkdir(parents=True, exist_ok=True)
        for spec in inputs:
            feeds[spec["name"]].tofile(sdir / f"{spec['name']}.bin")
        for name, ref in zip(out_names, refs):
            np.save(sdir / f"golden_{name}.npy", np.asarray(ref, dtype=np.float32))

    meta = {
        "partition": args.partition,
        "samples": len(pairs),
        "inputs": [{"name": s["name"], "shape": [int(x) for x in s["concrete_shape"]]} for s in inputs],
        "outputs": [{"name": o["name"], "shape": [int(x) for x in o["concrete_shape"]]} for o in contract["outputs"]],
        "source_root": str(args.source_root),
        "onnx_path": contract["onnx_path"],
    }
    (pack / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote eval pack: {pack} ({len(pairs)} real samples)")
    print(f"copy to board, then: validate_hbm_real.py compare --pack {args.pack} --partition {args.partition} --hbm <hbm>")
    return 0


def run_hbm(hbm: Path, sdir: Path, input_names: list[str], n_out: int, core_id: str) -> list[np.ndarray]:
    out_dir = sdir / "hbm"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_files = [str(sdir / f"{name}.bin") for name in input_names]
    cmd = [
        "/usr/hobot/bin/hrt_model_exec", "infer",
        "--model_file", str(hbm),
        "--core_id", core_id,
        "--frame_count", "1",
        "--input_file", ",".join(input_files),
        "--enable_dump", "true",
        "--dump_format", "bin",
        "--dequantize_process", "true",
        "--remove_padding_process", "true",
        "--dump_path", str(out_dir),
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    if result.returncode != 0 or "failed" in result.stdout.lower():
        lines = [x for x in result.stdout.splitlines() if any(s in x.lower() for s in ["infer time", "failed", "error", "load"])]
        raise RuntimeError(f"hrt_model_exec failed for {hbm}:\n" + "\n".join(lines))
    outputs = []
    for idx in range(n_out):
        files = sorted(out_dir.glob(f"model_infer_output_{idx}_*.bin"))
        if not files:
            raise RuntimeError(f"missing output {idx} in {out_dir}")
        outputs.append(np.fromfile(files[0], dtype=np.float32))
    return outputs


def run_persistent_hbm(runner: PersistentBpuModelRunner, sdir: Path) -> list[np.ndarray]:
    feeds = {}
    for spec in runner.input_specs:
        feeds[spec.name] = np.fromfile(sdir / f"{spec.name}.bin", dtype=np.float32).reshape(spec.shape)
    got = runner.infer(feeds)
    return [got[spec.name].reshape(-1) for spec in runner.output_specs]


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
        "max_abs": float(np.max(np.abs(got - ref))),
        "mean_abs": float(np.mean(np.abs(got - ref))),
    }


def cmd_compare(args: argparse.Namespace) -> int:
    pack = (args.pack if args.pack.is_absolute() else Path.cwd() / args.pack) / args.partition
    meta = json.loads((pack / "meta.json").read_text(encoding="utf-8"))
    input_names = [s["name"] for s in meta["inputs"]]
    out_names = [o["name"] for o in meta["outputs"]]
    hbm = args.hbm if args.hbm.is_absolute() else Path.cwd() / args.hbm
    runner = None
    if args.backend == "persistent":
        runner = PersistentBpuModelRunner.from_contract(
            Path.cwd(),
            args.partition,
            hbm,
            key=args.partition,
            runner_bin=args.bpu_runner_bin,
            core_id=args.core_id,
        )

    n = meta["samples"] if not args.limit else min(args.limit, meta["samples"])
    records = []
    try:
        for i in range(n):
            sdir = pack / f"sample_{i:06d}"
            gots = run_persistent_hbm(runner, sdir) if runner is not None else run_hbm(hbm, sdir, input_names, len(out_names), args.core_id)
            if args.partition.startswith("refine_net"):
                rec: dict[str, Any] = {"sample": i}
                for name, got in zip(out_names, gots):
                    ref = np.load(sdir / f"golden_{name}.npy").reshape(-1)
                    got = got.reshape(-1)
                    diff = got - ref
                    rec[name] = {"max_abs": float(np.max(np.abs(diff))), "l2": float(np.linalg.norm(diff)),
                                 "ref": ref.tolist(), "got": got.tolist()}
            else:
                ref = np.load(sdir / f"golden_{out_names[0]}.npy")
                rec = {"sample": i, "score_logit": rank_metrics(ref, gots[0])}
            records.append(rec)
            print(json.dumps(rec, ensure_ascii=False, sort_keys=True))
    finally:
        if runner is not None:
            runner.session.close()

    if args.partition.startswith("refine_net"):
        for name in out_names:
            ma = [r[name]["max_abs"] for r in records]
            l2 = [r[name]["l2"] for r in records]
            print(f"SUMMARY {name} max_abs_max={max(ma):.6g} max_abs_mean={np.mean(ma):.6g} "
                  f"l2_max={max(l2):.6g} l2_mean={np.mean(l2):.6g}")
    else:
        ms = [r["score_logit"] for r in records]
        print("SUMMARY score_logit "
              f"top1={sum(m['top1'] for m in ms)}/{len(ms)} "
              f"top5_mean={np.mean([m['top5'] for m in ms]):.3f} "
              f"top10_mean={np.mean([m['top10'] for m in ms]):.3f} "
              f"spearman_mean={np.mean([m['spearman'] for m in ms]):.6g} "
              f"rho_mean={np.mean([m['rho'] for m in ms]):.6g} "
              f"max_abs_max={max(m['max_abs'] for m in ms):.6g}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("golden", help="x86 host: build eval pack from real tensors via float ONNX")
    g.add_argument("--root", type=Path, default=Path.cwd())
    g.add_argument("--partition", required=True, help="partition name, e.g. refine_net or score_net_L20 (L16 fallback; L24+ diagnostic)")
    g.add_argument("--source-root", type=Path, required=True, help="real capture root, e.g. configs/calibration/data/raw_capture_driller")
    g.add_argument("--pack", type=Path, default=Path("build/foundationpose_export/eval_pack"))
    g.add_argument("--limit", type=int, default=0)
    g.set_defaults(func=cmd_golden)

    c = sub.add_parser("compare", help="S600 board: run HBM over eval pack, compare to golden")
    c.add_argument("--partition", required=True, help="partition name, e.g. refine_net or score_net_L20 (L16 fallback; L24+ diagnostic)")
    c.add_argument("--pack", type=Path, default=Path("build/foundationpose_export/eval_pack"))
    c.add_argument("--hbm", type=Path, required=True)
    c.add_argument("--core-id", default="1")
    c.add_argument("--backend", choices=["hrt", "persistent"], default="hrt", help="Runtime backend for board compare; persistent loads the HBM once via foundationpose_bpu_runner")
    c.add_argument("--bpu-runner-bin", type=Path, default=None, help="Path to foundationpose_bpu_runner for --backend persistent")
    c.add_argument("--limit", type=int, default=0)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
