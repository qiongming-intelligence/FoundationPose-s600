#!/usr/bin/env python3
"""Prepare hb_compile calibration directories from captured FoundationPose tensors.

The D-Robotics hb_compile YAML expects one calibration directory per model input,
provided as a semicolon-separated ``cal_data_dir`` value. This helper validates a
capture tree against the partition contracts and copies (or symlinks) tensors into
that layout:

    <out_root>/<partition>/<input_name>/000000.npy
    <out_root>/<partition>/<input_name>/000001.npy

Input captures can already be in the same partition/input layout. Files are not
committed; see ``configs/calibration/README.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

TENSOR_SUFFIXES = {".npy", ".bin"}


def load_contracts(contracts_dir: Path) -> list[dict]:
    contracts: list[dict] = []
    for path in sorted(contracts_dir.glob("*.json")):
        if path.name == "export_index.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "inputs" in data and "name" in data:
            contracts.append(data)
    if not contracts:
        raise SystemExit(f"no partition contracts found in: {contracts_dir}")
    return contracts


def tensor_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in TENSOR_SUFFIXES)


def install_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:  # pragma: no cover - argparse choices prevents this
        raise ValueError(mode)


def prepare_partition(contract: dict, source_root: Path, out_root: Path, mode: str, limit: int | None) -> dict:
    name = contract["name"]
    part_report: dict = {"inputs": {}, "cal_data_dir": []}
    for input_spec in contract["inputs"]:
        input_name = input_spec["name"]
        src_dir = source_root / name / input_name
        files = tensor_files(src_dir)
        if not files:
            # Also accept a single-batch smoke layout: <source>/<partition>/A.bin
            # or A.npy. Real calibration captures should prefer the directory form.
            part_dir = source_root / name
            files = sorted(
                p for p in (part_dir / f"{input_name}.npy", part_dir / f"{input_name}.bin") if p.is_file()
            )
        if not files:
            raise SystemExit(
                f"missing captured tensors for {name}/{input_name}: expected .npy/.bin files under "
                f"{src_dir} (or a single {source_root / name / (input_name + '.npy')})"
            )
        if limit is not None:
            files = files[:limit]
        dst_dir = out_root / name / input_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        # Remove stale tensor files for deterministic cal_data_dir contents.
        for stale in tensor_files(dst_dir):
            stale.unlink()
        for index, src in enumerate(files):
            dst = dst_dir / f"{index:06d}{src.suffix.lower()}"
            install_file(src, dst, mode)
        part_report["inputs"][input_name] = {
            "source_dir": str(src_dir),
            "output_dir": str(dst_dir),
            "count": len(files),
            "shape": input_spec.get("concrete_shape"),
            "dtype": input_spec.get("dtype"),
        }
        part_report["cal_data_dir"].append(str(dst_dir.resolve()))
    part_report["cal_data_dir"] = ";".join(part_report["cal_data_dir"])
    return part_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare FoundationPose hb_compile calibration directories.")
    parser.add_argument("--contracts-dir", type=Path, default=Path("build/foundationpose_export/contracts"))
    parser.add_argument("--source-root", type=Path, required=True, help="capture tree: <partition>/<input_name>/*.npy|*.bin")
    parser.add_argument("--out-root", type=Path, default=Path("configs/calibration/data"))
    parser.add_argument("--partition", action="append", help="only prepare this partition; repeatable")
    parser.add_argument("--mode", choices=("copy", "symlink"), default="symlink")
    parser.add_argument("--limit", type=int, default=None, help="max files per input, useful for smoke tests")
    args = parser.parse_args()

    contracts = load_contracts(args.contracts_dir)
    wanted = set(args.partition or [])
    report: dict = {"source_root": str(args.source_root), "out_root": str(args.out_root), "partitions": {}}
    for contract in contracts:
        if wanted and contract["name"] not in wanted:
            continue
        report["partitions"][contract["name"]] = prepare_partition(
            contract, args.source_root, args.out_root, args.mode, args.limit
        )
    if wanted:
        missing = wanted - set(report["partitions"])
        if missing:
            raise SystemExit(f"requested partitions not found in contracts: {sorted(missing)}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = args.out_root / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
