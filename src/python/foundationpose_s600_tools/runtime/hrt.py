#!/usr/bin/env python3
"""Thin Python wrapper around ``hrt_model_exec infer``.

This is intentionally simple and dependency-light: it writes NCHW float32 input
buffers to temporary ``.bin`` files, invokes the board-side HRT CLI, and reads the
dequantized / de-padded float32 output dumps back into numpy arrays.

It is suitable for validation and Python-hybrid smoke tests. It is **not** the
final low-latency runtime because each call starts a process and reloads the HBM;
production should use the planned C++/UCP runner that keeps models loaded.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass(frozen=True)
class PartitionContract:
    name: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]


def load_contract(root: Path, partition: str) -> PartitionContract:
    """Load a generated FoundationPose partition contract."""
    path = root / "build/foundationpose_export/contracts" / f"{partition}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PartitionContract(
        name=str(raw["name"]),
        inputs=tuple(TensorSpec(str(t["name"]), tuple(int(x) for x in t["concrete_shape"]), str(t.get("dtype", "float32"))) for t in raw["inputs"]),
        outputs=tuple(TensorSpec(str(t["name"]), tuple(int(x) for x in t["concrete_shape"]), str(t.get("dtype", "float32"))) for t in raw["outputs"]),
    )


def as_float32_nchw(array: object, shape: Sequence[int] | None = None) -> np.ndarray:
    """Convert numpy/torch-like tensors to contiguous float32 numpy arrays."""
    if hasattr(array, "detach") and callable(getattr(array, "detach")):
        # Torch tensor path, without importing torch at module load time.
        array = array.detach().float().cpu().numpy()
    arr = np.asarray(array, dtype=np.float32)
    if shape is not None and tuple(arr.shape) != tuple(int(x) for x in shape):
        raise ValueError(f"input shape {arr.shape} != expected {tuple(shape)}")
    return np.ascontiguousarray(arr)


def maybe_torch_output(template: object, outputs: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Return torch tensors on the template device if the input was a torch tensor."""
    if not (hasattr(template, "detach") and hasattr(template, "device")):
        return dict(outputs)
    try:
        import torch  # type: ignore
    except Exception:
        return dict(outputs)
    return {name: torch.as_tensor(value, dtype=torch.float32, device=template.device) for name, value in outputs.items()}


class HrtModelExecRunner:
    """Run one HBM via ``/usr/hobot/bin/hrt_model_exec infer``."""

    def __init__(
        self,
        hbm: str | Path,
        input_specs: Sequence[TensorSpec],
        output_specs: Sequence[TensorSpec],
        *,
        core_id: str = "1",
        hrt: str | Path = "/usr/hobot/bin/hrt_model_exec",
        keep_tmp: bool = False,
        remote: str | None = None,
        remote_root: str | Path | None = None,
        remote_tmp: str | Path = "/tmp",
        host_alias: str | None = None,
    ) -> None:
        self.hbm = Path(hbm)
        self.input_specs = tuple(input_specs)
        self.output_specs = tuple(output_specs)
        self.core_id = str(core_id)
        self.hrt = Path(hrt)
        self.keep_tmp = keep_tmp
        self.remote = remote
        self.remote_root = Path(remote_root) if remote_root is not None else None
        self.remote_tmp = Path(remote_tmp)
        self.host_alias = host_alias

    @classmethod
    def from_contract(
        cls,
        root: str | Path,
        partition: str,
        hbm: str | Path,
        *,
        core_id: str = "1",
        hrt: str | Path = "/usr/hobot/bin/hrt_model_exec",
        keep_tmp: bool = False,
        remote: str | None = None,
        remote_root: str | Path | None = None,
        remote_tmp: str | Path = "/tmp",
        host_alias: str | None = None,
    ) -> "HrtModelExecRunner":
        root_path = Path(root).resolve()
        contract = load_contract(root_path, partition)
        hbm_path = Path(hbm)
        if not hbm_path.is_absolute():
            hbm_path = root_path / hbm_path
        return cls(
            hbm_path,
            contract.inputs,
            contract.outputs,
            core_id=core_id,
            hrt=hrt,
            keep_tmp=keep_tmp,
            remote=remote,
            remote_root=remote_root,
            remote_tmp=remote_tmp,
            host_alias=host_alias,
        )

    def _remote_path(self, local_path: Path) -> Path:
        """Map a local repo path to the corresponding remote board path."""
        if self.remote_root is None:
            return local_path
        try:
            rel = local_path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            # Common case for two-host use: hbm was resolved under the x86 repo root
            # in from_contract(). Map by path suffix under FoundationPose-s600.
            parts = local_path.resolve().parts
            if "FoundationPose-s600" in parts:
                idx = parts.index("FoundationPose-s600")
                rel = Path(*parts[idx + 1 :])
            else:
                rel = Path(local_path.name)
        return self.remote_root / rel

    def _ssh(self, args: Sequence[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
        if not self.remote:
            raise RuntimeError("remote SSH target is not configured")
        remote_cmd = " ".join(shlex.quote(str(arg)) for arg in args)
        cmd = ["ssh", "-n", "-o", "BatchMode=yes", self.remote, remote_cmd]
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=timeout)

    def _copy_inputs_to_remote(self, local_paths: Sequence[Path], remote_dir: str) -> None:
        """Upload input buffers via ``ssh 'cat > dest'`` instead of scp.

        Same rationale as :meth:`_copy_outputs_from_remote`: a board login shell
        that prints non-interactive warnings can make scp report ``lost
        connection`` even when plain ssh commands succeed. Streaming bytes over
        ssh stdin sidesteps scp's remote-shell handshake.
        """
        if not self.remote:
            raise RuntimeError("remote SSH target is not configured")
        chunk_bytes = 1 << 20
        for local_path in local_paths:
            remote_path = f"{remote_dir}/{local_path.name}"
            first = True
            with open(local_path, "rb") as handle:
                while True:
                    chunk = handle.read(chunk_bytes)
                    if not chunk:
                        break
                    op = ">" if first else ">>"
                    first = False
                    cmd = ["ssh", "-T", "-o", "BatchMode=yes", self.remote, f"cat {op} {shlex.quote(remote_path)}"]
                    result = subprocess.run(cmd, input=chunk, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
                    if result.returncode != 0:
                        raise RuntimeError("remote input upload failed:\n" + result.stderr.decode("utf-8", errors="replace"))

    def _copy_outputs_from_remote(self, remote_paths: Sequence[str], local_dir: Path) -> None:
        """Fetch remote output dumps without scp's fragile remote-shell protocol.

        Some board login shells print non-interactive setup warnings (for example
        ``resize: can't open terminal``). That can break ``scp remote:file .`` even
        though ordinary ``ssh remote command`` works. Fetch each binary dump with
        ``ssh cat`` and keep stderr separate so shell warnings cannot corrupt the
        output bytes.
        """
        if not self.remote:
            raise RuntimeError("remote SSH target is not configured")
        if self.host_alias and remote_paths:
            sources = " ".join(shlex.quote(path) for path in remote_paths)
            dest = shlex.quote(self.host_alias + ":" + str(local_dir) + "/")
            push_cmd = f"rsync -a {sources} {dest}"
            result = self._ssh(["sh", "-lc", push_cmd], timeout=300)
            if result.returncode != 0:
                raise RuntimeError("remote rsync output push failed:\n" + result.stdout)
            return
        for remote_path in remote_paths:
            local_path = local_dir / Path(remote_path).name
            cmd = ["ssh", "-o", "BatchMode=yes", self.remote, "cat", remote_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, timeout=300)
            if result.returncode != 0:
                raise RuntimeError("remote output fetch failed:\n" + result.stderr.decode("utf-8", errors="replace"))
            local_path.write_bytes(result.stdout)

    def _run_local_hrt(self, input_files: Sequence[str], dump_dir: Path) -> None:
        cmd = [
            str(self.hrt),
            "infer",
            "--model_file",
            str(self.hbm),
            "--core_id",
            self.core_id,
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
            str(dump_dir),
        ]
        attempts = int(os.environ.get("FOUNDATIONPOSE_S600_HRT_RETRIES", "5"))
        attempts = max(1, attempts)
        last_result: subprocess.CompletedProcess[str] | None = None
        for attempt in range(attempts):
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            last_result = result
            lower = result.stdout.lower()
            if not ("iova addr not equal" in lower or "hbrt4_status_invalid_argument" in lower):
                self._check_hrt_result(result)
                return
            if attempt + 1 < attempts:
                time.sleep(1.0)
        assert last_result is not None
        self._check_hrt_result(last_result)

    def _run_remote_hrt(self, input_files: Sequence[Path], dump_dir: Path) -> None:
        if not self.remote:
            raise RuntimeError("remote SSH target is not configured")
        remote_tag = f"foundationpose_s600_hrt_{os.getpid()}_{next(tempfile._get_candidate_names())}"
        remote_dir = str(self.remote_tmp / remote_tag)
        mkdir = self._ssh(["mkdir", "-p", remote_dir, f"{remote_dir}/dump"])
        if mkdir.returncode != 0:
            raise RuntimeError("remote mkdir failed:\n" + mkdir.stdout)
        try:
            self._copy_inputs_to_remote(input_files, remote_dir)
            remote_inputs = [f"{remote_dir}/{p.name}" for p in input_files]
            remote_dump = f"{remote_dir}/dump"
            remote_hbm = str(self._remote_path(self.hbm))
            remote_cmd = " ".join(
                shlex.quote(x)
                for x in [
                    str(self.hrt),
                    "infer",
                    "--model_file",
                    remote_hbm,
                    "--core_id",
                    self.core_id,
                    "--frame_count",
                    "1",
                    "--input_file",
                    ",".join(remote_inputs),
                    "--enable_dump",
                    "true",
                    "--dump_format",
                    "bin",
                    "--dequantize_process",
                    "true",
                    "--remove_padding_process",
                    "true",
                    "--dump_path",
                    remote_dump,
                ]
            )
            result = self._ssh(["sh", "-lc", remote_cmd])
            self._check_hrt_result(result)
            list_cmd = f"find {shlex.quote(remote_dump)} -maxdepth 1 -type f -name 'model_infer_output_*.bin' | sort"
            listed = self._ssh(["sh", "-lc", list_cmd])
            if listed.returncode != 0:
                raise RuntimeError("remote output listing failed:\n" + listed.stdout)
            remote_outputs = [
                line.strip()
                for line in listed.stdout.splitlines()
                if line.strip().startswith(remote_dump + "/")
            ]
            if not remote_outputs:
                raise RuntimeError("remote HRT produced no output dumps:\n" + listed.stdout)
            self._copy_outputs_from_remote(remote_outputs, dump_dir)
        finally:
            if not self.keep_tmp:
                self._ssh(["rm", "-rf", remote_dir], timeout=60)

    @staticmethod
    def _check_hrt_result(result: subprocess.CompletedProcess[str]) -> None:
        lower = result.stdout.lower()
        if result.returncode != 0 or "failed" in lower or "error code" in lower:
            interesting = [
                line
                for line in result.stdout.splitlines()
                if any(token in line.lower() for token in ["infer time", "failed", "error", "load", "iova"])
            ]
            raise RuntimeError("hrt_model_exec failed:\n" + "\n".join(interesting[-80:]))

    def infer(self, inputs: Mapping[str, object]) -> dict[str, np.ndarray]:
        missing = [spec.name for spec in self.input_specs if spec.name not in inputs]
        if missing:
            raise KeyError(f"missing HBM inputs: {missing}")
        tmp_root = Path(tempfile.mkdtemp(prefix="foundationpose_s600_hrt_"))
        try:
            input_files: list[Path] = []
            for spec in self.input_specs:
                arr = as_float32_nchw(inputs[spec.name], spec.shape)
                path = tmp_root / f"{spec.name}.bin"
                arr.tofile(path)
                input_files.append(path)

            dump_dir = tmp_root / "dump"
            dump_dir.mkdir(parents=True, exist_ok=True)
            if self.remote:
                self._run_remote_hrt(input_files, dump_dir)
            else:
                self._run_local_hrt([str(p) for p in input_files], dump_dir)

            outputs: dict[str, np.ndarray] = {}
            for index, spec in enumerate(self.output_specs):
                files = sorted(dump_dir.glob(f"model_infer_output_{index}_*.bin"))
                if not files:
                    raise RuntimeError(f"missing HRT output {index} ({spec.name}) in {dump_dir}")
                arr = np.fromfile(files[0], dtype=np.float32)
                outputs[spec.name] = arr.reshape(spec.shape)
            return outputs
        finally:
            if not self.keep_tmp:
                shutil.rmtree(tmp_root, ignore_errors=True)
