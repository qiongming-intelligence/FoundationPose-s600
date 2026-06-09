#!/usr/bin/env python3
"""Persistent C++/UCP BPU runner wrapper.

This module mirrors the small ``HrtModelExecRunner.infer(...)`` API used by the
validation adapters, but keeps a native ``foundationpose_bpu_runner`` subprocess
alive so HBMs are loaded once per process instead of once per inference call.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from foundationpose_s600_tools.runtime.hrt import PartitionContract, TensorSpec, as_float32_nchw, load_contract


@dataclass(frozen=True)
class PersistentModelSpec:
    key: str
    hbm: Path
    partition: str


class PersistentBpuSession:
    """Long-lived subprocess hosting one or more loaded BPU HBMs."""

    def __init__(
        self,
        models: Sequence[PersistentModelSpec],
        *,
        runner_bin: str | Path | None = None,
        core_id: str = "1",
    ) -> None:
        if not models:
            raise ValueError("PersistentBpuSession requires at least one model")
        self.models = tuple(models)
        self.runner_bin = Path(runner_bin) if runner_bin is not None else default_runner_bin()
        self.core_id = str(core_id)
        self._stderr: deque[str] = deque(maxlen=200)
        self._stdout_noise: deque[str] = deque(maxlen=200)
        self._lock = threading.Lock()
        self._proc = self._start()
        atexit.register(self.close)

    def _start(self) -> subprocess.Popen[str]:
        if not self.runner_bin.is_file():
            raise FileNotFoundError(f"persistent BPU runner not found: {self.runner_bin}")
        cmd = [str(self.runner_bin), "--protocol", "jsonl", "--core-id", self.core_id]
        for spec in self.models:
            cmd.extend(["--model", f"{spec.key}={spec.hbm}:{spec.partition}"])
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True).start()
        return proc

    def _drain_stderr(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            self._stderr.append(line.rstrip("\n"))

    def stderr_tail(self) -> str:
        parts = []
        if self._stdout_noise:
            parts.append("stdout noise:\n" + "\n".join(self._stdout_noise))
        if self._stderr:
            parts.append("stderr:\n" + "\n".join(self._stderr))
        return "\n".join(parts)

    def request(self, payload: Mapping[str, object], *, timeout: float | None = None) -> dict[str, object]:
        # ``timeout`` is accepted for API clarity; readline itself is blocking. The
        # native runner is used only for board validation and each command is bounded
        # by HBRT/UCP. If this ever needs hard timeouts, wrap stdout reads in a worker.
        del timeout
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError(f"persistent BPU runner exited rc={self._proc.returncode}\n{self.stderr_tail()}")
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            self._proc.stdin.write(json.dumps(dict(payload), separators=(",", ":")) + "\n")
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"persistent BPU runner produced no response rc={self._proc.poll()}\n{self.stderr_tail()}")
                stripped = line.lstrip()
                if not stripped.startswith("{"):
                    self._stdout_noise.append(line.rstrip("\n"))
                    continue
                try:
                    response = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid persistent BPU runner response: {line!r}\n{self.stderr_tail()}") from exc
                if not response.get("ok", False):
                    raise RuntimeError(f"persistent BPU runner failed: {response.get('error', response)}\n{self.stderr_tail()}")
                return response

    def model_info(self, key: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {"cmd": "model_info"}
        if key is not None:
            payload["model"] = key
        return self.request(payload)

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            self.request({"cmd": "shutdown"})
        except Exception:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


def _scratch_parent() -> Path | None:
    """Prefer tmpfs scratch for BPU input/output files when available."""
    override = os.environ.get("FOUNDATIONPOSE_S600_BPU_SCRATCH")
    if override:
        path = Path(override)
        path.mkdir(parents=True, exist_ok=True)
        return path
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK | os.X_OK):
        return shm
    return None


class PersistentBpuModelRunner:
    """Run one loaded model in a :class:`PersistentBpuSession`."""

    def __init__(
        self,
        session: PersistentBpuSession,
        key: str,
        input_specs: Sequence[TensorSpec],
        output_specs: Sequence[TensorSpec],
        *,
        keep_tmp: bool = False,
    ) -> None:
        self.session = session
        self.key = key
        self.input_specs = tuple(input_specs)
        self.output_specs = tuple(output_specs)
        self.keep_tmp = keep_tmp
        self._io_lock = threading.Lock()
        self._tmp_root: Path | None = None
        self._output_dir: Path | None = None
        atexit.register(self.close)

    @classmethod
    def from_contract(
        cls,
        root: str | Path,
        partition: str,
        hbm: str | Path,
        *,
        key: str,
        session: PersistentBpuSession | None = None,
        runner_bin: str | Path | None = None,
        core_id: str = "1",
        keep_tmp: bool = False,
    ) -> "PersistentBpuModelRunner":
        root_path = Path(root).resolve()
        contract = load_contract(root_path, partition)
        hbm_path = resolve_hbm(root_path, hbm)
        if session is None:
            session = PersistentBpuSession(
                [PersistentModelSpec(key=key, hbm=hbm_path, partition=partition)],
                runner_bin=runner_bin,
                core_id=core_id,
            )
        return cls(session, key, contract.inputs, contract.outputs, keep_tmp=keep_tmp)

    def _ensure_scratch(self) -> tuple[Path, Path]:
        if self._tmp_root is None:
            self._tmp_root = Path(tempfile.mkdtemp(prefix=f"foundationpose_s600_persistent_{self.key}_", dir=_scratch_parent()))
            self._output_dir = self._tmp_root / "out"
            self._output_dir.mkdir(parents=True, exist_ok=True)
        assert self._output_dir is not None
        return self._tmp_root, self._output_dir

    def close(self) -> None:
        tmp_root = self._tmp_root
        self._tmp_root = None
        self._output_dir = None
        if tmp_root is not None and not self.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def infer(self, inputs: Mapping[str, object]) -> dict[str, np.ndarray]:
        missing = [spec.name for spec in self.input_specs if spec.name not in inputs]
        if missing:
            raise KeyError(f"missing HBM inputs: {missing}")
        with self._io_lock:
            tmp_root, output_dir = self._ensure_scratch()
            del tmp_root
            input_files: dict[str, str] = {}
            for spec in self.input_specs:
                arr = as_float32_nchw(inputs[spec.name], spec.shape)
                path = output_dir.parent / f"{spec.name}.bin"
                arr.tofile(path)
                input_files[spec.name] = str(path)
            response = self.session.request(
                {
                    "cmd": "infer",
                    "model": self.key,
                    "inputs": input_files,
                    "output_dir": str(output_dir),
                }
            )
            output_paths = response.get("outputs", {})
            if not isinstance(output_paths, dict):
                raise RuntimeError(f"persistent runner response missing outputs: {response}")
            outputs: dict[str, np.ndarray] = {}
            for spec in self.output_specs:
                path_value = output_paths.get(spec.name)
                if not isinstance(path_value, str):
                    raise RuntimeError(f"persistent runner missing output {spec.name}: {response}")
                arr = np.fromfile(path_value, dtype=np.float32)
                outputs[spec.name] = arr.reshape(spec.shape).copy()
            return outputs


def resolve_hbm(root: Path, hbm: str | Path) -> Path:
    path = Path(hbm)
    return path if path.is_absolute() else root / path


def default_runner_bin() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / "build" / "cmake" / "src" / "csrc" / "foundationpose_bpu_runner",
        repo_root / "build" / "src" / "csrc" / "foundationpose_bpu_runner",
        repo_root / "build" / "foundationpose_bpu_runner",
        repo_root / "cmake-build-debug" / "src" / "csrc" / "foundationpose_bpu_runner",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def make_session_from_contracts(
    root: str | Path,
    model_specs: Mapping[str, tuple[str | Path, str]],
    *,
    runner_bin: str | Path | None = None,
    core_id: str = "1",
) -> PersistentBpuSession:
    root_path = Path(root).resolve()
    specs = [PersistentModelSpec(key=key, hbm=resolve_hbm(root_path, hbm), partition=partition) for key, (hbm, partition) in model_specs.items()]
    return PersistentBpuSession(specs, runner_bin=runner_bin, core_id=core_id)


__all__ = [
    "PersistentBpuModelRunner",
    "PersistentBpuSession",
    "PersistentModelSpec",
    "default_runner_bin",
    "make_session_from_contracts",
]
