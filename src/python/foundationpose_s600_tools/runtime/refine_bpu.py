#!/usr/bin/env python3
"""RefineNet BPU adapter for Python hybrid validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from foundationpose_s600_tools.runtime.hrt import HrtModelExecRunner, as_float32_nchw, maybe_torch_output
from foundationpose_s600_tools.runtime.persistent_bpu import PersistentBpuModelRunner, PersistentBpuSession

BpuBackend = Literal["hrt", "persistent"]


class RefineNetBpu:
    """Callable RefineNet replacement returning ``{'trans': ..., 'rot': ...}``.

    The deployable RefineNet contract is currently fixed at batch N=1. If callers
    pass N>1, this adapter runs the HBM once per sample and concatenates outputs.
    Input tail shape and output widths are read from the selected contract, so
    variants such as ``rot_dim=6`` remain compatible.

    With ``backend='hrt'`` every sample goes through ``hrt_model_exec`` and
    reloads the HBM; with ``backend='persistent'`` the same per-sample loop reuses
    a native C++/UCP runner process that has loaded the HBM once.

    Inputs may be numpy arrays or torch tensors. Torch inputs produce torch
    outputs on the same device to match upstream predictor expectations.
    """

    def __init__(
        self,
        hbm: str | Path = "models/hbm_real_int16/foundationpose_refine_net_real_int16_no_output.hbm",
        *,
        root: str | Path = ".",
        partition: str = "refine_net",
        core_id: str = "1",
        backend: BpuBackend = "hrt",
        runner: Any | None = None,
        persistent_session: PersistentBpuSession | None = None,
        runner_bin: str | Path | None = None,
        remote: str | None = None,
        remote_root: str | Path | None = None,
        remote_tmp: str | Path = "/tmp",
        host_alias: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.partition = partition
        self.backend = backend
        if runner is not None:
            self.runner = runner
        elif backend == "hrt":
            self.runner = HrtModelExecRunner.from_contract(
                self.root,
                partition,
                hbm,
                core_id=core_id,
                remote=remote,
                remote_root=remote_root,
                remote_tmp=remote_tmp,
                host_alias=host_alias,
            )
        elif backend == "persistent":
            if remote is not None or remote_root is not None or host_alias is not None:
                raise ValueError("persistent BPU backend is local-only; use backend='hrt' for remote SSH validation")
            self.runner = PersistentBpuModelRunner.from_contract(
                self.root,
                partition,
                hbm,
                key="refine",
                session=persistent_session,
                runner_bin=runner_bin,
                core_id=core_id,
            )
        else:
            raise ValueError(f"unsupported BPU backend: {backend}")
        self._load_contract_specs()

    @staticmethod
    def _named_specs(specs: Any, kind: str) -> dict[str, Any]:
        out = {str(spec.name): spec for spec in specs}
        if not out:
            raise ValueError(f"RefineNet runner exposes no {kind} specs")
        return out

    def _load_contract_specs(self) -> None:
        input_specs = self._named_specs(getattr(self.runner, "input_specs", ()), "input")
        output_specs = self._named_specs(getattr(self.runner, "output_specs", ()), "output")
        missing_inputs = [name for name in ("A", "B") if name not in input_specs]
        missing_outputs = [name for name in ("trans", "rot") if name not in output_specs]
        if missing_inputs:
            raise ValueError(f"RefineNet contract missing input spec(s): {missing_inputs}")
        if missing_outputs:
            raise ValueError(f"RefineNet contract missing output spec(s): {missing_outputs}")
        a_shape = tuple(int(x) for x in input_specs["A"].shape)
        b_shape = tuple(int(x) for x in input_specs["B"].shape)
        if a_shape != b_shape:
            raise ValueError(f"RefineNet A/B contract shapes differ: {a_shape} != {b_shape}")
        if len(a_shape) < 2:
            raise ValueError(f"RefineNet input contract shape must include N and tail dims, got {a_shape}")
        self.contract_input_shape = a_shape
        self.contract_n = int(a_shape[0])
        self.contract_tail_shape = a_shape[1:]
        if self.contract_n != 1:
            raise ValueError(f"RefineNetBpu expects an N=1 compiled contract for per-sample looping; {self.partition} has N={self.contract_n}")
        self.trans_shape = tuple(int(x) for x in output_specs["trans"].shape)
        self.rot_shape = tuple(int(x) for x in output_specs["rot"].shape)
        if not self.trans_shape or self.trans_shape[0] != 1:
            raise ValueError(f"RefineNet trans output must have batch 1 for per-sample looping, got {self.trans_shape}")
        if not self.rot_shape or self.rot_shape[0] != 1:
            raise ValueError(f"RefineNet rot output must have batch 1 for per-sample looping, got {self.rot_shape}")

    def _validate_inputs(self, A: object, B: object) -> tuple[np.ndarray, np.ndarray]:
        a = as_float32_nchw(A)
        b = as_float32_nchw(B)
        if a.shape != b.shape:
            raise ValueError(f"A shape {a.shape} != B shape {b.shape}")
        if a.ndim != len(self.contract_input_shape):
            raise ValueError(f"expected A/B rank {len(self.contract_input_shape)} from {self.partition}, got {a.ndim}")
        if tuple(a.shape[1:]) != self.contract_tail_shape:
            raise ValueError(f"expected A/B tail shape {self.contract_tail_shape} from {self.partition}, got {tuple(a.shape[1:])}")
        if int(a.shape[0]) <= 0:
            raise ValueError("RefineNet input batch is empty")
        return a, b

    def eval(self) -> "RefineNetBpu":
        return self

    def train(self, mode: bool = True) -> "RefineNetBpu":  # API compatibility; no-op
        return self

    def cuda(self, *args: object, **kwargs: object) -> "RefineNetBpu":  # API compatibility; no-op
        return self

    def to(self, *args: object, **kwargs: object) -> "RefineNetBpu":  # API compatibility; no-op
        return self

    def predict_numpy(self, A: object, B: object) -> dict[str, np.ndarray]:
        a, b = self._validate_inputs(A, B)
        outs: dict[str, list[np.ndarray]] = {"trans": [], "rot": []}
        trans_tail = self.trans_shape[1:]
        rot_tail = self.rot_shape[1:]
        for i in range(int(a.shape[0])):
            got = self.runner.infer({"A": a[i : i + 1], "B": b[i : i + 1]})
            outs["trans"].append(np.asarray(got["trans"], dtype=np.float32).reshape((1, *trans_tail)))
            outs["rot"].append(np.asarray(got["rot"], dtype=np.float32).reshape((1, *rot_tail)))
        return {name: np.concatenate(parts, axis=0).astype(np.float32, copy=False) for name, parts in outs.items()}

    def __call__(self, A: object, B: object) -> dict[str, object]:
        return maybe_torch_output(A, self.predict_numpy(A, B))

    forward = __call__
