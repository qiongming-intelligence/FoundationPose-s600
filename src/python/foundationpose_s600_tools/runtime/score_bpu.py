#!/usr/bin/env python3
"""ScoreNet BPU adapters for fixed-L contracts.

FoundationPose upstream calls ``ScoreNetMultiPair.forward(A, B, L=len(A))``
with the whole current candidate set. The network applies attention across the
``L`` candidates, so a fixed-L ScoreNet HBM is semantically equivalent only when
the caller provides exactly that compiled candidate count.

By default this adapter runs in ``strict`` mode and rejects non-exact candidate
counts. ``smoke_pad`` mode is available for board/HBM loadability and
control-flow smoke only: it pads a shorter candidate set to the fixed HBM L,
runs one inference, and clips back to the real candidate count. That mode is not
a ScoreNet correctness gate because padded candidates participate in attention.

Earlier runs used L20 as the preferred target, and L32 is the latest strict
32-hypothesis board target. Current S600/HBRT validation is board-state
sensitive: L20 fails normal HBM parse with the HBRT 4.7.5 cross-core IOVA check
but loads with the one-core preload shim, while L32 loads both normally and with
the same preload policy. Always validate the selected HBM with
``hrt_model_exec model_info`` on the target board before treating it as
deployable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from foundationpose_s600_tools.runtime.hrt import HrtModelExecRunner, as_float32_nchw, maybe_torch_output
from foundationpose_s600_tools.runtime.persistent_bpu import PersistentBpuModelRunner, PersistentBpuSession

PadMode = Literal["repeat_last", "zero"]
ScoreMode = Literal["strict", "smoke_pad"]
BpuBackend = Literal["hrt", "persistent"]


def default_score_hbm(chunk_size: int) -> str:
    return f"models/hbm_real_int16/foundationpose_score_net_L{int(chunk_size)}_real_int16_no_output.hbm"


def default_score_partition(chunk_size: int) -> str:
    return f"score_net_L{int(chunk_size)}"


class ScoreNetBpu:
    """Callable ScoreNet replacement returning ``{'score_logit': ...}``.

    Inputs may be numpy arrays or torch tensors matching the selected contract's
    ``A``/``B`` tail shape. Outputs are returned as torch tensors on the input
    device when torch inputs are provided, otherwise numpy. ``chunk_size`` must
    match the fixed-L HBM contract, for example 20 for ``score_net_L20`` or 32
    for ``score_net_L32``.

    ``mode='strict'`` requires exactly the contract L candidates and is the only
    semantic CPU/BPU ScoreNet validation path. ``mode='smoke_pad'`` pads shorter
    candidate sets for board load/control-flow smoke only.

    ``backend='persistent'`` keeps the selected fixed-L HBM loaded in the native
    C++/UCP runner across calls; ``backend='hrt'`` preserves the old validation
    path through ``hrt_model_exec``.
    """

    def __init__(
        self,
        hbm: str | Path | None = None,
        *,
        root: str | Path = ".",
        partition: str | None = None,
        core_id: str = "1",
        chunk_size: int = 32,
        mode: ScoreMode = "strict",
        pad_mode: PadMode = "repeat_last",
        backend: BpuBackend = "hrt",
        runner: Any | None = None,
        persistent_session: PersistentBpuSession | None = None,
        runner_bin: str | Path | None = None,
        remote: str | None = None,
        remote_root: str | Path | None = None,
        remote_tmp: str | Path = "/tmp",
        host_alias: str | None = None,
    ) -> None:
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if mode not in ("strict", "smoke_pad"):
            raise ValueError(f"unsupported ScoreNet BPU mode: {mode}")
        self.mode = mode
        self.root = Path(root)
        self.partition = partition or default_score_partition(self.chunk_size)
        self.pad_mode = pad_mode
        self.backend = backend
        hbm_path = default_score_hbm(self.chunk_size) if hbm is None else hbm
        if runner is not None:
            self.runner = runner
        elif backend == "hrt":
            self.runner = HrtModelExecRunner.from_contract(
                self.root,
                self.partition,
                hbm_path,
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
                self.partition,
                hbm_path,
                key="score",
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
            raise ValueError(f"ScoreNet runner exposes no {kind} specs")
        return out

    def _load_contract_specs(self) -> None:
        input_specs = self._named_specs(getattr(self.runner, "input_specs", ()), "input")
        output_specs = self._named_specs(getattr(self.runner, "output_specs", ()), "output")
        missing_inputs = [name for name in ("A", "B") if name not in input_specs]
        if missing_inputs:
            raise ValueError(f"ScoreNet contract missing input spec(s): {missing_inputs}")
        if "score_logit" not in output_specs:
            raise ValueError("ScoreNet contract missing output spec: score_logit")
        a_shape = tuple(int(x) for x in input_specs["A"].shape)
        b_shape = tuple(int(x) for x in input_specs["B"].shape)
        if a_shape != b_shape:
            raise ValueError(f"ScoreNet A/B contract shapes differ: {a_shape} != {b_shape}")
        if len(a_shape) < 2:
            raise ValueError(f"ScoreNet input contract shape must include L and tail dims, got {a_shape}")
        self.contract_input_shape = a_shape
        self.contract_l = int(a_shape[0])
        self.contract_tail_shape = a_shape[1:]
        self.contract_output_shape = tuple(int(x) for x in output_specs["score_logit"].shape)
        if self.contract_l <= 0:
            raise ValueError(f"ScoreNet contract L must be positive, got {self.contract_l}")
        if self.chunk_size != self.contract_l:
            raise ValueError(
                f"chunk_size={self.chunk_size} does not match {self.partition} contract L={self.contract_l}"
            )

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
            raise ValueError("ScoreNet input batch is empty")
        return a, b

    def _pad_to_contract_l(self, arr: np.ndarray) -> tuple[np.ndarray, int]:
        n = int(arr.shape[0])
        if n == self.contract_l:
            return arr, n
        if n <= 0:
            raise ValueError("ScoreNet input batch is empty")
        pad = self.contract_l - n
        if pad < 0:
            raise ValueError(
                f"ScoreNet smoke_pad mode supports at most {self.contract_l} candidates for {self.partition}; got {n}"
            )
        if self.pad_mode == "repeat_last":
            pad_values = np.repeat(arr[-1:, ...], pad, axis=0)
        elif self.pad_mode == "zero":
            pad_values = np.zeros((pad, *arr.shape[1:]), dtype=np.float32)
        else:
            raise ValueError(f"unsupported pad_mode: {self.pad_mode}")
        return np.concatenate([arr, pad_values], axis=0), n

    def eval(self) -> "ScoreNetBpu":
        return self

    def train(self, mode: bool = True) -> "ScoreNetBpu":  # API compatibility; no-op
        return self

    def cuda(self, *args: object, **kwargs: object) -> "ScoreNetBpu":  # API compatibility; no-op
        return self

    def to(self, *args: object, **kwargs: object) -> "ScoreNetBpu":  # API compatibility; no-op
        return self

    def predict_numpy(self, A: object, B: object) -> np.ndarray:
        """Return score logits as shape ``(1, N)`` numpy float32."""
        a, b = self._validate_inputs(A, B)
        total = int(a.shape[0])

        if self.mode == "strict":
            if total != self.contract_l:
                raise ValueError(
                    f"ScoreNetBpu strict mode requires exactly {self.contract_l} candidates for {self.partition}; got {total}. "
                    "Use a matching fixed-L HBM for semantic validation, or mode='smoke_pad' for board smoke only."
                )
            out = self.runner.infer({"A": a, "B": b})["score_logit"]
            return np.asarray(out, dtype=np.float32).reshape(self.contract_output_shape)

        a_run, valid = self._pad_to_contract_l(a)
        b_run, _ = self._pad_to_contract_l(b)
        out = np.asarray(self.runner.infer({"A": a_run, "B": b_run})["score_logit"], dtype=np.float32).reshape(-1)
        if out.size < self.contract_l:
            raise ValueError(f"ScoreNet runner returned {out.size} logits, expected at least {self.contract_l}")
        return out[:valid].astype(np.float32, copy=False).reshape(1, valid)

    def __call__(self, A: object, B: object, L: int | None = None) -> dict[str, object]:
        if L is not None and self.mode == "strict" and int(L) != self.contract_l:
            raise ValueError(f"ScoreNetBpu strict mode requires L={self.contract_l} for {self.partition}, got L={L}")
        scores = self.predict_numpy(A, B)
        if L is not None and int(L) != scores.shape[1]:
            raise ValueError(f"caller passed L={L}, but A/B contain {scores.shape[1]} candidates")
        return maybe_torch_output(A, {"score_logit": scores})

    # Torch modules use forward(); making it an alias lets callers install this as
    # ``score_predictor.model`` in the upstream FoundationPose predictor.
    forward = __call__


class ScoreNetBpuL16(ScoreNetBpu):
    """ScoreNet adapter for the deployable L16 fallback HBM."""

    def __init__(
        self,
        hbm: str | Path | None = None,
        *,
        root: str | Path = ".",
        partition: str | None = None,
        core_id: str = "1",
        chunk_size: int = 16,
        mode: ScoreMode = "strict",
        pad_mode: PadMode = "repeat_last",
        backend: BpuBackend = "hrt",
        runner: Any | None = None,
        persistent_session: PersistentBpuSession | None = None,
        runner_bin: str | Path | None = None,
        remote: str | None = None,
        remote_root: str | Path | None = None,
        remote_tmp: str | Path = "/tmp",
        host_alias: str | None = None,
    ) -> None:
        if int(chunk_size) != 16:
            raise ValueError("ScoreNetBpuL16 is compiled for chunk_size=16")
        super().__init__(
            hbm,
            root=root,
            partition=partition or "score_net_L16",
            core_id=core_id,
            chunk_size=16,
            mode=mode,
            pad_mode=pad_mode,
            backend=backend,
            runner=runner,
            persistent_session=persistent_session,
            runner_bin=runner_bin,
            remote=remote,
            remote_root=remote_root,
            remote_tmp=remote_tmp,
            host_alias=host_alias,
        )


class ScoreNetBpuL20(ScoreNetBpu):
    """ScoreNet adapter for the L20 HBM when it loads on the board."""

    def __init__(
        self,
        hbm: str | Path | None = None,
        *,
        root: str | Path = ".",
        partition: str | None = None,
        core_id: str = "1",
        chunk_size: int = 20,
        mode: ScoreMode = "strict",
        pad_mode: PadMode = "repeat_last",
        backend: BpuBackend = "hrt",
        runner: Any | None = None,
        persistent_session: PersistentBpuSession | None = None,
        runner_bin: str | Path | None = None,
        remote: str | None = None,
        remote_root: str | Path | None = None,
        remote_tmp: str | Path = "/tmp",
        host_alias: str | None = None,
    ) -> None:
        if int(chunk_size) != 20:
            raise ValueError("ScoreNetBpuL20 is compiled for chunk_size=20")
        super().__init__(
            hbm,
            root=root,
            partition=partition or "score_net_L20",
            core_id=core_id,
            chunk_size=20,
            mode=mode,
            pad_mode=pad_mode,
            backend=backend,
            runner=runner,
            persistent_session=persistent_session,
            runner_bin=runner_bin,
            remote=remote,
            remote_root=remote_root,
            remote_tmp=remote_tmp,
            host_alias=host_alias,
        )


class ScoreNetBpuL32(ScoreNetBpu):
    """ScoreNet adapter for the real-int16 L32 HBM fallback validated in E2E smoke."""

    def __init__(
        self,
        hbm: str | Path | None = None,
        *,
        root: str | Path = ".",
        partition: str | None = None,
        core_id: str = "1",
        chunk_size: int = 32,
        mode: ScoreMode = "strict",
        pad_mode: PadMode = "repeat_last",
        backend: BpuBackend = "hrt",
        runner: Any | None = None,
        persistent_session: PersistentBpuSession | None = None,
        runner_bin: str | Path | None = None,
        remote: str | None = None,
        remote_root: str | Path | None = None,
        remote_tmp: str | Path = "/tmp",
        host_alias: str | None = None,
    ) -> None:
        if int(chunk_size) != 32:
            raise ValueError("ScoreNetBpuL32 is compiled for chunk_size=32")
        super().__init__(
            hbm,
            root=root,
            partition=partition or "score_net_L32",
            core_id=core_id,
            chunk_size=32,
            mode=mode,
            pad_mode=pad_mode,
            backend=backend,
            runner=runner,
            persistent_session=persistent_session,
            runner_bin=runner_bin,
            remote=remote,
            remote_root=remote_root,
            remote_tmp=remote_tmp,
            host_alias=host_alias,
        )
