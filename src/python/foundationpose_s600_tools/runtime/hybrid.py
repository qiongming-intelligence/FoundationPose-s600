#!/usr/bin/env python3
"""Helpers for installing BPU adapters into upstream FoundationPose predictors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from foundationpose_s600_tools.runtime.persistent_bpu import PersistentBpuSession, make_session_from_contracts
from foundationpose_s600_tools.runtime.refine_bpu import RefineNetBpu
from foundationpose_s600_tools.runtime.score_bpu import ScoreNetBpu, default_score_hbm, default_score_partition

ScoreMode = Literal["strict", "smoke_pad"]
ScorePadMode = Literal["repeat_last", "zero"]
BpuBackend = Literal["hrt", "persistent"]


def install_bpu_adapters(
    refine_predictor: Any | None = None,
    score_predictor: Any | None = None,
    *,
    root: str | Path = ".",
    refine_hbm: str | Path = "models/hbm_real_int16/foundationpose_refine_net_real_int16_no_output.hbm",
    refine_partition: str = "refine_net",
    score_hbm: str | Path | None = None,
    score_partition: str | None = None,
    score_chunk_size: int = 32,
    score_mode: ScoreMode = "strict",
    score_pad_mode: ScorePadMode = "repeat_last",
    core_id: str = "1",
    bpu_backend: BpuBackend = "hrt",
    bpu_runner_bin: str | Path | None = None,
    persistent_session: PersistentBpuSession | None = None,
    remote: str | None = None,
    remote_root: str | Path | None = None,
    remote_tmp: str | Path = "/tmp",
    host_alias: str | None = None,
) -> dict[str, Any]:
    """Replace predictor ``.model`` attributes with BPU-backed adapters.

    Call this after constructing upstream ``PoseRefinePredictor`` and
    ``ScorePredictor`` instances (or after constructing ``FoundationPose`` if it
    exposes ``est.refiner`` / ``est.scorer``). The adapters preserve the upstream
    dict outputs (``trans``/``rot`` and ``score_logit``) and return torch tensors
    when called with torch tensors.

    ``score_chunk_size`` selects the fixed-size ScoreNet contract/HBM. L32 is the
    current strict Score BPU target on this S600/HBRT state, while L20 needs the
    one-core preload shim on HBRT 4.7.5; validate the selected L value with
    ``hrt_model_exec model_info`` using the deployment preload policy. ``score_mode='strict'`` requires the caller's
    candidate count to exactly match that fixed L and is the semantic validation
    path. ``score_mode='smoke_pad'`` pads a shorter candidate set for board
    load/control-flow smoke only; it is not equivalent to upstream ScoreNet
    attention over the unpadded candidate set.

    ``bpu_backend='persistent'`` starts one native C++/UCP JSONL runner and loads
    all selected HBMs once. This directly avoids the per-inference HBM reload done
    by the ``hrt_model_exec`` validation backend.
    """
    installed: dict[str, Any] = {}
    chunk_size = int(score_chunk_size)
    resolved_score_hbm = score_hbm or default_score_hbm(chunk_size)
    resolved_score_partition = score_partition or default_score_partition(chunk_size)

    if bpu_backend == "persistent" and (refine_predictor is not None or score_predictor is not None):
        if remote is not None or remote_root is not None or host_alias is not None:
            raise ValueError("persistent BPU backend is local-only; use bpu_backend='hrt' for remote SSH validation")
        if persistent_session is None:
            model_specs: dict[str, tuple[str | Path, str]] = {}
            # Load Score before Refine when both models share one HBRT process. This
            # order is harmless for L32 and keeps the shared-session policy stable
            # across HBRT 4.7.5 loader states where cross-core IOVA checks can be
            # sensitive to process-local load history.
            if score_predictor is not None:
                model_specs["score"] = (resolved_score_hbm, resolved_score_partition)
            if refine_predictor is not None:
                model_specs["refine"] = (refine_hbm, refine_partition)
            persistent_session = make_session_from_contracts(root, model_specs, runner_bin=bpu_runner_bin, core_id=core_id)
        # Force a startup handshake so HBM load failures surface during adapter
        # installation instead of at the first FoundationPose inference call.
        persistent_session.model_info()
        installed["session"] = persistent_session

    if refine_predictor is not None:
        adapter = RefineNetBpu(
            refine_hbm,
            root=root,
            partition=refine_partition,
            core_id=core_id,
            backend=bpu_backend,
            persistent_session=persistent_session,
            runner_bin=bpu_runner_bin,
            remote=remote,
            remote_root=remote_root,
            remote_tmp=remote_tmp,
            host_alias=host_alias,
        )
        if hasattr(refine_predictor, "model"):
            refine_predictor.model = adapter
        installed["refine"] = adapter
    if score_predictor is not None:
        adapter = ScoreNetBpu(
            resolved_score_hbm,
            root=root,
            partition=resolved_score_partition,
            core_id=core_id,
            chunk_size=chunk_size,
            mode=score_mode,
            pad_mode=score_pad_mode,
            backend=bpu_backend,
            persistent_session=persistent_session,
            runner_bin=bpu_runner_bin,
            remote=remote,
            remote_root=remote_root,
            remote_tmp=remote_tmp,
            host_alias=host_alias,
        )
        if hasattr(score_predictor, "model"):
            score_predictor.model = adapter
        installed["score"] = adapter
    return installed
