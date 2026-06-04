#!/usr/bin/env python3
"""Capture real FoundationPose RefineNet/ScoreNet A/B tensors for calibration.

This module intentionally avoids patching ``third_party/FoundationPose`` files.
Wrap the predictor modules after construction, or install the optional monkey
patch before constructing predictors:

    from foundationpose_s600_tools.debug.dump_intermediates import wrap_foundationpose_predictors
    est = FoundationPose(...)
    wrap_foundationpose_predictors(est.refiner, est.scorer, out_root="configs/calibration/data/raw_capture")

or:

    from foundationpose_s600_tools.debug.dump_intermediates import install_auto_capture
    install_auto_capture(out_root="configs/calibration/data/raw_capture")
    # create PoseRefinePredictor / ScorePredictor after this

The captured layout is directly accepted by ``prepare_calibration.py``:

    <out_root>/refine_net/A/000000.npy
    <out_root>/refine_net/B/000000.npy
    <out_root>/score_net_L64/A/000000.npy
    <out_root>/score_net_L64/B/000000.npy

Environment knobs for ad-hoc runs:

    FOUNDATIONPOSE_S600_CAPTURE_DIR      output root
    FOUNDATIONPOSE_S600_CAPTURE_LIMIT    max files per partition (default 1024)
    FOUNDATIONPOSE_S600_CAPTURE_SCORE_L  comma-separated ScoreNet L values to stage (default 16,64)
    FOUNDATIONPOSE_S600_CAPTURE_CHUNK_SCORE=0 disables chunking larger score groups
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class CaptureConfig:
    out_root: Path
    limit: int = 1024
    score_lengths: tuple[int, ...] = (16, 64)
    chunk_score: bool = True
    save_format: str = "npy"
    counters: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_env(cls, out_root: str | os.PathLike[str] | None = None, limit: int | None = None) -> "CaptureConfig":
        root = out_root or os.environ.get("FOUNDATIONPOSE_S600_CAPTURE_DIR") or "configs/calibration/data/raw_capture"
        raw_l = os.environ.get("FOUNDATIONPOSE_S600_CAPTURE_SCORE_L", "16,64")
        score_lengths = tuple(int(x) for x in raw_l.split(",") if x.strip())
        chunk_score = os.environ.get("FOUNDATIONPOSE_S600_CAPTURE_CHUNK_SCORE", "1") != "0"
        env_limit = os.environ.get("FOUNDATIONPOSE_S600_CAPTURE_LIMIT")
        resolved_limit = int(env_limit) if env_limit is not None else (limit if limit is not None else 1024)
        return cls(out_root=Path(root), limit=resolved_limit, score_lengths=score_lengths, chunk_score=chunk_score)


def _to_numpy(tensor: Any):
    """Detach a torch tensor as CPU float32 numpy without importing torch at module load."""
    return tensor.detach().float().cpu().numpy()


def _next_index(cfg: CaptureConfig, partition: str) -> int | None:
    current = cfg.counters.get(partition, 0)
    if current >= cfg.limit:
        return None
    cfg.counters[partition] = current + 1
    return current


def _write_pair(cfg: CaptureConfig, partition: str, A: Any, B: Any, meta: dict[str, Any]) -> bool:
    index = _next_index(cfg, partition)
    if index is None:
        return False
    part_dir = cfg.out_root / partition
    a_dir = part_dir / "A"
    b_dir = part_dir / "B"
    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np  # lazy; only needed when capture is enabled

    a_np = _to_numpy(A)
    b_np = _to_numpy(B)
    suffix = ".npy" if cfg.save_format == "npy" else ".bin"
    a_path = a_dir / f"{index:06d}{suffix}"
    b_path = b_dir / f"{index:06d}{suffix}"
    if cfg.save_format == "npy":
        np.save(a_path, a_np)
        np.save(b_path, b_np)
    elif cfg.save_format == "bin":
        a_np.astype("float32", copy=False).tofile(a_path)
        b_np.astype("float32", copy=False).tofile(b_path)
    else:
        raise ValueError(f"unsupported save_format: {cfg.save_format}")

    record = {
        "index": index,
        "partition": partition,
        "A": str(a_path),
        "B": str(b_path),
        "A_shape": list(a_np.shape),
        "B_shape": list(b_np.shape),
        "dtype": "float32",
        "time_unix": time.time(),
        **meta,
    }
    with (part_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return True


def capture_refine_pair(cfg: CaptureConfig, A: Any, B: Any, source: str = "refine_net.forward") -> int:
    """Capture RefineNet inputs as N=1 samples matching the default contract."""
    n = int(A.shape[0])
    written = 0
    for i in range(n):
        if _write_pair(cfg, "refine_net", A[i : i + 1], B[i : i + 1], {"source": source, "sample_in_batch": i}):
            written += 1
    return written


def capture_score_pair(cfg: CaptureConfig, A: Any, B: Any, L: int | None = None, source: str = "score_net.forward") -> int:
    """Capture ScoreNet inputs for fixed-L contracts.

    If the upstream group length equals a requested L, the full group is dumped.
    If it is larger and ``chunk_score`` is enabled, non-overlapping chunks are
    dumped for each requested L. Chunking is for calibration range coverage, not
    for parity labels.
    """
    total = int(A.shape[0])
    group_l = int(L or total)
    written = 0
    targets = [group_l] if group_l in cfg.score_lengths else []
    if cfg.chunk_score:
        for target in cfg.score_lengths:
            if target not in targets and total >= target:
                targets.append(target)
    for target in targets:
        partition = f"score_net_L{target}"
        chunks = total // target
        for chunk in range(chunks):
            lo = chunk * target
            hi = lo + target
            if _write_pair(
                cfg,
                partition,
                A[lo:hi],
                B[lo:hi],
                {"source": source, "upstream_L": group_l, "chunk": chunk, "chunk_size": target},
            ):
                written += 1
    return written


def _score_l_from_args(args: tuple[Any, ...], kwargs: dict[str, Any], A: Any) -> int:
    if "L" in kwargs and kwargs["L"] is not None:
        return int(kwargs["L"])
    if len(args) >= 3 and args[2] is not None:
        return int(args[2])
    return int(A.shape[0])


def wrap_module(module: Any, kind: str, cfg: CaptureConfig | None = None) -> Any:
    """Return a torch.nn.Module wrapper that dumps A/B before forwarding.

    ``kind`` is ``"refine"`` or ``"score"``.
    """
    import torch  # lazy: S600 export tooling can import this module without torch until wrapping

    cfg = cfg or CaptureConfig.from_env()
    if getattr(module, "__foundationpose_s600_capture_wrapped__", False):
        return module

    class _CaptureWrapper(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped
            self.__foundationpose_s600_capture_wrapped__ = True

        def forward(self, *args: Any, **kwargs: Any):
            if len(args) >= 2:
                A, B = args[0], args[1]
                try:
                    if kind == "refine":
                        capture_refine_pair(cfg, A, B)
                    elif kind == "score":
                        capture_score_pair(cfg, A, B, L=_score_l_from_args(args, kwargs, A))
                except Exception as exc:  # capture must never break the pose pipeline
                    print(f"[foundationpose_s600 capture] WARNING: failed to dump {kind} tensors: {exc}")
            return self.wrapped(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.wrapped, name)

    wrapper = _CaptureWrapper(module)
    wrapper.train(module.training)
    return wrapper


def wrap_foundationpose_predictors(refine_predictor: Any | None = None, score_predictor: Any | None = None, out_root: str | os.PathLike[str] | None = None, limit: int | None = None) -> CaptureConfig:
    """Wrap already-created upstream predictor instances in-place."""
    cfg = CaptureConfig.from_env(out_root=out_root, limit=limit)
    if refine_predictor is not None and hasattr(refine_predictor, "model"):
        refine_predictor.model = wrap_module(refine_predictor.model, "refine", cfg=cfg)
    if score_predictor is not None and hasattr(score_predictor, "model"):
        score_predictor.model = wrap_module(score_predictor.model, "score", cfg=cfg)
    return cfg


def _patch_predictor_class(cls: type, kind: str, cfg_factory: Callable[[], CaptureConfig]) -> None:
    if getattr(cls, "__foundationpose_s600_capture_patched__", False):
        return
    original_init = cls.__init__

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        if hasattr(self, "model"):
            self.model = wrap_module(self.model, kind, cfg=cfg_factory())

    cls.__init__ = __init__  # type: ignore[method-assign]
    cls.__foundationpose_s600_capture_patched__ = True  # type: ignore[attr-defined]


def install_auto_capture(out_root: str | os.PathLike[str] | None = None, limit: int | None = None) -> CaptureConfig:
    """Monkey-patch upstream predictor constructors to wrap models after init.

    Call this before creating ``PoseRefinePredictor`` / ``ScorePredictor``. It
    imports upstream modules, so the normal FoundationPose environment must be
    active (CUDA/render deps available).
    """
    base_cfg = CaptureConfig.from_env(out_root=out_root, limit=limit)

    def cfg_factory() -> CaptureConfig:
        return base_cfg

    try:
        from learning.training.predict_pose_refine import PoseRefinePredictor  # type: ignore
        _patch_predictor_class(PoseRefinePredictor, "refine", cfg_factory)
    except Exception as exc:
        print(f"[foundationpose_s600 capture] WARNING: could not patch PoseRefinePredictor: {exc}")
    try:
        from learning.training.predict_score import ScorePredictor  # type: ignore
        _patch_predictor_class(ScorePredictor, "score", cfg_factory)
    except Exception as exc:
        print(f"[foundationpose_s600 capture] WARNING: could not patch ScorePredictor: {exc}")
    return base_cfg


__all__ = [
    "CaptureConfig",
    "capture_refine_pair",
    "capture_score_pair",
    "install_auto_capture",
    "wrap_foundationpose_predictors",
    "wrap_module",
]
