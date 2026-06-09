#!/usr/bin/env python3
"""Run a board-side FoundationPose hybrid smoke with BPU adapters.

This launcher is intentionally non-invasive: it does not modify vendored
FoundationPose files. It prepares the local S600 Python path, maps upstream
CUDA-only tensor construction to CPU so startup can proceed on the aarch64 board,
installs selectable RefineNet/ScoreNet BPU adapters, and can use a placeholder
renderer to exercise the full Python control flow through BPU inference.

The placeholder renderer is for environment/control-flow smoke only. The current
implementation is a CPU triangle/point smoke renderer that produces non-zero
rendered A-side crops, but it is still only an approximation of nvdiffrast and is
not suitable for ADD/ADD-S accuracy without renderer/tensor alignment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def add_paths(repo_root: Path) -> Path:
    deps = repo_root / ".deps" / "s600-foundationpose"
    upstream = repo_root / "third_party" / "FoundationPose"
    src = repo_root / "src" / "python"
    for path in [src, deps, upstream]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    missing_render_deps = any(importlib.util.find_spec(name) is None for name in ["pytorch3d", "nvdiffrast"])
    if missing_render_deps:
        compat = src / "foundationpose_s600_tools" / "compat_shims"
        if str(compat) not in sys.path:
            sys.path.insert(0, str(compat))
    return upstream


def patch_torch_cpu_cuda_compat() -> None:
    """Map common upstream CUDA-only torch calls to CPU for S600 startup."""
    import torch

    # Kornia decorates some helpers with torch.jit.script at import time. Keep the
    # original torch constructors visible during that import; otherwise TorchScript
    # tries to compile the varargs wrappers installed below.
    try:
        import kornia  # noqa: F401
    except Exception:
        pass

    def map_device(device: Any) -> Any:
        if isinstance(device, str) and device.startswith("cuda"):
            return "cpu"
        if isinstance(device, torch.device) and device.type == "cuda":
            return torch.device("cpu")
        return device

    def wrap_ctor(fn):
        def wrapped(*args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = map_device(kwargs["device"])
            return fn(*args, **kwargs)
        return wrapped

    for name in [
        "tensor",
        "as_tensor",
        "zeros",
        "ones",
        "empty",
        "full",
        "eye",
        "arange",
        "randn",
        "rand",
        "zeros_like",
        "ones_like",
    ]:
        if hasattr(torch, name):
            setattr(torch, name, wrap_ctor(getattr(torch, name)))

    torch.nn.Module.cuda = lambda self, *args, **kwargs: self  # type: ignore[method-assign]
    torch.Tensor.cuda = lambda self, *args, **kwargs: self  # type: ignore[method-assign]

    orig_tensor_to = torch.Tensor.to

    def tensor_to(self, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)):
            args = (map_device(args[0]), *args[1:])
        if "device" in kwargs:
            kwargs["device"] = map_device(kwargs["device"])
        return orig_tensor_to(self, *args, **kwargs)

    torch.Tensor.to = tensor_to  # type: ignore[method-assign]

    orig_module_to = torch.nn.Module.to

    def module_to(self, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)):
            args = (map_device(args[0]), *args[1:])
        if "device" in kwargs:
            kwargs["device"] = map_device(kwargs["device"])
        return orig_module_to(self, *args, **kwargs)

    torch.nn.Module.to = module_to  # type: ignore[method-assign]

    orig_load = torch.load

    def cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return orig_load(*args, **kwargs)

    torch.load = cpu_load  # type: ignore[assignment]

    orig_set_default_tensor_type = torch.set_default_tensor_type

    def set_default_tensor_type(tensor_type):
        if tensor_type in ("torch.cuda.FloatTensor", getattr(torch.cuda, "FloatTensor", object())):
            tensor_type = torch.FloatTensor
        return orig_set_default_tensor_type(tensor_type)

    torch.set_default_tensor_type = set_default_tensor_type  # type: ignore[assignment]

    # CUDA module helpers used by upstream code. Keep them harmless on CPU-only torch.
    torch.cuda.empty_cache = lambda *args, **kwargs: None  # type: ignore[assignment]
    torch.cuda.manual_seed_all = lambda *args, **kwargs: None  # type: ignore[assignment]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PlaceholderGlctx:
    """Sentinel raster context for placeholder rendering."""


def tensor_to_numpy(value: Any) -> np.ndarray | None:
    """Detach a tensor-like value to CPU numpy for debug summaries."""
    if value is None:
        return None
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().float().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def tensor_stats(value: Any) -> dict[str, Any] | None:
    arr = tensor_to_numpy(value)
    if arr is None:
        return None
    finite = np.isfinite(arr)
    valid = arr[finite]
    out: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": int(finite.sum()),
        "nan": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        "nonzero": int(np.count_nonzero(arr)),
        "nonzero_frac": float(np.count_nonzero(arr) / arr.size) if arr.size else 0.0,
    }
    if valid.size:
        out.update(
            {
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "abs_mean": float(np.abs(valid).mean()),
            }
        )
    return out


def downsample_nchw(value: Any, size: int = 16) -> np.ndarray | None:
    arr = tensor_to_numpy(value)
    if arr is None or arr.ndim != 4:
        return None
    n, c, h, w = arr.shape
    if h < size or w < size:
        return arr.astype(np.float32, copy=False)
    sy = max(1, h // size)
    sx = max(1, w // size)
    hh = min(size, h // sy)
    ww = min(size, w // sx)
    trimmed = arr[:, :, : hh * sy, : ww * sx]
    low = trimmed.reshape(n, c, hh, sy, ww, sx).mean(axis=(3, 5))
    return low.astype(np.float32, copy=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class TimingProfiler:
    """Small opt-in wall-clock profiler for the board demo script."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.events: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            event = {
                "name": name,
                "duration_ms": (end - start) * 1000.0,
                "start_ms": (start - self._t0) * 1000.0,
            }
            event.update(metadata)
            self.events.append(event)

    def wrap_method(self, obj: Any, method_name: str, label: str) -> None:
        if not self.enabled or obj is None or not hasattr(obj, method_name):
            return
        import functools

        original = getattr(obj, method_name)

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self.span(label):
                return original(*args, **kwargs)

        setattr(obj, method_name, wrapped)

    def summary(self) -> dict[str, Any]:
        totals: dict[str, dict[str, Any]] = {}
        for event in self.events:
            name = str(event["name"])
            item = totals.setdefault(name, {"count": 0, "total_ms": 0.0, "min_ms": None, "max_ms": None})
            duration = float(event["duration_ms"])
            item["count"] += 1
            item["total_ms"] += duration
            item["min_ms"] = duration if item["min_ms"] is None else min(float(item["min_ms"]), duration)
            item["max_ms"] = duration if item["max_ms"] is None else max(float(item["max_ms"]), duration)
        for item in totals.values():
            count = int(item["count"])
            item["avg_ms"] = float(item["total_ms"]) / max(count, 1)
        return {"total_wall_ms": (time.perf_counter() - self._t0) * 1000.0, "totals": totals, "events": self.events}

    def write(self, path: Path) -> None:
        if self.enabled:
            write_json(path, self.summary())

    def print_summary(self) -> None:
        if not self.enabled:
            return
        data = self.summary()
        print("Timing profile summary:", file=sys.stderr, flush=True)
        print(f"  total_wall_ms={data['total_wall_ms']:.3f}", file=sys.stderr, flush=True)
        totals = data["totals"]
        for name, item in sorted(totals.items(), key=lambda kv: float(kv[1]["total_ms"]), reverse=True):
            print(
                f"  {name}: count={item['count']} total_ms={item['total_ms']:.3f} "
                f"avg_ms={item['avg_ms']:.3f} min_ms={item['min_ms']:.3f} max_ms={item['max_ms']:.3f}",
                file=sys.stderr,
                flush=True,
            )


def _profile_start(profiler: TimingProfiler | None) -> float | None:
    if profiler is None or not profiler.enabled:
        return None
    return time.perf_counter()


def _profile_end(profiler: TimingProfiler | None, name: str, start: float | None, **metadata: Any) -> None:
    if profiler is None or not profiler.enabled or start is None:
        return
    end = time.perf_counter()
    event = {
        "name": name,
        "duration_ms": (end - start) * 1000.0,
        "start_ms": (start - profiler._t0) * 1000.0,
    }
    event.update(metadata)
    profiler.events.append(event)


def _placeholder_renderer_mode_note(max_faces: int) -> str:
    if max_faces <= 0:
        return "points-only control-flow/BPU profiling renderer; not accuracy-valid"
    return "CPU triangle smoke renderer; not production latency or ADD/ADD-S evidence"


_PLACEHOLDER_RENDER_LIB: ctypes.CDLL | None = None
_PLACEHOLDER_RENDER_LIB_ATTEMPTED = False


def _load_placeholder_renderer_lib() -> ctypes.CDLL | None:
    global _PLACEHOLDER_RENDER_LIB, _PLACEHOLDER_RENDER_LIB_ATTEMPTED
    if _PLACEHOLDER_RENDER_LIB_ATTEMPTED:
        return _PLACEHOLDER_RENDER_LIB
    _PLACEHOLDER_RENDER_LIB_ATTEMPTED = True
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "build" / "cmake" / "src" / "csrc" / "libfoundationpose_placeholder_renderer.so",
        repo_root / "build" / "src" / "csrc" / "libfoundationpose_placeholder_renderer.so",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(candidate))
            fn = lib.foundationpose_placeholder_render_triangles
            ptr_f = ctypes.POINTER(ctypes.c_float)
            ptr_u8 = ctypes.POINTER(ctypes.c_uint8)
            ptr_i64 = ctypes.POINTER(ctypes.c_int64)
            if hasattr(lib, "foundationpose_depth_to_xyz_map"):
                depth_xyz_fn = lib.foundationpose_depth_to_xyz_map
                depth_xyz_fn.argtypes = [
                    ptr_f, ctypes.c_int, ctypes.c_int,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_float, ptr_f,
                ]
                depth_xyz_fn.restype = ctypes.c_int
            if hasattr(lib, "foundationpose_inline_transform_rgb_xyz"):
                inline_transform_fn = lib.foundationpose_inline_transform_rgb_xyz
                inline_transform_fn.argtypes = [
                    ptr_f, ptr_i64, ptr_f, ptr_i64, ptr_f, ptr_i64, ptr_f, ptr_i64,
                    ptr_f, ptr_f, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                    ctypes.c_float, ctypes.c_int,
                ]
                inline_transform_fn.restype = ctypes.c_int
            if hasattr(lib, "foundationpose_warp_perspective_hwc_batch"):
                warp_fn = lib.foundationpose_warp_perspective_hwc_batch
                warp_fn.argtypes = [
                    ptr_f, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ptr_f, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ptr_f,
                ]
                warp_fn.restype = ctypes.c_int
            fn.argtypes = [
                ptr_f, ptr_f, ptr_f, ptr_u8, ctypes.c_int64,
                ptr_f, ptr_f, ptr_f, ptr_i64, ctypes.c_int64,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr_f,
                ctypes.c_float, ctypes.c_float, ctypes.c_int, ptr_f,
                ctypes.c_int, ctypes.c_int, ptr_f, ctypes.c_int64, ptr_i64,
                ptr_f, ptr_f, ptr_f,
            ]
            fn.restype = ctypes.c_int
            point_fn = lib.foundationpose_placeholder_render_points
            point_fn.argtypes = [
                ptr_f, ptr_f, ptr_f, ptr_u8, ctypes.c_int64,
                ptr_f, ptr_f, ptr_i64, ctypes.c_int64,
                ctypes.c_int, ctypes.c_int, ptr_f, ptr_f, ptr_f,
            ]
            point_fn.restype = ctypes.c_int
            pose_fn = lib.foundationpose_placeholder_render_pose
            pose_argtypes = [
                ptr_f, ctypes.c_int64, ptr_f,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_float, ctypes.c_float, ptr_f, ctypes.c_int,
                ptr_f, ptr_f, ptr_i64, ctypes.c_int64,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr_f,
                ctypes.c_float, ctypes.c_float, ctypes.c_int, ptr_f,
                ctypes.c_int, ctypes.c_int, ptr_f, ctypes.c_int64, ptr_i64,
                ptr_i64, ctypes.c_int64, ptr_f, ptr_f, ptr_f,
            ]
            pose_fn.argtypes = pose_argtypes
            pose_fn.restype = ctypes.c_int
            if hasattr(lib, "foundationpose_placeholder_render_pose_batch"):
                batch_fn = lib.foundationpose_placeholder_render_pose_batch
                batch_fn.argtypes = [ptr_f, ctypes.c_int64, ptr_f, ctypes.c_int64, *pose_argtypes[3:]]
                batch_fn.restype = ctypes.c_int
            _PLACEHOLDER_RENDER_LIB = lib
            return lib
        except OSError:
            continue
    return None


def _update_placeholder_zbuffer(
    depth_image: np.ndarray,
    xyz_image: np.ndarray,
    color_image: np.ndarray,
    *,
    xmin: int,
    ymin: int,
    inside: np.ndarray,
    z_map: np.ndarray,
    xyz_map: np.ndarray,
    color_map: np.ndarray,
) -> int:
    """Vectorized z-buffer update for one placeholder-rendered triangle.

    ``inside``/``z_map``/``xyz_map``/``color_map`` are local triangle-bounds arrays;
    ``xmin``/``ymin`` translate local coordinates into the destination images.
    A pixel is updated only when the new triangle depth is positive and nearer than
    the current non-zero depth, matching the old per-pixel placeholder loop.
    """
    valid = np.asarray(inside, dtype=bool) & (np.asarray(z_map) > 1e-4)
    local_y, local_x = np.nonzero(valid)
    if local_y.size == 0:
        return 0

    y = local_y + int(ymin)
    x = local_x + int(xmin)
    z_values = np.asarray(z_map, dtype=np.float32)[local_y, local_x]
    current = depth_image[y, x]
    update = (current == 0) | (z_values < current)
    if not bool(update.any()):
        return 0

    y_update = y[update]
    x_update = x[update]
    local_y_update = local_y[update]
    local_x_update = local_x[update]
    depth_image[y_update, x_update] = z_values[update]
    xyz_image[y_update, x_update] = np.asarray(xyz_map, dtype=np.float32)[local_y_update, local_x_update]
    color_image[y_update, x_update] = np.asarray(color_map, dtype=np.float32)[local_y_update, local_x_update]
    return int(np.count_nonzero(update))


def save_pose_data_dump(dump_dir: Path, stage: str, index: int, pose_data: Any, *, save_full: bool) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    prefix = dump_dir / f"{stage}_{index:03d}"
    keys = [
        "rgbAs",
        "rgbBs",
        "xyz_mapAs",
        "xyz_mapBs",
        "depthAs",
        "depthBs",
        "normalAs",
        "normalBs",
        "poseA",
        "poseB",
        "tf_to_crops",
        "Ks",
        "mesh_diameters",
    ]
    arrays = {key: tensor_to_numpy(getattr(pose_data, key, None)) for key in keys}
    arrays = {key: value for key, value in arrays.items() if value is not None}

    model_inputs: dict[str, np.ndarray] = {}
    if "rgbAs" in arrays and "xyz_mapAs" in arrays:
        model_inputs["A"] = np.concatenate([arrays["rgbAs"], arrays["xyz_mapAs"]], axis=1).astype(np.float32, copy=False)
    if "rgbBs" in arrays and "xyz_mapBs" in arrays:
        model_inputs["B"] = np.concatenate([arrays["rgbBs"], arrays["xyz_mapBs"]], axis=1).astype(np.float32, copy=False)

    summary = {
        "stage": stage,
        "index": index,
        "tensors": {key: tensor_stats(value) for key, value in arrays.items()},
        "model_inputs": {key: tensor_stats(value) for key, value in model_inputs.items()},
    }
    write_json(prefix.with_suffix(".summary.json"), summary)

    lowres = {}
    for key in ["rgbAs", "rgbBs", "xyz_mapAs", "xyz_mapBs", "depthAs", "depthBs", "A", "B"]:
        value = model_inputs.get(key) if key in model_inputs else arrays.get(key)
        low = downsample_nchw(value)
        if low is not None:
            lowres[key] = low
    if lowres:
        np.savez_compressed(prefix.with_suffix(".lowres.npz"), **lowres)
    if save_full:
        full = {**arrays, **{f"model_{key}": value for key, value in model_inputs.items()}}
        np.savez_compressed(prefix.with_suffix(".full.npz"), **full)


def save_array_dump(dump_dir: Path, name: str, index: int, arrays: dict[str, Any]) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    prefix = dump_dir / f"{name}_{index:03d}"
    np_arrays = {key: tensor_to_numpy(value) for key, value in arrays.items()}
    np_arrays = {key: value for key, value in np_arrays.items() if value is not None}
    write_json(prefix.with_suffix(".summary.json"), {"name": name, "index": index, "tensors": {key: tensor_stats(value) for key, value in np_arrays.items()}})
    if np_arrays:
        np.savez_compressed(prefix.with_suffix(".npz"), **np_arrays)


def install_tensor_dump_hooks(dump_dir: Path, *, save_full: bool = False) -> None:
    """Wrap upstream crop/predict calls and save tensor summaries for alignment."""
    import functools
    import learning.training.predict_pose_refine as predict_pose_refine
    import learning.training.predict_score as predict_score

    dump_dir.mkdir(parents=True, exist_ok=True)
    counters = {"refine_crop": 0, "score_crop": 0}

    orig_refine_crop = predict_pose_refine.make_crop_data_batch

    @functools.wraps(orig_refine_crop)
    def refine_crop_wrapper(*args, **kwargs):
        pose_data = orig_refine_crop(*args, **kwargs)
        idx = counters["refine_crop"]
        counters["refine_crop"] += 1
        save_pose_data_dump(dump_dir, "refine_crop", idx, pose_data, save_full=save_full)
        return pose_data

    predict_pose_refine.make_crop_data_batch = refine_crop_wrapper

    orig_score_crop = predict_score.make_crop_data_batch

    @functools.wraps(orig_score_crop)
    def score_crop_wrapper(*args, **kwargs):
        pose_data = orig_score_crop(*args, **kwargs)
        idx = counters["score_crop"]
        counters["score_crop"] += 1
        save_pose_data_dump(dump_dir, "score_crop", idx, pose_data, save_full=save_full)
        return pose_data

    predict_score.make_crop_data_batch = score_crop_wrapper


def install_predictor_dump_hooks(refiner: Any, scorer: Any, dump_dir: Path) -> None:
    """Save raw refiner poses and raw per-candidate scores before FoundationPose sorts."""
    import functools

    counters = {"refine_predict": 0, "score_predict": 0}

    orig_refine_predict = refiner.predict

    @functools.wraps(orig_refine_predict)
    def refine_predict_wrapper(*args, **kwargs):
        poses, vis = orig_refine_predict(*args, **kwargs)
        idx = counters["refine_predict"]
        counters["refine_predict"] += 1
        save_array_dump(dump_dir, "refine_predict", idx, {"poses": poses, "last_trans_update": getattr(refiner, "last_trans_update", None), "last_rot_update": getattr(refiner, "last_rot_update", None)})
        return poses, vis

    refiner.predict = refine_predict_wrapper

    orig_score_predict = scorer.predict

    @functools.wraps(orig_score_predict)
    def score_predict_wrapper(*args, **kwargs):
        scores, vis = orig_score_predict(*args, **kwargs)
        idx = counters["score_predict"]
        counters["score_predict"] += 1
        scores_np = tensor_to_numpy(scores)
        order = None if scores_np is None else np.argsort(-scores_np.reshape(-1)).astype(np.int64)
        save_array_dump(dump_dir, "score_predict", idx, {"scores": scores, "order": order})
        return scores, vis

    scorer.predict = score_predict_wrapper


def install_timing_hooks(profiler: TimingProfiler, refiner: Any, scorer: Any) -> None:
    """Install opt-in timing wrappers around crop, predictor, and BPU adapter calls."""
    if not profiler.enabled:
        return
    import learning.training.predict_pose_refine as predict_pose_refine
    import learning.training.predict_score as predict_score

    profiler.wrap_method(predict_pose_refine, "make_crop_data_batch", "crop.refine_make_crop_data_batch")
    profiler.wrap_method(predict_score, "make_crop_data_batch", "crop.score_make_crop_data_batch")
    profiler.wrap_method(refiner, "predict", "predict.refine_total")
    profiler.wrap_method(scorer, "predict", "predict.score_total")

    for label, predictor in [("refine", refiner), ("score", scorer)]:
        model = getattr(predictor, "model", None)
        profiler.wrap_method(model, "predict_numpy", f"adapter.{label}.predict_numpy")
        runner = getattr(model, "runner", None)
        profiler.wrap_method(runner, "infer", f"adapter.{label}.runner_infer")


def save_frame_dump(dump_dir: Path, frame_id: str, pose: np.ndarray, est: Any) -> None:
    payload = {
        "frame_id": frame_id,
        "final_pose": np.asarray(pose).reshape(4, 4).tolist(),
        "best_id": int(tensor_to_numpy(getattr(est, "best_id", -1)).reshape(-1)[0]) if getattr(est, "best_id", None) is not None else None,
    }
    for key in ["scores", "poses"]:
        value = tensor_to_numpy(getattr(est, key, None))
        if value is not None:
            payload[key] = value.tolist()
    write_json(dump_dir / f"frame_{frame_id}.json", payload)


_XYZ_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_XYZ_FRAME_CACHE: dict[str, Any] = {}


def _xyz_k_key(K_np: np.ndarray) -> tuple[float, ...]:
    return tuple(float(v) for v in K_np.reshape(9))


def _begin_frame_xyz_cache() -> None:
    """Start a frame-local XYZ cache generation for register/score reuse."""
    _XYZ_FRAME_CACHE.clear()
    _XYZ_FRAME_CACHE["trust_same_frame"] = True


def _get_cached_frame_xyz(depth_np: np.ndarray, K_np: np.ndarray) -> np.ndarray | None:
    cached = _XYZ_FRAME_CACHE.get("last")
    if not cached:
        return None
    if cached.get("shape") != tuple(depth_np.shape) or cached.get("strides") != tuple(depth_np.strides):
        return None
    if cached.get("dtype") != str(depth_np.dtype) or cached.get("K") != _xyz_k_key(K_np):
        return None
    try:
        shares_depth = np.shares_memory(depth_np, cached["depth"])
    except Exception:
        shares_depth = False
    if not shares_depth and not bool(_XYZ_FRAME_CACHE.get("trust_same_frame", False)):
        return None
    return cached["xyz"]


def _set_cached_frame_xyz(depth_np: np.ndarray, K_np: np.ndarray, xyz_map: np.ndarray) -> None:
    _XYZ_FRAME_CACHE["last"] = {
        "depth": depth_np,
        "shape": tuple(depth_np.shape),
        "strides": tuple(depth_np.strides),
        "dtype": str(depth_np.dtype),
        "K": _xyz_k_key(K_np),
        "xyz": xyz_map,
    }


def depth2xyzmap_fast(depth: Any, K: Any, uvs: Any = None) -> np.ndarray:
    """Vectorized CPU depth->XYZ map for the S600 demo crop path.

    The common full-frame path reuses cached pixel grids.  The optional ``uvs``
    argument preserves the upstream ``Utils.depth2xyzmap`` contract for debug
    callers that request a sparse set of rounded pixels.
    """
    depth_np = tensor_to_numpy(depth).astype(np.float32, copy=False)
    h, w = depth_np.shape[:2]
    K_np = np.asarray(K, dtype=np.float32).reshape(3, 3)

    if uvs is not None:
        uvs_np = np.asarray(uvs).round().astype(np.int64, copy=False).reshape(-1, 2)
        us = np.clip(uvs_np[:, 0], 0, w - 1)
        vs = np.clip(uvs_np[:, 1], 0, h - 1)
        zs = depth_np[vs, us]
        xyz_map = np.zeros((h, w, 3), dtype=np.float32)
        xyz_map[vs, us, 0] = (us.astype(np.float32) - K_np[0, 2]) * zs / K_np[0, 0]
        xyz_map[vs, us, 1] = (vs.astype(np.float32) - K_np[1, 2]) * zs / K_np[1, 1]
        xyz_map[vs, us, 2] = zs
        xyz_map[depth_np < 0.001] = 0
        return xyz_map

    cached = _get_cached_frame_xyz(depth_np, K_np)
    if cached is not None:
        return cached

    native_lib = _load_placeholder_renderer_lib()
    native_depth_fn = getattr(native_lib, "foundationpose_depth_to_xyz_map", None) if native_lib is not None else None
    depth_contig = np.ascontiguousarray(depth_np, dtype=np.float32)
    xyz_map = np.empty((h, w, 3), dtype=np.float32)
    if native_depth_fn is not None:
        rc = native_depth_fn(
            depth_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(int(h)),
            ctypes.c_int(int(w)),
            ctypes.c_float(float(K_np[0, 0])),
            ctypes.c_float(float(K_np[1, 1])),
            ctypes.c_float(float(K_np[0, 2])),
            ctypes.c_float(float(K_np[1, 2])),
            ctypes.c_float(0.001),
            xyz_map.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"native depth-to-XYZ map failed rc={rc}")
    else:
        key = (int(h), int(w))
        grids = _XYZ_GRID_CACHE.get(key)
        if grids is None:
            vs, us = np.meshgrid(
                np.arange(h, dtype=np.float32),
                np.arange(w, dtype=np.float32),
                sparse=False,
                indexing="ij",
            )
            grids = (us, vs)
            _XYZ_GRID_CACHE[key] = grids
        us, vs = grids
        zs = depth_np
        xyz_map[..., 0] = (us - K_np[0, 2]) * zs / K_np[0, 0]
        xyz_map[..., 1] = (vs - K_np[1, 2]) * zs / K_np[1, 1]
        xyz_map[..., 2] = zs
        xyz_map[zs < 0.001] = 0
    _set_cached_frame_xyz(depth_np, K_np, xyz_map)
    return xyz_map


def install_fast_depth2xyzmap(profiler: TimingProfiler | None = None) -> None:
    """Patch imported FoundationPose modules to use cached-grid depth->XYZ."""
    import Utils
    import estimater

    def depth2xyzmap_s600(depth: Any, K: Any, uvs: Any = None) -> np.ndarray:
        start = _profile_start(profiler)
        try:
            return depth2xyzmap_fast(depth, K, uvs=uvs)
        finally:
            _profile_end(profiler, "frame.depth2xyzmap_fast", start, sparse=uvs is not None)

    for mod in [Utils, estimater]:
        if hasattr(mod, "depth2xyzmap"):
            setattr(mod, "depth2xyzmap", depth2xyzmap_s600)


def _warp_perspective_hwc_batch_cv2(source: Any, tf_to_crops: Any, dsize: Any, *, mode: str) -> Any:
    """Warp one HWC source image/map by a batch of crop transforms."""
    import cv2
    import torch

    src = tensor_to_numpy(source).astype(np.float32, copy=False)
    if src.ndim == 2:
        src = src[..., None]
    if src.ndim != 3:
        raise ValueError(f"expected HWC source for cv2 warp, got shape {src.shape}")
    matrices = tensor_to_numpy(tf_to_crops).astype(np.float32, copy=False).reshape(-1, 3, 3)
    src_h, src_w = int(src.shape[0]), int(src.shape[1])
    out_h, out_w = int(dsize[0]), int(dsize[1])
    # Kornia with align_corners=False uses PyTorch grid_sample half-pixel
    # normalization. Convert its source->destination pixel homography into the
    # equivalent OpenCV/native destination->source map.
    half_pixel = np.array([[src_w / max(src_w - 1.0, 1e-14), 0.0, -0.5], [0.0, src_h / max(src_h - 1.0, 1e-14), -0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    cv_matrices = np.empty_like(matrices, dtype=np.float32)
    for idx, matrix in enumerate(matrices):
        cv_matrices[idx] = (half_pixel @ np.linalg.inv(matrix)).astype(np.float32, copy=False)

    native_lib = _load_placeholder_renderer_lib()
    native_warp_fn = getattr(native_lib, "foundationpose_warp_perspective_hwc_batch", None) if native_lib is not None else None
    if native_warp_fn is not None:
        src_contig = np.ascontiguousarray(src, dtype=np.float32)
        out = np.empty((len(matrices), src_contig.shape[2], out_h, out_w), dtype=np.float32)
        rc = native_warp_fn(
            src_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(src_h),
            ctypes.c_int(src_w),
            ctypes.c_int(int(src_contig.shape[2])),
            np.ascontiguousarray(cv_matrices, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(len(matrices)),
            ctypes.c_int(out_h),
            ctypes.c_int(out_w),
            ctypes.c_int(1 if mode == "bilinear" else 0),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"native perspective warp failed rc={rc}")
        return torch.from_numpy(out).contiguous()

    flags = cv2.INTER_LINEAR if mode == "bilinear" else cv2.INTER_NEAREST
    warped = np.empty((len(matrices), out_h, out_w, src.shape[2]), dtype=np.float32)
    for idx, matrix in enumerate(cv_matrices):
        cur = cv2.warpPerspective(
            src,
            matrix.astype(np.float32, copy=False),
            (out_w, out_h),
            flags=flags | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if cur.ndim == 2:
            cur = cur[..., None]
        warped[idx] = cur
    return torch.from_numpy(warped).permute(0, 3, 1, 2).contiguous()


def _runtime_float_tensor(value: Any) -> Any:
    """Move a tensor-like value to the patched runtime device and make it float."""
    import torch

    if torch.is_tensor(value):
        try:
            return value.cuda().float()
        except (AssertionError, RuntimeError):
            # Unit tests may import this module without the board CPU/CUDA shim.
            return value.float()
    return torch.as_tensor(value, dtype=torch.float)


def _inline_precomputed_xyz_transform(batch: Any, cfg: Any, *, invalid_z_threshold: float) -> Any:
    """Inline the test-time transform when RGB and XYZ crops are already present.

    Upstream ``dataset.transform_batch`` always normalizes RGB and then calls
    ``transform_depth_to_xyzmap``.  The S600 crop patches already provide both
    ``xyz_mapAs`` and ``xyz_mapBs``, so the depth-warp branches, crop inverse, and
    generic dataset dispatch are hot-path overhead.  This keeps the same Pair/Triplet
    XYZ normalization semantics while avoiding that extra work.
    """
    import torch

    if batch.xyz_mapAs is None or batch.xyz_mapBs is None:
        raise ValueError("inline transform requires precomputed xyz_mapAs and xyz_mapBs")

    bs = len(batch.rgbAs)
    normalize_xyz = bool(cfg.get("normalize_xyz", False)) if hasattr(cfg, "get") else bool(cfg["normalize_xyz"])
    batch.rgbAs = _runtime_float_tensor(batch.rgbAs)
    batch.rgbBs = _runtime_float_tensor(batch.rgbBs)
    batch.poseA = _runtime_float_tensor(batch.poseA)
    batch.Ks = _runtime_float_tensor(batch.Ks)
    batch.mesh_diameters = _runtime_float_tensor(batch.mesh_diameters)
    batch.xyz_mapAs = _runtime_float_tensor(batch.xyz_mapAs)
    batch.xyz_mapBs = _runtime_float_tensor(batch.xyz_mapBs)

    native_lib = _load_placeholder_renderer_lib()
    native_transform_fn = getattr(native_lib, "foundationpose_inline_transform_rgb_xyz", None) if native_lib is not None else None
    if native_transform_fn is not None:
        tensors = [batch.rgbAs, batch.rgbBs, batch.xyz_mapAs, batch.xyz_mapBs]
        if all(torch.is_tensor(tensor) and tensor.dtype == torch.float32 and tensor.device.type == "cpu" for tensor in tensors):
            rgb_as = batch.rgbAs.contiguous()
            rgb_bs = batch.rgbBs.contiguous()
            xyz_as = batch.xyz_mapAs.contiguous()
            xyz_bs = batch.xyz_mapBs.contiguous()
            h, w = int(rgb_as.shape[-2]), int(rgb_as.shape[-1])
            pose_centers = batch.poseA[:, :3, 3].detach().contiguous().cpu().numpy().astype(np.float32, copy=False)
            mesh_diameters = batch.mesh_diameters.detach().contiguous().cpu().numpy().astype(np.float32, copy=False)
            stride_arrays = [
                np.ascontiguousarray(tensor.stride(), dtype=np.int64)
                for tensor in (rgb_as, rgb_bs, xyz_as, xyz_bs)
            ]
            rc = native_transform_fn(
                rgb_as.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                stride_arrays[0].ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                rgb_bs.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                stride_arrays[1].ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                xyz_as.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                stride_arrays[2].ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                xyz_bs.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                stride_arrays[3].ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                pose_centers.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                mesh_diameters.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int64(int(bs)),
                ctypes.c_int(h),
                ctypes.c_int(w),
                ctypes.c_float(float(invalid_z_threshold)),
                ctypes.c_int(1 if normalize_xyz else 0),
            )
            if rc != 0:
                raise RuntimeError(f"native inline RGB/XYZ transform failed rc={rc}")
            batch.rgbAs = rgb_as
            batch.rgbBs = rgb_bs
            batch.xyz_mapAs = xyz_as
            batch.xyz_mapBs = xyz_bs
            return batch

    batch.rgbAs.mul_(1.0 / 255.0)
    batch.rgbBs.mul_(1.0 / 255.0)
    pose_center = batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)
    scale = (2.0 / batch.mesh_diameters).reshape(bs, 1, 1, 1)

    def transform_xyz(xyz_value: Any) -> Any:
        xyz = _runtime_float_tensor(xyz_value)
        invalid = xyz[:, 2:3] < float(invalid_z_threshold) if normalize_xyz else None
        xyz.sub_(pose_center)
        if normalize_xyz:
            xyz.mul_(scale)
            xyz.masked_fill_(torch.abs(xyz) >= 2, 0)
            xyz.masked_fill_(invalid.expand_as(xyz), 0)
        return xyz

    batch.xyz_mapAs = transform_xyz(batch.xyz_mapAs)
    batch.xyz_mapBs = transform_xyz(batch.xyz_mapBs)
    return batch


def install_refine_observed_cv2_crop(profiler: TimingProfiler | None = None) -> None:
    """Patch Refine crop assembly to use OpenCV for observed RGB/XYZ warps."""
    import torch
    import learning.training.predict_pose_refine as predict_pose_refine

    def make_crop_data_batch_s600(
        render_size,
        ob_in_cams,
        mesh,
        rgb,
        depth,
        K,
        crop_ratio,
        xyz_map,
        normal_map=None,
        mesh_diameter=None,
        cfg=None,
        glctx=None,
        mesh_tensors=None,
        dataset=None,
    ):
        predict_pose_refine.logging.info("Welcome make_crop_data_batch_s600_refine")
        depth_np = tensor_to_numpy(depth).astype(np.float32, copy=False)
        rgb_np = tensor_to_numpy(rgb).astype(np.float32, copy=False)
        xyz_map_np = tensor_to_numpy(xyz_map).astype(np.float32, copy=False)
        normal_map_np = tensor_to_numpy(normal_map).astype(np.float32, copy=False) if normal_map is not None else None
        H, W = depth_np.shape[:2]
        method = "box_3d"
        tf_to_crops = predict_pose_refine.compute_crop_window_tf_batch(
            pts=mesh.vertices,
            H=H,
            W=W,
            poses=ob_in_cams,
            K=K,
            crop_ratio=crop_ratio,
            out_size=(render_size[1], render_size[0]),
            method=method,
            mesh_diameter=mesh_diameter,
        )
        predict_pose_refine.logging.info("make tf_to_crops done")

        B = len(ob_in_cams)
        poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device="cuda")
        rgb_rs = []
        normal_rs = []
        xyz_map_rs = []
        bbox2d_crop = torch.as_tensor(
            np.array([0, 0, cfg["input_resize"][0] - 1, cfg["input_resize"][1] - 1]).reshape(2, 2),
            device="cuda",
            dtype=torch.float,
        )
        bbox2d_ori = predict_pose_refine.transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1, 4)

        bs = 512
        for b in range(0, len(poseA), bs):
            extra: dict[str, Any] = {}
            rgb_r, _depth_r, normal_r = predict_pose_refine.nvdiffrast_render(
                K=K,
                H=H,
                W=W,
                ob_in_cams=poseA[b : b + bs],
                context="cuda",
                get_normal=cfg["use_normal"],
                glctx=glctx,
                mesh_tensors=mesh_tensors,
                output_size=cfg["input_resize"],
                bbox2d=bbox2d_ori[b : b + bs],
                use_light=True,
                extra=extra,
            )
            rgb_rs.append(rgb_r)
            if cfg["use_normal"]:
                normal_rs.append(normal_r)
            xyz_map_rs.append(extra["xyz_map"])
        rgb_rs = torch.cat(rgb_rs, dim=0).permute(0, 3, 1, 2) * 255
        xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0, 3, 1, 2)
        Ks = torch.as_tensor(K, device="cuda", dtype=torch.float).reshape(1, 3, 3)
        if cfg["use_normal"]:
            normal_rs = torch.cat(normal_rs, dim=0).permute(0, 3, 1, 2)
        predict_pose_refine.logging.info("render done")

        rgb_warp_start = _profile_start(profiler)
        rgbBs = _warp_perspective_hwc_batch_cv2(rgb_np, tf_to_crops, render_size, mode="bilinear")
        _profile_end(profiler, "crop.refine_rgbB_warp_cv2", rgb_warp_start, candidates=B)
        if rgb_rs.shape[-2:] != cfg["input_resize"]:
            rgbAs = predict_pose_refine.kornia.geometry.transform.warp_perspective(
                rgb_rs, tf_to_crops, dsize=render_size, mode="bilinear", align_corners=False
            )
        else:
            rgbAs = rgb_rs
        if xyz_map_rs.shape[-2:] != cfg["input_resize"]:
            xyz_mapAs = predict_pose_refine.kornia.geometry.transform.warp_perspective(
                xyz_map_rs, tf_to_crops, dsize=render_size, mode="nearest", align_corners=False
            )
        else:
            xyz_mapAs = xyz_map_rs
        xyz_warp_start = _profile_start(profiler)
        xyz_mapBs = _warp_perspective_hwc_batch_cv2(xyz_map_np, tf_to_crops, render_size, mode="nearest")
        _profile_end(profiler, "crop.refine_xyz_mapB_warp_cv2", xyz_warp_start, candidates=B)

        if cfg["use_normal"]:
            normalAs = predict_pose_refine.kornia.geometry.transform.warp_perspective(
                normal_rs, tf_to_crops, dsize=render_size, mode="nearest", align_corners=False
            )
            normalBs = _warp_perspective_hwc_batch_cv2(normal_map_np, tf_to_crops, render_size, mode="nearest")
        else:
            normalAs = None
            normalBs = None
        predict_pose_refine.logging.info("warp done")

        mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device="cuda") * mesh_diameter
        pose_data = predict_pose_refine.BatchPoseData(
            rgbAs=rgbAs,
            rgbBs=rgbBs,
            depthAs=None,
            depthBs=None,
            normalAs=normalAs,
            normalBs=normalBs,
            poseA=poseA,
            poseB=None,
            xyz_mapAs=xyz_mapAs,
            xyz_mapBs=xyz_mapBs,
            tf_to_crops=tf_to_crops,
            Ks=Ks,
            mesh_diameters=mesh_diameters,
        )
        transform_start = _profile_start(profiler)
        pose_data = _inline_precomputed_xyz_transform(pose_data, cfg, invalid_z_threshold=0.001)
        _profile_end(profiler, "crop.refine_inline_transform", transform_start, candidates=B)
        predict_pose_refine.logging.info("pose batch data done")
        return pose_data

    predict_pose_refine.make_crop_data_batch = make_crop_data_batch_s600


def install_score_observed_xyz_crop(profiler: TimingProfiler | None = None) -> None:
    """Patch Score crop assembly to crop observed XYZ directly from the frame map.

    Upstream Score builds observed XYZ by warping candidate depth crops back to the
    full frame, running depth2xyzmap_batch for every candidate, then warping back
    to crop resolution. The Refine path already uses the cheaper equivalent: build
    the full-frame XYZ map once and crop it by the candidate transforms. This patch
    applies that pattern to Score while preserving the BatchPoseData contract.
    """
    import torch
    import learning.training.predict_score as predict_score

    def make_crop_data_batch_s600(
        render_size,
        ob_in_cams,
        mesh,
        rgb,
        depth,
        K,
        crop_ratio,
        normal_map=None,
        mesh_diameter=None,
        glctx=None,
        mesh_tensors=None,
        dataset=None,
        cfg=None,
    ):
        del normal_map
        predict_score.logging.info("Welcome make_crop_data_batch_s600")
        depth_np = tensor_to_numpy(depth).astype(np.float32, copy=False)
        rgb_np = tensor_to_numpy(rgb).astype(np.float32, copy=False)
        H, W = depth_np.shape[:2]
        method = "box_3d"
        tf_to_crops = predict_score.compute_crop_window_tf_batch(
            pts=mesh.vertices,
            H=H,
            W=W,
            poses=ob_in_cams,
            K=K,
            crop_ratio=crop_ratio,
            out_size=(render_size[1], render_size[0]),
            method=method,
            mesh_diameter=mesh_diameter,
        )
        predict_score.logging.info("make tf_to_crops done")

        B = len(ob_in_cams)
        poseAs = torch.as_tensor(ob_in_cams, dtype=torch.float, device="cuda")

        rgb_rs = []
        xyz_map_rs = []
        bbox2d_crop = torch.as_tensor(
            np.array([0, 0, cfg["input_resize"][0] - 1, cfg["input_resize"][1] - 1]).reshape(2, 2),
            device="cuda",
            dtype=torch.float,
        )
        bbox2d_ori = predict_score.transform_pts(bbox2d_crop, tf_to_crops.inverse()[:, None]).reshape(-1, 4)

        bs = 512
        for b in range(0, len(ob_in_cams), bs):
            extra: dict[str, Any] = {}
            rgb_r, depth_r, _normal_r = predict_score.nvdiffrast_render(
                K=K,
                H=H,
                W=W,
                ob_in_cams=poseAs[b : b + bs],
                context="cuda",
                get_normal=cfg["use_normal"],
                glctx=glctx,
                mesh_tensors=mesh_tensors,
                output_size=cfg["input_resize"],
                bbox2d=bbox2d_ori[b : b + bs],
                use_light=True,
                extra=extra,
            )
            rgb_rs.append(rgb_r)
            xyz_map_rs.append(extra["xyz_map"])

        rgb_rs = torch.cat(rgb_rs, dim=0).permute(0, 3, 1, 2) * 255
        xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0, 3, 1, 2)
        predict_score.logging.info("render done")

        rgb_warp_start = _profile_start(profiler)
        rgbBs = _warp_perspective_hwc_batch_cv2(rgb_np, tf_to_crops, render_size, mode="bilinear")
        _profile_end(profiler, "crop.score_rgbB_warp_cv2", rgb_warp_start, candidates=B)
        if rgb_rs.shape[-2:] != cfg["input_resize"]:
            rgbAs = predict_score.kornia.geometry.transform.warp_perspective(
                rgb_rs, tf_to_crops, dsize=render_size, mode="bilinear", align_corners=False
            )
        else:
            rgbAs = rgb_rs

        if xyz_map_rs.shape[-2:] != cfg["input_resize"]:
            xyz_mapAs = predict_score.kornia.geometry.transform.warp_perspective(
                xyz_map_rs, tf_to_crops, dsize=render_size, mode="nearest", align_corners=False
            )
        else:
            xyz_mapAs = xyz_map_rs

        xyz_start = _profile_start(profiler)
        xyz_map_np = depth2xyzmap_fast(depth_np, K)
        _profile_end(profiler, "crop.score_depth2xyzmap_full", xyz_start, candidates=B)
        xyz_warp_start = _profile_start(profiler)
        xyz_mapBs = _warp_perspective_hwc_batch_cv2(xyz_map_np, tf_to_crops, render_size, mode="nearest")
        _profile_end(profiler, "crop.score_xyz_mapB_warp_cv2", xyz_warp_start, candidates=B)

        Ks = torch.as_tensor(K, dtype=torch.float).reshape(1, 3, 3).expand(len(rgbAs), 3, 3)
        mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device="cuda") * mesh_diameter
        pose_data = predict_score.BatchPoseData(
            rgbAs=rgbAs,
            rgbBs=rgbBs,
            depthAs=None,
            depthBs=None,
            normalAs=None,
            normalBs=None,
            poseA=poseAs,
            xyz_mapAs=xyz_mapAs,
            xyz_mapBs=xyz_mapBs,
            tf_to_crops=tf_to_crops,
            Ks=Ks,
            mesh_diameters=mesh_diameters,
        )
        transform_start = _profile_start(profiler)
        pose_data = _inline_precomputed_xyz_transform(pose_data, cfg, invalid_z_threshold=0.1)
        _profile_end(profiler, "crop.score_inline_transform", transform_start, candidates=B)
        predict_score.logging.info("pose batch data done")
        return pose_data

    predict_score.make_crop_data_batch = make_crop_data_batch_s600


_PLACEHOLDER_MESH_PREP_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _placeholder_mesh_cache_key(mesh_tensors: dict[str, Any], max_faces: int) -> tuple[Any, ...]:
    names = ("pos", "faces", "tex", "uv", "uv_idx", "vertex_color", "vnormals")
    return (id(mesh_tensors), int(max_faces), *tuple((name, id(mesh_tensors.get(name))) for name in names))


def _sample_texture_np(tex_np: np.ndarray | None, uv_values: np.ndarray) -> np.ndarray:
    uv_values = np.asarray(uv_values, dtype=np.float32)
    if tex_np is None:
        return np.full((*uv_values.shape[:-1], 3), 0.65, dtype=np.float32)
    shape = uv_values.shape[:-1]
    uv_flat = uv_values.reshape(-1, 2)
    # nvdiffrast uses the UVs after make_mesh_tensors() flips v=1-v.
    # Clamp instead of wrapping for this smoke renderer; demo OBJ UVs are in-range.
    uu = np.clip(uv_flat[:, 0], 0.0, 1.0)
    vv = np.clip(uv_flat[:, 1], 0.0, 1.0)
    x_tex = uu * (tex_np.shape[1] - 1)
    y_tex = vv * (tex_np.shape[0] - 1)
    x0 = np.floor(x_tex).astype(np.int64)
    y0 = np.floor(y_tex).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, tex_np.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, tex_np.shape[0] - 1)
    wx = (x_tex - x0).astype(np.float32)[:, None]
    wy = (y_tex - y0).astype(np.float32)[:, None]
    c00 = tex_np[y0, x0]
    c10 = tex_np[y0, x1]
    c01 = tex_np[y1, x0]
    c11 = tex_np[y1, x1]
    out = (1 - wx) * (1 - wy) * c00 + wx * (1 - wy) * c10 + (1 - wx) * wy * c01 + wx * wy * c11
    return out.reshape(*shape, 3).astype(np.float32, copy=False)


def _prepare_placeholder_render_mesh(mesh_tensors: dict[str, Any], max_faces: int) -> dict[str, Any]:
    """Convert immutable mesh renderer inputs once and reuse across crop calls."""
    import torch

    key = _placeholder_mesh_cache_key(mesh_tensors, max_faces)
    cached = _PLACEHOLDER_MESH_PREP_CACHE.get(key)
    if cached is not None:
        return cached

    pts_full = torch.as_tensor(mesh_tensors["pos"], dtype=torch.float32).cpu()
    empty_faces = torch.empty((0, 3), dtype=torch.long, device="cpu")
    faces = torch.as_tensor(mesh_tensors.get("faces", empty_faces), dtype=torch.long).cpu()
    tex_np = None
    uv_np = None
    uv_idx = faces
    has_tex = "tex" in mesh_tensors and "uv" in mesh_tensors

    if has_tex:
        tex = torch.as_tensor(mesh_tensors["tex"], dtype=torch.float32).cpu()
        if tex.ndim == 4:
            tex = tex[0]
        tex_np = tex.numpy().astype(np.float32, copy=False).clip(0, 1)
        uv_np = torch.as_tensor(mesh_tensors["uv"], dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False)
        if len(uv_np) != len(pts_full):
            uv_np = None
            has_tex = False
        else:
            uv_idx = torch.as_tensor(mesh_tensors.get("uv_idx", faces), dtype=torch.long).cpu()
            base_colors_full = torch.from_numpy(_sample_texture_np(tex_np, uv_np))
    if not has_tex and "vertex_color" in mesh_tensors:
        base_colors_full = torch.as_tensor(mesh_tensors["vertex_color"], dtype=torch.float32).cpu()
        if len(base_colors_full) != len(pts_full):
            base_colors_full = torch.full((len(pts_full), 3), 0.65, dtype=torch.float32, device="cpu")
    elif not has_tex:
        base_colors_full = torch.full((len(pts_full), 3), 0.65, dtype=torch.float32, device="cpu")

    normals_full = torch.as_tensor(mesh_tensors.get("vnormals", torch.zeros_like(pts_full)), dtype=torch.float32).cpu()
    pts_full_np = np.ascontiguousarray(pts_full.numpy().astype(np.float32, copy=False))
    normals_full_np = np.ascontiguousarray(normals_full.numpy().astype(np.float32, copy=False))

    # Triangle smoke path uses a bounded deterministic face subset for speed.
    # Passing --renderer-max-faces 0 disables triangle fill and keeps only the
    # point-splat path, which is useful for throughput profiling when exact
    # placeholder-renderer coverage is not the bottleneck under test.
    all_face_indices = torch.arange(len(faces), dtype=torch.long, device="cpu")
    if max_faces <= 0 or len(faces) == 0:
        face_indices_smoke = all_face_indices[:0]
        faces_smoke = faces[:0]
    elif len(faces) > max_faces:
        face_step = max(1, int(np.ceil(len(faces) / max_faces)))
        face_indices_smoke = all_face_indices[::face_step][:max_faces]
        faces_smoke = faces[face_indices_smoke]
    else:
        face_indices_smoke = all_face_indices
        faces_smoke = faces

    faces_smoke_np = np.ascontiguousarray(faces_smoke.numpy().astype(np.int64, copy=False))
    face_indices_np = face_indices_smoke.numpy().astype(np.int64, copy=False)
    base_colors_full_np = np.ascontiguousarray(base_colors_full.numpy().astype(np.float32, copy=False))
    uv_idx_np = uv_idx.numpy().astype(np.int64, copy=False) if has_tex else None
    native_triangle_lib = _load_placeholder_renderer_lib()
    tex_native_np = np.ascontiguousarray(tex_np, dtype=np.float32) if has_tex and tex_np is not None else None
    uv_native_np = np.ascontiguousarray(uv_np, dtype=np.float32) if has_tex and uv_np is not None else None
    uv_faces_native_np = None
    if has_tex and uv_idx_np is not None:
        uv_faces_native_np = np.ascontiguousarray(uv_idx_np[face_indices_np].astype(np.int64, copy=False))

    # Point-splat fallback/cleanup uses a bounded vertex subset.
    if len(pts_full) > 20000:
        vert_step = max(1, int(np.ceil(len(pts_full) / 20000)))
        point_indices_np = np.arange(0, len(pts_full), vert_step, dtype=np.int64)
    else:
        point_indices_np = np.arange(len(pts_full), dtype=np.int64)
    point_indices_np = np.ascontiguousarray(point_indices_np[:20000])
    pts = pts_full[torch.as_tensor(point_indices_np, dtype=torch.long)]
    base_colors = base_colors_full[torch.as_tensor(point_indices_np, dtype=torch.long)]
    pts_h = torch.cat([pts, torch.ones((len(pts), 1), dtype=torch.float32, device="cpu")], dim=1)
    pts_full_h = torch.cat([pts_full, torch.ones((len(pts_full), 1), dtype=torch.float32, device="cpu")], dim=1)

    prepared = {
        "_source": mesh_tensors,
        "pts_full": pts_full,
        "faces_smoke_np": faces_smoke_np,
        "face_indices_np": face_indices_np,
        "base_colors_full_np": base_colors_full_np,
        "uv_idx_np": uv_idx_np,
        "native_triangle_lib": native_triangle_lib,
        "tex_native_np": tex_native_np,
        "uv_native_np": uv_native_np,
        "uv_faces_native_np": uv_faces_native_np,
        "point_indices_np": point_indices_np,
        "pts": pts,
        "base_colors": base_colors,
        "pts_h": pts_h,
        "pts_full_h": pts_full_h,
        "normals_full": normals_full,
        "pts_full_np": pts_full_np,
        "normals_full_np": normals_full_np,
        "tex_np": tex_np,
        "uv_np": uv_np,
        "has_tex": has_tex,
    }
    if len(_PLACEHOLDER_MESH_PREP_CACHE) >= 8:
        _PLACEHOLDER_MESH_PREP_CACHE.clear()
    _PLACEHOLDER_MESH_PREP_CACHE[key] = prepared
    return prepared


def render_placeholder_cpu(
    *,
    K: Any = None,
    H: Any = None,
    W: Any = None,
    ob_in_cams: Any = None,
    get_normal: bool = False,
    output_size: Any = None,
    extra: dict[str, Any] | None = None,
    mesh_tensors: dict[str, Any] | None = None,
    bbox2d: Any = None,
    max_faces: int = 3000,
    splat_radius: int = 1,
    profiler: TimingProfiler | None = None,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """CPU triangle/point renderer used as the S600 placeholder host renderer.

    This preserves the small subset of ``Utils.nvdiffrast_render`` semantics used
    by the upstream FoundationPose crop builders. It is a smoke/control-flow
    renderer, not a production-accuracy renderer.
    """
    import torch

    render_start = _profile_start(profiler)
    try:
        prepare_start = _profile_start(profiler)
        try:
            if ob_in_cams is None:
                n = 1
                poses = torch.eye(4, dtype=torch.float32, device="cpu").reshape(1, 4, 4)
            else:
                poses = torch.as_tensor(ob_in_cams, dtype=torch.float32).cpu()
                n = int(len(poses))
            if output_size is None:
                output_size = (H, W)
            h, w = int(output_size[0]), int(output_size[1])
            color = torch.zeros((n, h, w, 3), dtype=torch.float32, device="cpu")
            depth = torch.zeros((n, h, w), dtype=torch.float32, device="cpu")
            xyz = torch.zeros((n, h, w, 3), dtype=torch.float32, device="cpu")
            normal = torch.zeros((n, h, w, 3), dtype=torch.float32, device="cpu") if get_normal else None

            if mesh_tensors is None or K is None:
                if extra is not None:
                    extra["xyz_map"] = xyz
                return color, depth, normal

            K_np = np.asarray(K, dtype=np.float32).reshape(3, 3)
            fx, fy, cx, cy = float(K_np[0, 0]), float(K_np[1, 1]), float(K_np[0, 2]), float(K_np[1, 2])
            use_light = bool(kwargs.get("use_light", False))
            light_dir = np.asarray(kwargs.get("light_dir", np.array([0, 0, 1])), dtype=np.float32).reshape(3)
            light_dir_neg = -light_dir / max(float(np.linalg.norm(light_dir)), 1e-8)
            w_ambient = float(kwargs.get("w_ambient", 0.8))
            w_diffuse = float(kwargs.get("w_diffuse", 0.5))

            prepared = _prepare_placeholder_render_mesh(mesh_tensors, max_faces)
            pts_full = prepared["pts_full"]
            faces_smoke_np = prepared["faces_smoke_np"]
            face_indices_np = prepared["face_indices_np"]
            base_colors_full_np = prepared["base_colors_full_np"]
            uv_idx_np = prepared["uv_idx_np"]
            native_triangle_lib = prepared["native_triangle_lib"]
            tex_native_np = prepared["tex_native_np"]
            uv_native_np = prepared["uv_native_np"]
            uv_faces_native_np = prepared["uv_faces_native_np"]
            point_indices_np = prepared["point_indices_np"]
            pts = prepared["pts"]
            base_colors = prepared["base_colors"]
            pts_h = prepared["pts_h"]
            pts_full_h = prepared["pts_full_h"]
            normals_full = prepared["normals_full"]
            pts_full_np = prepared["pts_full_np"]
            normals_full_np = prepared["normals_full_np"]
            tex_np = prepared["tex_np"]
            uv_np = prepared["uv_np"]
            has_tex = bool(prepared["has_tex"])

            bbox = None if bbox2d is None else torch.as_tensor(bbox2d, dtype=torch.float32).cpu().reshape(-1, 4)
            full_h = float(H if H is not None else h)
            full_w = float(W if W is not None else w)
        finally:
            _profile_end(profiler, "render.placeholder_prepare", prepare_start, max_faces=max_faces)

        native_pose_batch_fn = getattr(native_triangle_lib, "foundationpose_placeholder_render_pose_batch", None) if native_triangle_lib is not None else None
        if native_pose_batch_fn is not None and splat_radius == 0:
            native_start = _profile_start(profiler)
            try:
                poses_np = np.ascontiguousarray(poses.numpy().astype(np.float32, copy=False))
                depth_np = np.ascontiguousarray(depth.numpy())
                xyz_np = np.ascontiguousarray(xyz.numpy())
                color_np = np.ascontiguousarray(color.numpy())
                use_bbox = bool(bbox is not None and len(bbox) >= n)
                bboxes_np = np.ascontiguousarray(bbox[:n].numpy().astype(np.float32, copy=False)) if use_bbox else None
                use_texture = bool(has_tex and tex_native_np is not None and uv_native_np is not None and uv_faces_native_np is not None)
                light_ptr = (
                    np.ascontiguousarray(light_dir_neg, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    if use_light else ctypes.POINTER(ctypes.c_float)()
                )
                bbox_ptr = (
                    bboxes_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    if bboxes_np is not None else ctypes.POINTER(ctypes.c_float)()
                )
                texture_ptr = (
                    tex_native_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    if use_texture else ctypes.POINTER(ctypes.c_float)()
                )
                uv_ptr = (
                    uv_native_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    if use_texture else ctypes.POINTER(ctypes.c_float)()
                )
                uv_faces_ptr = (
                    uv_faces_native_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
                    if use_texture else ctypes.POINTER(ctypes.c_int64)()
                )
                rc = native_pose_batch_fn(
                    pts_full_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int64(len(pts_full_np)),
                    poses_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int64(n),
                    ctypes.c_float(fx),
                    ctypes.c_float(fy),
                    ctypes.c_float(cx),
                    ctypes.c_float(cy),
                    ctypes.c_float(full_h),
                    ctypes.c_float(full_w),
                    bbox_ptr,
                    ctypes.c_int(1 if use_bbox else 0),
                    base_colors_full_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    normals_full_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    faces_smoke_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int64(len(faces_smoke_np)),
                    ctypes.c_int(h),
                    ctypes.c_int(w),
                    ctypes.c_int(1 if use_light else 0),
                    light_ptr,
                    ctypes.c_float(w_ambient),
                    ctypes.c_float(w_diffuse),
                    ctypes.c_int(1 if use_texture else 0),
                    texture_ptr,
                    ctypes.c_int(int(tex_native_np.shape[0]) if use_texture else 0),
                    ctypes.c_int(int(tex_native_np.shape[1]) if use_texture else 0),
                    uv_ptr,
                    ctypes.c_int64(len(uv_native_np) if use_texture else 0),
                    uv_faces_ptr,
                    point_indices_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int64(len(point_indices_np)),
                    depth_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    xyz_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    color_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )
                if rc != 0:
                    raise RuntimeError(f"native placeholder batch pose renderer failed rc={rc}")
                depth.copy_(torch.from_numpy(depth_np))
                xyz.copy_(torch.from_numpy(xyz_np))
                color.copy_(torch.from_numpy(color_np))
            finally:
                _profile_end(profiler, "render.placeholder_native_pose_batch", native_start, poses=n, faces=int(len(faces_smoke_np)), points=int(len(point_indices_np)))
            if extra is not None:
                extra["xyz_map"] = xyz
            return color, depth, normal

        for i in range(n):
            if native_triangle_lib is not None and splat_radius == 0:
                native_start = _profile_start(profiler)
                try:
                    depth_np = depth[i].numpy()
                    xyz_np = xyz[i].numpy()
                    color_np = color[i].numpy()
                    pose_np = np.ascontiguousarray(poses[i].numpy().astype(np.float32, copy=False))
                    use_bbox = bool(bbox is not None and i < len(bbox))
                    bbox_np = np.ascontiguousarray(bbox[i].numpy().astype(np.float32, copy=False)) if use_bbox else None
                    use_texture = bool(has_tex and tex_native_np is not None and uv_native_np is not None and uv_faces_native_np is not None)
                    light_ptr = (
                        np.ascontiguousarray(light_dir_neg, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        if use_light else ctypes.POINTER(ctypes.c_float)()
                    )
                    bbox_ptr = (
                        bbox_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        if bbox_np is not None else ctypes.POINTER(ctypes.c_float)()
                    )
                    texture_ptr = (
                        tex_native_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        if use_texture else ctypes.POINTER(ctypes.c_float)()
                    )
                    uv_ptr = (
                        uv_native_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        if use_texture else ctypes.POINTER(ctypes.c_float)()
                    )
                    uv_faces_ptr = (
                        uv_faces_native_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
                        if use_texture else ctypes.POINTER(ctypes.c_int64)()
                    )
                    rc = native_triangle_lib.foundationpose_placeholder_render_pose(
                        pts_full_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.c_int64(len(pts_full_np)),
                        pose_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.c_float(fx),
                        ctypes.c_float(fy),
                        ctypes.c_float(cx),
                        ctypes.c_float(cy),
                        ctypes.c_float(full_h),
                        ctypes.c_float(full_w),
                        bbox_ptr,
                        ctypes.c_int(1 if use_bbox else 0),
                        base_colors_full_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        normals_full_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        faces_smoke_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                        ctypes.c_int64(len(faces_smoke_np)),
                        ctypes.c_int(h),
                        ctypes.c_int(w),
                        ctypes.c_int(1 if use_light else 0),
                        light_ptr,
                        ctypes.c_float(w_ambient),
                        ctypes.c_float(w_diffuse),
                        ctypes.c_int(1 if use_texture else 0),
                        texture_ptr,
                        ctypes.c_int(int(tex_native_np.shape[0]) if use_texture else 0),
                        ctypes.c_int(int(tex_native_np.shape[1]) if use_texture else 0),
                        uv_ptr,
                        ctypes.c_int64(len(uv_native_np) if use_texture else 0),
                        uv_faces_ptr,
                        point_indices_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                        ctypes.c_int64(len(point_indices_np)),
                        depth_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        xyz_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        color_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    )
                    if rc != 0:
                        raise RuntimeError(f"native placeholder pose renderer failed rc={rc}")
                finally:
                    _profile_end(profiler, "render.placeholder_native_pose", native_start, pose=i, faces=int(len(faces_smoke_np)), points=int(len(point_indices_np)))
                continue

            # Coarse triangle fill first.
            cam_all = (poses[i].cpu() @ pts_full_h.T).T[:, :3]
            z_all = cam_all[:, 2]
            valid_all = z_all > 1e-4
            u_all = fx * cam_all[:, 0] / z_all.clamp_min(1e-4) + cx
            v_all = fy * cam_all[:, 1] / z_all.clamp_min(1e-4) + cy
            if bbox is not None and i < len(bbox):
                left, top, right, bottom = [float(x) for x in bbox[i]]
                bw = max(right - left, 1.0)
                bh = max(bottom - top, 1.0)
                x_all = (u_all - left) / bw * (w - 1)
                y_all = (v_all - top) / bh * (h - 1)
            else:
                x_all = u_all / max(full_w - 1, 1.0) * (w - 1)
                y_all = v_all / max(full_h - 1, 1.0) * (h - 1)

            x_np = np.ascontiguousarray(x_all.numpy().astype(np.float32, copy=False))
            y_np = np.ascontiguousarray(y_all.numpy().astype(np.float32, copy=False))
            z_np = np.ascontiguousarray(z_all.numpy().astype(np.float32, copy=False))
            valid_np = np.ascontiguousarray(valid_all.numpy().astype(np.uint8, copy=False))
            cam_np = np.ascontiguousarray(cam_all.numpy().astype(np.float32, copy=False))
            normals_cam_np = None
            if use_light:
                normals_cam_np = np.ascontiguousarray((poses[i, :3, :3].cpu() @ normals_full.T).T.numpy().astype(np.float32, copy=False))
            depth_np = depth[i].numpy()
            xyz_np = xyz[i].numpy()
            color_np = color[i].numpy()

            triangle_start = _profile_start(profiler)
            try:
                if len(faces_smoke_np) > 0:
                    if native_triangle_lib is not None:
                        use_texture = bool(has_tex and tex_native_np is not None and uv_native_np is not None and uv_faces_native_np is not None)
                        normal_ptr = (
                            normals_cam_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                            if normals_cam_np is not None else ctypes.POINTER(ctypes.c_float)()
                        )
                        light_ptr = (
                            np.ascontiguousarray(light_dir_neg, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                            if use_light else ctypes.POINTER(ctypes.c_float)()
                        )
                        texture_ptr = (
                            tex_native_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                            if use_texture else ctypes.POINTER(ctypes.c_float)()
                        )
                        uv_ptr = (
                            uv_native_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                            if use_texture else ctypes.POINTER(ctypes.c_float)()
                        )
                        uv_faces_ptr = (
                            uv_faces_native_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
                            if use_texture else ctypes.POINTER(ctypes.c_int64)()
                        )
                        rc = native_triangle_lib.foundationpose_placeholder_render_triangles(
                            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            y_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            z_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            valid_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                            ctypes.c_int64(len(x_np)),
                            cam_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            base_colors_full_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            normal_ptr,
                            faces_smoke_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                            ctypes.c_int64(len(faces_smoke_np)),
                            ctypes.c_int(h),
                            ctypes.c_int(w),
                            ctypes.c_int(1 if use_light else 0),
                            light_ptr,
                            ctypes.c_float(w_ambient),
                            ctypes.c_float(w_diffuse),
                            ctypes.c_int(1 if use_texture else 0),
                            texture_ptr,
                            ctypes.c_int(int(tex_native_np.shape[0]) if use_texture else 0),
                            ctypes.c_int(int(tex_native_np.shape[1]) if use_texture else 0),
                            uv_ptr,
                            ctypes.c_int64(len(uv_native_np) if use_texture else 0),
                            uv_faces_ptr,
                            depth_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            xyz_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            color_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        )
                        if rc != 0:
                            raise RuntimeError(f"native placeholder triangle renderer failed rc={rc}")
                    else:
                        for face_i, (f0, f1, f2) in enumerate(faces_smoke_np):
                            if not (valid_np[f0] and valid_np[f1] and valid_np[f2]):
                                continue
                            xs = np.array([x_np[f0], x_np[f1], x_np[f2]], dtype=np.float32)
                            ys = np.array([y_np[f0], y_np[f1], y_np[f2]], dtype=np.float32)
                            xmin = max(0, int(np.floor(xs.min())))
                            xmax = min(w - 1, int(np.ceil(xs.max())))
                            ymin = max(0, int(np.floor(ys.min())))
                            ymax = min(h - 1, int(np.ceil(ys.max())))
                            if xmax < xmin or ymax < ymin:
                                continue
                            area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (ys[1] - ys[0]) * (xs[2] - xs[0])
                            if abs(float(area)) < 1e-6:
                                continue
                            px = np.arange(xmin, xmax + 1, dtype=np.float32) + 0.5
                            py = np.arange(ymin, ymax + 1, dtype=np.float32) + 0.5
                            gx, gy = np.meshgrid(px, py)
                            w0 = ((xs[1] - gx) * (ys[2] - gy) - (ys[1] - gy) * (xs[2] - gx)) / area
                            w1 = ((xs[2] - gx) * (ys[0] - gy) - (ys[2] - gy) * (xs[0] - gx)) / area
                            w2 = 1.0 - w0 - w1
                            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
                            if not inside.any():
                                continue
                            tri_z = (w0 * z_np[f0] + w1 * z_np[f1] + w2 * z_np[f2]).astype(np.float32, copy=False)
                            tri_xyz = (
                                w0[..., None] * cam_np[f0]
                                + w1[..., None] * cam_np[f1]
                                + w2[..., None] * cam_np[f2]
                            ).astype(np.float32, copy=False)
                            if has_tex and uv_np is not None and uv_idx_np is not None:
                                tu0, tu1, tu2 = [int(v) for v in uv_idx_np[int(face_indices_np[face_i])]]
                                tri_uv = (
                                    w0[..., None] * uv_np[tu0]
                                    + w1[..., None] * uv_np[tu1]
                                    + w2[..., None] * uv_np[tu2]
                                )
                                tri_color_map = _sample_texture_np(tex_np, tri_uv)
                            else:
                                tri_color_map = (
                                    w0[..., None] * base_colors_full_np[f0]
                                    + w1[..., None] * base_colors_full_np[f1]
                                    + w2[..., None] * base_colors_full_np[f2]
                                ).astype(np.float32, copy=False)
                            if use_light and normals_cam_np is not None:
                                tri_normal = (
                                    w0[..., None] * normals_cam_np[f0]
                                    + w1[..., None] * normals_cam_np[f1]
                                    + w2[..., None] * normals_cam_np[f2]
                                )
                                norm = np.linalg.norm(tri_normal, axis=-1, keepdims=True)
                                tri_normal = tri_normal / np.maximum(norm, 1e-8)
                                diffuse = np.clip((tri_normal * light_dir_neg.reshape(1, 1, 3)).sum(axis=-1, keepdims=True), 0, 1)
                                tri_color_map = np.clip(tri_color_map * w_ambient + diffuse * tri_color_map * w_diffuse, 0, 1)
                            _update_placeholder_zbuffer(
                                depth_np,
                                xyz_np,
                                color_np,
                                xmin=xmin,
                                ymin=ymin,
                                inside=inside,
                                z_map=tri_z,
                                xyz_map=tri_xyz,
                                color_map=tri_color_map,
                            )
            finally:
                _profile_end(profiler, "render.placeholder_triangles", triangle_start, pose=i, faces=int(len(faces_smoke_np)))

            # Point splats fill holes left by the bounded triangle subset.
            point_start = _profile_start(profiler)
            try:
                if native_triangle_lib is not None and splat_radius == 0:
                    depth_np = depth[i].numpy()
                    xyz_np = xyz[i].numpy()
                    color_np = color[i].numpy()
                    rc = native_triangle_lib.foundationpose_placeholder_render_points(
                        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        y_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        z_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        valid_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                        ctypes.c_int64(len(x_np)),
                        cam_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        base_colors_full_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        point_indices_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                        ctypes.c_int64(len(point_indices_np)),
                        ctypes.c_int(h),
                        ctypes.c_int(w),
                        depth_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        xyz_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        color_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    )
                    if rc != 0:
                        raise RuntimeError(f"native placeholder point renderer failed rc={rc}")
                    continue
                cam = (poses[i].cpu() @ pts_h.T).T[:, :3]
                z = cam[:, 2]
                valid = z > 1e-4
                if not bool(valid.any()):
                    continue
                cam = cam[valid]
                z = z[valid]
                cols = base_colors[valid]
                u = fx * cam[:, 0] / z + cx
                v = fy * cam[:, 1] / z + cy
                if bbox is not None and i < len(bbox):
                    left, top, right, bottom = [float(x) for x in bbox[i]]
                    bw = max(right - left, 1.0)
                    bh = max(bottom - top, 1.0)
                    x = (u - left) / bw * (w - 1)
                    y = (v - top) / bh * (h - 1)
                else:
                    x = u / max(full_w - 1, 1.0) * (w - 1)
                    y = v / max(full_h - 1, 1.0) * (h - 1)
                xi = torch.round(x).long()
                yi = torch.round(y).long()
                keep = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                if not bool(keep.any()):
                    continue
                xi = xi[keep]
                yi = yi[keep]
                z_keep = z[keep]
                cam_keep = cam[keep]
                cols_keep = cols[keep]
                if splat_radius == 0:
                    # Fast center-point path: for each projected pixel, keep the nearest
                    # z. This replaces a Python loop over up to 20k vertices per pose.
                    xi_np = xi.numpy().astype(np.int64, copy=False)
                    yi_np = yi.numpy().astype(np.int64, copy=False)
                    z_np = z_keep.numpy().astype(np.float32, copy=False)
                    pix = yi_np * w + xi_np
                    order = np.lexsort((z_np, pix))  # pixel asc, z asc -> first is nearest
                    pix_sorted = pix[order]
                    first = np.ones(len(order), dtype=bool)
                    first[1:] = pix_sorted[1:] != pix_sorted[:-1]
                    sel = order[first]
                    xs = xi_np[sel]
                    ys = yi_np[sel]
                    zs = z_np[sel]
                    depth_np = depth[i].numpy()
                    update = (depth_np[ys, xs] == 0) | (zs < depth_np[ys, xs])
                    if update.any():
                        xs = xs[update]
                        ys = ys[update]
                        depth_np[ys, xs] = zs[update]
                        xyz[i].numpy()[ys, xs] = cam_keep.numpy()[sel][update]
                        color[i].numpy()[ys, xs] = cols_keep.numpy()[sel][update]
                else:
                    order = torch.argsort(z_keep, descending=True)  # far first, near overwrites
                    for idx in order.tolist():
                        xx, yy = int(xi[idx]), int(yi[idx])
                        zz = float(z_keep[idx])
                        # Optional splat cleanup for a less sparse smoke render.
                        for dy in range(-splat_radius, splat_radius + 1):
                            y2 = yy + dy
                            if y2 < 0 or y2 >= h:
                                continue
                            for dx in range(-splat_radius, splat_radius + 1):
                                x2 = xx + dx
                                if x2 < 0 or x2 >= w:
                                    continue
                                if depth[i, y2, x2] == 0 or zz < float(depth[i, y2, x2]):
                                    depth[i, y2, x2] = zz
                                    xyz[i, y2, x2] = cam_keep[idx]
                                    color[i, y2, x2] = cols_keep[idx]
            finally:
                _profile_end(profiler, "render.placeholder_points", point_start, pose=i, splat_radius=splat_radius)
        if extra is not None:
            extra["xyz_map"] = xyz
        return color, depth, normal
    finally:
        _profile_end(profiler, "render.placeholder_total", render_start, max_faces=max_faces, splat_radius=splat_radius)

def install_placeholder_host_ops(max_faces: int = 3000, splat_radius: int = 1, profiler: TimingProfiler | None = None) -> None:
    """Patch CUDA renderer/depth filters with CPU smoke implementations."""
    import torch
    import Utils
    import estimater
    import learning.training.predict_pose_refine as predict_pose_refine
    import learning.training.predict_score as predict_score

    if getattr(Utils, "mycpp", None) is None or not hasattr(Utils.mycpp, "cluster_poses"):
        class MycppFallback:
            @staticmethod
            def cluster_poses(angle_bin, dist_bin, poses, symmetry_tfs):
                return poses
        Utils.mycpp = MycppFallback()
        estimater.mycpp = Utils.mycpp

    def identity_depth(depth, *args, **kwargs):
        return depth

    def placeholder_render(
        K=None,
        H=None,
        W=None,
        ob_in_cams=None,
        glctx=None,
        get_normal=False,
        output_size=None,
        extra=None,
        mesh_tensors=None,
        bbox2d=None,
        **kwargs,
    ):
        del glctx
        return render_placeholder_cpu(
            K=K,
            H=H,
            W=W,
            ob_in_cams=ob_in_cams,
            get_normal=get_normal,
            output_size=output_size,
            extra=extra,
            mesh_tensors=mesh_tensors,
            bbox2d=bbox2d,
            max_faces=max_faces,
            splat_radius=splat_radius,
            profiler=profiler,
            **kwargs,
        )

    for mod in [Utils, estimater, predict_pose_refine, predict_score]:
        if hasattr(mod, "erode_depth"):
            setattr(mod, "erode_depth", identity_depth)
        if hasattr(mod, "bilateral_filter_depth"):
            setattr(mod, "bilateral_filter_depth", identity_depth)
        if hasattr(mod, "nvdiffrast_render"):
            setattr(mod, "nvdiffrast_render", placeholder_render)


def resolve_repo_hbm(repo_root: Path, hbm: str | Path) -> Path:
    """Resolve an HBM path relative to the repo root."""
    path = Path(hbm)
    return path if path.is_absolute() else repo_root / path


def strip_ansi(text: str) -> str:
    """Remove terminal color/control sequences from HRT log snippets."""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def hrt_model_info_ok(hbm: Path, *, hrt: str = "/usr/hobot/bin/hrt_model_exec") -> tuple[bool, str]:
    """Return whether hrt_model_exec model_info can load an HBM, plus a compact note."""
    if not hbm.is_file():
        return False, f"missing HBM: {hbm}"
    cmd = [hrt, "model_info", "--model_file", str(hbm)]
    try:
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    except Exception as exc:
        return False, f"model_info execution failed: {exc}"
    lower = result.stdout.lower()
    if result.returncode == 0 and "iova addr not equal" not in lower and "load hbm failed" not in lower:
        model_line = next((line.strip() for line in result.stdout.splitlines() if line.startswith("[model name]")), "model_info OK")
        return True, model_line
    interesting = [
        strip_ansi(line).strip()
        for line in result.stdout.splitlines()
        if any(token in line.lower() for token in ["iova", "load hbm failed", "load model failed", "hbrt4_status", "error code"])
    ]
    return False, " | ".join(interesting[-6:]) or f"model_info failed rc={result.returncode}"


def select_bpu_adapters(args: argparse.Namespace) -> tuple[str, bool, bool]:
    """Resolve CLI BPU mode plus per-subgraph overrides."""
    bpu_mode = args.bpu_mode
    if args.bpu_adapters is not None:
        bpu_mode = "full" if args.bpu_adapters else "cpu"
    refine_bpu = bpu_mode in {"refine", "full"}
    score_bpu = bpu_mode in {"score", "full"}
    if args.refine_bpu_adapter is not None:
        refine_bpu = args.refine_bpu_adapter
    if args.score_bpu_adapter is not None:
        score_bpu = args.score_bpu_adapter
    return bpu_mode, bool(refine_bpu), bool(score_bpu)


def configure_bpu_preload(args: argparse.Namespace, repo_root: Path, *, refine_bpu: bool, score_bpu: bool) -> None:
    """Optionally make child HBRT processes see one BPU core for single-core HBMs."""
    if not (refine_bpu or score_bpu):
        return
    preload = str(args.bpu_core1_preload or "")
    if not preload:
        return
    if preload == "auto":
        candidate = repo_root / "build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so"
        # Only auto-enable the shim for Refine/full paths; Score L32 is normally
        # loadable without it, while the full Refine HBM needs it on HBRT 4.7.5.
        if not refine_bpu or not candidate.is_file():
            return
        preload_path = candidate
    else:
        preload_path = Path(preload)
        if not preload_path.is_absolute():
            preload_path = repo_root / preload_path
        if not preload_path.is_file():
            raise SystemExit(f"--bpu-core1-preload not found: {preload_path}")
    existing = os.environ.get("LD_PRELOAD", "")
    preload_value = str(preload_path)
    if existing:
        paths = existing.split(":")
        if preload_value not in paths:
            preload_value = preload_value + ":" + existing
        else:
            preload_value = existing
    os.environ["LD_PRELOAD"] = preload_value
    print(f"BPU core1 preload enabled for child HBRT processes: {preload_path}", file=sys.stderr, flush=True)


def preflight_bpu_hbms(args: argparse.Namespace, repo_root: Path, *, refine_bpu: bool, score_bpu: bool) -> None:
    """Fail early if selected BPU HBMs cannot be parsed by HBRT on this board."""
    if not args.bpu_preflight:
        return
    checks: list[tuple[str, Path]] = []
    if refine_bpu:
        checks.append(("refine", resolve_repo_hbm(repo_root, args.refine_hbm)))
    if score_bpu:
        score_hbm = args.score_hbm or f"models/hbm_real_int16/foundationpose_score_net_L{int(args.score_chunk_size)}_real_int16_no_output.hbm"
        checks.append(("score", resolve_repo_hbm(repo_root, score_hbm)))
    for name, hbm in checks:
        ok, note = hrt_model_info_ok(hbm)
        print(f"BPU preflight {name}: {'OK' if ok else 'FAIL'} {hbm} :: {note}", file=sys.stderr, flush=True)
        if not ok:
            raise SystemExit(
                f"BPU preflight failed for {name} HBM. "
                "Run src/python/scripts/probe_hbm_loadability.sh and choose --bpu-mode/--score-hbm accordingly."
            )


def run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    upstream = add_paths(repo_root)
    os.chdir(upstream)
    profiler = TimingProfiler(args.profile_timing)

    # Prefer an explicit partial-BPU mode over the older all-or-nothing default.
    # RefineNet is exported as N=1; the hrt_model_exec validation adapter reloads
    # that HBM once per candidate, which can trigger HBRT IOVA load failures even
    # at small hypothesis counts once the board runtime is in a bad IOVA state.
    # Keep the default CPU-only and require users to opt into BPU modes explicitly.
    bpu_mode, refine_bpu, score_bpu = select_bpu_adapters(args)
    print(
        "BPU adapters: "
        f"mode={bpu_mode} runtime={args.bpu_runtime} refine={refine_bpu} score={score_bpu} "
        f"refine_hbm={args.refine_hbm} score_chunk={args.score_chunk_size} "
        f"score_bpu_mode={args.score_bpu_mode} score_hbm={args.score_hbm or '<default>'}",
        file=sys.stderr,
        flush=True,
    )
    with profiler.span("setup.configure_bpu_preload"):
        configure_bpu_preload(args, repo_root, refine_bpu=refine_bpu, score_bpu=score_bpu)
    with profiler.span("setup.preflight_bpu_hbms"):
        preflight_bpu_hbms(args, repo_root, refine_bpu=refine_bpu, score_bpu=score_bpu)

    if args.cpu_cuda_compat:
        with profiler.span("setup.cpu_cuda_compat"):
            patch_torch_cpu_cuda_compat()

    with profiler.span("setup.imports"):
        from foundationpose_s600_tools.runtime import install_bpu_adapters
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        from datareader import YcbineoatReader
        from Utils import set_logging_format, set_seed, depth2xyzmap, draw_posed_3d_box, draw_xyz_axis
        import cv2
        import trimesh

    with profiler.span("setup.fast_depth2xyzmap"):
        install_fast_depth2xyzmap(profiler=profiler)

    if args.placeholder_renderer:
        print(
            "Placeholder renderer mode: "
            f"{_placeholder_renderer_mode_note(args.renderer_max_faces)} "
            f"(max_faces={args.renderer_max_faces}, splat_radius={args.renderer_splat_radius})",
            file=sys.stderr,
            flush=True,
        )
        with profiler.span("setup.placeholder_host_ops"):
            install_placeholder_host_ops(max_faces=args.renderer_max_faces, splat_radius=args.renderer_splat_radius, profiler=profiler)
        with profiler.span("setup.refine_observed_cv2_crop"):
            install_refine_observed_cv2_crop(profiler=profiler)
        with profiler.span("setup.score_observed_xyz_crop"):
            install_score_observed_xyz_crop(profiler=profiler)
    tensor_dump_dir = Path(args.tensor_dump_dir) if args.tensor_dump_dir else None
    if tensor_dump_dir is not None:
        with profiler.span("setup.tensor_dump_hooks"):
            install_tensor_dump_hooks(tensor_dump_dir, save_full=args.tensor_dump_full)

    with profiler.span("setup.logging_seed"):
        set_logging_format()
        set_seed(0)

    mesh_file = Path(args.mesh_file)
    test_scene_dir = Path(args.test_scene_dir)
    if not mesh_file.is_absolute():
        mesh_file = upstream / mesh_file
    if not test_scene_dir.is_absolute():
        test_scene_dir = upstream / test_scene_dir

    with profiler.span("setup.load_mesh"):
        mesh = trimesh.load(mesh_file)
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "ob_in_cam").mkdir(parents=True, exist_ok=True)

    with profiler.span("setup.construct_predictors"):
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()

    if refine_bpu or score_bpu:
        with profiler.span("setup.install_bpu_adapters"):
            install_bpu_adapters(
                refiner if refine_bpu else None,
                scorer if score_bpu else None,
                root=repo_root,
                core_id=args.core_id,
                refine_hbm=args.refine_hbm,
                refine_partition=args.refine_partition,
                score_hbm=args.score_hbm,
                score_partition=args.score_partition,
                score_chunk_size=args.score_chunk_size,
                score_mode=args.score_bpu_mode.replace("-", "_"),
                score_pad_mode=args.score_pad_mode,
                bpu_backend=args.bpu_runtime,
                bpu_runner_bin=args.bpu_runner_bin,
            )
    if tensor_dump_dir is not None:
        with profiler.span("setup.predictor_dump_hooks"):
            install_predictor_dump_hooks(refiner, scorer, tensor_dump_dir)
    install_timing_hooks(profiler, refiner, scorer)

    glctx = PlaceholderGlctx() if args.placeholder_renderer else None
    with profiler.span("setup.construct_foundationpose"):
        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=str(debug_dir),
            debug=args.debug,
            glctx=glctx,
        )
    if args.placeholder_renderer:
        with profiler.span("setup.placeholder_mesh_prewarm"):
            _prepare_placeholder_render_mesh(est.mesh_tensors, args.renderer_max_faces)
    if args.max_hypotheses and len(est.rot_grid) > args.max_hypotheses:
        est.rot_grid = est.rot_grid[: args.max_hypotheses]
        print(f"limited rot_grid to {len(est.rot_grid)} hypotheses")

    with profiler.span("setup.reader_init"):
        reader = YcbineoatReader(video_dir=str(test_scene_dir), shorter_side=None, zfar=np.inf)
    n = min(len(reader.color_files), args.max_frames)
    print(f"reader frames={len(reader.color_files)} running={n}")

    for i in range(n):
        with profiler.span("frame.read_inputs", frame=i):
            color = reader.get_color(i)
            depth = reader.get_depth(i)
            mask = reader.get_mask(0).astype(bool) if i == 0 else None
        _begin_frame_xyz_cache()
        if i == 0:
            with profiler.span("frame.register", frame=i):
                pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        else:
            with profiler.span("frame.track_one", frame=i):
                pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
        with profiler.span("frame.write_outputs", frame=i):
            out_path = debug_dir / "ob_in_cam" / f"{reader.id_strs[i]}.txt"
            np.savetxt(out_path, pose.reshape(4, 4))
            if tensor_dump_dir is not None:
                save_frame_dump(tensor_dump_dir, reader.id_strs[i], pose, est)
        print(f"wrote {out_path}")

        if args.debug >= 1:
            with profiler.span("frame.debug_visualization", frame=i):
                to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imwrite(str(debug_dir / f"vis_{reader.id_strs[i]}.png"), vis[..., ::-1])

    profiler.write(debug_dir / "timing_profile.json")
    profiler.print_summary()
    print("OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default="/home/sunrise/Projects/FoundationPose-s600")
    ap.add_argument("--mesh-file", default="demo_data/kinect_driller_seq/mesh/textured_mesh.obj")
    ap.add_argument("--test-scene-dir", default="demo_data/kinect_driller_seq")
    ap.add_argument("--debug-dir", default="/tmp/foundationpose_s600_board_hybrid_debug")
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--est-refine-iter", type=int, default=1)
    ap.add_argument("--track-refine-iter", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=1)
    ap.add_argument("--max-hypotheses", type=int, default=4)
    ap.add_argument("--core-id", default="1")
    ap.add_argument("--renderer-max-faces", type=int, default=3000, help="Maximum mesh faces used by the CPU triangle smoke renderer")
    ap.add_argument("--renderer-splat-radius", type=int, default=0, help="Point-splat cleanup radius after triangle fill; 0 disables cleanup splats and best matches nvdiffrast in current alignment runs")
    ap.add_argument("--cpu-cuda-compat", action=argparse.BooleanOptionalAction, default=True, help="Map upstream CUDA-only torch calls to CPU; use --no-cpu-cuda-compat on CUDA hosts")
    ap.add_argument("--bpu-mode", choices=["cpu", "refine", "score", "full"], default="cpu", help="Which neural subgraphs use BPU adapters. Default 'cpu' is the safe board baseline; use 'score' with L32 for current Score BPU smoke; use 'refine'/'full' only after Refine HBM model_info probes OK")
    ap.add_argument("--bpu-adapters", action=argparse.BooleanOptionalAction, default=None, help="Backward-compatible all-or-nothing override: --bpu-adapters => --bpu-mode full, --no-bpu-adapters => --bpu-mode cpu")
    ap.add_argument("--bpu-runtime", choices=["hrt", "persistent"], default="hrt", help="BPU adapter backend: hrt_model_exec per inference, or persistent native C++/UCP runner that loads selected HBMs once")
    ap.add_argument("--bpu-runner-bin", default=None, help="Path to foundationpose_bpu_runner for --bpu-runtime persistent; default searches build/src/csrc/foundationpose_bpu_runner")
    ap.add_argument("--bpu-core1-preload", default="auto", help="LD_PRELOAD shim for HBRT 4.7.5 cross-core IOVA loadability. 'auto' enables build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so for Refine/full BPU modes when present; set empty string to disable or pass an explicit .so path.")
    ap.add_argument("--refine-bpu-adapter", action=argparse.BooleanOptionalAction, default=None, help="Override BPU mode for RefineNet only")
    ap.add_argument("--score-bpu-adapter", action=argparse.BooleanOptionalAction, default=None, help="Override BPU mode for ScoreNet only")
    ap.add_argument("--refine-hbm", default="models/hbm_real_int16/foundationpose_refine_net_real_int16_no_output.hbm", help="RefineNet HBM path for BPU refine/full modes; default is the full all-node-int16/FLOAT32-output HBM; on HBRT 4.7.5 use LD_PRELOAD=build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so if normal preflight reports cross-core IOVA failure")
    ap.add_argument("--refine-partition", default="refine_net", help="RefineNet contract partition name")
    ap.add_argument("--score-chunk-size", type=int, default=32, help="Fixed ScoreNet BPU L/HBM size. Strict mode requires this many candidates; smoke-pad can pad shorter candidate sets for load/control-flow smoke only")
    ap.add_argument("--score-bpu-mode", choices=["strict", "smoke-pad"], default="strict", help="Score BPU semantics: strict requires candidate count to match the fixed-L HBM; smoke-pad pads shorter sets for board smoke only and is not ScoreNet-equivalent")
    ap.add_argument("--score-hbm", default=None, help="Optional ScoreNet HBM path override; default follows --score-chunk-size")
    ap.add_argument("--score-partition", default=None, help="Optional ScoreNet contract partition override; default follows --score-chunk-size")
    ap.add_argument("--score-pad-mode", choices=["repeat_last", "zero"], default="repeat_last", help="Padding strategy used only by --score-bpu-mode smoke-pad")
    ap.add_argument("--bpu-preflight", action=argparse.BooleanOptionalAction, default=True, help="Run hrt_model_exec model_info before installing selected BPU adapters; catches HBRT IOVA failures early")
    ap.add_argument("--placeholder-renderer", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tensor-dump-dir", default=None, help="Optional directory for renderer/crop/model-input tensor summaries")
    ap.add_argument("--tensor-dump-full", action="store_true", help="Also save full-size compressed tensor npz files; summaries and 16x16 lowres dumps are always saved")
    ap.add_argument("--profile-timing", action="store_true", help="Print and save timing_profile.json with setup/frame/predictor/BPU-adapter wall-clock timings")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
