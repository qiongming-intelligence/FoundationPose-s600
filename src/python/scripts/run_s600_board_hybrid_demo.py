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
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def install_placeholder_host_ops(max_faces: int = 3000, splat_radius: int = 1) -> None:
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
        """CPU triangle/point renderer for board smoke.

        This approximates nvdiffrast output enough to exercise crop assembly and
        BPU inference on non-zero rendered A-side tensors. It projects vertices,
        approximately fills a bounded number of projected triangles with a z-buffer,
        and then adds point splats to reduce holes. It does not do full texture UV
        interpolation or exact nvdiffrast semantics, so pose accuracy is not
        meaningful yet.
        """
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

        pts_full = torch.as_tensor(mesh_tensors["pos"], dtype=torch.float32).cpu()
        empty_faces = torch.empty((0, 3), dtype=torch.long, device="cpu")
        faces = torch.as_tensor(mesh_tensors.get("faces", empty_faces), dtype=torch.long).cpu()
        K_np = np.asarray(K, dtype=np.float32).reshape(3, 3)
        fx, fy, cx, cy = float(K_np[0, 0]), float(K_np[1, 1]), float(K_np[0, 2]), float(K_np[1, 2])
        use_light = bool(kwargs.get("use_light", False))
        light_dir = np.asarray(kwargs.get("light_dir", np.array([0, 0, 1])), dtype=np.float32).reshape(3)
        light_dir_neg = -light_dir / max(float(np.linalg.norm(light_dir)), 1e-8)
        w_ambient = float(kwargs.get("w_ambient", 0.8))
        w_diffuse = float(kwargs.get("w_diffuse", 0.5))

        tex_np = None
        uv_np = None
        uv_idx = faces
        has_tex = "tex" in mesh_tensors and "uv" in mesh_tensors

        def sample_texture_np(uv_values: np.ndarray) -> np.ndarray:
            if tex_np is None:
                uv_values = np.asarray(uv_values, dtype=np.float32)
                return np.full((*uv_values.shape[:-1], 3), 0.65, dtype=np.float32)
            uv_values = np.asarray(uv_values, dtype=np.float32)
            shape = uv_values.shape[:-1]
            uv_flat = uv_values.reshape(-1, 2)
            # nvdiffrast uses the UVs after make_mesh_tensors() flips v=1-v.
            # Clamp instead of wrapping for this smoke renderer; demo OBJ UVs are in-range.
            uu = np.clip(uv_flat[:, 0], 0.0, 1.0)
            vv = np.clip(uv_flat[:, 1], 0.0, 1.0)
            x = uu * (tex_np.shape[1] - 1)
            y = vv * (tex_np.shape[0] - 1)
            x0 = np.floor(x).astype(np.int64)
            y0 = np.floor(y).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, tex_np.shape[1] - 1)
            y1 = np.clip(y0 + 1, 0, tex_np.shape[0] - 1)
            wx = (x - x0).astype(np.float32)[:, None]
            wy = (y - y0).astype(np.float32)[:, None]
            c00 = tex_np[y0, x0]
            c10 = tex_np[y0, x1]
            c01 = tex_np[y1, x0]
            c11 = tex_np[y1, x1]
            out = (1 - wx) * (1 - wy) * c00 + wx * (1 - wy) * c10 + (1 - wx) * wy * c01 + wx * wy * c11
            return out.reshape(*shape, 3).astype(np.float32, copy=False)

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
                base_colors_full = torch.from_numpy(sample_texture_np(uv_np))
        if not has_tex and "vertex_color" in mesh_tensors:
            base_colors_full = torch.as_tensor(mesh_tensors["vertex_color"], dtype=torch.float32).cpu()
            if len(base_colors_full) != len(pts_full):
                base_colors_full = torch.full((len(pts_full), 3), 0.65, dtype=torch.float32, device="cpu")
        elif not has_tex:
            base_colors_full = torch.full((len(pts_full), 3), 0.65, dtype=torch.float32, device="cpu")
        normals_full = torch.as_tensor(mesh_tensors.get("vnormals", torch.zeros_like(pts_full)), dtype=torch.float32).cpu()

        # Triangle smoke path uses a bounded deterministic face subset for speed.
        all_face_indices = torch.arange(len(faces), dtype=torch.long, device="cpu")
        if len(faces) > max_faces:
            face_step = max(1, int(np.ceil(len(faces) / max_faces)))
            face_indices_smoke = all_face_indices[::face_step][:max_faces]
            faces_smoke = faces[face_indices_smoke]
        else:
            face_indices_smoke = all_face_indices
            faces_smoke = faces

        # Point-splat fallback/cleanup uses a bounded vertex subset.
        if len(pts_full) > 20000:
            vert_step = max(1, int(np.ceil(len(pts_full) / 20000)))
            pts = pts_full[::vert_step]
            base_colors = base_colors_full[::vert_step]
        else:
            pts = pts_full
            base_colors = base_colors_full
        pts_h = torch.cat([pts, torch.ones((len(pts), 1), dtype=torch.float32, device="cpu")], dim=1)
        pts_full_h = torch.cat([pts_full, torch.ones((len(pts_full), 1), dtype=torch.float32, device="cpu")], dim=1)

        bbox = None if bbox2d is None else torch.as_tensor(bbox2d, dtype=torch.float32).cpu().reshape(-1, 4)
        full_h = float(H if H is not None else h)
        full_w = float(W if W is not None else w)

        for i in range(n):
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

            if len(faces_smoke) > 0:
                x_np = x_all.numpy()
                y_np = y_all.numpy()
                z_np = z_all.numpy()
                valid_np = valid_all.numpy()
                cam_np = cam_all.numpy()
                col_np = base_colors_full.numpy()
                normals_cam_np = (poses[i, :3, :3].cpu() @ normals_full.T).T.numpy()
                uv_np_local = uv_np
                uv_idx_np = uv_idx.numpy() if has_tex else None
                face_indices_np = face_indices_smoke.numpy()
                for face_i, (f0, f1, f2) in enumerate(faces_smoke.tolist()):
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
                    tri_z = w0 * z_np[f0] + w1 * z_np[f1] + w2 * z_np[f2]
                    tri_xyz = (
                        w0[..., None] * cam_np[f0]
                        + w1[..., None] * cam_np[f1]
                        + w2[..., None] * cam_np[f2]
                    )
                    if has_tex and uv_np_local is not None and uv_idx_np is not None:
                        tu0, tu1, tu2 = [int(v) for v in uv_idx_np[int(face_indices_np[face_i])]]
                        tri_uv = (
                            w0[..., None] * uv_np_local[tu0]
                            + w1[..., None] * uv_np_local[tu1]
                            + w2[..., None] * uv_np_local[tu2]
                        )
                        tri_color_map = sample_texture_np(tri_uv)
                    else:
                        tri_color_map = (
                            w0[..., None] * col_np[f0]
                            + w1[..., None] * col_np[f1]
                            + w2[..., None] * col_np[f2]
                        ).astype(np.float32, copy=False)
                    if use_light:
                        tri_normal = (
                            w0[..., None] * normals_cam_np[f0]
                            + w1[..., None] * normals_cam_np[f1]
                            + w2[..., None] * normals_cam_np[f2]
                        )
                        norm = np.linalg.norm(tri_normal, axis=-1, keepdims=True)
                        tri_normal = tri_normal / np.maximum(norm, 1e-8)
                        diffuse = np.clip((tri_normal * light_dir_neg.reshape(1, 1, 3)).sum(axis=-1, keepdims=True), 0, 1)
                        tri_color_map = np.clip(tri_color_map * w_ambient + diffuse * tri_color_map * w_diffuse, 0, 1)
                    yy, xx = np.where(inside)
                    for local_y, local_x in zip(yy.tolist(), xx.tolist()):
                        y2 = ymin + local_y
                        x2 = xmin + local_x
                        zz = float(tri_z[local_y, local_x])
                        if zz <= 1e-4:
                            continue
                        if depth[i, y2, x2] == 0 or zz < float(depth[i, y2, x2]):
                            depth[i, y2, x2] = zz
                            xyz[i, y2, x2] = torch.from_numpy(tri_xyz[local_y, local_x])
                            color[i, y2, x2] = torch.from_numpy(tri_color_map[local_y, local_x])

            # Point splats fill holes left by the bounded triangle subset.
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
        if extra is not None:
            extra["xyz_map"] = xyz
        return color, depth, normal

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
    configure_bpu_preload(args, repo_root, refine_bpu=refine_bpu, score_bpu=score_bpu)
    preflight_bpu_hbms(args, repo_root, refine_bpu=refine_bpu, score_bpu=score_bpu)

    if args.cpu_cuda_compat:
        patch_torch_cpu_cuda_compat()

    from foundationpose_s600_tools.runtime import install_bpu_adapters
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    from datareader import YcbineoatReader
    from Utils import set_logging_format, set_seed, depth2xyzmap, draw_posed_3d_box, draw_xyz_axis
    import cv2
    import trimesh

    if args.placeholder_renderer:
        install_placeholder_host_ops(max_faces=args.renderer_max_faces, splat_radius=args.renderer_splat_radius)
    tensor_dump_dir = Path(args.tensor_dump_dir) if args.tensor_dump_dir else None
    if tensor_dump_dir is not None:
        install_tensor_dump_hooks(tensor_dump_dir, save_full=args.tensor_dump_full)

    set_logging_format()
    set_seed(0)

    mesh_file = Path(args.mesh_file)
    test_scene_dir = Path(args.test_scene_dir)
    if not mesh_file.is_absolute():
        mesh_file = upstream / mesh_file
    if not test_scene_dir.is_absolute():
        test_scene_dir = upstream / test_scene_dir

    mesh = trimesh.load(mesh_file)
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "ob_in_cam").mkdir(parents=True, exist_ok=True)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()

    if refine_bpu or score_bpu:
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
        install_predictor_dump_hooks(refiner, scorer, tensor_dump_dir)

    glctx = PlaceholderGlctx() if args.placeholder_renderer else None
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
    if args.max_hypotheses and len(est.rot_grid) > args.max_hypotheses:
        est.rot_grid = est.rot_grid[: args.max_hypotheses]
        print(f"limited rot_grid to {len(est.rot_grid)} hypotheses")

    reader = YcbineoatReader(video_dir=str(test_scene_dir), shorter_side=None, zfar=np.inf)
    n = min(len(reader.color_files), args.max_frames)
    print(f"reader frames={len(reader.color_files)} running={n}")

    for i in range(n):
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
        out_path = debug_dir / "ob_in_cam" / f"{reader.id_strs[i]}.txt"
        np.savetxt(out_path, pose.reshape(4, 4))
        if tensor_dump_dir is not None:
            save_frame_dump(tensor_dump_dir, reader.id_strs[i], pose, est)
        print(f"wrote {out_path}")

        if args.debug >= 1:
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imwrite(str(debug_dir / f"vis_{reader.id_strs[i]}.png"), vis[..., ::-1])

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
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
