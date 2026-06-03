#!/usr/bin/env python3
"""Shared PyTorch→ONNX export helpers for FoundationPose S600 subgraphs.

These build ``RefineNet`` / ``ScoreNetMultiPair`` directly from the upstream
config + checkpoint, on CPU, **without** going through the ``PoseRefinePredictor``
/ ``ScorePredictor`` classes. The predictors hard-code ``.cuda()`` and pull in
nvdiffrast, H5 datasets, and the full crop pipeline -- none of which we want in
an export graph. We only need the ``nn.Module`` and its weights.

torch / omegaconf are imported lazily so this module can be inspected on the
aarch64 S600 box (which has no torch); the actual export must run on the x86
export host.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import types
from pathlib import Path
from typing import Any


def upstream_root(explicit: str | None = None) -> Path:
    """Locate the vendored upstream FoundationPose checkout."""
    if explicit:
        root = Path(explicit)
    elif os.environ.get("FOUNDATIONPOSE_ROOT"):
        root = Path(os.environ["FOUNDATIONPOSE_ROOT"])
    else:
        # repo_root/third_party/FoundationPose
        repo_root = Path(__file__).resolve().parents[4]
        root = repo_root / "third_party" / "FoundationPose"
    if not (root / "learning" / "models").is_dir():
        raise SystemExit(
            f"upstream FoundationPose not found at {root}; run scripts/fetch_upstream.sh "
            "or set FOUNDATIONPOSE_ROOT"
        )
    return root


def add_upstream_to_path(root: Path) -> None:
    """Make upstream importable (it uses bare ``from Utils import *`` etc.)."""
    for sub in (str(root), str(root / "learning" / "models")):
        if sub not in sys.path:
            sys.path.insert(0, sub)


def install_utils_shim(force: bool = False) -> bool:
    """Pre-register a lightweight stand-in for upstream ``Utils``.

    ``learning/models/{refine,score}_network.py`` and ``network_modules`` open
    with ``from Utils import *``. Upstream ``Utils.py`` eagerly imports the whole
    rendering/geometry stack (pytorch3d, nvdiffrast, open3d, warp, kornia, ...),
    none of which the two conv+transformer subnets actually use at construction
    or in ``forward``. Installing a stub module under the name ``Utils`` *before*
    importing the model files lets the export host stay slim (just torch + onnx +
    omegaconf), instead of provisioning the full CUDA renderer toolchain.

    Set ``FOUNDATIONPOSE_NO_UTILS_SHIM=1`` (or pass ``force=False`` with the env
    unset) to skip and use the real Utils, e.g. to validate the assumption.
    Returns True if a shim was installed.
    """
    if os.environ.get("FOUNDATIONPOSE_NO_UTILS_SHIM") == "1" and not force:
        return False
    if "Utils" in sys.modules and not force:
        return False
    shim = types.ModuleType("Utils")
    shim.__doc__ = "Export-only stub for FoundationPose Utils (no rendering deps)."
    shim.__FOUNDATIONPOSE_S600_SHIM__ = True
    sys.modules["Utils"] = shim
    return True


def load_config(root: Path, run_name: str, overrides: dict[str, Any] | None = None) -> Any:
    """Load ``weights/{run_name}/config.yml`` and apply backward-compat defaults.

    Mirrors the default-filling done in the upstream predictor ``__init__`` so the
    constructed module matches the shipped checkpoint.
    """
    from omegaconf import OmegaConf  # lazy

    cfg_path = root / "weights" / run_name / "config.yml"
    if not cfg_path.is_file():
        raise SystemExit(f"missing upstream config: {cfg_path} (download authorized weights first)")
    cfg = OmegaConf.load(cfg_path)

    defaults = {
        "use_normal": False,
        "use_mask": False,
        "use_BN": False,
        "c_in": 4,
        "crop_ratio": 1.2,
        "n_view": 1,
        "trans_rep": "tracknet",
        "rot_rep": "axis_angle",
        "normalize_xyz": False,
    }
    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = value
    if overrides:
        for key, value in overrides.items():
            cfg[key] = value
    return cfg


def load_checkpoint_state(root: Path, run_name: str, model_name: str = "model_best.pth") -> dict[str, Any]:
    import torch  # lazy

    ckpt_path = root / "weights" / run_name / model_name
    if not ckpt_path.is_file():
        raise SystemExit(f"missing upstream checkpoint: {ckpt_path} (download authorized weights first)")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    return ckpt


class NamedTupleOutput:
    """Wrap a module so its dict output becomes an ordered tuple of named tensors.

    ``torch.onnx.export`` needs positional tensor outputs whose names we control.
    The contract fixes the order; ``output_keys`` selects+orders the dict.
    """

    def __new__(cls, torch_mod: Any, wrapped: Any, output_keys: list[str], call_adapter):
        class _Wrapper(torch_mod.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.wrapped = wrapped
                self.output_keys = list(output_keys)
                self._call = call_adapter

            def forward(self, *args: Any):
                out = self._call(self.wrapped, *args)
                if isinstance(out, dict):
                    return tuple(out[k] for k in self.output_keys)
                if isinstance(out, (tuple, list)):
                    return tuple(out)
                return (out,)

        module = _Wrapper()
        module.eval()
        return module


def dummy_inputs(torch_mod: Any, contract: dict[str, Any]) -> tuple[Any, ...]:
    """Build zero tensors matching the contract input specs (CPU, float32)."""
    tensors = []
    for item in contract["inputs"]:
        dtype = getattr(torch_mod, str(item["dtype"]))
        tensors.append(torch_mod.zeros(list(item["concrete_shape"]), dtype=dtype))
    return tuple(tensors)


def read_contract(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_onnx(
    torch_mod: Any,
    wrapper: Any,
    inputs: tuple[Any, ...],
    contract: dict[str, Any],
    opset: int = 17,
    dynamo: bool = False,
) -> Path:
    onnx_path = Path(contract["onnx_path"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    input_names = [t["name"] for t in contract["inputs"]]
    output_names = [t["name"] for t in contract["outputs"]]
    if hasattr(wrapper, "eval"):
        wrapper.eval()
    # PyTorch eval mode may route TransformerEncoderLayer/MultiheadAttention
    # through fused fastpaths such as aten::_transformer_encoder_layer_fwd, which
    # the legacy ONNX exporter cannot lower to opset 17. Disable the fastpath so
    # the graph decomposes into standard MatMul/Softmax/Gemm/LayerNorm ops.
    mha_backend = getattr(getattr(torch_mod, "backends", None), "mha", None)
    if mha_backend is not None and hasattr(mha_backend, "set_fastpath_enabled"):
        mha_backend.set_fastpath_enabled(False)
    export_kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamic_axes": None,  # fixed shapes for BPU
    }
    if "dynamo" in inspect.signature(torch_mod.onnx.export).parameters:
        export_kwargs["dynamo"] = dynamo  # False = legacy exporter (keeps opset 17 on torch 2.12)
    with torch_mod.no_grad():
        torch_mod.onnx.export(wrapper, inputs, str(onnx_path), **export_kwargs)
    return onnx_path


def verify_against_torch(
    torch_mod: Any,
    wrapper: Any,
    inputs: tuple[Any, ...],
    onnx_path: Path,
    output_names: list[str],
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> dict[str, float]:
    """Compare ONNX Runtime outputs against the eager wrapper. Returns max abs diff per output."""
    import numpy as np  # lazy
    import onnxruntime as ort  # lazy

    with torch_mod.no_grad():
        ref = wrapper(*inputs)
    ref = [r.detach().cpu().numpy() for r in ref]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[i].name: inputs[i].detach().cpu().numpy() for i in range(len(inputs))}
    got = sess.run(None, feed)

    diffs: dict[str, float] = {}
    for name, r, g in zip(output_names, ref, got):
        max_abs = float(np.max(np.abs(r - g)))
        diffs[name] = max_abs
        ok = np.allclose(r, g, rtol=rtol, atol=atol)
        print(f"  {name}: max|Δ|={max_abs:.3e} {'OK' if ok else 'MISMATCH'}")
    return diffs
