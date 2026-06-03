# Model export: FoundationPose subgraphs → ONNX

This covers exporting the two static FoundationPose subnetworks to ONNX so they
can be compiled to S600 HBM. Runs on the **x86 export host** (needs PyTorch,
omegaconf, and onnxruntime for `--verify`); contract generation alone is pure
Python and runs anywhere.

## The contract is the source of truth

`foundationpose_s600_tools.export.contract` emits one JSON per partition with
exact tensor names, concrete shapes, and dtypes. ONNX export, HBM compile, the
C++ runtime, and the Python hybrid path all read these names, so the chain stays
consistent. Regenerate after any shape change:

```bash
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.contract \
  --out-dir build/foundationpose_export --print-plan
```

Flags: `--image-size` (crop H=W, default 160), `--c-in` (6 = rgb+xyz shipped,
4 = legacy rgb+depth), `--rot-dim` (3 axis_angle / 6 6d), `--score-pairs`
(repeatable L values, default 16 and 64), `--partition` (subset).

### Verified upstream signatures (pinned SHA, see docs/upstream_pin.md)

```
RefineNet.forward(A, B)                # learning/models/refine_network.py
  A,B : (N, 6, 160, 160)               # cat(rgb 3, xyz_map 3)
  -> {'trans': (N,3), 'rot': (N,3|6)}  # axis_angle=3 (default) / 6d=6

ScoreNetMultiPair.forward(A, B, L)     # learning/models/score_network.py
  A,B : (B*L, 6, 160, 160)
  -> {'score_logit': (B, L)}           # export pins B=1, so N==L
```

The 6-channel input is assembled in the predictors as
`A = torch.cat([rgbAs, xyz_mapAs], dim=1)` — *not* the `c_in=4` constructor
fallback, which is a legacy rgb+depth path. Always set `--c-in` to match the
checkpoint you load.

## Export

```bash
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.refine \
  --contract build/foundationpose_export/contracts/refine_net.json --verify

PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.score \
  --contract build/foundationpose_export/contracts/score_net_L16.json --verify
```

If the official `model_best.pth` files are temporarily unavailable, you can smoke-test
the export/ABI path with random weights:

```bash
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.refine \
  --contract build/foundationpose_export/contracts/refine_net.json \
  --allow-random-init --verify
```

`--allow-random-init` is **not deployable**; it only validates that the upstream
module import, ONNX export, fixed tensor names/shapes, and ONNX Runtime parity all
work before the authorized weights are installed.

What the exporters do:

1. Locate vendored upstream (`third_party/FoundationPose`, or `--upstream-root`
   / `$FOUNDATIONPOSE_ROOT`) and put it on `sys.path`.
2. Pre-install an export-only lightweight `Utils` shim. Upstream model files do
   `from Utils import *`, but real `Utils.py` eagerly imports nvdiffrast,
   pytorch3d, open3d, warp, etc.; the subnets do not use those symbols during
   construction or forward export, so the shim keeps the x86 export env slim.
3. Load `weights/{run_name}/config.yml`, fill the same backward-compat defaults
   the upstream predictors do, then **build the `nn.Module` directly on CPU** —
   bypassing `PoseRefinePredictor`/`ScorePredictor`, which hard-code `.cuda()`
   and pull in nvdiffrast + H5 datasets we don't want in the graph.
4. Load `model_best.pth` weights, `eval()`.
5. Wrap the module so its dict output becomes an ordered named-tensor tuple
   (`trans,rot` / `score_logit`); for ScoreNet, `L` is constant-folded into the
   graph (no dynamic control flow).
6. `torch.onnx.export(opset=17, do_constant_folding=True, dynamic_axes=None)` —
   fixed shapes for the BPU. The default is PyTorch's legacy exporter with MHA
   fastpath disabled so Transformer layers decompose into standard ONNX ops.
7. `--verify`: run the ONNX under onnxruntime and compare to eager outputs.

Default run names (from the predictors): RefineNet `2023-10-28-18-33-37`,
ScoreNet `2024-01-11-20-02-45`. Override with `--run-name`.

## Optional precision conversion

The subgraph outputs are tiny, so output-only FP16 buys ~nothing here (unlike
SAM3 mask logits). The meaningful lever is whole-graph FP16/BF16:

```bash
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.convert.precision \
  models/onnx/foundationpose_refine_net.onnx \
  models/onnx/foundationpose_refine_net_fp16.onnx --keep-io-types
```

`output_precision.py` is kept for parity/experiments. Any reduced-precision
graph must pass the accuracy gate in [verification.md](verification.md) before
use — RefineNet is a continuous regressor iterated 5×, so it accumulates drift.

## Gotchas

- **No custom CUDA ops in the graph.** The subnets are plain conv + transformer;
  if export pulls in anything from `mycpp`/nvdiffrast, you exported too much —
  re-check you built the bare `nn.Module`, not the predictor.
- **BatchNorm vs no-BN.** `cfg.use_BN` decides norm layers; it comes from the
  checkpoint's config.yml. Don't override it.
- **Opset/exporter.** Default export uses the legacy PyTorch ONNX exporter with
  `opset=17`, because S600/HBDK compatibility is usually better there. On PyTorch
  2.12 the new dynamo exporter may auto-upgrade to opset 18 and can fail version
  conversion back to 17; use `--dynamo-exporter` only if the compile host accepts
  opset 18. The tooling disables `torch.backends.mha` fastpath for legacy export
  so `TransformerEncoderLayer` / `MultiheadAttention` become standard ONNX ops
  (`MatMul`, `Softmax`, `Gemm`, `LayerNormalization`) instead of fused aten ops.
