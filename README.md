# FoundationPose S600

A workspace for adapting [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose)
to the Horizon Robotics / D-Robotics **S600 BPU**, following the engineering
pattern proven in the prior [`SAM_s600`](https://github.com/qiongming-intelligence/SAM_s600)
project.

## What this is (and isn't)

FoundationPose is a **hybrid 6D-pose pipeline**, not a single neural network. Its
`estimater.py` drives registration/tracking control flow and depends on
`nvdiffrast` CUDA rasterization, mesh/depth geometry, and two learning
subnetworks. Only the two **static** subnetworks are BPU candidates:

| Subgraph | Upstream | Role | Input | Output |
|---|---|---|---|---|
| `RefineNet` | `learning/models/refine_network.py` | pose refinement | `A,B (N,6,160,160)` | `trans (N,3)`, `rot (N,3)` |
| `ScoreNetMultiPair` | `learning/models/score_network.py` | candidate scoring | `A,B (L,6,160,160)` | `score_logit (1,L)` |

`A`/`B` are crop pairs: **RGB(3) + XYZ-map(3) = 6 channels**, resized to
`160x160`. Everything else — rendering, depth→XYZ, crop/warp, pose decode, SE(3)
composition, candidate selection — **stays on CPU/GPU/Python**. See
[docs/rendering_split.md](docs/rendering_split.md).

This repo does **not** redistribute FoundationPose source or weights (NVIDIA
proprietary). Upstream is vendored locally but git-ignored; see
[docs/upstream_pin.md](docs/upstream_pin.md).

## Two-host workflow

The toolchain splits across two machines (same as `SAM_s600`):

- **x86_64 export/compile host** — PyTorch ONNX export + `hb_compile` (AI
  Toolchain 3.7.0+). Neither runs on aarch64.
- **aarch64 S600 runtime box** — `hrt_model_exec` benchmarking, the C++ runtime,
  and the Python hybrid path. Contract generation (pure Python) also runs here.

## Pipeline at a glance

```
upstream weights ──► export ONNX (x86) ──► [opt] precision cast ──► hb_compile HBM (x86)
                                                                          │
                                          ┌───────────────────────────────┘
                                          ▼
                          inspect + benchmark on S600  ──►  C++ raw runner / Python hybrid
```

## Quick start

### 1. Vendor upstream (any host)

```bash
scripts/fetch_upstream.sh        # clones the pinned SHA into third_party/ (git-ignored)
```

### 2. Generate export contracts (any host — pure Python)

```bash
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.contract \
  --out-dir build/foundationpose_export --print-plan
```

Produces `contracts/refine_net.json`, `contracts/score_net_L16.json`,
`contracts/score_net_L64.json`, and `export_index.json`. Override shapes with
`--image-size`, `--c-in`, `--rot-dim`, `--score-pairs` to match your checkpoint.

### 3. Export ONNX (x86 host, needs torch + authorized weights)

```bash
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.refine \
  --contract build/foundationpose_export/contracts/refine_net.json --verify

PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.score \
  --contract build/foundationpose_export/contracts/score_net_L16.json --verify
```

`--verify` compares ONNX Runtime vs PyTorch outputs. See
[docs/model_export.md](docs/model_export.md).

### 4. Compile HBM (x86 toolchain host)

```bash
src/python/scripts/compile_hbm.sh build/foundationpose_export/contracts models/hbm
# or sweep variants:
src/python/scripts/compile_matrix.sh \
  build/foundationpose_export/contracts/refine_net.json models/hbm
```

FP32 baseline first (`calibration_type: skip`), `march nash-p`, `core_num=2`,
`O2 latency`. Quantization only after the accuracy gate
([docs/verification.md](docs/verification.md)).

### 5. Inspect + benchmark on S600

```bash
src/python/scripts/inspect_hbm.sh \
  models/hbm/foundationpose_refine_net.hbm \
  build/foundationpose_export/contracts/refine_net.json

src/python/scripts/bench_subgraph.sh \
  models/hbm/foundationpose_refine_net.hbm refine_net <input_dir> 30
```

## Repository layout

```
src/python/foundationpose_s600_tools/
  export/   contract.py  wrappers.py  refine.py  score.py
  convert/  output_precision.py  precision.py
  runtime/  debug/         (hybrid BPU adapters — to come)
src/python/scripts/   compile_hbm.sh  compile_matrix.sh  bench_subgraph.sh  inspect_hbm.sh
configs/manifests/    foundationpose_refine.yaml  foundationpose_score_L16.yaml  foundationpose_tracking_hybrid.yaml
configs/calibration/  README.md      (real-intermediate capture policy)
models/onnx  models/hbm  (git-ignored artifacts)
docs/        upstream_pin.md  model_export.md  s600_runtime.md  rendering_split.md  verification.md
third_party/FoundationPose/   (vendored upstream, git-ignored)
```

## Status

- [x] Upstream vendored + pinned (`docs/upstream_pin.md`)
- [x] Export contracts (RefineNet, ScoreNet L16/L64) — generated + smoke-tested
- [x] PyTorch→ONNX export wrappers (x86 host)
- [x] ONNX precision-conversion tooling
- [x] `hb_compile` scripts, manifests, calibration policy
- [ ] C++ raw-tensor BPU runner (plan §5)
- [ ] Python hybrid integration into the FoundationPose loop (plan §6)
- [ ] End-to-end profile and variant selection

## Documentation

- [docs/upstream_pin.md](docs/upstream_pin.md) — vendored upstream SHA + signatures
- [docs/model_export.md](docs/model_export.md) — contract → ONNX export details
- [docs/s600_runtime.md](docs/s600_runtime.md) — HBM compile + S600 runtime notes
- [docs/rendering_split.md](docs/rendering_split.md) — what runs on BPU vs host, and why
- [docs/verification.md](docs/verification.md) — accuracy gates and milestones

## License

Upstream FoundationPose is NVIDIA-proprietary and governed by its own license;
this repo neither includes nor redistributes it. The S600 adaptation tooling
here is the original contribution of this workspace.
