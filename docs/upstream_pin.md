# Upstream FoundationPose pin

This workspace adapts [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose)
to the Horizon / D-Robotics S600 BPU. The upstream source is **vendored locally but
not committed** to this repository: it is NVIDIA-proprietary and governed by the
upstream license. `third_party/FoundationPose/` is git-ignored, mirroring the
non-redistribution policy used by the `SAM_s600` reference project.

## Pinned commit

| Field  | Value |
|--------|-------|
| Remote | `https://github.com/NVlabs/FoundationPose` |
| SHA    | `a1b694b83e633c2cb6115b9063d940a687759392` |
| Title  | `Local Conda install: fix mycpp import, streamline dependencies, refresh readme (#407)` |
| Date   | `2026-04-29` |

## Reproduce locally

```bash
scripts/fetch_upstream.sh
```

This clones the pinned SHA into `third_party/FoundationPose/`. Re-run with a
different SHA via `FOUNDATIONPOSE_SHA=<sha> scripts/fetch_upstream.sh`.

## Why we don't import the whole pipeline onto the BPU

FoundationPose is a hybrid 6D-pose pipeline, not a single network:

- `estimater.py` drives registration/tracking control flow (dynamic Python).
- mesh setup, depth→XYZ, crop/warp, and **`nvdiffrast` CUDA rasterization** are
  not static neural graphs and cannot be compiled to HBM.
- `mycpp.cluster_poses`, SE(3) composition, candidate sort/top-k stay on CPU/GPU.

Only the two static learning subgraphs are BPU candidates:

- **`RefineNet`** — `learning/models/refine_network.py`, `forward(A, B)`.
- **`ScoreNetMultiPair`** — `learning/models/score_network.py`, `forward(A, B, L)`.

See `docs/rendering_split.md` for the full CPU/GPU-vs-BPU partition rationale.

## Ground-truth signatures (read from the pinned source)

Both subnets take a **6-channel** crop pair (RGB 3 + XYZ map 3), resized to
`160x160`. The legacy `c_in=4` default in the predictors is an RGB+depth
fallback for old checkpoints; the shipped checkpoints below use RGB+XYZ.

### RefineNet (`weights/2023-10-28-18-33-37/`)

```
forward(A, B)
  A, B : (N, 6, 160, 160)   # rgb(3) + xyz_map(3)
  ->
  trans : (N, 3)            # decoded with trans_rep='tracknet' (tanh * trans_normalizer)
  rot   : (N, 3)            # rot_rep='axis_angle' default (3); '6d' -> 6
```

### ScoreNetMultiPair (`weights/2024-01-11-20-02-45/`)

```
forward(A, B, L)
  A, B : (B*L, 6, 160, 160)   # rgb(3) + xyz_map(3)
  L    : pairs per group
  ->
  score_logit : (B, L)
```

Config (`input_resize`, `c_in`, `rot_rep`, `use_BN`, `crop_ratio`, normalizers)
is loaded from `weights/{run_name}/config.yml` at runtime. The export tooling
reads those values and writes them into the export contract so ONNX/HBM/C++
shapes stay consistent.
