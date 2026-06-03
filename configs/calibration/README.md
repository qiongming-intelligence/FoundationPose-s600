# Calibration data for FoundationPose S600 quantization

Calibration tensors are **not committed** (`configs/calibration/data/` is
git-ignored). This file documents how to collect and lay them out.

## Why real intermediates, not random images

`RefineNet` and `ScoreNetMultiPair` consume 6-channel crop pairs
(RGB 3 + XYZ map 3, 160x160) produced by nvdiffrast rendering + kornia warping
inside the FoundationPose pipeline. The XYZ channels are metric object-frame
coordinates with a very specific distribution — nothing like a natural image.
INT8/PTQ calibrated on random data will badly mis-scale the XYZ range and wreck
pose accuracy. Capture **real** `A`/`B` tensors from the running pipeline.

## What to capture

Dump the exact tensors fed to each network in
`learning/training/predict_pose_refine.py` / `predict_score.py`
(the `A = torch.cat([rgbAs, xyz_mapAs])`, `B = torch.cat([rgbBs, xyz_mapBs])`
lines), across a representative spread:

- multiple objects / meshes and scales;
- multiple viewpoints and distances (XYZ range coverage);
- the full refine-iteration trajectory (iteration 0..N deltas);
- good *and* bad pose hypotheses for ScoreNet (it must rank both);
- occlusion, depth noise, and clipping cases;
- both registration and tracking frames.

Aim for a few hundred to a few thousand pairs per network.
`src/python/foundationpose_s600_tools/debug/dump_intermediates.py` (hybrid path)
is the intended capture hook.

## Layout

```
configs/calibration/data/
  refine_net/
    A/000000.npy ... (N, 6, 160, 160) float32, one file per batch or per-sample
    B/000000.npy ...
  score_net_L16/
    A/000000.npy ... (16, 6, 160, 160)
    B/000000.npy ...
```

`.npy`/`.bin`/`.npz` are git-ignored. Record provenance (which dataset, which
frames, upstream SHA) in a sidecar `manifest.json` you keep out of git.

## Accuracy gate (before trusting any quantized variant)

- **RefineNet**: per-iteration trans/rot delta error vs FP32, and final
  ADD / ADD-S / pose drift over the tracking sequence.
- **ScoreNet**: top-1 and top-k agreement with FP32, rank correlation, and the
  final selected-pose outcome.

A quantized HBM that fails these gates is not deployable regardless of latency —
see docs/verification.md. Do not assume INT8 is faster: on the SAM_s600 detector
graph, whole-graph INT8 PTQ was *slower* than FP32 on S600.
