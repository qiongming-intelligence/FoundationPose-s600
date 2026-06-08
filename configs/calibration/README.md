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

## Capture hook

Wrap already-created upstream predictors:

```python
from foundationpose_s600_tools.debug.dump_intermediates import wrap_foundationpose_predictors

# after constructing the FoundationPose estimator / predictors
wrap_foundationpose_predictors(
    refine_predictor=est.refiner,     # or your PoseRefinePredictor instance
    score_predictor=est.scorer,       # or your ScorePredictor instance
    out_root="configs/calibration/data/raw_capture",
    limit=2000,
)
```

Or patch predictor constructors before they are created:

```python
from foundationpose_s600_tools.debug.dump_intermediates import install_auto_capture
install_auto_capture(out_root="configs/calibration/data/raw_capture", limit=2000)
```

Useful environment knobs:

```bash
export FOUNDATIONPOSE_S600_CAPTURE_DIR=configs/calibration/data/raw_capture
export FOUNDATIONPOSE_S600_CAPTURE_LIMIT=2000
export FOUNDATIONPOSE_S600_CAPTURE_SCORE_L=20   # historical real-tensor gate target; capture L32 too for the strict board target
```

The hook dumps `.npy` float32 tensors and never raises into the pose pipeline if a
capture write fails.

## Layout

```
configs/calibration/data/
  refine_net/
    A/000000.npy ... (N, 6, 160, 160) float32, one file per batch or per-sample
    B/000000.npy ...
  score_net_L20/
    A/000000.npy ... (20, 6, 160, 160)
    B/000000.npy ...
  score_net_L16/          # optional fallback
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

## Preparing `cal_data_dir` for hb_compile

Once captures exist in the layout above, validate and stage them with:

```bash
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.debug.prepare_calibration \
  --source-root configs/calibration/data/raw_capture \
  --out-root configs/calibration/data/hb_compile_real \
  --mode symlink
```

The helper writes one directory per partition input, e.g.
`configs/calibration/data/hb_compile_real/score_net_L20/A/*.npy`, and a
`manifest.json` containing the exact semicolon-separated `cal_data_dir` strings.

Compile calibrated candidates on the x86 toolchain host with:

```bash
HB_COMPILE_CALIBRATION_TYPE=max \
HB_COMPILE_CALIB_DATA_ROOT=configs/calibration/data/hb_compile_real \
src/python/scripts/compile_hbm.sh build/foundationpose_export/contracts models/hbm_real_calib
```

Do **not** treat `calibration_type: skip` as deployable. With hb_compile 3.5.3 it
still ran fixed/random calibration in our tests; the resulting ScoreNet HBM
collapsed logits to a constant and failed top-1/rank gates.
