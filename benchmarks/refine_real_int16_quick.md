# Refine/Score real-tensor quick accuracy and board benchmark

Context: the user hard constraint is full BPU placement for the learning subgraphs, especially RefineNet ("必须得都放bpu"). This note records the precision-first S600 path using real captured FoundationPose driller tensors, not synthetic random input.

## Precision policy

Deployable BPU candidates use:

- `calibration_type=max`
- real captured calibration tensors
- `optimization=set_all_nodes_int16`
- `core_num=1`
- FLOAT32 model outputs for the preferred ABI (`set_model_output_int16` is diagnostic only)

S600 `hb_compile` has no true all-FP32 BPU path for these graphs, so the highest-precision deployable path is all-node int16 with real calibration and FLOAT32 outputs.

## RefineNet default: full all-node-int16 / FLOAT32-output HBM

HBM:

```text
models/hbm_real_int16/foundationpose_refine_net_real_int16_no_output.hbm
```

Compile metadata from `hrt_model_exec model_info`:

- `BUILDER_VERSION`: 3.5.3
- `HBDK_VERSION`: 4.7.5
- `HMCT_VERSION`: 2.6.5
- `MARCH`: nash-p
- `OPTIMIZATION`: `set_all_nodes_int16`
- `CALI_TYPE`: max
- `CORE_NUM`: 1
- Inputs: `A`, `B`, each valid shape `(1,6,160,160)`
- Outputs: `trans`, `rot`, each valid shape `(1,3)`, FLOAT32
- HBM size: ~24 MB

Board loadability on UCP 3.13.6 / HBRT 4.7.5:

```text
normal model_info: IOVA_FAIL 0/8
with build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so: OK 8/8
```

The preload shim overrides only `hb_bpu_core_num()` to report one visible BPU core. It works around HBRT 4.7.5's pre-scheduling cross-core IOVA equality check for single-core deployment; it does not modify the HBM, graph placement, or precision.

Board real-tensor gate (`validate_hbm_real.py compare`, 8 real driller samples, persistent backend + preload):

```text
SUMMARY trans max_abs_max=0.0158056
SUMMARY rot   max_abs_max=0.0317621
```

Board HBM perf (`hrt_model_exec perf`, `core_id=1`, preload):

```text
Frame count: 100
Average latency: 1.957 ms
Frame rate: 507.666 FPS
BPU_inference_time_cost avg: 1.93282 ms
CPU_inference_time_cost: 0.0 ms
```

This is the current default Refine BPU target. The older micro-BPU recovery HBM remains fallback/control only because node inspection reports CPU fallback for most RefineNet nodes.

## Split all-BPU Refine probes

The split all-BPU HBMs also load with the one-core preload:

```text
models/hbm_refine_split_20260608/refine_encodeA.hbm
models/hbm_refine_split_20260608/refine_encodeAB_pos.hbm
models/hbm_refine_split_20260608/refine_trans_head.hbm
models/hbm_refine_split_20260608/refine_rot_head.hbm
```

Normal `model_info` fails with the same IOVA check; preload `model_info` is `OK 8/8` for each. Manual hrt chaining showed the split path only works when padded/aligned intermediate tensors are preserved. Stripping intermediate padding caused large numerical errors, so the full Refine HBM is preferred for runtime simplicity.

## ScoreNet L20 and L32

### L20 historical gate

HBM:

```text
models/hbm_real_int16/foundationpose_score_net_L20_real_int16_no_output.hbm
```

Board loadability:

```text
normal model_info: IOVA_FAIL 0/8
with one-core preload: OK 8/8
```

Real-driller ranking gate over 9 L20 groups with preload:

```text
SUMMARY score_logit top1=8/9 top5_mean=4.556 top10_mean=9.444 spearman_mean=0.986466 rho_mean=0.992366 max_abs_max=1.47508
```

Compile perf reference: ~24.73 ms.

### L32 strict board target

HBM:

```text
models/hbm_real_int16/foundationpose_score_net_L32_real_int16_no_output.hbm
```

Board loadability is state-sensitive; latest policy is to use the same one-core preload for benchmark/deployment consistency. With preload, `model_info` passed.

ONNX export/verify on ws-wan matched eager PyTorch:

```text
score_logit max|Δ|=7.629e-06
```

Strict-32 real-driller tensor gate over 6 groups via persistent HBM compare:

```text
SUMMARY score_logit top1=6/6 top5_mean=5.000 top10_mean=9.333 spearman_mean=0.978861 rho_mean=0.992902 max_abs_max=1.70461
```

Board HBM perf (`hrt_model_exec perf`, `core_id=1`, preload):

```text
Frame count: 50
Average latency: 41.213 ms
Frame rate: 24.253 FPS
BPU_inference_time_cost avg: 41.18178 ms
CPU_inference_time_cost: 0.0 ms
```

## Single-frame E2E smoke and benchmark

Command family:

```text
run_s600_board_hybrid_demo.py
  --max-frames 1
  --max-hypotheses 32
  --renderer-splat-radius 0
  --est-refine-iter 1
  --track-refine-iter 1
  --score-chunk-size 32
  --score-partition score_net_L32
  --score-bpu-mode strict
  --bpu-runtime persistent
```

One-frame wall-clock includes Python/import setup, model/preflight/HBM load, renderer/crop/host control flow, inference, output writing, and process cleanup. It is not pure neural-subgraph latency.

| Mode | Preload | Wall-clock |
|---|---:|---:|
| CPU-only | no | 118.976 s |
| Refine CPU + Score BPU L32 | yes | 117.258 s |
| Full BPU: Refine BPU + Score BPU L32 | yes | 145.105 s |

Output pose deltas vs CPU-only on the driller smoke frame:

```text
Score-BPU vs CPU-only:
  translation_m  = 0
  rotation_deg   = 0
  matrix_max_abs = 0

Full-BPU vs CPU-only:
  translation_m  = 0.000803929445721049
  rotation_deg   = 0.496347800284933
  matrix_max_abs = 0.00618880987167358
```

Interpretation:

- Precision smoke is OK for the current single frame: full-BPU is ~0.804 mm / 0.496° from CPU-only.
- Score-BPU alone selected the same final pose as CPU-only.
- Current demo-level full-BPU wall-clock is slower than CPU-only because the pipeline is dominated by Python/startup/host rendering and the Refine adapter loops an `N=1` HBM over hypotheses through JSONL + temporary `.bin` files.

## First performance optimization pass

Added `--profile-timing` to `run_s600_board_hybrid_demo.py` to split setup, register/track, crop, predictor, and BPU-adapter wall time. With full-BPU, 32 hypotheses, no preflight, default placeholder renderer (`renderer_max_faces=3000`, `renderer_splat_radius=0`), the profile showed the bottleneck was host crop/render, not BPU inference:

```text
frame.register                         99.475 s
crop.refine_make_crop_data_batch        48.613 s
crop.score_make_crop_data_batch         50.489 s
adapter.refine.runner_infer total        0.218 s  (32 calls, avg 6.80 ms/call)
adapter.score.runner_infer total         0.066 s  (1 call)
```

A fast profiling mode now accepts `--renderer-max-faces 0` to disable placeholder triangle fill and keep only point projection. The point projection path for `--renderer-splat-radius 0` was vectorized with numpy nearest-z grouping instead of looping in Python over projected vertices. Measured effects:

| Full-BPU mode, 32 hypotheses, no preflight | Wall-clock | `frame.register` | Refine crop | Score crop |
|---|---:|---:|---:|---:|
| Before timing/renderer optimization, default renderer | 120.741 s | ~99.475 s | 48.613 s | 50.489 s |
| Triangle fill disabled before vectorized point path | 89.612 s | 69.663 s | 34.416 s | 34.875 s |
| Triangle fill disabled + vectorized point path | 22.022 s | 2.078 s | 0.531 s | 1.178 s |
| Default 3000-face triangle fill + vectorized point path | 54.613 s | 34.632 s | 16.220 s | 18.037 s |

Default-renderer optimized pose remained close to CPU-only on the smoke frame:

```text
Full-BPU optimized default renderer vs CPU-only:
  translation_m  = 0.000529465377433024
  rotation_deg   = 0.459817005893148
  matrix_max_abs = 0.00710770487785339
```

The remaining default-renderer wall time is mostly the Python triangle-fill placeholder path. For throughput profiling, `--renderer-max-faces 0 --renderer-splat-radius 0` isolates the BPU/host-control overhead and reduces one-frame wall-clock to ~22 s including imports/setup/HBM load.

### No-setup internal frame timings

For performance comparisons, prefer the internal `frame.register` span from `--profile-timing`; this excludes Python import, model construction, preflight, HBM load, and process teardown. Latest one-frame / 32-hypothesis timings:

Default placeholder renderer (`--renderer-max-faces 3000 --renderer-splat-radius 0`):

| Mode | `frame.register` | Refine predict | Score predict | BPU runner time |
|---|---:|---:|---:|---:|
| CPU-only | 41.261 s | 19.863 s | 21.334 s | n/a |
| Refine CPU + Score BPU L32 | 37.827 s | 19.931 s | 17.833 s | Score 65.9 ms |
| Full BPU: Refine BPU + Score BPU L32 | 34.486 s | 16.504 s | 17.920 s | Refine 215.6 ms across 32 calls; Score 65.2 ms |

Points-only profiling renderer (`--renderer-max-faces 0 --renderer-splat-radius 0`):

| Mode | `frame.register` | Refine predict | Score predict | BPU runner time |
|---|---:|---:|---:|---:|
| CPU-only | 9.466 s | 4.582 s | 4.821 s | n/a |
| Refine CPU + Score BPU L32 | 5.383 s | 3.996 s | 1.324 s | Score 65.6 ms |
| Full BPU: Refine BPU + Score BPU L32 | 2.083 s | 0.745 s | 1.277 s | Refine 213.2 ms across 32 calls; Score 66.5 ms |

No-setup interpretation:

- With the default placeholder renderer, full-BPU is now faster than CPU-only for the measured frame (`34.486 s` vs `41.261 s`), but the majority of time is still host crop/render.
- With the points-only profiling renderer, full-BPU is ~4.5× faster than CPU-only at the `frame.register` level (`2.083 s` vs `9.466 s`).
- Refine BPU runner IPC still costs ~6.7 ms per N=1 call, but total Refine BPU runner time is only ~0.21 s for 32 hypotheses; the larger remaining target is host render/crop and then batched Refine/IPC cleanup.

### Persistent runner scratch reuse probe

For real deployment work, setup is excluded and the next relevant BPU-adapter target is per-inference IPC. A low-risk change made `PersistentBpuModelRunner` reuse one scratch/input/output directory per loaded model instead of creating/removing a temporary directory for every inference. Microbenchmark over repeated Refine N=1 calls:

```text
before scratch reuse: avg 6.684 ms/call over 32 calls
 after scratch reuse: avg 6.628 ms/call over 64 calls
```

The gain is negligible (~0.06 ms/call). The remaining ~4.7 ms/call over raw HBM latency is dominated by tensor `.bin` write/read, JSONL request/response, and C++ runner file I/O rather than directory lifecycle. Meaningful real deployment optimization should therefore use a deeper IPC/batching change, not just filesystem scratch reuse.

### Batched Refine HBM evaluation

The export/runtime path now supports batched Refine variants such as `refine_net_N8`, `refine_net_N16`, and `refine_net_N32`:

```bash
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.contract \
  --out-dir build/foundationpose_export \
  --refine-batch 1 --refine-batch 8 --refine-batch 16 \
  --partition refine_net_N16 --print-plan
```

Generated shape family:

```text
refine_net_N{K}:
  A,B   [K, 6, 160, 160]
  trans [K, 3]
  rot   [K, 3]
```

`RefineNetBpu` supports compiled `N>1` contracts by chunking input batches by the HBM's contract N, padding only the final partial chunk, and clipping padded outputs. Board follow-up found that batched Refine outputs can have per-row padding even when the valid output shape is small. For example, `refine_net_N8` and `refine_net_N16` report `trans/rot` valid shapes `(K,3)` but output stride `(16,4)`, i.e. four float slots per row. The persistent native runner now writes compact output tensors by walking HBM stride metadata instead of copying the first `K*3*sizeof(float)` bytes blindly.

Compiled precision-first batched candidates on ws-wan:

| Candidate | HBM | sha256 | Size | Calibration samples |
|---|---|---|---:|---:|
| N8 | `models/hbm_real_int16_batch/foundationpose_refine_net_N8.hbm` | `abca443dc48a3da221b65fe191877e375d9776e6757498756080675adfde0ed5` | 41 MB | 3 N8 batches / 24 real samples |
| N16 | `models/hbm_real_int16_batch/foundationpose_refine_net_N16.hbm` | `2c1df67aa59761793b605a4bca3ec8ed3a71cfde17165286a521b4975c9d21cb` | 70 MB | 6 N16 batches / 96 real samples available; gate below used 3 batches / 48 real samples |

Both use `calibration_type=max`, real captured tensors, `optimization=set_all_nodes_int16`, `core_num=1`, `compile_mode=latency`, `O2`, and FLOAT32 outputs. Both load on S600 with the one-core preload shim.

S600 preload `model_info` highlights:

```text
refine_net_N8:
  A,B   valid shape (8,6,160,160),  F32
  trans valid shape (8,3),          F32, stride (16,4)
  rot   valid shape (8,3),          F32, stride (16,4)

refine_net_N16:
  A,B   valid shape (16,6,160,160), F32
  trans valid shape (16,3),         F32, stride (16,4)
  rot   valid shape (16,3),         F32, stride (16,4)
```

S600 raw HBM perf (`hrt_model_exec perf`, `core_id=1`, preload, 100 frames):

| Candidate | Average latency | Per-hypothesis raw equivalent | FPS |
|---|---:|---:|---:|
| N1 full Refine | 1.957 ms / batch1 | 1.957 ms | 507.666 |
| N8 | 12.533 ms / batch8 | 1.57 ms | 79.689 |
| N16 | 24.502 ms / batch16 | 1.53 ms | 40.781 |

Real-tensor gates after output depadding fix:

| Candidate | Real samples gated | `trans max_abs_max` | `rot max_abs_max` | Notes |
|---|---:|---:|---:|---|
| N1 full Refine | 8 | 0.0158056 | 0.0317621 | historical quick gate |
| N8 | 24 | 0.0222071 | 0.0608479 | slightly looser than N1 |
| N16 | 48 | 0.0222071 | 0.0608479 | essentially same max as N8 on covered samples |

Full-BPU no-setup E2E timing with ScoreNet L32, one frame / 32 hypotheses:

| Refine HBM | Renderer | `frame.register` | Refine predict | Refine runner | Score predict | Score runner |
|---|---|---:|---:|---:|---:|---:|
| N1 | points-only | 2.083 s | 0.745 s | 213.2 ms across 32 calls | 1.277 s | 66.5 ms |
| N8 | points-only | 1.961 s | 0.641 s | 92.5 ms across 4 calls | 1.259 s | 62.1 ms |
| N16 | points-only | 2.088 s | 0.684 s | 83.4 ms across 2 calls | 1.342 s | 62.6 ms |
| N1 | default placeholder | 34.486 s | 16.504 s | 215.6 ms across 32 calls | 17.920 s | 65.2 ms |
| N8 | default placeholder | 34.859 s | 16.533 s | 93.5 ms across 4 calls | 18.265 s | 61.9 ms |
| N16 | default placeholder | 34.591 s | 16.507 s | 83.4 ms across 2 calls | 18.022 s | 63.0 ms |

Interpretation:

- N8 removes ~120 ms of Refine adapter overhead in the points-only profiling run and reduces calls from 32 to 4.
- N16 reduces calls further to 2, but only improves Refine runner time by another ~9 ms versus N8; run-to-run host crop/render variance is larger than this delta.
- Default placeholder renderer remains dominated by CPU crop/render (~16–18 s per Refine/Score crop), so batched Refine does not materially move the default-renderer `frame.register` number.
- N8 is the better practical batched Refine candidate right now: much smaller HBM (41 MB vs 70 MB), nearly all adapter-call reduction benefit, same single-frame final pose as N1/N16, and less memory pressure. N16 is valid but offers marginal additional runtime benefit for 32 hypotheses.

### Native placeholder renderer optimization

The Python placeholder triangle loop was moved behind a small native C++ raster helper (`src/csrc/tools/placeholder_renderer.cpp`, built as `build/cmake/src/csrc/libfoundationpose_placeholder_renderer.so`). This keeps the same smoke-renderer API used by the upstream crop builders and still labels the output as non-production / not ADD-valid, but removes the pose × face × pixel Python loop. If the shared library is absent, the Python fallback remains available.

Targeted verification:

```text
PYTHONPATH=src/python /home/sunrise/miniconda3/envs/sam3-export/bin/python -m pytest \
  src/python/tests/test_placeholder_renderer.py \
  src/python/tests/test_runtime_adapters.py \
  src/python/tests/test_contract_defaults.py
# 22 passed
```

Full-BPU no-setup E2E timing with N8 Refine + ScoreNet L32, one frame / 32 hypotheses, after the native placeholder renderer:

| Renderer | `frame.register` | Refine crop | Score crop | Render total | Triangle span | Point span | Refine runner | Score runner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native default placeholder (`--renderer-max-faces 3000 --renderer-splat-radius 0`) | 2.100 s | 0.597 s | 1.261 s | 1.042 s across 2 calls | 150.0 ms across 64 pose renders | 506.3 ms across 64 pose renders | 92.7 ms across 4 calls | 62.3 ms |
| points-only profiling (`--renderer-max-faces 0 --renderer-splat-radius 0`) | 2.144 s | 0.590 s | 1.310 s | 0.936 s across 2 calls | 0.7 ms across 64 pose renders | 533.9 ms across 64 pose renders | 93.5 ms across 4 calls | 62.0 ms |

Follow-up host/runtime optimizations then:

- patched upstream `depth2xyzmap` in the demo to use the cached-grid/native `depth2xyzmap_fast` helper;
- reused the register-time frame XYZ map in the Score crop patch, reducing `crop.score_depth2xyzmap_full` from ~13 ms to ~0.12 ms;
- stopped rendering/warping unused Score depth tensors;
- cached immutable placeholder mesh preparation across Refine/Score render calls and prewarmed it during setup;
- added a batched native placeholder renderer entry point so all crop poses render through one native call per crop builder;
- moved persistent BPU runner scratch files to `/dev/shm` when available;
- replaced observed RGB/XYZ crop warps with a native/OpenMP batch perspective warp using the same half-pixel transform convention as the OpenCV control path;
- inlined RGB/XYZ crop normalization with a native helper for already-materialized `xyz_mapAs`/`xyz_mapBs`.

Latest measured full-BPU no-setup timing, one driller smoke frame, default placeholder renderer (`--renderer-max-faces 3000 --renderer-splat-radius 0`):

| Runtime path | Hypotheses | `frame.register` | Refine crop | Score crop | Render total | Refine runner | Score runner |
|---|---:|---:|---:|---:|---:|---:|---:|
| N8 Refine + Score L32, native batch placeholder + frame XYZ reuse + tmpfs scratch | 32 | 0.464 s | 150.2 ms | 125.8 ms | 111.3 ms across 2 calls | 76.3 ms across 4 calls | 55.8 ms |
| N8 Refine + Score L32, + OpenMP render + OpenCV crop warps | 32 | 0.313 s | 71.9 ms | 52.1 ms | 36.7 ms across 2 calls | 76.9 ms across 4 calls | 55.8 ms |
| N16 Refine + Score L32, + prewarmed mesh + native depth/transform/warp | 32 | **0.215 s** | 32.1 ms | 27.3 ms | 23.1 ms across 2 calls | 68.0 ms across 2 calls | 56.5 ms |
| N8 Refine + Score L20, same latest native path | 20 | **0.171 s** | 25.2 ms | 21.0 ms | 18.6 ms across 2 calls | 57.6 ms across 3 calls | 36.3 ms |

Detailed latest N16 + L32 profile:

```text
frame.register: 214.988 ms
predict.refine_total: 108.455 ms
predict.score_total: 89.932 ms
adapter.refine.runner_infer: 67.975 ms across 2 calls
adapter.score.runner_infer: 56.451 ms
crop.refine_make_crop_data_batch: 32.125 ms
crop.score_make_crop_data_batch: 27.267 ms
render.placeholder_total: 23.133 ms across 2 calls
crop.refine_rgbB_warp_cv2: 4.606 ms
crop.score_rgbB_warp_cv2: 4.398 ms
crop.refine_xyz_mapB_warp_cv2: 3.232 ms
crop.score_xyz_mapB_warp_cv2: 3.232 ms
crop.refine_inline_transform: 4.048 ms
crop.score_inline_transform: 4.017 ms
frame.depth2xyzmap_fast: 1.332 ms
```

The latest native batch warp path changed the smoke-frame final pose only slightly versus the previous OpenCV-warp path:

```text
trans_m = 4.6890675e-05
rot_deg = 0.056244
max_abs = 0.000607371
```

The 20-hypothesis L20 speed path measured `frame.register = 171.107 ms` and selected effectively the same final pose as the 32-hypothesis L32 path on this smoke frame:

```text
L20 vs L32 trans_m = 0
L20 vs L32 rot_deg = 0.011685
L20 vs L32 max_abs = 5.96e-08
```

Intermediate default-renderer checkpoints:

| Optimization checkpoint | `frame.register` | Notes |
|---|---:|---|
| native placeholder first pass | 2.100 s | native raster/point pose path, before Score XYZ/depth cleanup |
| native pose path + Score observed-XYZ/depth-skip | 0.556 s | `crop.score_depth2xyzmap_full` still recomputed ~13 ms |
| frame XYZ reuse | 0.504 s | Score full-frame XYZ recompute ~0.12 ms |
| native batched renderer | 0.486 s | one native renderer call per crop builder |
| tmpfs BPU scratch | 0.464 s | BPU file I/O moved from ext4 scratch to `/dev/shm` |
| OpenMP renderer + OpenCV crop warps | 0.313 s | observed RGB/XYZ warps moved from CPU Kornia to OpenCV control path |
| prewarmed mesh + native depth/transform/warp | 0.215 s | latest N16/L32 path; native full-frame depth2XYZ, crop normalization, and batch perspective warp |
| 20-hypothesis L20 speed path | 0.171 s | N8/L20 path; faster but lower hypothesis count than strict L32/32 board target |

Interpretation:

- The default smoke renderer dropped from ~34.9 s/frame to ~0.215 s/frame at the setup-excluded `frame.register` level for the 32-hypothesis L32 path.
- The 20-hypothesis L20 speed path reached ~0.171 s/frame on the same smoke frame, with effectively identical final pose versus the L32/32 result for this single case.
- For the latest L32/32 path, BPU runner time is now ~124 ms total for Refine+Score (`68.0 ms + 56.5 ms`) and remains fully on BPU for the learning subgraphs.
- Host crop/render is now tens of milliseconds per crop builder rather than seconds; the remaining bottleneck is mostly BPU runtime/file IPC plus fixed ScoreNet latency.
- This is still a placeholder renderer result. It is useful for board-side full-BPU throughput/control-flow work, not final ADD/ADD-S accuracy evidence.

Pose deltas from batched Refine to the earlier N1 full-BPU run with matching renderer:

```text
N8 default vs N1 default:
  translation_m  = 0
  rotation_deg   ~= 0.018  # acos/numerical noise
  matrix_max_abs = 1.49e-08

N8 points-only vs N1 points-only:
  translation_m  = 0
  rotation_deg   ~= 0.027  # acos/numerical noise
  matrix_max_abs = 1.49e-08

N16 default vs N1 default:
  translation_m  = 0
  rotation_deg   ~= 0.021  # acos/numerical noise
  matrix_max_abs = 5.96e-08

N16 points-only vs N1 points-only:
  translation_m  = 0
  rotation_deg   ~= 0.026  # acos/numerical noise
  matrix_max_abs = 2.98e-08
```

`refine_net_N32` export and calibration reached `hbdk.compile` on ws-wan but did not produce an HBM after a long compile stall; it was stopped and remains experimental. Given N16's marginal gain over N8, N32 is not currently worth pursuing unless the IPC path is otherwise solved and memory/loadability can be guaranteed.

## Next performance work

1. Keep using `--profile-timing` for every optimization run and report both whole-process wall-clock and internal timed spans.
2. Treat `refine_net_N8` as the current practical batched Refine candidate for 32-hypothesis runs; keep N1 as the tighter tensor-gate baseline and N16 as a valid but heavier control.
3. Reduce or replace the placeholder CPU triangle renderer/crop path for benchmark runs; it dominates the default one-frame wall-clock and is not representative of a production renderer.
4. Reduce persistent-runner per-inference overhead further only if the deployment path still needs it after batching; the remaining file/JSONL IPC matters most for small-batch or repeated subgraph calls.
5. Run multi-frame pose drift plus ADD / ADD-S before final deployment sign-off.
