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

## Next performance work

1. Add fine-grained timers around register/track stages: render, crop assembly, Refine adapter, Score adapter, candidate selection, and output write.
2. Reduce persistent-runner per-inference overhead by avoiding temporary file round-trips where practical.
3. Evaluate a batched Refine HBM (`N=32` or selected batch sizes) so 32 hypotheses do not require 32 independent N=1 adapter calls.
4. Run multi-frame pose drift plus ADD / ADD-S before final deployment sign-off.
