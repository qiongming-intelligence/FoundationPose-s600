# S600 runtime and HBM compile notes

This doc covers ONNX→HBM compilation and S600-side inspection/benchmarking for
FoundationPose's RefineNet and ScoreNet subgraphs.

## Compile host (x86_64)

`hb_compile` is an x86_64 host tool. It is not available on the aarch64 S600
runtime board. Run inside the D-Robotics/OpenExplorer AI Toolchain environment
(3.7.0+), then copy `.hbm` files to the S600 target.

Baseline compile:

```bash
src/python/scripts/compile_hbm.sh build/foundationpose_export/contracts models/hbm
```

The script reads every `*.json` partition contract, creates a per-model YAML under
`models/hbm/hb_compile_configs/`, and invokes `hb_compile`.

Default YAML knobs:

```yaml
model_parameters:
  march: "nash-p"
  core_num: 2
  compile_mode: "latency"
input_parameters:
  input_type_rt: "featuremap;featuremap"
  input_type_train: "featuremap;featuremap"
  input_layout_train: "NCHW;NCHW"
  norm_type: "no_preprocess"
calibration_parameters:
  calibration_type: "skip"
compiler_parameters:
  optimize_level: "O2"
```

Override with environment variables:

```bash
HB_COMPILE_CORE_NUM=1 \
HB_COMPILE_OPTIMIZE_LEVEL=O3 \
HB_COMPILE_MODE=bandwidth \
src/python/scripts/compile_hbm.sh build/foundationpose_export/contracts models/hbm
```

Matrix sweep for one contract:

```bash
src/python/scripts/compile_matrix.sh \
  build/foundationpose_export/contracts/refine_net.json models/hbm
```

By default this sweeps `O1 O2 O3` × `core 1/2` × `latency/bandwidth` and names
HBMs like `refine_net_core2_O2_latency.hbm`.

## Runtime target (aarch64 S600)

Probe loadability before any E2E BPU run:

```bash
# One model_info attempt per HBM by default; set FOUNDATIONPOSE_S600_PROBE_RETRIES=N for stress probing.
src/python/scripts/probe_hbm_loadability.sh \
  models/hbm_real_int16/foundationpose_score_net_L20_real_int16_no_output.hbm \
  models/hbm_real_int16/foundationpose_score_net_L32_real_int16_no_output.hbm
```

Inspect ABI:

```bash
src/python/scripts/inspect_hbm.sh \
  models/hbm/foundationpose_refine_net.hbm \
  build/foundationpose_export/contracts/refine_net.json
```

Benchmark with raw A/B tensors:

```bash
src/python/scripts/bench_subgraph.sh \
  models/hbm/foundationpose_refine_net.hbm refine_net \
  models/hbm/perf_inputs_refine_f32 30
```

`bench_subgraph.sh` expects raw files in contract input order:

```
INPUT_DIR/
  A_f32.bin
  B_f32.bin
```

Shapes:

- RefineNet: `A,B = 1x6x160x160` by default contract.
- ScoreNet L16/L20/L32/L64: fixed-L ScoreNet contracts use `A,B = Lx6x160x160` and output `score_logit = 1xL`.
- ScoreNet scores candidates jointly with attention over that fixed `L`. Semantic CPU/BPU validation therefore requires the candidate count to exactly match the selected HBM (`--score-bpu-mode strict`, the launcher default). `--score-bpu-mode smoke-pad` can pad a shorter set for HBM load/control-flow smoke only; padded entries participate in attention, so it is not a ScoreNet correctness gate.
- Current local S600/HBRT 4.7.5 loadability has a loader-side cross-core IOVA check: some single-core HBMs fail normal `model_info` with `iova addr not equal` before `--core_id` scheduling. For single-core deployment, build/use `build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so`; with that preload, the full Refine all-node-int16 HBM and Score L20 both load 8/8 on this board.
- Always run `hrt_model_exec model_info` on the exact target board before choosing a ScoreNet L for validation; include the same preload policy that the runtime will use.

The raw bytes are NCHW float32, no preprocessing; the FoundationPose Python
pipeline has already rendered/cropped/concatenated the tensors.

## Manifests

Manifests under `configs/manifests/` bind HBM files to exact tensor names and
host-stage responsibilities:

- `foundationpose_refine.yaml`
- `foundationpose_score_L20.yaml` (historical real-tensor gate target; use the one-core preload on HBRT 4.7.5)
- `foundationpose_score_L16.yaml` (small fixed-L fallback; revalidate loadability on the board)
- `foundationpose_score_L32.yaml` (latest strict 32-hypothesis Score BPU target)
- `foundationpose_tracking_hybrid.yaml`

The native C++/UCP persistent runner loads the selected HBMs once, allocates BPU
input/output tensors from the same contract names, copies raw `.bin` buffers into
A/B, runs inference, and dumps `trans`/`rot` or `score_logit`.

Current local S600 board observation (UCP 3.13.6 / HBRT 4.7.5): several
single-core HBMs fail normal `model_info` because HBRT checks that one buffer has
identical IOVA mappings on every reported core before scheduling is restricted to
`--core_id 1`. The helper target `foundationpose_bpu_core1_preload` overrides
`hb_bpu_core_num()` to report one visible core for this single-core deployment.
With `LD_PRELOAD=build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so`:

- `foundationpose_refine_net_real_int16_no_output.hbm`: normal `model_info` fails with `iova addr not equal` (`0/8`), but preload `model_info` passes `8/8`. This is the full all-node-int16/FLOAT32-output Refine HBM.
- Full Refine real-tensor compare over 8 samples via persistent runner: `trans max_abs_max=0.0158056`, `rot max_abs_max=0.0317621`.
- Full Refine `hrt_model_exec perf --frame_count 10`: average latency `1.950 ms` (`509.061 FPS`).
- Four split all-BPU Refine HBMs (`refine_encodeA`, `refine_encodeAB_pos`, `refine_trans_head`, `refine_rot_head`) also pass preload `model_info 8/8`. If chained externally, padded intermediate tensor strides must be preserved; the full HBM is simpler and is the preferred runtime target.
- `refine_bpu_encodeab_conv_only_real_int16.hbm` remains a loadable fallback/control (`model_info 8/8`, real gate `trans max_abs_max=0.00174491`, `rot max_abs_max=0.00425202`) but it is not the full-BPU path because node_info reports CPU fallback for most RefineNet nodes.
- `foundationpose_score_net_L20_real_int16_no_output.hbm`: normal `model_info` fails with `iova addr not equal` (`0/8`), but preload `model_info` passes `8/8`; L20 real ranking gate remains `top1=8/9`, `spearman_mean=0.986466`, `rho_mean=0.992366`, `max_abs_max=1.47508`.
- `foundationpose_score_net_L32_real_int16_no_output.hbm`: normal and preload `model_info` pass `8/8`; L32 ONNX export/verify on ws-wan matched eager PyTorch (`score_logit max|Δ|=7.629e-06`).
- A six-sample strict-32 real-driller tensor gate for Score L32 via persistent HBM compare produced `top1=6/6`, `top5_mean=5.000`, `top10_mean=9.333`, `spearman_mean=0.978861`, `rho_mean=0.992902`, `max_abs_max=1.70461`.
- Strict 32-hypothesis `Refine CPU + Score BPU L32` via persistent runner matched CPU-only exactly on the latest driller smoke frame (translation delta 0 m, rotation delta 0°, matrix max-abs delta 0).
- Strict 32-hypothesis full `Refine BPU + Score BPU L32` with preload and persistent runner completed; final pose delta vs CPU-only on the smoke frame was `0.000803929 m / 0.496348°`, matrix max-abs `0.00618881`.

This was verified with local HBM files, so treat the IOVA issue as an HBRT loader
multi-core visibility problem rather than a Python tensor-shape issue. Revalidate
after any board reboot, HBRT/UCP update, or HBM recompile.

For Python hybrid validation, `foundationpose_s600_tools.runtime` provides
`RefineNetBpu` and generic `ScoreNetBpu` wrappers plus `install_bpu_adapters(...)`:

```python
from foundationpose_s600_tools.runtime import install_bpu_adapters
install_bpu_adapters(est.refiner, est.scorer, root="/path/to/FoundationPose-s600")
```

The legacy backend calls `hrt_model_exec` and therefore reloads the HBM per
inference. The persistent backend starts `foundationpose_bpu_runner`, loads the
selected HBMs once, and reuses them via `hbDNNInferV2`/UCP JSONL infer requests.
Use `--bpu-runtime persistent` when exercising Refine/full BPU modes. For HBRT
4.7.5 full Refine, the launcher auto-enables the one-core preload shim when
`build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so` exists; otherwise
set `LD_PRELOAD` manually or pass `--bpu-core1-preload <path>`. The runner parses
`--core-id` with `hrt_model_exec` semantics (`1` = BPU core 0) and submits UCP
with the matching explicit backend mask.

Launcher modes:

```bash
# Safe default baseline: CPU neural nets, CPU renderer/control flow.
PYTHONPATH=src/python:.deps/s600-foundationpose \
python src/python/scripts/run_s600_board_hybrid_demo.py \
  --bpu-mode cpu --max-hypotheses 20 --renderer-splat-radius 0

# Probe before BPU runs. On HBRT 4.7.5, use the one-core preload for the
# full Refine HBM and Score L20; Score L32 also loads with the same policy.
LD_PRELOAD=$PWD/build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so \
src/python/scripts/probe_hbm_loadability.sh \
  models/hbm_real_int16/foundationpose_refine_net_real_int16_no_output.hbm \
  models/hbm_real_int16/foundationpose_score_net_L20_real_int16_no_output.hbm \
  models/hbm_real_int16/foundationpose_score_net_L32_real_int16_no_output.hbm

# Score BPU strict/smoke path: Refine CPU + Score BPU L32.
# `smoke-pad` preserves the old 20-hypothesis L32 load/control-flow smoke;
# use --max-hypotheses 32 with --score-bpu-mode strict for semantic ScoreNet checks.
PYTHONPATH=src/python:.deps/s600-foundationpose \
/home/sunrise/miniconda3/envs/sam3-export/bin/python src/python/scripts/run_s600_board_hybrid_demo.py \
  --bpu-mode score --bpu-runtime persistent \
  --max-hypotheses 20 --renderer-splat-radius 0 \
  --score-chunk-size 32 --score-partition score_net_L32 \
  --score-bpu-mode smoke-pad \
  --score-hbm models/hbm_real_int16/foundationpose_score_net_L32_real_int16_no_output.hbm

# Refine/full BPU modes are intentionally explicit. Use the persistent runner,
# keep preflight enabled, and let the launcher auto-enable the one-core preload
# for the full Refine HBM when the built .so is present. Make ScoreNet semantic
# by matching --max-hypotheses to the fixed-L score HBM.
PYTHONPATH=src/python:.deps/s600-foundationpose \
/home/sunrise/miniconda3/envs/sam3-export/bin/python src/python/scripts/run_s600_board_hybrid_demo.py \
  --bpu-mode full --bpu-runtime persistent \
  --max-hypotheses 32 \
  --refine-hbm models/hbm_real_int16/foundationpose_refine_net_real_int16_no_output.hbm \
  --score-chunk-size 32 --score-partition score_net_L32 \
  --score-bpu-mode strict \
  --score-hbm models/hbm_real_int16/foundationpose_score_net_L32_real_int16_no_output.hbm
```

`run_s600_board_hybrid_demo.py` runs `hrt_model_exec model_info` preflight for
selected BPU HBMs by default (`--bpu-preflight`). Use `--no-bpu-preflight` only if
you intentionally want to skip this early guard.

## Profiling requirements

For every HBM variant record:

- `hrt_model_exec model_info` tensor names, shapes, dtypes;
- `hrt_model_exec perf` average latency;
- `profiler.csv` split: `BPU_inference` vs `CPU_inference`.

If a variant is faster in wall-clock but has unexpected CPU fallback, treat it as
suspect. In the SAM_s600 reference, whole-graph INT8 PTQ was slower than FP32 on
one graph; do not assume lower precision is faster on S600.

## Current machine note

The current development box is the S600/aarch64 runtime side (has
`hrt_model_exec` / Hobot runtime / onnx). The default `python3` lacks torch and
onnxruntime, but the local `sam3-export` conda env has enough PyTorch/ONNX
Runtime support to export/verify Score L32 ONNX and run small ONNX golden smoke
checks. `hb_compile` is still x86/toolchain-host only, and broader eval-pack
generation should stay on ws-wan where the real capture/calibration trees live.
