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
- ScoreNet L16: `A,B = 16x6x160x160`.
- ScoreNet L64: `A,B = 64x6x160x160`.

The raw bytes are NCHW float32, no preprocessing; the FoundationPose Python
pipeline has already rendered/cropped/concatenated the tensors.

## Manifests

Manifests under `configs/manifests/` bind HBM files to exact tensor names and
host-stage responsibilities:

- `foundationpose_refine.yaml`
- `foundationpose_score_L16.yaml`
- `foundationpose_tracking_hybrid.yaml`

The future C++ raw-tensor runner should load these manifests, allocate BPU input
and output tensors, copy raw `.bin` buffers into A/B, run inference, and dump
`trans/rot` or `score_logit`.

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
`hrt_model_exec` / Hobot runtime / onnx, lacks torch / onnxruntime / hb_compile).
So here we can run contract generation, ONNX utility smoke tests, and HBM
inspection/benchmarking *after* the x86 host produces HBMs. ONNX export and
`hb_compile` must run elsewhere.
