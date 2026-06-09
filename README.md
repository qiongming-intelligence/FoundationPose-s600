# FoundationPose S600

Adapt NVlabs/FoundationPose to the Horizon / D-Robotics **S600 BPU** by exporting
only the static learning subgraphs and keeping the geometry/rendering pipeline on
host CPU/GPU/Python.

This repo does **not** redistribute FoundationPose source, weights, ONNX, HBM, or
calibration dumps. Those artifacts stay local and are git-ignored.

## Current result

The precision-first path is now established on real FoundationPose intermediate
tensors:

- **RefineNet**: the full all-node-int16/FLOAT32-output HBM is now the default
  Refine BPU target when the one-core HBRT preload shim is used on S600 HBRT 4.7.5.
  The shim reports one visible BPU core to the loader so a single-core HBM is not
  rejected by a pre-scheduling cross-core IOVA equality check; it does not change
  RefineNet precision or HBM contents.
  - Default full-BPU HBM: `models/hbm_real_int16/foundationpose_refine_net_real_int16_no_output.hbm`
    - Normal `model_info`: `IOVA_FAIL 0/8` on HBRT 4.7.5 because the loader checks
      IOVA equality across all reported BPU cores before `--core_id` scheduling.
    - With `build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so`:
      `model_info OK 8/8`.
    - Real-tensor Refine gate over 8 real driller samples:
      `trans max_abs_max=0.0158056`, `rot max_abs_max=0.0317621`.
    - `hrt_model_exec perf --frame_count 10` with preload: average latency
      `1.950 ms` (`509.061 FPS`).
    - Full strict `Refine BPU + Score BPU L32` smoke through the persistent runner
      completed at 32 hypotheses; final pose delta vs CPU-only on the smoke frame:
      `0.000803929 m / 0.496348°`, matrix max-abs `0.00618881`.
  - The former micro-BPU recovery HBM
    `models/hbm_refine_recovery_micro_bpu/refine_bpu_encodeab_conv_only_real_int16.hbm`
    remains a loadable fallback/control but is **not** the user's requested path:
    it reports CPU fallback for most RefineNet nodes.
  - Four split all-BPU Refine HBMs (`encodeA`, `encodeAB_pos`, `trans_head`,
    `rot_head`) also become `model_info OK 8/8` with the preload. Direct chaining
    must preserve padded intermediate tensor strides, so the simpler full HBM is
    preferred for runtime.
- **ScoreNetMultiPair**: fixed-L all-node-int16 HBMs with FLOAT32 output.
  - Historical real-tensor gate target: `models/hbm_real_int16/foundationpose_score_net_L20_real_int16_no_output.hbm`
    - Normal `model_info`: `IOVA_FAIL 0/8`; with the one-core preload shim:
      `model_info OK 8/8`.
    - Board real ranking gate over 9 L20 groups with preload:
      `top1=8/9`, `top5_mean=4.556`, `top10_mean=9.444`,
      `spearman_mean=0.986466`, `rho_mean=0.992366`, `max_abs_max=1.47508`.
    - Latency from hb_compile perf: ~24.73 ms
  - Latest strict-32 board target: `models/hbm_real_int16/foundationpose_score_net_L32_real_int16_no_output.hbm`
    - `hrt_model_exec model_info`: latest repeated probe passed `8/8`.
    - L32 ONNX export/verify on ws-wan matched eager PyTorch with
      `score_logit max|Δ|=7.629e-06`.
    - Broader real-driller strict-32 eval pack through persistent HBM compare:
      `top1=6/6`, `top5_mean=5.000`, `top10_mean=9.333`,
      `spearman_mean=0.978861`, `rho_mean=0.992902`, `max_abs_max=1.70461`.
    - Strict 32-hypothesis `Refine CPU + Score BPU L32` via persistent runner matched CPU-only exactly on the driller smoke frame (translation delta 0 m, rotation delta 0°, matrix max-abs delta 0).
    - Padded 20-hypothesis `Refine CPU + Score BPU L32` via persistent runner remains a load/control-flow smoke path and differs from CPU-only by 0 m / 0.0146° on the same frame.
    - Strict full `Refine BPU + Score BPU L32` via persistent runner now completes with the full all-node-int16/FLOAT32-output Refine HBM and one-core preload; final pose delta vs CPU-only was `0.000803929 m / 0.496348°`, matrix max-abs `0.00618881`.

Important board-runtime limit: on the current S600 runtime (UCP 3.13.6 / HBRT
4.7.5), HBM loadability is state/layout sensitive. HBRT rejects some valid
single-core HBMs during a pre-scheduling multi-core IOVA equality check:

```text
hbrt4_loader/src/hbm4/parsing.rs:236: iova addr not equal for different core
HBRT4_STATUS_INVALID_ARGUMENT
```

For this board/HBRT combination, build and use the small preload shim when
running full Refine or L20 on a single BPU core:

```bash
cmake -S . -B build/cmake
cmake --build build/cmake --target foundationpose_bpu_core1_preload foundationpose_bpu_runner
export LD_PRELOAD=$PWD/build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so
```

`run_s600_board_hybrid_demo.py` auto-enables that shim for Refine/full BPU modes
when it is present. Treat ScoreNet L selection as a board-validation result, not
a fixed truth: run `hrt_model_exec model_info` on the exact target before choosing
L16/L20/L32. For semantic ScoreNet validation, the candidate count must match the
fixed-L HBM; shorter padded runs are load/control-flow smoke only.

## What runs on BPU

FoundationPose is a hybrid 6D pose pipeline, not one static neural net. Only these
subgraphs are exported to ONNX/HBM:

| Subgraph | Upstream module | BPU contract | Output |
|---|---|---|---|
| RefineNet | `learning/models/refine_network.py` | `A,B = (N,6,160,160)`; N1 precision baseline, N8/N16 batched throughput HBMs | `trans (N,3)`, `rot (N,3)` |
| ScoreNetMultiPair | `learning/models/score_network.py` | fixed `A,B = (L,6,160,160)` exact candidate group; L20 fast 20-hypothesis path, L32 strict 32-hypothesis board target | `score_logit (1,L)` |

`A` and `B` are crop pairs: RGB(3) + XYZ-map(3) = 6 channels, resized to 160x160.
Rendering, depth preprocessing, crop/warp, pose decode, SE(3) composition, and
candidate selection stay on host.

## Two-host workflow

- **ws-wan / x86_64**: PyTorch export, ONNX Runtime golden generation,
  `hb_compile`. Use separate conda envs:
  - `sam3-hbm` for capture/export utilities needing numpy 2.x
  - `sam3-compile` for `hb_compile` / hmct needing numpy 1.23.0
- **S600 board / aarch64**: HBM load/infer via `/usr/hobot/bin/hrt_model_exec`.
  No onnxruntime on the board, so golden outputs are generated on ws-wan and copied
  over as eval packs.

The board link is flaky for large files. Use retrying `rsync --append-verify` and
sha256 validation; plain `scp` can leave truncated HBM files.

## Rebuild / validate

Generate default contracts (RefineNet + ScoreNet L20 historical gate + L32 strict board target):

```bash
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.contract \
  --out-dir build/foundationpose_export --print-plan
```

Export ONNX on x86 with authorized weights:

```bash
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.refine \
  --contract build/foundationpose_export/contracts/refine_net.json --verify

PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.score \
  --contract build/foundationpose_export/contracts/score_net_L20.json --verify
```

Compile precision-first deployable HBMs on x86:

```bash
HB_COMPILE_CALIBRATION_TYPE=max \
HB_COMPILE_CALIB_DATA_ROOT=configs/calibration/data/real_driller \
HB_COMPILE_OPTIMIZATION=set_all_nodes_int16 \
HB_COMPILE_CORE_NUM=1 \
HB_COMPILE_OPTIMIZE_LEVEL=O2 \
HB_COMPILE_MODE=latency \
  bash src/python/scripts/compile_hbm.sh build/foundationpose_export/contracts models/hbm_real_int16
```

Generate real golden eval packs on ws-wan and compare on the board. First probe
HBM loadability on the target S600; chunk selection is board/HBRT-state sensitive:

```bash
# S600 board: model_info loadability probe
src/python/scripts/probe_hbm_loadability.sh \
  models/hbm_real_int16/foundationpose_score_net_L20_real_int16_no_output.hbm \
  models/hbm_real_int16/foundationpose_score_net_L32_real_int16_no_output.hbm

# x86 / ws-wan
PYTHONPATH=src/python python src/python/foundationpose_s600_tools/debug/validate_hbm_real.py golden \
  --partition score_net_L20 \
  --source-root configs/calibration/data/real_driller \
  --pack build/foundationpose_export/eval_pack

# S600 board (choose --partition/--hbm according to the probe result)
PYTHONPATH=src/python python3 src/python/foundationpose_s600_tools/debug/validate_hbm_real.py compare \
  --partition score_net_L20 \
  --pack build/foundationpose_export/eval_pack \
  --hbm models/hbm_real_int16/foundationpose_score_net_L20_real_int16_no_output.hbm \
  --core-id 1
```

See `benchmarks/refine_real_int16_quick.md` for the recorded board gates.

## Python hybrid validation adapters

For validation, `foundationpose_s600_tools.runtime` provides both the legacy
`hrt_model_exec` wrappers and a native persistent C++/UCP runner:

```python
from foundationpose_s600_tools.runtime import install_bpu_adapters

# after constructing upstream FoundationPose / predictors
install_bpu_adapters(est.refiner, est.scorer, root="/home/sunrise/Projects/FoundationPose-s600")
```

The adapters preserve upstream-style dict outputs (`trans`/`rot`, `score_logit`).
`ScoreNetBpu` is strict by default: candidate count must exactly match the
fixed-L HBM contract because upstream ScoreNet attends across the whole candidate
set. `--score-bpu-mode smoke-pad` can pad a shorter set for board/HBM
loadability and control-flow smoke, but that mode is not a ScoreNet correctness
gate.

Backends:

- `--bpu-runtime hrt` / `bpu_backend="hrt"`: validation fallback around
  `hrt_model_exec`; it starts a process and reloads the HBM for every inference.
- `--bpu-runtime persistent` / `bpu_backend="persistent"`: native
  `foundationpose_bpu_runner` subprocess that loads selected HBMs once with
  `hbDNNInitializeFromFiles` and reuses them via `hbDNNInferV2`/UCP JSONL calls.

Current validation caveat: full Refine on HBRT 4.7.5 needs the one-core preload
shim because normal `model_info` can fail before scheduling reaches `--core_id 1`;
the launcher auto-enables the shim for Refine/full modes when the built `.so`
exists. For runtime, use the persistent C++/UCP runner so selected HBMs are loaded
once and reused across Refine/Score calls. Validated E2E paths are:

- CPU-only board baseline: safe default for renderer/control-flow smoke.
- `Refine CPU + Score BPU L32`: strict 32-hypothesis path matched CPU-only exactly
  on the driller smoke frame. Use strict mode with 32 hypotheses for semantic
  ScoreNet validation.
- Full `Refine BPU + Score BPU L32`: all-node-int16 Refine HBM + Score L32,
  persistent backend, one-core preload. The optimized 32-hypothesis no-setup
  board path uses `refine_net_N16` and measures `frame.register ~= 215 ms` with
  the native placeholder crop/render path. The latest smoke pose delta vs the
  prior OpenCV-warp path was `4.69e-05 m / 0.0562°`.
- Full `Refine BPU + Score BPU L20`: fast 20-hypothesis option with `refine_net_N8`
  + Score L20, persistent backend, one-core preload. It measures
  `frame.register ~= 171 ms` on the same smoke frame. L20 vs L32 final pose was
  effectively unchanged on that frame (`0 m / 0.0117°`, matrix max-abs `5.96e-08`),
  but L32 remains the strict 32-hypothesis board target for semantic validation.
- `Score BPU L20`: available again with the one-core preload (`model_info OK 8/8`)
  and retains the 9-sample ranking gate.

These timings are setup-excluded `--profile-timing` spans using the placeholder
host renderer. They are useful for board-side full-BPU throughput/control-flow
work, not final ADD/ADD-S accuracy evidence.

Latest one-frame driller smoke timings (`--profile-timing`, setup/preflight/model
load excluded from `frame.register`):

| Full-BPU path | Hypotheses | `frame.register` | Refine runner | Score runner | Crop/render notes |
|---|---:|---:|---:|---:|---|
| Refine N16 + Score L32 | 32 | **214.988 ms** | 67.975 ms across 2 calls | 56.451 ms | Refine crop 32.1 ms, Score crop 27.3 ms, render 23.1 ms |
| Refine N8 + Score L20 | 20 | **171.107 ms** | 57.632 ms across 3 calls | 36.336 ms | Refine crop 25.2 ms, Score crop 21.0 ms, render 18.6 ms |

Recommended board smoke commands:

```bash
# Safe default baseline: CPU neural nets, CPU renderer/control flow.
PYTHONPATH=src/python:.deps/s600-foundationpose \
python src/python/scripts/run_s600_board_hybrid_demo.py \
  --bpu-mode cpu --max-hypotheses 20 --renderer-splat-radius 0

# Probe before BPU runs. For HBRT 4.7.5 full Refine/L20 runs, use the
# one-core preload shim that the launcher auto-enables for Refine/full modes.
LD_PRELOAD=$PWD/build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so \
src/python/scripts/probe_hbm_loadability.sh \
  models/hbm_real_int16/foundationpose_refine_net_real_int16_no_output.hbm \
  models/hbm_real_int16/foundationpose_score_net_L20_real_int16_no_output.hbm \
  models/hbm_real_int16/foundationpose_score_net_L32_real_int16_no_output.hbm

# Score BPU strict/smoke path with L32: Refine CPU + Score BPU L32.
# The launcher also runs model_info preflight by default before installing BPU adapters.
# `smoke-pad` preserves the old 20-hypothesis L32 load/control-flow smoke; use
# --max-hypotheses 32 with --score-bpu-mode strict for semantic ScoreNet checks.
PYTHONPATH=src/python:.deps/s600-foundationpose \
/home/sunrise/miniconda3/envs/sam3-export/bin/python src/python/scripts/run_s600_board_hybrid_demo.py \
  --bpu-mode score --bpu-runtime persistent \
  --max-hypotheses 20 --renderer-splat-radius 0 \
  --score-chunk-size 32 --score-partition score_net_L32 \
  --score-bpu-mode smoke-pad \
  --score-hbm models/hbm_real_int16/foundationpose_score_net_L32_real_int16_no_output.hbm

# Strict full-BPU 32-hypothesis path: batched Refine N16 + Score L32 through
# the persistent runner. The launcher auto-enables the one-core preload shim for
# Refine/full modes when build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so exists.
PYTHONPATH=src/python:.deps/s600-foundationpose \
/home/sunrise/miniconda3/envs/sam3-export/bin/python src/python/scripts/run_s600_board_hybrid_demo.py \
  --bpu-mode full --bpu-runtime persistent \
  --max-hypotheses 32 --renderer-splat-radius 0 \
  --refine-partition refine_net_N16 \
  --refine-hbm models/hbm_real_int16_batch/foundationpose_refine_net_N16.hbm \
  --score-chunk-size 32 --score-partition score_net_L32 \
  --score-bpu-mode strict \
  --score-hbm models/hbm_real_int16/foundationpose_score_net_L32_real_int16_no_output.hbm \
  --profile-timing

# Faster 20-hypothesis path: batched Refine N8 + Score L20. This is useful for
# deployment throughput experiments; keep L32/32 for strict 32-candidate ScoreNet validation.
PYTHONPATH=src/python:.deps/s600-foundationpose \
/home/sunrise/miniconda3/envs/sam3-export/bin/python src/python/scripts/run_s600_board_hybrid_demo.py \
  --bpu-mode full --bpu-runtime persistent \
  --max-hypotheses 20 --renderer-splat-radius 0 \
  --refine-partition refine_net_N8 \
  --refine-hbm models/hbm_real_int16_batch/foundationpose_refine_net_N8.hbm \
  --score-chunk-size 20 --score-partition score_net_L20 \
  --score-bpu-mode strict \
  --score-hbm models/hbm_real_int16/foundationpose_score_net_L20_real_int16_no_output.hbm \
  --profile-timing
```

Full Refine+Score BPU at normal hypothesis counts should use the persistent
C++/UCP runner. On HBRT 4.7.5, keep preflight enabled and use the one-core
preload shim so `model_info` validates only the selected single-core deployment
path. Refine `N>1` HBMs reduce repeated Refine calls; ScoreNet remains fixed-L,
so candidate count must match the selected Score HBM for semantic validation.

## Repository layout

```text
src/python/foundationpose_s600_tools/
  export/      contracts + ONNX export wrappers
  debug/       capture, calibration prep, validation gates
  runtime/     Python hrt_model_exec + persistent C++/UCP adapters for hybrid validation
  convert/     ONNX diagnostic transforms
src/python/scripts/      hb_compile / inspect / benchmark scripts
src/csrc/                persistent C++/UCP BPU runner (`foundationpose_bpu_runner`)
configs/manifests/       refine + fixed-L score manifests (L20 historical gate, L32 strict board target)
configs/calibration/     calibration data policy (actual data git-ignored)
docs/                    export/runtime/rendering/verification notes
benchmarks/              recorded accuracy/perf notes (no large tensors/models)
third_party/FoundationPose/  local vendored upstream (git-ignored)
```

## Status / stopping point

Done for this checkpoint:

- RefineNet + fixed-L ScoreNet HBMs exported/compiled with real calibration and all-node int16; deployable candidates preserve FLOAT32 outputs.
- The default Refine BPU target is now the full all-node-int16/FLOAT32-output HBM. Normal HBRT 4.7.5 `model_info` fails `IOVA_FAIL 0/8`, but the one-core preload shim makes it `OK 8/8`; real-tensor gate is `trans max_abs_max=0.0158056`, `rot max_abs_max=0.0317621`, and perf is `1.950 ms` / `509.061 FPS`.
- Score L20 also recovers with the preload (`model_info OK 8/8`) and keeps the 9-group ranking gate: `top1=8/9`, `spearman_mean=0.986466`, `rho_mean=0.992366`.
- Score L32 loads normally and with preload (`model_info OK 8/8`); strict-32 real-driller compare gives `top1=6/6`, `spearman_mean=0.978861`, `rho_mean=0.992902`.
- Python hybrid validation adapters support partial BPU toggles, strict fixed-L ScoreNet validation, explicit ScoreNet smoke padding, a persistent native C++/UCP backend, explicit `--core-id` backend masks, and auto-preload for Refine/full modes.
- Full strict `Refine BPU + Score BPU L32` now runs with the full all-node Refine HBM on BPU. The latest smoke frame delta vs CPU-only is `0.000803929 m / 0.496348°`, matrix max-abs `0.00618881`.
- The micro-BPU Refine recovery HBM and progressive recompiles remain fallback/control artifacts only; they are no longer the default path because the user's required all-BPU Refine path is recovered via the one-core preload shim.

Next work when resuming:

1. Run broader FoundationPose end-to-end pose drift plus ADD / ADD-S gates across CPU-only, Score-BPU strict, and full-BPU strict variants on identical frames/hypotheses/seed.
2. Revalidate exact HBM loadability after board reboot, HBRT/UCP changes, or any HBM recompile before treating a variant as deployable.
3. Keep the micro-BPU recovery HBM only as a fallback/control; do not promote it over the full all-node Refine HBM unless future measurements force that trade-off.
