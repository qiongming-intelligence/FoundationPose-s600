# Verification plan

This is the gate for every FoundationPose S600 variant. A fast HBM is not useful
unless it preserves pose accuracy and avoids unexpected CPU fallback.

## Milestone 1 — upstream + contract sanity

Run on any host:

```bash
scripts/fetch_upstream.sh
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.contract \
  --out-dir build/foundationpose_export --print-plan
```

Expected default shapes:

```text
refine_net:     A,B [1,6,160,160] -> trans [1,3], rot [1,3]
score_net_L20:  A,B [20,6,160,160] -> score_logit [1,20]  # historical real-tensor gate target
score_net_L16:  A,B [16,6,160,160] -> score_logit [1,16]  # small fixed-L fallback, revalidate model_info
score_net_L32:  A,B [32,6,160,160] -> score_logit [1,32]  # latest strict board Score BPU target
```

If your checkpoint config differs (`c_in`, `input_resize`, `rot_rep`), regenerate
contracts with matching flags and document the deviation.

## Milestone 2 — ONNX export correctness

Run on the x86 export host:

```bash
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.refine --verify
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.score \
  --contract build/foundationpose_export/contracts/score_net_L20.json --verify
```

Checks:

- ONNX checker passes.
- ONNX Runtime outputs match eager PyTorch (`max|Δ|` printed per output).
- ONNX graph inputs/outputs have names exactly matching the contract.
- No nvdiffrast / mycpp / crop pipeline ops appear in the graph.

## Milestone 3 — HBM ABI + performance

Compile on x86, then inspect and benchmark on S600:

```bash
src/python/scripts/inspect_hbm.sh <hbm> <contract>
src/python/scripts/bench_subgraph.sh <hbm> <model_name> <input_dir> 30
```

Checks:

- `hrt_model_exec model_info` names/shapes/dtypes match the contract.
- `profiler.csv` has the expected BPU/CPU split; unexpected CPU fallback is a
  blocker until explained.
- Sweep `core_num=1/2`, `O1/O2/O3`, `latency/bandwidth`; choose by measured
  latency and CPU fallback, not by assumption.

## Milestone 4 — PyTorch/ONNX vs HBM tensor parity

Using real captured intermediate tensors (not random inputs), compare:

- RefineNet PyTorch/ONNX output vs HBM output for `trans` and `rot`.
- ScoreNet PyTorch/ONNX output vs HBM output for `score_logit`.

Start with `atol=1e-3, rtol=1e-3` for smoke tests, then judge by final pose
impact. For S600 hb_compile 3.5.3, `calibration_type: skip` is **not** a true
FP32 pass-through in practice: it may run fixed/random calibration when no
calibration data is provided. Treat skip-calibrated HBM as ABI/perf smoke only.

Current real-calibrated S600 results to keep in mind:

- Precision policy: do not quantize just for the sake of quantization. On S600,
  `hb_compile` has no true all-FP32 BPU path, so a BPU HBM is necessarily a
  reduced-precision deployment artifact. The highest-precision deployable path is
  `calibration_type=max` with real captured tensors and
  `optimization=set_all_nodes_int16` (preserve FLOAT32 outputs by not adding
  `set_model_output_int16`).
- HBRT 4.7.5 loader caveat: some valid single-core HBMs fail normal
  `model_info` before `--core_id` scheduling because HBRT checks IOVA equality
  across every reported BPU core. The local preload shim
  `build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so` overrides only
  `hb_bpu_core_num()` to report one visible core for single-core deployment. It
  changes loader visibility, not HBM contents or graph precision.
- RefineNet default BPU target:
  `models/hbm_real_int16/foundationpose_refine_net_real_int16_no_output.hbm`.
  This is the full all-node-int16/FLOAT32-output HBM. Normal `model_info` fails
  on this HBRT with IOVA errors (`0/8`), but with the one-core preload it passes
  `model_info OK 8/8`. The 8-sample real-driller tensor gate via persistent HBM
  compare produced `trans max_abs_max=0.0158056` and
  `rot max_abs_max=0.0317621`. `hrt_model_exec perf --frame_count 10` with
  preload measured `1.950 ms` average latency (`509.061 FPS`). Strict full
  `Refine BPU + Score BPU L32` E2E completed through the persistent runner at 32
  hypotheses with final pose delta vs CPU-only `0.000803929 m / 0.496348°`,
  matrix max-abs `0.00618881`.
- Refine split all-BPU probes:
  `models/hbm_refine_split_20260608/refine_encodeA.hbm`,
  `refine_encodeAB_pos.hbm`, `refine_trans_head.hbm`, and `refine_rot_head.hbm`
  fail normal `model_info` without preload and pass `OK 8/8` with preload.
  Direct external chaining must preserve padded/aligned intermediate tensors;
  stripping padding caused large numerical errors. The full Refine HBM is the
  preferred runtime target because it avoids this padded handoff complexity.
- RefineNet fallback/control artifacts:
  `models/hbm_refine_recovery_micro_bpu/refine_bpu_encodeab_conv_only_real_int16.hbm`
  remains loadable and accurate as a control (`trans max_abs_max=0.00174491`,
  `rot max_abs_max=0.00425202`), but it is not the requested all-BPU path because
  node inspection reports CPU fallback for most RefineNet nodes. The progressive
  `encodeA2` recompile remains superseded by the full all-node HBM plus preload.
- ScoreNet historical real-tensor gate target:
  `models/hbm_real_int16/foundationpose_score_net_L20_real_int16_no_output.hbm`.
  Normal `model_info` currently fails with the same IOVA error (`0/8`), but with
  the one-core preload it passes `OK 8/8`. The 9-group real-driller ranking gate
  with preload produced `top1=8/9`, `top5_mean=4.556`, `top10_mean=9.444`,
  `spearman_mean=0.986466`, `rho_mean=0.992366`, `max_abs_max=1.47508`; profiler
  latency from the compile report is ≈24.73 ms.
- ScoreNet latest strict-32 board precision target:
  `models/hbm_real_int16/foundationpose_score_net_L32_real_int16_no_output.hbm`.
  Normal and preload `model_info` both passed 8/8 attempts. L32 ONNX export/verify
  on ws-wan matched eager PyTorch with `score_logit max|Δ|=7.629e-06`. A
  six-sample strict-32 real-driller tensor gate via persistent HBM compare
  produced `top1=6/6`, `top5_mean=5.000`, `top10_mean=9.333`,
  `spearman_mean=0.978861`, `rho_mean=0.992902`, `max_abs_max=1.70461`. Strict
  32-hypothesis `Refine CPU + Score BPU L32` matched CPU-only exactly on the
  driller smoke frame (translation delta 0 m, rotation delta 0°, matrix max-abs
  delta 0). Padded shorter-L runs remain load/control-flow smoke only, not a
  ScoreNet correctness gate.
- ScoreNet HBM loadability is board/HBRT-state sensitive and not monotonic in L.
  Revalidate the exact HBM on the exact board with the same preload policy that
  deployment will use.

The all-BPU Refine path is recovered for the current board/HBRT state via the
one-core preload shim. Before final sign-off, broaden from the single-frame smoke
to multi-frame pose drift and FoundationPose-level ADD/ADD-S gates.

## Milestone 5 — FoundationPose-level accuracy gate

Run the upstream FoundationPose pipeline with only the subnet implementation
changed (PyTorch baseline vs BPU hybrid), using the same input frames and mesh.

### RefineNet metrics

- per-iteration `trans`/`rot` delta error;
- translation drift after all refine iterations;
- angular error after all refine iterations;
- final ADD / ADD-S / pose trajectory drift.

RefineNet is iterated (commonly 5×), so a tiny per-step error can accumulate.

### ScoreNet metrics

- top-1 agreement with FP32;
- top-k agreement (e.g. top-5);
- rank correlation of candidate logits;
- final selected-pose agreement / ADD-S impact.

A ScoreNet variant can have small raw-logit error but still be wrong if it swaps
nearby candidates at the top of the ranking.

## Quantization gate

Only attempt INT8/PTQ after real calibration intermediates are captured (see
`configs/calibration/README.md`). For every reduced-precision candidate, re-run
Milestones 3–5.

Rules:

- Random calibration data is invalid for this model (XYZ distribution matters).
- A quantized variant that fails pose/ranking gates is not deployable even if it
  is faster.
- A quantized variant that introduces CPU fallback is usually not useful even if
  compile succeeds.

## Report template

Record each variant in `benchmarks/results/<variant>/README.md` (or a future CSV):

```text
variant: refine_net_core2_O2_latency_fp32
upstream_sha: a1b694b83e633c2cb6115b9063d940a687759392
contract: build/foundationpose_export/contracts/refine_net.json
onnx_sha256: ...
hbm_sha256: ...
hb_compile: march=nash-p core_num=2 mode=latency optimize=O2 precision=fp32
hrt_latency_avg_ms: ...
profiler_bpu_ms: ...
profiler_cpu_ms: ...
pt_vs_hbm_max_abs: trans=... rot=...
end_to_end_pose: ADD=... ADD-S=... drift=...
notes: ...
```
