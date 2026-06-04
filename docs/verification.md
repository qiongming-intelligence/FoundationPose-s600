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
score_net_L16:  A,B [16,6,160,160] -> score_logit [1,16]
score_net_L64:  A,B [64,6,160,160] -> score_logit [1,64]
```

If your checkpoint config differs (`c_in`, `input_resize`, `rot_rep`), regenerate
contracts with matching flags and document the deviation.

## Milestone 2 — ONNX export correctness

Run on the x86 export host:

```bash
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.refine --verify
PYTHONPATH=src/python python3 -m foundationpose_s600_tools.export.score \
  --contract build/foundationpose_export/contracts/score_net_L16.json --verify
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

Current S600 smoke results to keep in mind:

- Precision policy: do not quantize just for the sake of quantization. On S600,
  `hb_compile` has no true all-FP32 BPU path, so a BPU HBM is necessarily a
  reduced-precision deployment artifact. True FP32 can be preserved only by
  forcing nodes to the CPU with `node_info`; that is valid as an accuracy golden
  or as an intentional hybrid fallback for small, sensitive layers.
- CPU-float HBM is an accuracy golden only, not a deployable target. RefineNet
  CPU-float matched ONNX to `trans max_abs≈2.8e-8`, `rot max_abs≈3.1e-7`, but
  ran at about 17.2 s/infer.
- RefineNet deployable baseline is currently
  `models/hbm_refine_formula64/foundationpose_refine_net_formula64_int16_no_output_core1.hbm`:
  full BPU (`NODE_INFO={}`, `CORE_NUM=1`) with no CPU fallback,
  `calibration_type=max`, 64 formula calibration samples, and
  `optimization=set_all_nodes_int16` **without** `set_model_output_int16`.
  Against the board-side CPU-float HBM golden over 8 formula samples:
  `trans L2_mean≈0.0056` (`max_abs_max≈0.0134`) and `rot L2_mean≈0.0098`
  (`max_abs_max≈0.0137`). Profiler latency is ≈1.95 ms single-core /
  ≈1.96 ms dual-core (BPU≈1.92–1.93 ms, CPU=0 ms). The older
  `foundationpose_refine_net_opt_int16_core1.hbm` used single-sample calibration
  plus `set_model_output_int16` and clipped `trans` outputs, so it is no longer
  the deployable RefineNet candidate.
- ScoreNet skip/random-calibrated L16/L64 collapsed logits to a constant, so
  top-1/rank gates failed even though ABI and performance looked good.
- ScoreNet full-BPU int16 core1 runs, but is not precision-aligned on current
  formula-smoke ranking gates: L16 top-5=3/5, Spearman≈0.61; L64 top-1 failed,
  top-5=2/5, Spearman≈0.32. Mean-centering the output fixed output scale but
  not internal attention/head quantization error.
- Small CPU-fallback ScoreNet sweeps (7/11/12/15/18/34 CPU nodes) either missed
  top-1 or top-5 and were not better than the stable candidate.
- The current ScoreNet deployable candidate is
  `models/hbm_core1_sweep/foundationpose_score_net_L16_cpu_cross_head_core1.hbm`
  / `foundationpose_score_net_L64_cpu_cross_head_core1.hbm`: encoder/self-attn
  stays on BPU, cross-attention/head stays float on CPU. Formula-smoke metrics:
  L16 top-1/top-5/top-10 all match, Spearman≈0.994; profiler latency is
  ≈44.25 ms dual-core (BPU≈20.65 ms, CPU≈23.49 ms). L64 top-1/top-10 match,
  top-5=4/5, Spearman≈0.986; profiler latency is ≈178.15 ms single-core
  (BPU≈83.49 ms, CPU≈94.39 ms).

Real captured `cal_data_dir` tensors are still mandatory for final deployment
sign-off, because formula tensors do not prove FoundationPose pose accuracy.

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
