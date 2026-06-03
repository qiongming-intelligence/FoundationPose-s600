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

## Milestone 4 — PyTorch vs HBM tensor parity

Using real captured intermediate tensors (not random inputs), compare:

- RefineNet PyTorch output vs HBM output for `trans` and `rot`.
- ScoreNet PyTorch output vs HBM output for `score_logit`.

Recommended tolerances for FP32 HBM: start with `atol=1e-3, rtol=1e-3`, then
tighten/relax based on actual BPU numeric behavior and final pose impact.

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
