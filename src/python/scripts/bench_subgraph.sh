#!/usr/bin/env bash
# Benchmark one FoundationPose subgraph HBM on the local S600 across single-core
# and dual-core BPU scheduling, with profiling, and print a compact summary.
#
# Runs on the aarch64 S600 (local box), NOT the x86 compile host. The subgraphs
# take two inputs A,B in contract order; provide raw .bin tensors matching the
# compiled shapes.
#
# usage: bench_subgraph.sh HBM_FILE MODEL_NAME INPUT_DIR [FRAME_COUNT]
#   INPUT_DIR must contain A*.bin and B*.bin (contract input order)
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: bench_subgraph.sh HBM_FILE MODEL_NAME INPUT_DIR [FRAME_COUNT]

  HBM_FILE     path to the .hbm to benchmark
  MODEL_NAME   model name inside the HBM (see `hrt_model_exec model_info`)
  INPUT_DIR    dir with raw input .bin files; needs A*.bin then B*.bin
  FRAME_COUNT  frames per perf run (default 30)

Runs model_info, then perf on core 1 (single) and cores 1,2 (dual), each with
--profile_path, and prints latency + BPU/CPU split.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ $# -lt 3 || $# -gt 4 ]]; then usage; exit 2; fi

hbm="$1"
model_name="$2"
input_dir="$3"
frames="${4:-30}"
hrt="/usr/hobot/bin/hrt_model_exec"

if [[ ! -f "$hbm" ]]; then echo "missing HBM: $hbm" >&2; exit 1; fi
if [[ ! -d "$input_dir" ]]; then echo "missing input dir: $input_dir" >&2; exit 1; fi

# A, B in contract order.
inputs=""
for key in A B; do
  f=$(ls "$input_dir"/${key}*.bin 2>/dev/null | head -n1 || true)
  if [[ -z "$f" ]]; then echo "missing input file for '$key' in $input_dir" >&2; exit 1; fi
  inputs+="${inputs:+,}$f"
done

variant="$(basename "$hbm" .hbm)"
prof_root="benchmarks/results/$variant"
mkdir -p "$prof_root"

echo "================= model_info: $variant ================="
"$hrt" model_info --model_file "$hbm" 2>&1 | grep -E '^\[model name\]|^name:|valid shape|tensor type' || true

run_one() {
  local label="$1" core="$2" threads="$3"
  local dir="$prof_root/$label"
  rm -rf "$dir"; mkdir -p "$dir"
  echo "----------------- perf $label (core_id=$core thread=$threads) -----------------"
  "$hrt" perf --model_file "$hbm" --model_name "$model_name" \
    --core_id "$core" --frame_count "$frames" --perf_time 0 --thread_num "$threads" \
    --input_file "$inputs" --profile_path "$dir" 2>&1 \
    | grep -E 'Average|Frame      rate|Frame totally|Program run time' || true
  sed -n '/processor_latency/,/task_latency/p' "$dir/profiler.csv" 2>/dev/null \
    | grep -E 'BPU_inference|CPU_inference' || true
}

run_one single_core 1 1
run_one dual_core "1,2" 1

echo "================= done: $variant ================="
