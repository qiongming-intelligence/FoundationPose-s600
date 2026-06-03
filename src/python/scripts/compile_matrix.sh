#!/usr/bin/env bash
# Sweep hb_compile settings for one FoundationPose subgraph contract and emit a
# matrix of HBM variants (optimize_level x core_num x compile_mode).
#
# x86_64 toolchain host only. Thin wrapper over compile_hbm.sh: it re-invokes the
# compiler once per cell with a distinct output prefix so variants don't clobber
# each other.
#
# usage: compile_matrix.sh CONTRACT_JSON [OUTPUT_DIR]
#   e.g. compile_matrix.sh build/foundationpose_export/contracts/refine_net.json models/hbm
#
# Env knobs (space-separated lists):
#   MATRIX_OPT     default "O1 O2 O3"
#   MATRIX_CORES   default "1 2"
#   MATRIX_MODES   default "latency bandwidth"
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: compile_matrix.sh CONTRACT_JSON [OUTPUT_DIR]" >&2
  exit 2
fi

contract="$1"
output_dir="${2:-models/hbm}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$contract" ]]; then
  echo "contract not found: $contract" >&2
  exit 1
fi

opt_levels=(${MATRIX_OPT:-O1 O2 O3})
core_nums=(${MATRIX_CORES:-1 2})
modes=(${MATRIX_MODES:-latency bandwidth})

name="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$contract")"

# compile_hbm.sh scans a directory of contracts; isolate this one contract in a
# temp dir, and re-point its hbm_name/onnx_path-derived prefix per cell.
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

for opt in "${opt_levels[@]}"; do
  for cores in "${core_nums[@]}"; do
    for mode in "${modes[@]}"; do
      variant="${name}_core${cores}_${opt}_${mode}"
      # Rewrite the contract so the produced HBM carries the variant name.
      python3 - "$contract" "$tmp_dir/$name.json" "$variant" <<'PY'
import json, sys
src, dst, variant = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(src, encoding="utf-8"))
c["hbm_name"] = f"{variant}.hbm"
c["hbm_path"] = f"models/hbm/{variant}.hbm"
json.dump(c, open(dst, "w", encoding="utf-8"), indent=2, sort_keys=True)
PY
      echo "######## compiling $variant ########"
      HB_COMPILE_OPTIMIZE_LEVEL="$opt" \
      HB_COMPILE_CORE_NUM="$cores" \
      HB_COMPILE_MODE="$mode" \
      "$script_dir/compile_hbm.sh" "$tmp_dir" "$output_dir"
    done
  done
done

echo "matrix complete for $name -> $output_dir"
