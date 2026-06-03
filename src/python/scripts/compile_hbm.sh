#!/usr/bin/env bash
# Compile FoundationPose subgraph ONNX -> S600 HBM via hb_compile.
#
# Reads each partition contract JSON for tensor names/shapes and emits a per-
# partition hb_compile YAML, then runs hb_compile. hb_compile is an x86_64 host
# tool from the D-Robotics AI Toolchain (3.7.0+); run this inside the toolchain
# Docker image or on an x86_64 host with the OpenExplorer wheels installed.
#
# usage: compile_hbm.sh CONTRACT_DIR [OUTPUT_DIR]
#   CONTRACT_DIR : build/foundationpose_export/contracts (from export.contract)
#   OUTPUT_DIR   : where HBM + generated configs land (default models/hbm)
#
# Env knobs (FP32 baseline by default):
#   HB_COMPILE_OPTIMIZE_LEVEL  O0|O1|O2|O3   (default O2)
#   HB_COMPILE_CORE_NUM        1|2           (default 2; both S600 BPU cores)
#   HB_COMPILE_MODE            latency|bandwidth (default latency)
#   HB_COMPILE_MARCH           default nash-p
set -euo pipefail

usage() {
  sed -n '3,17p' "$0" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

contract_dir="$1"
output_dir="${2:-models/hbm}"

if [[ ! -d "$contract_dir" ]]; then
  echo "contract directory does not exist: $contract_dir" >&2
  exit 1
fi

if ! command -v hb_compile >/dev/null 2>&1; then
  echo "hb_compile not found in PATH; install D-Robotics AI Toolchain 3.7.0+ on an x86_64 host or run inside the AI Toolchain Docker image" >&2
  exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "hb_compile is distributed as an x86_64 host tool; current architecture is $(uname -m)" >&2
  exit 1
fi

march="${HB_COMPILE_MARCH:-nash-p}"
optimize_level="${HB_COMPILE_OPTIMIZE_LEVEL:-O2}"
core_num="${HB_COMPILE_CORE_NUM:-2}"
compile_mode="${HB_COMPILE_MODE:-latency}"

mkdir -p "$output_dir"
config_dir="$output_dir/hb_compile_configs"
mkdir -p "$config_dir"
shopt -s nullglob
contracts=("$contract_dir"/*.json)
if [[ ${#contracts[@]} -eq 0 ]]; then
  echo "no contract JSON files found in: $contract_dir" >&2
  exit 1
fi

for contract in "${contracts[@]}"; do
  # export_index.json is not a partition contract; skip it.
  base="$(basename "$contract")"
  if [[ "$base" == "export_index.json" ]]; then
    continue
  fi
  mapfile -t fields < <(python3 - "$contract" <<'PY'
import json, sys
from pathlib import Path
contract = json.load(open(sys.argv[1], encoding="utf-8"))
print(contract["name"])
print(contract["onnx_path"])
print(contract["hbm_name"])
print(Path(contract["hbm_name"]).stem)
inputs = contract["inputs"]
print(";".join(t["name"] for t in inputs))
print(";".join("x".join(str(d) for d in t["concrete_shape"]) for t in inputs))
print(";".join(["featuremap"] * len(inputs)))
print(";".join(["NCHW"] * len(inputs)))
PY
  )
  name="${fields[0]}"
  onnx_path="${fields[1]}"
  hbm_name="${fields[2]}"
  prefix="${fields[3]}"
  input_names="${fields[4]}"
  input_shapes="${fields[5]}"
  input_types="${fields[6]}"
  input_layouts="${fields[7]}"
  config_path="$config_dir/${name}.yaml"

  if [[ ! -f "$onnx_path" ]]; then
    echo "missing ONNX for $base: $onnx_path (export it first on the x86 host)" >&2
    exit 1
  fi

  cat >"$config_path" <<EOF
model_parameters:
  onnx_model: "$onnx_path"
  march: "$march"
  working_dir: "$output_dir"
  output_model_file_prefix: "$prefix"
  core_num: $core_num
  compile_mode: "$compile_mode"

input_parameters:
  input_name: "$input_names"
  input_type_rt: "$input_types"
  input_type_train: "$input_types"
  input_layout_train: "$input_layouts"
  input_shape: "$input_shapes"
  norm_type: "no_preprocess"

calibration_parameters:
  calibration_type: "skip"

compiler_parameters:
  optimize_level: "$optimize_level"
EOF
  echo "converting $onnx_path -> $output_dir/$hbm_name (march=$march core_num=$core_num mode=$compile_mode $optimize_level)"
  hb_compile --config "$config_path"
done
