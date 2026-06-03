#!/usr/bin/env bash
# Inspect an HBM's input/output ABI on the local S600 and check it against a
# partition contract. Helps confirm the compiled tensor names/shapes/dtypes
# match what the C++ runtime and Python hybrid path expect.
#
# usage: inspect_hbm.sh HBM_FILE [CONTRACT_JSON]
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: inspect_hbm.sh HBM_FILE [CONTRACT_JSON]" >&2
  exit 2
fi

hbm="$1"
contract="${2:-}"
hrt="/usr/hobot/bin/hrt_model_exec"

if [[ ! -f "$hbm" ]]; then echo "missing HBM: $hbm" >&2; exit 1; fi

echo "================= model_info: $(basename "$hbm") ================="
"$hrt" model_info --model_file "$hbm"

if [[ -n "$contract" ]]; then
  if [[ ! -f "$contract" ]]; then echo "missing contract: $contract" >&2; exit 1; fi
  echo ""
  echo "================= contract expectation: $(basename "$contract") ================="
  python3 - "$contract" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"partition: {c['name']}  ({c['upstream_callable']})")
for kind in ("inputs", "outputs"):
    print(f"  {kind}:")
    for t in c[kind]:
        print(f"    {t['name']:<14} {t['dtype']:<8} {t['concrete_shape']}")
PY
  echo ""
  echo "Compare the [model name]/valid-shape/tensor-type lines above against the contract."
fi
