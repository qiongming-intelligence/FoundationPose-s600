#!/usr/bin/env bash
# Probe FoundationPose HBM loadability on the local S600 board.
#
# This intentionally uses only `hrt_model_exec model_info` (no inference) so it
# checks HBRT/HBM parse/load behavior without consuming input tensors. It is meant
# to catch the board-state/layout-sensitive IOVA failures observed with some
# ScoreNet real-int16 HBMs before running an end-to-end hybrid command.
#
# usage:
#   probe_hbm_loadability.sh [HBM_FILE ...]
#
# If no HBM files are provided, probes the local FoundationPose real-int16 Refine
# and Score HBMs under models/hbm_real_int16/.
#
# Env knobs:
#   FOUNDATIONPOSE_S600_PROBE_RETRIES=N   model_info attempts per HBM (default 1)
#   HRT_MODEL_EXEC=/path/to/hrt_model_exec (default /usr/hobot/bin/hrt_model_exec)
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: probe_hbm_loadability.sh [HBM_FILE ...]

Probe hrt_model_exec model_info loadability for FoundationPose S600 HBMs.
With no arguments, probes models/hbm_real_int16/foundationpose_{refine,score}_*.hbm.

Environment:
  FOUNDATIONPOSE_S600_PROBE_RETRIES=N   attempts per HBM (default 1)
  HRT_MODEL_EXEC=/path/to/hrt_model_exec (default /usr/hobot/bin/hrt_model_exec)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

hrt="${HRT_MODEL_EXEC:-/usr/hobot/bin/hrt_model_exec}"
retries="${FOUNDATIONPOSE_S600_PROBE_RETRIES:-1}"
if [[ ! "$retries" =~ ^[0-9]+$ || "$retries" -lt 1 ]]; then
  echo "FOUNDATIONPOSE_S600_PROBE_RETRIES must be a positive integer, got: $retries" >&2
  exit 2
fi
if [[ ! -x "$hrt" ]]; then
  echo "hrt_model_exec not found/executable: $hrt" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  hbms=("$@")
else
  shopt -s nullglob
  hbms=(
    models/hbm_real_int16/foundationpose_refine_net*.hbm
    models/hbm_real_int16/foundationpose_score_net_*.hbm
  )
  shopt -u nullglob
fi

if [[ ${#hbms[@]} -eq 0 ]]; then
  echo "no HBM files to probe" >&2
  exit 1
fi

printf '%-84s  %-12s  %-8s  %s\n' "HBM" "STATUS" "ATTEMPTS" "NOTE"
overall=0
for hbm in "${hbms[@]}"; do
  if [[ ! -f "$hbm" ]]; then
    printf '%-84s  %-12s  %-8s  %s\n' "$hbm" "MISSING" "0/$retries" "file not found"
    overall=1
    continue
  fi

  ok=0
  iova=0
  other=0
  note=""
  for attempt in $(seq 1 "$retries"); do
    log="/tmp/foundationpose_s600_probe_$(basename "$hbm" .hbm)_${attempt}_$$.log"
    set +e
    "$hrt" model_info --model_file "$hbm" >"$log" 2>&1
    rc=$?
    set -e
    if grep -qi "iova addr not equal" "$log"; then
      iova=$((iova + 1))
      note=$(grep -i "iova addr not equal" "$log" | tail -1 | sed -E 's/.*parsing\.rs:[0-9]+: //; s/\x1b\[[0-9;]*m//g')
    elif [[ $rc -eq 0 ]]; then
      ok=$((ok + 1))
      if [[ -z "$note" ]]; then
        note=$(grep -E '^\[model name\]' "$log" | head -1 | sed -E 's/\x1b\[[0-9;]*m//g' || true)
      fi
    else
      other=$((other + 1))
      note=$(grep -iE 'Load hbm failed|Load model failed|HBRT|failed|error' "$log" | tail -2 | tr '\n' ' ' | sed -E 's/\x1b\[[0-9;]*m//g' || true)
      [[ -n "$note" ]] || note="hrt_model_exec rc=$rc"
    fi
    rm -f "$log"
  done

  if [[ $ok -eq $retries ]]; then
    status="OK"
  elif [[ $iova -gt 0 && $ok -eq 0 && $other -eq 0 ]]; then
    status="IOVA_FAIL"
    overall=1
  elif [[ $ok -gt 0 ]]; then
    status="FLAKY"
    overall=1
  else
    status="FAIL"
    overall=1
  fi
  printf '%-84s  %-12s  %-8s  %s\n' "$hbm" "$status" "$ok/$retries" "$note"
done

exit "$overall"
