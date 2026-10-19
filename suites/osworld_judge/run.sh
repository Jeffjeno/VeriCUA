#!/usr/bin/env bash
set -Eeuo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SUITE_DIR}/lib.sh"

KEEP_JUDGE=0
WITH_SCORE=0
EVAL_EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <checkpoint_dir> [--with-score] [--keep-judge] [--reuse-judge] [eval_args...]
  $(basename "$0") score <checkpoint_dir>
  $(basename "$0") score_raw <checkpoint_dir>
  $(basename "$0") stop-judge
  $(basename "$0") status

Example:
  bash suites/osworld_judge/run.sh /path/to/qwen3vl4b-checkpoint --with-score
EOF
}

on_exit() {
  local code=$?
  if [[ "${KEEP_JUDGE}" -eq 0 && "${SUITE_STARTED_JUDGE:-0}" -eq 1 ]]; then
    suite_stop_judge || true
  fi
  exit "${code}"
}

cmd="${1:-}"
case "${cmd}" in
  -h|--help|help) usage; exit 0 ;;
  "") usage >&2; exit 1 ;;
  stop-judge) suite_stop_judge; exit 0 ;;
  status) suite_judge_status; exit 0 ;;
  score|score_raw)
    ckpt="$(suite_resolve_model_path "${2:-}")"
    result_dir="$(suite_result_dir "${ckpt}")"
    if [[ ! -d "${result_dir}" ]]; then
      echo "ERROR: result dir not found: ${result_dir}" >&2
      exit 1
    fi
    if [[ "${cmd}" == "score" ]]; then
      suite_score "${result_dir}"
    else
      suite_score_raw "${result_dir}"
    fi
    exit 0
    ;;
esac

ckpt="$(suite_resolve_model_path "${cmd}")"
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-score) WITH_SCORE=1; shift ;;
    --keep-judge) KEEP_JUDGE=1; shift ;;
    --reuse-judge) REUSE_JUDGE=1; KEEP_JUDGE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) EVAL_EXTRA_ARGS+=("$1"); shift ;;
  esac
done

trap on_exit EXIT
suite_prepare_eval "$(suite_result_dir "${ckpt}")" "${ckpt}"
SUITE_STARTED_JUDGE=1
suite_start_judge "${ckpt}"
suite_run_eval "${ckpt}" ${EVAL_EXTRA_ARGS[@]+"${EVAL_EXTRA_ARGS[@]}"}
result_dir="$(suite_result_dir "${ckpt}")"
echo "Results: ${result_dir}"
if [[ "${WITH_SCORE}" -eq 1 ]]; then
  suite_score "${result_dir}"
fi
