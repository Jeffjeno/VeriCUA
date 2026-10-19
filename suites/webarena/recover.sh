#!/usr/bin/env bash
set -euo pipefail
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SUITE_DIR}/../.." && pwd)"
METHOD="${1:-guirl_qwen4b_holo}"
if [[ "${METHOD}" == "guirl" ]]; then
  METHOD="guirl_qwen4b_holo"
fi
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/webarena/${METHOD}}"
mapfile -t IDS < <(python "${SUITE_DIR}/score.py" --result-root "${RESULT_ROOT}" --error-tasks)
if [[ "${#IDS[@]}" -eq 0 ]]; then
  echo "No error episodes found."
  exit 0
fi
ARGS=()
for candidate_id in "${IDS[@]}"; do
  ARGS+=(--candidate-id "${candidate_id}")
done
RESULT_ROOT="${RESULT_ROOT}" exec bash "${SUITE_DIR}/run.sh" "${METHOD}" --no-resume "${ARGS[@]}"
