#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SUITE_DIR}/../.." && pwd)"
DEFAULT_PYTHON="${PROJECT_DIR}/../../envs/agent_runtime/bin/python"
[[ -x "${DEFAULT_PYTHON}" ]] || DEFAULT_PYTHON="$(command -v python)"
PYTHON_BIN="${PYTHON:-${DEFAULT_PYTHON}}"
METHOD="${1:-guirl_qwen4b_gui_owl}"
shift || true

case "${METHOD}" in
  guirl) DOCTOR_METHOD="guirl_qwen4b_gui_owl" ;;
  *) DOCTOR_METHOD="${METHOD}" ;;
esac

SMOKE_LIMIT="${SMOKE_LIMIT:-1}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/mobileworld_smoke/${METHOD}}"
export RESULT_ROOT

"${PYTHON_BIN}" "${SUITE_DIR}/doctor.py" --method "${DOCTOR_METHOD}" ${DOCTOR_ARGS:-}

TASK_ARGS=(--limit "${SMOKE_LIMIT}")
if [[ -n "${SMOKE_TASK:-}" ]]; then
  TASK_ARGS=(--tasks "${SMOKE_TASK}")
fi

"${SUITE_DIR}/run.sh" "${METHOD}" \
  --no-resume \
  "${TASK_ARGS[@]}" \
  "$@"

"${PYTHON_BIN}" "${SUITE_DIR}/score.py" --result-root "${RESULT_ROOT}"
