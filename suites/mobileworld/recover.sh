#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SUITE_DIR}/../.." && pwd)"
DEFAULT_PYTHON="${PROJECT_DIR}/../../envs/agent_runtime/bin/python"
[[ -x "${DEFAULT_PYTHON}" ]] || DEFAULT_PYTHON="$(command -v python)"
PYTHON_BIN="${PYTHON:-${DEFAULT_PYTHON}}"
METHOD="${1:-guirl_qwen4b_gui_owl}"
if [[ -z "${METHOD}" ]]; then
  echo "Usage: $0 [guirl|guirl_qwen4b_gui_owl] [runner args...]" >&2
  exit 2
fi
shift

if [[ "${METHOD}" == "guirl" ]]; then
  METHOD="guirl_qwen4b_gui_owl"
fi

RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/mobileworld/${METHOD}}"
export RESULT_ROOT
if [[ ! -d "${RESULT_ROOT}" ]]; then
  echo "Result root does not exist: ${RESULT_ROOT}" >&2
  exit 1
fi

TASK_FILE="${RESULT_ROOT}/recovery_tasks.txt"
"${PYTHON_BIN}" "${SUITE_DIR}/score.py" \
  --result-root "${RESULT_ROOT}" \
  --error-tasks > "${TASK_FILE}"

if [[ ! -s "${TASK_FILE}" ]]; then
  echo "No error tasks to recover."
  exit 0
fi

echo "Recovering $(wc -l < "${TASK_FILE}") task(s) from ${TASK_FILE}"
"${SUITE_DIR}/run.sh" "${METHOD}" \
  --task-file "${TASK_FILE}" \
  --no-resume \
  "$@"

"${PYTHON_BIN}" "${SUITE_DIR}/score.py" --result-root "${RESULT_ROOT}"
