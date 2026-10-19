#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SUITE_DIR}/../.." && pwd)"

METHOD="${1:-}"
if [[ -z "${METHOD}" ]]; then
  echo "Usage: $0 {guirl|guirl_qwen4b_gui_owl} [runner args...]" >&2
  exit 2
fi
shift

case "${METHOD}" in
  guirl|guirl_qwen4b_gui_owl)
    METHOD="guirl_qwen4b_gui_owl"
    CONFIG="${SUITE_DIR}/configs/guirl_qwen4b_gui_owl.yaml"
    ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac

if [[ -z "${PYTHON:-}" ]]; then
  PYTHON="${PROJECT_DIR}/third_party/MobileWorld/.venv/bin/python"
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found or not executable: ${PYTHON}" >&2
  exit 1
fi

RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/mobileworld/${METHOD}}"
CONCURRENCY="${CONCURRENCY:-1}"
STEP_WAIT_TIME="${STEP_WAIT_TIME:-3}"
RECOVERY_TIMEOUT="${RECOVERY_TIMEOUT:-180}"
TASK_RETRIES="${TASK_RETRIES:-2}"
MOBILE_WORLD_BACKENDS="${MOBILE_WORLD_BACKENDS:-http://127.0.0.1:6800}"

ARGS=(
  --dataset mobileworld
  --evaluator-config "${CONFIG}"
  --result-root "${RESULT_ROOT}"
  --concurrency "${CONCURRENCY}"
  --step-wait-time "${STEP_WAIT_TIME}"
  --recovery-timeout "${RECOVERY_TIMEOUT}"
  --task-retries "${TASK_RETRIES}"
)

normalized_backends="${MOBILE_WORLD_BACKENDS//,/ }"
for backend in ${normalized_backends}; do
  ARGS+=(--backend "${backend}")
done

cd "${PROJECT_DIR}"
exec "${PYTHON}" -m eval_runtime.mobile_world.runner "${ARGS[@]}" "$@"
