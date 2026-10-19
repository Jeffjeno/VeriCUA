#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SUITE_DIR}/../.." && pwd)"
METHOD="${1:-}"
DATASET="${DATASET:-webarena}"
if [[ -z "${METHOD}" ]]; then
  echo "Usage: DATASET=webarena $0 {guirl|guirl_qwen4b_holo} [runner args...]" >&2
  exit 2
fi
shift

case "${METHOD}" in
  guirl|guirl_qwen4b_holo)
    METHOD=guirl_qwen4b_holo
    CONFIG="${SUITE_DIR}/configs/guirl_qwen4b_holo.yaml"
    ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac

CONFIG="${EVALUATOR_CONFIG:-${CONFIG}}"

if [[ -z "${PYTHON:-}" ]]; then
  PYTHON="${PROJECT_DIR}/third_party/webarena-verified/.venv/bin/python"
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found: ${PYTHON}" >&2
  exit 1
fi

RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/${DATASET}_eval/${METHOD}}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

cd "${PROJECT_DIR}"
exec "${PYTHON}" -m eval_runtime.web_arena.runner \
  --dataset "${DATASET}" \
  --evaluator-config "${CONFIG}" \
  --result-root "${RESULT_ROOT}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  "$@"
