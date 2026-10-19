#!/usr/bin/env bash
set -euo pipefail
SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${1:-guirl_qwen4b_holo}"
shift || true
if [[ -n "${SMOKE_CANDIDATE:-}" ]]; then
  exec bash "${SUITE_DIR}/run.sh" "${METHOD}" --candidate-id "${SMOKE_CANDIDATE}" --no-resume "$@"
fi
exec bash "${SUITE_DIR}/run.sh" "${METHOD}" --limit 1 --no-resume "$@"
