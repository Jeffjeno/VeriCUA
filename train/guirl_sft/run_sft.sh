#!/usr/bin/env bash

set -euo pipefail

RUN_CONFIG="${1:-}"
if [[ -z "${RUN_CONFIG}" ]]; then
  echo "Usage: bash $0 <sft_*.yaml>" >&2
  echo "  e.g. bash $0 train/guirl_sft/configs/sft_qwen3vl4b.yaml" >&2
  exit 1
fi

export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,::1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,::1"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

AGENT_RUNTIME_VENV="${AGENT_RUNTIME_VENV:-/path/to/environment-venv}"
if [[ -f "${AGENT_RUNTIME_VENV}/bin/activate" ]]; then
  source "${AGENT_RUNTIME_VENV}/bin/activate"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

export AGENT_RUNTIME_PATH="${AGENT_RUNTIME_PATH:-${PROJECT_DIR}}"
export PYTHONPATH="${AGENT_RUNTIME_PATH}:${PROJECT_DIR}:${PYTHONPATH:-}"
export OSWORLD_PATH="${OSWORLD_PATH:-${PROJECT_DIR}}"
export OSWORLD_VM_PATH="${OSWORLD_VM_PATH:-/path/to/Ubuntu.qcow2}"

if [[ "${RUN_CONFIG}" != /* ]]; then
  RUN_CONFIG="${PROJECT_DIR}/${RUN_CONFIG}"
fi
if [[ ! -f "${RUN_CONFIG}" ]]; then
  echo "ERROR: config not found: ${RUN_CONFIG}" >&2
  exit 2
fi

if ! python -c "import pyregister" &>/dev/null; then
  echo "ERROR: Install train/rft/requirements-env.txt before collecting data." >&2
  exit 1
fi

if ! python -c "import rollout_runtime; import sys; from pathlib import Path; r=str(Path('.').resolve()); sys.path.insert(0,r) if r not in sys.path else None; from mm_agents.holo3_probe_module import Holo3ProbeAgent" &>/dev/null; then
  echo "ERROR: Holo3 probe import failed. Check PYTHONPATH and pyregister." >&2
  exit 1
fi

DRY_RUN="${DRY_RUN:-0}"
LIST_ONLY="${LIST_ONLY:-0}"
DEBUG="${DEBUG:-0}"
LIMIT="${LIMIT:-}"

if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" && "${LIST_ONLY}" != "1" && "${LIST_ONLY}" != "true" ]]; then
  if ! docker info &>/dev/null; then
    echo "ERROR: Cannot access Docker." >&2
    exit 1
  fi
fi

mapfile -t _COLLECT_ARGS < <(
python - "${RUN_CONFIG}" "${PROJECT_DIR}" "${DRY_RUN}" "${LIST_ONLY}" "${DEBUG}" "${LIMIT}" <<'PY'
import sys
from pathlib import Path

import yaml

run_cfg_path = Path(sys.argv[1])
project_dir = Path(sys.argv[2])
dry_run = sys.argv[3] in ("1", "true")
list_only = sys.argv[4] in ("1", "true")
debug = sys.argv[5] in ("1", "true")
limit = sys.argv[6].strip()

cfg = yaml.safe_load(run_cfg_path.read_text(encoding="utf-8")) or {}
actor = cfg.get("actor") or {}
run = cfg.get("run") or {}

guirl_rel = cfg.get("guirl_config", "train/guirl_sft/configs/shared_judge_probe.yaml")
guirl_path = Path(guirl_rel)
if not guirl_path.is_absolute():
    guirl_path = project_dir / guirl_path
if not guirl_path.is_file():
    raise SystemExit(f"guirl_config not found: {guirl_path}")

output_dir = run.get("output_dir")
if output_dir and not Path(output_dir).is_absolute():
    output_dir = str(project_dir / output_dir)

args = [
    "--osworld-path", str(project_dir if not run.get("osworld_path") else run["osworld_path"]),
    "--vm-path", run.get("vm_path", "/path/to/Ubuntu.qcow2"),
    "--provider", run.get("provider", "docker"),
    "--num-vms", str(run.get("num_vms", 8)),
    "--max-steps", str(run.get("max_steps", 30)),
    "--output-dir", output_dir or str(project_dir / "results" / f"guirl_sft_{cfg.get('name', 'data')}"),
    "--vm-upload-dir", run.get("vm_upload_dir", "/home/user/Desktop"),
    "--settle-time", str(run.get("settle_time", 20)),
    "--sleep-after-exec", str(run.get("sleep_after_exec", 2)),
    "--client-password", actor.get("client_password", "password"),
    "--harness", actor["harness"],
    "--model", actor["model"],
    "--api-base", actor["api_base"],
    "--api-key", actor.get("api_key", "EMPTY"),
    "--max-tokens", str(actor.get("max_tokens", 8192)),
    "--temperature", str(actor.get("temperature", 0.0)),
    "--request-timeout", str(actor.get("request_timeout", 1800)),
    "--guirl-config", str(guirl_path),
    "--max-eval-rounds", str(run.get("max_eval_rounds", 3)),
    "--num-rollouts", str(run.get("num_rollouts", 2)),
    "--split", run.get("split", "sft_tasks"),
]

harness = actor.get("harness", "")
if harness.startswith("holo3") or harness.startswith("holo31"):
    args.extend([
        "--max-trajectory-length", str(actor.get("max_trajectory_length", 20)),
        "--reasoning-effort", str(actor.get("reasoning_effort", "medium")),
        "--holo3-max-images", str(actor.get("holo3_max_images", 3)),
    ])
    if actor.get("holo3_enable_thinking", True):
        args.append("--holo3-enable-thinking")
    else:
        args.append("--no-holo3-enable-thinking")

if harness in ("qwen35vl", "qwen3vl"):
    args.extend([
        "--coordinate-type", str(actor.get("coordinate_type", "absolute")),
        "--history-n", str(actor.get("history_n", 4)),
        "--image-max", str(actor.get("image_max", 5)),
        "--fold-size", str(actor.get("fold_size", 2)),
    ])
    if actor.get("enable_thinking", False):
        args.append("--enable-thinking")
    else:
        args.append("--no-enable-thinking")

if run.get("headless", True):
    args.append("--headless")

if limit:
    args.extend(["--limit", limit])
elif run.get("limit") is not None:
    args.extend(["--limit", str(run["limit"])])

if dry_run:
    args.append("--dry-run")
if list_only:
    args.append("--list-only")
if debug:
    args.append("--debug")

for arg in args:
    print(arg)
PY
)

echo "=== GUIRL SFT (${RUN_CONFIG}) ==="
echo "Project:  ${PROJECT_DIR}"
echo "Config:   ${RUN_CONFIG}"
echo ""

exec python -m train.guirl_sft.collect "${_COLLECT_ARGS[@]}"
