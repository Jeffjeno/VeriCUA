#!/usr/bin/env bash

set -Eeuo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SUITE_DIR}/../.." && pwd)"

JUDGE_PORT="${JUDGE_PORT:-40000}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:${JUDGE_PORT}/v1}"
JUDGE_MODELS_URL="${JUDGE_MODELS_URL:-${JUDGE_BASE_URL}/models}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-4B-Thinking-sft-judge}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-osworld-4b-judge}"
BASE_WORKER_PORT="${BASE_WORKER_PORT:-40001}"
WORKER_GPU_GROUPS="${WORKER_GPU_GROUPS:-0;1;2;3;4;5;6;7}"

SGLANG_IMAGE="${SGLANG_IMAGE:-slimerl/sglang:v0.5.12.post1-cu129}"
ROUTER_IMAGE="${ROUTER_IMAGE:-lmsysorg/sgl-model-gateway:latest}"
TP_SIZE=1
ROUTER_POLICY="${ROUTER_POLICY:-round_robin}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
WORKER_READY_TIMEOUT_SEC="${WORKER_READY_TIMEOUT_SEC:-600}"
REUSE_JUDGE="${REUSE_JUDGE:-0}"

export AGENT_RUNTIME_PATH="${AGENT_RUNTIME_PATH:-${PROJECT_DIR}}"
export OSWORLD_PATH="${OSWORLD_PATH:-${PROJECT_DIR}}"
export PYTHONPATH="${AGENT_RUNTIME_PATH}:${PROJECT_DIR}:${PYTHONPATH:-}"

AGENT_RUNTIME_VENV="${AGENT_RUNTIME_VENV:-/path/to/environment-venv}"
if [[ -x "${AGENT_RUNTIME_VENV}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${AGENT_RUNTIME_VENV}/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
fi

export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,::1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,::1"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DATASET_CONFIG="${DATASET_CONFIG:-config/eval_datasets/osworld.yaml}"
REPLAY_EVAL_CONFIG="${REPLAY_EVAL_CONFIG:-${SUITE_DIR}/judge.yaml}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/osworld-qwen4b_sft_judge_n4}"
EXPECTED_ROLLOUTS="${EXPECTED_ROLLOUTS:-4}"

export GUIRL_EXPORT_SFT_DATA="${GUIRL_EXPORT_SFT_DATA:-0}"

PROBE_BASE_URL="${PROBE_BASE_URL:-${PROBE_URL:-}}"
PROBE_BASE_URL="${PROBE_BASE_URL%/models}"
PATH_TO_VM="${PATH_TO_VM:-/path/to/Ubuntu.qcow2}"
NUM_ENVS="${NUM_ENVS:-16}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_EVAL_ROUNDS="${MAX_EVAL_ROUNDS:-3}"
SAMPLER_MAX_STEPS="${SAMPLER_MAX_STEPS:-15}"
SLEEP_AFTER_EXEC="${SLEEP_AFTER_EXEC:-2}"
SLEEP_AFTER_RESET="${SLEEP_AFTER_RESET:-20}"
SLEEP_BEFORE_EVAL="${SLEEP_BEFORE_EVAL:-20}"
ROLLOUT_DOMAIN="${ROLLOUT_DOMAIN:-all}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
EFFECTIVE_REPLAY_EVAL_CONFIG=""

IFS=';' read -r -a GPU_GROUPS <<< "${WORKER_GPU_GROUPS}"
NUM_INSTANCES=${#GPU_GROUPS[@]}

worker_port() { echo $(( BASE_WORKER_PORT + $1 )); }
worker_name() { echo "${CONTAINER_PREFIX}-worker-$1"; }
router_name()  { echo "${CONTAINER_PREFIX}-router"; }

suite_ckpt_tag() {
  local ckpt="$1"
  basename "$(cd "${ckpt}" && pwd)" | tr '/ ' '__'
}

suite_resolve_model_path() {
  local arg="${1:-}"
  local path="${arg}"

  if [[ -z "${path}" ]]; then
    echo "ERROR: model path required" >&2
    return 1
  fi
  if [[ "${path}" != /* ]]; then
    path="${PROJECT_DIR}/${path}"
  fi
  path="$(cd "${path}" && pwd)"
  suite_validate_ckpt "${path}"
  echo "${path}"
}

suite_model_path_from_ref() {
  echo "$1"
}

suite_result_dir() {
  echo "${RESULT_ROOT}/$(suite_ckpt_tag "$1")"
}

suite_validate_ckpt() {
  local ckpt="$1"
  if [[ ! -d "${ckpt}" ]]; then
    echo "ERROR: checkpoint directory not found: ${ckpt}" >&2
    return 1
  fi
  if [[ ! -f "${ckpt}/config.json" ]]; then
    echo "ERROR: missing config.json under ${ckpt}" >&2
    return 1
  fi
  if [[ ! -f "${ckpt}/model.safetensors" && ! -f "${ckpt}/model.safetensors.index.json" ]]; then
    echo "ERROR: no model weights (model.safetensors) under ${ckpt}" >&2
    return 1
  fi
}

suite_stop_all_judges() {
  echo "=== Stopping judge services on port ${JUDGE_PORT} ==="
  local prefixes=(
    "${CONTAINER_PREFIX}"
  )
  for prefix in "${prefixes[@]}"; do
    for i in $(seq 0 $(( NUM_INSTANCES - 1 ))); do
      local name="${prefix}-worker-${i}"
      if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -Fxq "${name}"; then
        echo "  stop ${name}"
        docker rm -f "${name}" >/dev/null 2>&1 || true
      fi
    done
    local rname="${prefix}-router"
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -Fxq "${rname}"; then
      echo "  stop ${rname}"
      docker rm -f "${rname}" >/dev/null 2>&1 || true
    fi
  done
}

suite_judge_matches_checkpoint() {
  local ckpt="$1"
  local expected_ckpt mounted_ckpt name port
  expected_ckpt="$(cd "${ckpt}" && pwd)"

  for i in $(seq 0 $(( NUM_INSTANCES - 1 ))); do
    name="$(worker_name "$i")"
    port="$(worker_port "$i")"
    if [[ "$(docker inspect -f '{{.State.Running}}' "${name}" 2>/dev/null || true)" != "true" ]]; then
      return 1
    fi
    mounted_ckpt="$(
      docker inspect -f \
        '{{range .Mounts}}{{if eq .Destination "/model"}}{{.Source}}{{end}}{{end}}' \
        "${name}" 2>/dev/null || true
    )"
    if [[ "${mounted_ckpt}" != "${expected_ckpt}" ]]; then
      return 1
    fi
    if ! curl -sf --connect-timeout 3 "http://127.0.0.1:${port}/health" &>/dev/null; then
      return 1
    fi
  done

  if [[ "$(docker inspect -f '{{.State.Running}}' "$(router_name)" 2>/dev/null || true)" != "true" ]]; then
    return 1
  fi
  curl -sf --connect-timeout 3 "${JUDGE_MODELS_URL}" 2>/dev/null | \
    python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if any(m['id']=='${SERVED_MODEL_NAME}' for m in d.get('data',[])) else 1)" \
    2>/dev/null
}

suite_start_judge() {
  local ckpt_ref="$1"
  local ckpt
  ckpt="$(suite_model_path_from_ref "${ckpt_ref}")"
  suite_validate_ckpt "${ckpt}"

  if ! docker info &>/dev/null; then
    echo "ERROR: Cannot access Docker. Run: newgrp docker" >&2
    return 1
  fi

  if [[ "${REUSE_JUDGE}" == "1" ]]; then
    if suite_judge_matches_checkpoint "${ckpt}"; then
      echo "=== Reusing healthy judge sglang ==="
      echo "  checkpoint:   ${ckpt}"
      echo "  served name:  ${SERVED_MODEL_NAME}"
      echo "  router:       ${JUDGE_BASE_URL}"
      return 0
    fi
    echo "Reusable judge not found or does not match checkpoint; starting a fresh one."
  fi

  suite_stop_all_judges

  echo "=== Starting judge sglang ==="
  echo "  checkpoint:   ${ckpt}"
  echo "  served name:  ${SERVED_MODEL_NAME}"
  echo "  router port:  ${JUDGE_PORT}"
  echo ""

  for i in $(seq 0 $(( NUM_INSTANCES - 1 ))); do
    local name port gpus
    name=$(worker_name "$i")
    port=$(worker_port "$i")
    gpus="${GPU_GROUPS[$i]}"
    docker rm -f "${name}" 2>/dev/null || true
    echo "[worker-$i] GPU ${gpus} port ${port}"
    docker run -d \
      --name "${name}" \
      --gpus "\"device=${gpus}\"" \
      --network host \
      --ipc host \
      --shm-size 32g \
      -v "${ckpt}:/model:ro" \
      "${SGLANG_IMAGE}" \
      python -m sglang.launch_server \
        --model-path /model \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --host 0.0.0.0 \
        --port "${port}" \
        --tp-size "${TP_SIZE}" \
        --dtype bfloat16 \
        --mem-fraction-static 0.9 \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --attention-backend flashinfer \
        --trust-remote-code
  done

  echo "Waiting for workers ..."
  for i in $(seq 0 $(( NUM_INSTANCES - 1 ))); do
    local port deadline
    port=$(worker_port "$i")
    deadline=$(( $(date +%s) + WORKER_READY_TIMEOUT_SEC ))
    printf "  worker-$i "
    while true; do
      if curl -sf "http://127.0.0.1:${port}/health" &>/dev/null; then
        echo "ready"
        break
      fi
      if (( $(date +%s) > deadline )); then
        echo "TIMEOUT" >&2
        return 1
      fi
      sleep 5
    done
  done

  local rname worker_urls=()
  rname=$(router_name)
  docker rm -f "${rname}" 2>/dev/null || true
  for i in $(seq 0 $(( NUM_INSTANCES - 1 ))); do
    worker_urls+=("http://127.0.0.1:$(worker_port "$i")")
  done
  docker run -d \
    --name "${rname}" \
    --network host \
    "${ROUTER_IMAGE}" \
    --worker-urls "${worker_urls[@]}" \
    --policy "${ROUTER_POLICY}" \
    --host 0.0.0.0 \
    --port "${JUDGE_PORT}"

  sleep 3
  local router_deadline
  router_deadline=$(( $(date +%s) + 60 ))
  printf "  router "
  while true; do
    if curl -sf --connect-timeout 3 "${JUDGE_MODELS_URL}" 2>/dev/null | \
         python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if any(m['id']=='${SERVED_MODEL_NAME}' for m in d.get('data',[])) else 1)" 2>/dev/null; then
      echo "ready"
      break
    fi
    if (( $(date +%s) > router_deadline )); then
      echo "TIMEOUT: router not serving ${SERVED_MODEL_NAME} at ${JUDGE_MODELS_URL}" >&2
      return 1
    fi
    sleep 3
  done
  echo "Judge ready: ${JUDGE_BASE_URL}  model=${SERVED_MODEL_NAME}"
}

suite_stop_judge() {
  suite_stop_all_judges
}

suite_judge_status() {
  echo "=== Judge status (prefix=${CONTAINER_PREFIX}, port=${JUDGE_PORT}) ==="
  for i in $(seq 0 $(( NUM_INSTANCES - 1 ))); do
    local name port
    name=$(worker_name "$i")
    port=$(worker_port "$i")
    if docker inspect -f '{{.State.Running}}' "${name}" 2>/dev/null | grep -q true; then
      local health
      health=$(curl -sf "http://127.0.0.1:${port}/health" &>/dev/null && echo healthy || echo starting)
      echo "  worker-$i: running (${health}) port ${port}"
    else
      echo "  worker-$i: stopped"
    fi
  done
  if docker inspect -f '{{.State.Running}}' "$(router_name)" 2>/dev/null | grep -q true; then
    echo "  router: running ${JUDGE_BASE_URL}"
    curl -s "${JUDGE_MODELS_URL}" 2>/dev/null | python3 -m json.tool 2>/dev/null || true
  else
    echo "  router: stopped"
  fi
}

suite_prepare_eval() {
  local result_dir="$1"
  local checkpoint="$2"
  if [[ -z "${PYTHON_BIN}" ]] || ! command -v "${PYTHON_BIN}" >/dev/null; then
    echo "ERROR: Python not found: ${PYTHON_BIN}" >&2
    return 1
  fi
  if ! docker info &>/dev/null; then
    echo "ERROR: Cannot access Docker." >&2
    return 1
  fi
  if [[ "${JUDGE_MODELS_URL}" != "${JUDGE_BASE_URL%/}/models" ]]; then
    echo "ERROR: JUDGE_MODELS_URL must use the configured JUDGE_BASE_URL." >&2
    return 1
  fi
  if [[ -n "${PROBE_URL:-}" && "${PROBE_URL%/models}" != "${PROBE_BASE_URL%/}" ]]; then
    echo "ERROR: PROBE_URL and PROBE_BASE_URL refer to different endpoints." >&2
    return 1
  fi
  local dataset="${DATASET_CONFIG}" config="${REPLAY_EVAL_CONFIG}"
  [[ "${dataset}" != /* ]] && dataset="${PROJECT_DIR}/${dataset}"
  [[ "${config}" != /* ]] && config="${PROJECT_DIR}/${config}"
  EFFECTIVE_REPLAY_EVAL_CONFIG="${result_dir}/effective_replay_eval.yaml"
  "${PYTHON_BIN}" -c 'import eval_runtime.run_replay_eval'
  "${PYTHON_BIN}" -m eval_runtime.preflight \
    --project-dir "${PROJECT_DIR}" --dataset "${dataset}" \
    --domain "${ROLLOUT_DOMAIN}" --rollouts "${EXPECTED_ROLLOUTS}" \
    --vm "${PATH_TO_VM}" --config "${config}" \
    --output "${EFFECTIVE_REPLAY_EVAL_CONFIG}" \
    --judge-url "${JUDGE_BASE_URL}" --model "${SERVED_MODEL_NAME}" \
    --probe-url "${PROBE_BASE_URL}" --checkpoint "${checkpoint}"
  "${PYTHON_BIN}" -m eval_runtime.preflight \
    --check-endpoints "${EFFECTIVE_REPLAY_EVAL_CONFIG}" --probe-only
}

suite_preflight_eval() {
  if ! docker info &>/dev/null; then
    echo "ERROR: Cannot access Docker." >&2
    return 1
  fi
  "${PYTHON_BIN}" -m eval_runtime.preflight \
    --check-endpoints "${EFFECTIVE_REPLAY_EVAL_CONFIG}"
}

suite_write_manifest() {
  local ckpt="$1" result_dir="$2"
  mkdir -p "${result_dir}"
  python3 - <<PY
import json, datetime, os
payload = {
    "suite": "osworld_judge",
    "checkpoint": os.path.abspath("${ckpt}"),
    "served_model_name": "${SERVED_MODEL_NAME}",
    "judge_base_url": "${JUDGE_BASE_URL}",
    "result_dir": os.path.abspath("${result_dir}"),
    "dataset_config": "${DATASET_CONFIG}",
    "replay_eval_config": "${EFFECTIVE_REPLAY_EVAL_CONFIG}",
    "expected_rollouts": int("${EXPECTED_ROLLOUTS}"),
    "started_at": datetime.datetime.utcnow().isoformat() + "Z",
}
with open(os.path.join("${result_dir}", "suite_manifest.json"), "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

suite_run_eval() {
  local ckpt_ref="$1"
  shift
  local ckpt result_dir
  ckpt="$(suite_model_path_from_ref "${ckpt_ref}")"
  result_dir="$(suite_result_dir "${ckpt_ref}")"

  if [[ -z "${EFFECTIVE_REPLAY_EVAL_CONFIG}" ]]; then
    suite_prepare_eval "${result_dir}" "${ckpt}"
  fi
  suite_preflight_eval
  suite_write_manifest "${ckpt}" "${result_dir}"

  echo "=== Qwen3-VL-4B judge evaluation ==="
  echo "  Checkpoint:        ${ckpt}"
  echo "  Dataset:           ${DATASET_CONFIG}"
  echo "  Judge cfg:         ${EFFECTIVE_REPLAY_EVAL_CONFIG}"
  echo "  Result:            ${result_dir}"
  echo "  Domain:            ${ROLLOUT_DOMAIN}"
  echo "  Num envs:          ${NUM_ENVS}"
  echo "  Repeats per task:  ${EXPECTED_ROLLOUTS}"
  echo ""

  cd "${PROJECT_DIR}"
  "${PYTHON_BIN}" -m eval_runtime.run_replay_eval \
    --osworld_path "${OSWORLD_PATH}" \
    --provider_name docker \
    --path_to_vm "${PATH_TO_VM}" \
    --project_dir "${PROJECT_DIR}" \
    --eval_dataset_config "${DATASET_CONFIG}" \
    --replay_eval_config_path "${EFFECTIVE_REPLAY_EVAL_CONFIG}" \
    --result_dir "${result_dir}" \
    --domain "${ROLLOUT_DOMAIN}" \
    --num_envs "${NUM_ENVS}" \
    --max_steps "${MAX_STEPS}" \
    --max_eval_rounds "${MAX_EVAL_ROUNDS}" \
    --sampler_max_steps "${SAMPLER_MAX_STEPS}" \
    --sleep_after_execution "${SLEEP_AFTER_EXEC}" \
    --sleep_after_reset "${SLEEP_AFTER_RESET}" \
    --sleep_before_eval "${SLEEP_BEFORE_EVAL}" \
    --log_level "${LOG_LEVEL}" \
    --headless \
    --no-export_guirl_sft_data \
    "$@" \
    --replay_eval_config_path "${EFFECTIVE_REPLAY_EVAL_CONFIG}"
}

suite_score() {
  local result_dir="$1"
  cd "${PROJECT_DIR}"
  "${PYTHON_BIN}" scripts/compute_replay_eval_majority_consistency.py \
    "${result_dir}" --expected-rollouts "${EXPECTED_ROLLOUTS}"
}

suite_score_raw() {
  local result_dir="$1"
  cd "${PROJECT_DIR}"
  "${PYTHON_BIN}" scripts/compute_replay_eval_reward_consistency.py "${result_dir}"
}
