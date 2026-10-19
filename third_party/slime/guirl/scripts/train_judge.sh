#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/third_party/slime}"
SAVE_BASE="${SAVE_BASE:-${GUIRL_RUN_DIR:-${PROJECT_ROOT}/outputs/judge_rl}/checkpoints}"
MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR}"
GUIRL_REWARD_MODE="${GUIRL_REWARD_MODE:-shaped}"
case "$GUIRL_REWARD_MODE" in shaped|binary) ;; *) echo "Invalid GUIRL_REWARD_MODE" >&2; exit 1;; esac
HF_CHECKPOINT="${HF_CHECKPOINT:?Set HF_CHECKPOINT}"
FULL_MANIFEST="${FULL_MANIFEST:-${PROJECT_ROOT}/evaluation_examples/rl_tasks.json}"
GUIRL_RUN_DIR="${GUIRL_RUN_DIR:-${PROJECT_ROOT}/outputs/judge_rl}"
TRAIN_LOG="${GUIRL_TRAIN_LOG:-${GUIRL_RUN_DIR}/guirl_full_train.log}"
RUNTIME_CONFIG="${GUIRL_RUNTIME_CONFIG:-${GUIRL_RUN_DIR}/guirl_full_config.yaml}"
RUN_TAG="${RUN_TAG:-judge_rl}"
MEGATRON_SAVE_DIR="${MEGATRON_SAVE_DIR:-${SAVE_BASE}/guirl_full_${RUN_TAG}}"
HF_SAVE_DIR="${HF_SAVE_DIR:-${SAVE_BASE}/guirl_full_hf_${RUN_TAG}}"
GUIRL_ROLLOUT_HOST_V2="${GUIRL_ROLLOUT_HOST_V2:-1}"
GUIRL_V2_TRAINER_STATE_DIR="${GUIRL_V2_TRAINER_STATE_DIR:-${GUIRL_LOCAL_STATE_ROOT:-/var/tmp/guirl-$USER/${RUN_TAG}}/trainer}"
GUIRL_V2_TRAINER_ARTIFACT_DIR="${GUIRL_V2_TRAINER_ARTIFACT_DIR:-${GUIRL_RUN_DIR}/artifacts/trainer}"
GUIRL_TMP_ROOT="${GUIRL_TMP_ROOT:-${GUIRL_RUN_DIR}/tmp}"
export GUIRL_TMP_ROOT GUIRL_V2_TRAINER_ARTIFACT_DIR RAY_TMPDIR="${RAY_TMPDIR:-/var/tmp/gr-$(printf '%s' "$RUN_TAG" | sha256sum | cut -c1-12)}"
export TMPDIR="${TMPDIR:-/var/tmp/gt-$(printf '%s' "$RUN_TAG" | sha256sum | cut -c1-12)}"
export TEMP="${TEMP:-$TMPDIR}" TMP="${TMP:-$TMPDIR}"
mkdir -p "$GUIRL_V2_TRAINER_STATE_DIR" "$RAY_TMPDIR" "$TMPDIR"
mkdir -p "${GUIRL_RUN_DIR}/logs" "${HF_SAVE_DIR}"
if [[ "$GUIRL_ROLLOUT_HOST_V2" == "1" ]]; then
   mkdir -p "$GUIRL_V2_TRAINER_STATE_DIR"
   _run_id_file="$GUIRL_V2_TRAINER_STATE_DIR/run_id"
   if [[ -n "${GUIRL_V2_RUN_ID:-}" ]]; then
      if [[ -s "$_run_id_file" && "$(<"$_run_id_file")" != "$GUIRL_V2_RUN_ID" ]]; then
         echo "ERROR: $GUIRL_V2_TRAINER_STATE_DIR belongs to run_id $(<"$_run_id_file"), not $GUIRL_V2_RUN_ID." >&2
         echo "Choose a new GUIRL_V2_RUN_ID and GUIRL_V2_TRAINER_STATE_DIR." >&2
         exit 1
      fi
   elif [[ -s "$_run_id_file" ]]; then
      GUIRL_V2_RUN_ID="$(<"$_run_id_file")"
   else
      GUIRL_V2_RUN_ID="run-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
   fi
   if [[ "${GUIRL_RENDER_ONLY:-0}" != "1" ]]; then
      printf '%s\n' "$GUIRL_V2_RUN_ID" > "$_run_id_file"
   fi
else
   GUIRL_V2_RUN_ID="${GUIRL_V2_RUN_ID:-legacy}"
fi
: > "${TRAIN_LOG}"

TRAIN_BACKEND="megatron"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-4B-Thinking}"
TRANSFORMERS_RUNTIME_DIR="${TRANSFORMERS_RUNTIME_DIR:-}"
MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-}"
MODEL_ARGS_ROTARY_BASE="${MODEL_ARGS_ROTARY_BASE:-5000000}"
TRAIN_ROTARY_BASE="${TRAIN_ROTARY_BASE:-${MODEL_ARGS_ROTARY_BASE}}"
ROLLOUT_STOP_TOKEN_IDS="${ROLLOUT_STOP_TOKEN_IDS:-}"
SLIME_SGLANG_MODEL_PATH="${SLIME_SGLANG_MODEL_PATH:-}"

if [[ -n "$TRANSFORMERS_RUNTIME_DIR" ]]; then
   export PYTHONPATH="${TRANSFORMERS_RUNTIME_DIR}:${SLIME_DIR}:${PROJECT_ROOT}:${MEGATRON_DIR}:${PYTHONPATH:-}"
else
   export PYTHONPATH="${SLIME_DIR}:${PROJECT_ROOT}:${MEGATRON_DIR}:${PYTHONPATH:-}"
fi

if [[ "${MINICPMV46_PREFLIGHT:-0}" == "1" ]]; then
   MINICPMV46_PREFLIGHT_ARGS=(
      --checkpoint "$HF_CHECKPOINT"
      --runtime-dir "$TRANSFORMERS_RUNTIME_DIR"
   )
   if [[ -n "${MINICPMV46_MAX_SLICE_NUMS:-}" ]]; then
      MINICPMV46_PREFLIGHT_ARGS+=(
         --max-slice-nums "$MINICPMV46_MAX_SLICE_NUMS"
      )
   fi
   python3 -m guirl.minicpmv46_runtime preflight \
      "${MINICPMV46_PREFLIGHT_ARGS[@]}"
fi

NUM_GPUS="${SLIME_SCRIPT_NUM_GPUS:-16}"

COLOCATE="0"
ACTOR_NUM_GPUS="${ACTOR_NUM_GPUS:-8}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-8}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_DASHBOARD_AGENT_PORT="${RAY_DASHBOARD_AGENT_PORT:-52365}"

WORKER_NODE_IP="${WORKER_NODE_IP:?ERROR: WORKER_NODE_IP must be set to the rollout node's IP}"

HEAD_NODE_IP="${HEAD_NODE_IP:-$(hostname -I | awk '{print $1}')}"

NET_IFACES="${NET_IFACES:-}"
if [ -z "$NET_IFACES" ]; then
   HEAD_IFACE=$(python3 - "$HEAD_NODE_IP" <<'PY'
import socket
import sys

import psutil

target = sys.argv[1]
for name, addresses in psutil.net_if_addrs().items():
    if any(a.family == socket.AF_INET and a.address == target for a in addresses):
        print(name)
        break
PY
)
   if [ -z "$HEAD_IFACE" ]; then
      HEAD_IFACE=$(ip route get "${WORKER_NODE_IP}" 2>/dev/null | grep -oP 'dev \K\S+' || true)
   fi
   if [ -z "$HEAD_IFACE" ]; then
      HEAD_IFACE=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' || echo "eth0")
   fi
   NET_IFACES="${HEAD_IFACE}"
   if [ -n "${NET_IFACES_WORKER:-}" ]; then
      NET_IFACES="${NET_IFACES},${NET_IFACES_WORKER}"
   fi
fi
echo "Network interfaces for cross-node traffic: ${NET_IFACES}"

HEAD_NODE_IP="${HEAD_NODE_IP%%[!0-9.]*}"
WORKER_NODE_IP="${WORKER_NODE_IP%%[!0-9.]*}"
[[ "$HEAD_NODE_IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || {
   echo "ERROR: invalid HEAD_NODE_IP: $HEAD_NODE_IP" >&2
   exit 1
}
[[ "$WORKER_NODE_IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || {
   echo "ERROR: invalid WORKER_NODE_IP: $WORKER_NODE_IP" >&2
   exit 1
}

export GLOO_SOCKET_IFNAME="${NET_IFACES}"
export NCCL_SOCKET_IFNAME="${NET_IFACES}"
export TP_SOCKET_IFNAME="${NET_IFACES}"

if [ -z "${SWAP_TRAIN_ROLLOUT_GPUS:-}" ]; then
   if [[ "$HEAD_NODE_IP" < "$WORKER_NODE_IP" ]]; then
      SWAP_TRAIN_ROLLOUT_GPUS=0
   else
      SWAP_TRAIN_ROLLOUT_GPUS=1
   fi
fi
export SLIME_SWAP_TRAIN_ROLLOUT_GPUS="$SWAP_TRAIN_ROLLOUT_GPUS"
echo "Swap train/rollout: ${SLIME_SWAP_TRAIN_ROLLOUT_GPUS} (head=${HEAD_NODE_IP}, worker=${WORKER_NODE_IP})"

export SLIME_PIN_ENGINE_CVD="0"

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"

SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.5}"
TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-1073741824}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-1}"

TRAIN_GPU_COUNT="$ACTOR_NUM_GPUS"
if [ "$(( ACTOR_NUM_GPUS + ROLLOUT_NUM_GPUS ))" -ne "$NUM_GPUS" ]; then
   echo "ERROR: ACTOR_NUM_GPUS(${ACTOR_NUM_GPUS}) + ROLLOUT_NUM_GPUS(${ROLLOUT_NUM_GPUS}) must equal NUM_GPUS(${NUM_GPUS})"
   exit 1
fi

PARALLEL_WORLD=$(( TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE ))
if [ "$PARALLEL_WORLD" -ne "$TRAIN_GPU_COUNT" ]; then
   echo "ERROR: TP(${TENSOR_MODEL_PARALLEL_SIZE}) * CP(${CONTEXT_PARALLEL_SIZE}) * PP(${PIPELINE_MODEL_PARALLEL_SIZE}) = ${PARALLEL_WORLD}, must equal training GPU count (${TRAIN_GPU_COUNT})"
   exit 1
fi

FULL_TASKS_PER_STEP="${FULL_TASKS_PER_STEP:-16}"
NUM_ACTORS="${NUM_ACTORS:-4}"
FULL_ROLLOUT_BATCH_SIZE="${FULL_ROLLOUT_BATCH_SIZE:-$(( FULL_TASKS_PER_STEP * NUM_ACTORS ))}"
FULL_GLOBAL_BATCH_SIZE="${FULL_GLOBAL_BATCH_SIZE:-${FULL_ROLLOUT_BATCH_SIZE}}"
FULL_EPOCHS="${FULL_EPOCHS:-2}"
if [[ "$GUIRL_ROLLOUT_HOST_V2" != "0" && "$GUIRL_ROLLOUT_HOST_V2" != "1" ]]; then
   echo "ERROR: GUIRL_ROLLOUT_HOST_V2 must be 0 or 1" >&2
   exit 1
fi
if (( NUM_ACTORS != 4 || FULL_TASKS_PER_STEP <= 0 )); then
   echo "ERROR: this production path requires 4 actors and a positive M=FULL_TASKS_PER_STEP" >&2
   exit 1
fi
if (( FULL_ROLLOUT_BATCH_SIZE != NUM_ACTORS * FULL_TASKS_PER_STEP || FULL_GLOBAL_BATCH_SIZE != FULL_ROLLOUT_BATCH_SIZE )); then
   echo "ERROR: v2 requires rollout_batch_size == global_batch_size == N(4) * M(${FULL_TASKS_PER_STEP})" >&2
   exit 1
fi

export FULL_MANIFEST
TASK_COUNT="$(python3 - <<'PY'
import json, os
manifest = os.environ["FULL_MANIFEST"]
with open(manifest, "r", encoding="utf-8") as f:
    data = json.load(f)
print(sum(len(v) for v in data.values()))
PY
)"
STEPS_PER_EPOCH=$(( (TASK_COUNT + FULL_TASKS_PER_STEP - 1) / FULL_TASKS_PER_STEP ))
FULL_NUM_ROLLOUT="${FULL_NUM_ROLLOUT:-$(( STEPS_PER_EPOCH * FULL_EPOCHS ))}"
SAVE_INTERVAL="${SAVE_INTERVAL:-$STEPS_PER_EPOCH}"

SKIP_ENV_START="${SKIP_ENV_START:-1}"
ENV_MASTER_HOST="${ENV_MASTER_HOST:-127.0.0.1}"
ENV_MASTER_PORT="${ENV_MASTER_PORT:-9900}"
ROLLOUT_MASTER_HOST="${ROLLOUT_MASTER_HOST:-$ENV_MASTER_HOST}"
ROLLOUT_MASTER_PORT="${ROLLOUT_MASTER_PORT:-9910}"
NUM_ENV_WORKERS="${NUM_ENV_WORKERS:-16}"
GUIRL_EXPECTED_ENV_WORKERS="${GUIRL_EXPECTED_ENV_WORKERS:-${NUM_ENV_WORKERS}}"
GUIRL_HOST_WAIT_TIMEOUT_SECONDS="${GUIRL_HOST_WAIT_TIMEOUT_SECONDS:-3600}"
GUIRL_HOST_WAIT_POLL_SECONDS="${GUIRL_HOST_WAIT_POLL_SECONDS:-5}"
GUIRL_V2_EXPECTED_ACTORS_JSON="${GUIRL_V2_EXPECTED_ACTORS_JSON:-[\"evocua\",\"qwen3vl4bthinking\",\"holo3\",\"qwen36\"]}"

if [[ "$GUIRL_ROLLOUT_HOST_V2" == "1" ]]; then
   _trainer_fingerprint="$(python3 - "$FULL_MANIFEST" "$GUIRL_V2_EXPECTED_ACTORS_JSON" "$FULL_TASKS_PER_STEP" "$FULL_ROLLOUT_BATCH_SIZE" "$FULL_GLOBAL_BATCH_SIZE" "$ROLLOUT_MASTER_HOST" "$ROLLOUT_MASTER_PORT" "$GUIRL_REWARD_MODE" "$HF_CHECKPOINT" <<'PY'
import hashlib, json, pathlib, sys
payload = {
    "manifest_sha256": hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(),
    "actor_order": json.loads(sys.argv[2]),
    "slots_per_actor": int(sys.argv[3]),
    "rollout_batch_size": int(sys.argv[4]),
    "global_batch_size": int(sys.argv[5]),
    "rollout_master": f"{sys.argv[6]}:{sys.argv[7]}",
    "reward_mode": sys.argv[8],
    "initial_checkpoint": sys.argv[9],
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
)"
   _trainer_fingerprint_file="$GUIRL_V2_TRAINER_STATE_DIR/script_config.sha256"
   if [[ -s "$_trainer_fingerprint_file" && "$(<"$_trainer_fingerprint_file")" != "$_trainer_fingerprint" ]]; then
      echo "ERROR: actor manifest/config fingerprint changed for run_id $GUIRL_V2_RUN_ID." >&2
      echo "Changing actor order, M, batches, endpoint, or manifest requires a new GUIRL_V2_RUN_ID and GUIRL_V2_TRAINER_STATE_DIR." >&2
      exit 1
   fi
   printf '%s\n' "$_trainer_fingerprint" > "$GUIRL_RUN_DIR/script_config.preview.sha256"
   if [[ "${GUIRL_RENDER_ONLY:-0}" != "1" ]]; then
      printf '%s\n' "$_trainer_fingerprint" > "$_trainer_fingerprint_file"
   fi
fi

PROBE_API_BASE="${PROBE_API_BASE:?Set PROBE_API_BASE}"
PROBE_MODEL="${PROBE_MODEL:-holo3-1-35b-a3b}"
PROBE_MAX_STEPS="${PROBE_MAX_STEPS:-15}"

GUIRL_PULL_TOTAL_DEADLINE_SECONDS="${GUIRL_PULL_TOTAL_DEADLINE_SECONDS:-21600}"

if [ -z "${SLIME_SCRIPT_EXTERNAL_RAY:-}" ] || [ "${SLIME_SCRIPT_EXTERNAL_RAY:-0}" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,::1,${ENV_MASTER_HOST},${ROLLOUT_MASTER_HOST},${HEAD_NODE_IP},${WORKER_NODE_IP}"
export NO_PROXY="${no_proxy}"
export CUDA_VISIBLE_DEVICES="${SLIME_SCRIPT_GPUS:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"


cd "${SLIME_DIR}"
echo "════════════════════════════════════════════════════════════════"
echo "  GUIRL 2-Node Disaggregated Training — 4-ACTOR (evocua + qwen3vl4b + holo3 + qwen36)"
echo "════════════════════════════════════════════════════════════════"
echo "GUIRL run dir:    ${GUIRL_RUN_DIR}"
echo "Train log:        ${TRAIN_LOG}"
echo "Head node (train): ${HEAD_NODE_IP}  (${GPUS_PER_NODE} GPUs, TP=${TENSOR_MODEL_PARALLEL_SIZE})"
echo "Worker node (roll): ${WORKER_NODE_IP}  (${GPUS_PER_NODE} GPUs, ${ROLLOUT_NUM_GPUS} sglang engines)"
echo "Total GPUs:       ${NUM_GPUS}  (across 2 nodes)"
echo "GPU mode:         DISAGGREGATED (2-node)"
echo "Network iface:    ${NET_IFACES}"
echo "Swap train/roll:  ${SLIME_SWAP_TRAIN_ROLLOUT_GPUS}"
echo "SGLang mem frac:  ${SGLANG_MEM_FRACTION}"
echo "Manifest tasks:   ${TASK_COUNT}"
echo "Tasks / step:     ${FULL_TASKS_PER_STEP}  ×  ${NUM_ACTORS} actors = ${FULL_ROLLOUT_BATCH_SIZE} samples/step"
echo "Env VMs:          ${NUM_ENV_WORKERS}  (waves of $(( (FULL_ROLLOUT_BATCH_SIZE + NUM_ENV_WORKERS - 1) / NUM_ENV_WORKERS )) per step)"
echo "Steps / epoch:    ${STEPS_PER_EPOCH}"
echo "Total rollouts:   ${FULL_NUM_ROLLOUT}  (${FULL_EPOCHS} epochs)"
echo "Save interval:    ${SAVE_INTERVAL}  (once per epoch)"
echo "════════════════════════════════════════════════════════════════"

if [[ "$SKIP_ENV_START" == "1" && "${GUIRL_RENDER_ONLY:-0}" != "1" ]]; then
   echo "Waiting for GUIRL host services..."
   echo "  Env master:     http://${ENV_MASTER_HOST}:${ENV_MASTER_PORT}"
   echo "  Rollout master: http://${ROLLOUT_MASTER_HOST}:${ROLLOUT_MASTER_PORT}"
   echo "  Expected healthy workers: ${GUIRL_EXPECTED_ENV_WORKERS}"
   python3 - \
      "$ENV_MASTER_HOST" "$ENV_MASTER_PORT" \
      "$ROLLOUT_MASTER_HOST" "$ROLLOUT_MASTER_PORT" \
      "$GUIRL_EXPECTED_ENV_WORKERS" \
      "$GUIRL_HOST_WAIT_TIMEOUT_SECONDS" \
      "$GUIRL_HOST_WAIT_POLL_SECONDS" \
      "$GUIRL_ROLLOUT_HOST_V2" \
      "$GUIRL_V2_EXPECTED_ACTORS_JSON" <<'PY'
import json
import sys
import time
import urllib.request

env_host, env_port, rollout_host, rollout_port = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
expected_workers, timeout_s, poll_s = int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])
expect_v2 = sys.argv[8] == "1"
expected_actors = json.loads(sys.argv[9])
deadline = time.time() + timeout_s
last_msg = ""


def fetch_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw else {}


def fetch_ok(url: str, timeout: float = 5.0) -> bool:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return 200 <= resp.status < 300


while time.time() < deadline:
    env_health = False
    rollout_health = False
    healthy_workers = None
    registered_workers = None
    rollout_status = None
    rollout_mode = None
    actor_order = None
    errors = []

    try:
        env_health = fetch_ok(f"http://{env_host}:{env_port}/healthz")
    except Exception as exc:
        errors.append(f"env health: {type(exc).__name__}")

    if env_health:
        try:
            status = fetch_json(f"http://{env_host}:{env_port}/status")
            master = status.get("master") or {}
            healthy_workers = int(master.get("healthy_workers") or 0)
            registered_workers = int(master.get("worker_count") or 0)
        except Exception as exc:
            errors.append(f"env status: {type(exc).__name__}")

    try:
        rollout_health = fetch_ok(f"http://{rollout_host}:{rollout_port}/healthz")
    except Exception as exc:
        errors.append(f"rollout health: {type(exc).__name__}")

    if rollout_health:
        try:
            rollout_status = fetch_json(f"http://{rollout_host}:{rollout_port}/status")
            rollout_master = rollout_status.get("master") or {}
            rollout_mode = str(rollout_master.get("mode") or "legacy").lower()
            actor_order = list(rollout_master.get("actor_order") or [])
        except Exception:
            rollout_status = None

    mode_ready = (rollout_mode == "v2") if expect_v2 else (rollout_mode != "v2")
    actors_ready = (actor_order == expected_actors) if expect_v2 else True
    if env_health and rollout_health and (healthy_workers or 0) >= expected_workers and mode_ready and actors_ready:
        print(
            f"GUIRL host ready: healthy_workers={healthy_workers}/{expected_workers}, "
            f"registered={registered_workers}, rollout_mode={rollout_mode}, actors={actor_order}",
            flush=True,
        )
        raise SystemExit(0)

    msg = (
        f"waiting: env_health={env_health}, rollout_health={rollout_health}, "
        f"healthy_workers={healthy_workers}, registered_workers={registered_workers}, "
        f"rollout_mode={rollout_mode}, actors={actor_order}"
    )
    if errors:
        msg += f", last_errors={'; '.join(errors)}"
    if msg != last_msg:
        print(msg, flush=True)
        last_msg = msg
    time.sleep(poll_s)

print(
    f"ERROR: GUIRL host services not ready after {timeout_s}s. "
    f"Need healthy_workers >= {expected_workers}, rollout mode "
    f"{'v2' if expect_v2 else 'legacy'}, and actor order {expected_actors if expect_v2 else 'legacy manifest'}.",
    file=sys.stderr,
)
raise SystemExit(1)
PY
fi

NVLINK_COUNT=$( { nvidia-smi topo -m 2>/dev/null || true; } | { grep -o 'NV[0-9][0-9]*' || true; } | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK"

cat > "$RUNTIME_CONFIG" <<EOF
guirl_env_master_url: "http://${ENV_MASTER_HOST}:${ENV_MASTER_PORT}"
guirl_rollout_master_url: "http://${ROLLOUT_MASTER_HOST}:${ROLLOUT_MASTER_PORT}"

guirl_reward_mode: "${GUIRL_REWARD_MODE}"

guirl_rollout_host_v2: $([[ "$GUIRL_ROLLOUT_HOST_V2" == "1" ]] && echo true || echo false)
guirl_v2_trainer_state_dir: "$GUIRL_V2_TRAINER_STATE_DIR"
guirl_v2_run_id: "$GUIRL_V2_RUN_ID"
guirl_v2_expected_actors: ${GUIRL_V2_EXPECTED_ACTORS_JSON}
guirl_v2_slots_per_actor: ${FULL_TASKS_PER_STEP}
guirl_v2_allow_batch_size_compatibility: false

guirl_stream_pull: true
guirl_pull_total_deadline_seconds: ${GUIRL_PULL_TOTAL_DEADLINE_SECONDS}

guirl_probe_agent_type: "Holo3_Probe"
guirl_probe_base_url: "$PROBE_API_BASE"
guirl_probe_model_name: "$PROBE_MODEL"
guirl_probe_api_key: "EMPTY"
guirl_probe_max_tokens: 8192
guirl_probe_max_steps: ${PROBE_MAX_STEPS}
guirl_probe_temperature: 0.0
guirl_probe_enable_thinking: true
guirl_probe_reasoning_effort: "medium"
guirl_probe_max_trajectory_length: 15
guirl_probe_max_images: 3
guirl_probe_client_password: "password"
guirl_probe_sleep_after_execution: 2.0

guirl_judge_max_rounds: 3
guirl_judge_history_n: 3
guirl_judge_max_tokens: 24576
guirl_judge_temperature: 1.0
guirl_judge_enable_thinking: true
guirl_judge_strip_history_thinking: true
guirl_judge_fanout_turns: true
guirl_allow_missing_thinking_template: false
guirl_judge_parse_max_retries: 3

guirl_mm_pool_size: ${GUIRL_MM_POOL_SIZE:-32}
guirl_mm_per_image_cache: ${GUIRL_MM_PER_IMAGE_CACHE:-true}

guirl_manifest_path: "$FULL_MANIFEST"
guirl_examples_dir: "${PROJECT_ROOT}/evaluation_examples/examples"

guirl_batch_size: ${NUM_ENV_WORKERS}
guirl_env_capacity: ${NUM_ENV_WORKERS}
guirl_close_env_after_generate: true
EOF

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --save          "${MEGATRON_SAVE_DIR}"
   --save-hf       "${HF_SAVE_DIR}/rollout_{rollout_id}"
   --save-interval "${SAVE_INTERVAL}"
   --rotary-base   "${TRAIN_ROTARY_BASE}"
)
if [[ "${ENABLE_RESUME_LOAD:-0}" == "1" ]]; then
   CKPT_ARGS+=(--load "${MEGATRON_SAVE_DIR}")
else
   CKPT_ARGS+=(--load "${HF_CHECKPOINT}")
fi

ROLLOUT_ARGS=(
   --data-source-path              "guirl.master_datasource.data_source.RolloutDataSource"
   --rollout-function-path         "guirl.master_datasource.streaming_rollout.generate_rollout"
   --custom-generate-function-path "guirl.slime_custom.custom_generate"
   --custom-rm-path                "guirl.slime_custom.custom_rm"
   --custom-config-path            "$RUNTIME_CONFIG"
   --num-rollout               "${FULL_NUM_ROLLOUT}"
   --rollout-batch-size        "${FULL_ROLLOUT_BATCH_SIZE}"
   --over-sampling-batch-size  "${FULL_ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt      1
   --rollout-max-response-len  24576
   --rollout-temperature       1.0
   --global-batch-size         "${FULL_GLOBAL_BATCH_SIZE}"
)
if [[ -n "$ROLLOUT_STOP_TOKEN_IDS" ]]; then
   read -r -a _rollout_stop_ids <<< "$ROLLOUT_STOP_TOKEN_IDS"
   ROLLOUT_ARGS+=(--rollout-stop-token-ids "${_rollout_stop_ids[@]}")
fi

MULTIMODAL_KEYS='{"image": "images"}'

GRPO_ARGS=(
   --advantage-estimator reinforce_plus_plus
   --normalize-advantages
   --eps-clip       0.2
   --eps-clip-high  0.28
)

OPTIMIZER_ARGS=(
   --optimizer      adam
   --lr             "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay   0.1
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static  "${SGLANG_MEM_FRACTION}"
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64
)
if [[ -n "$SLIME_SGLANG_MODEL_PATH" ]]; then
   SGLANG_ARGS+=(--sglang-model-path "$SLIME_SGLANG_MODEL_PATH")
fi

PLACEMENT_ARGS=(
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
)

MISC_ARGS=(
   "${PLACEMENT_ARGS[@]}"
   --use-wandb
   --wandb-project          "${WANDB_PROJECT:-guirl}"
   --wandb-group            "${WANDB_GROUP:-judge_rl}"
   --wandb-dir              "${GUIRL_RUN_DIR}/wandb"
)

BACKEND_ARGS=(
   --train-backend              megatron
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
   --context-parallel-size      "${CONTEXT_PARALLEL_SIZE}"
   --calculate-per-token-loss
   --expert-model-parallel-size  1
   --expert-tensor-parallel-size 1
   --recompute-granularity      full
   --recompute-method           uniform
   --recompute-num-layers       1
   --recompute-loss-function
   --micro-batch-size           1
   --log-probs-chunk-size       128
   --attention-dropout          0.0
   --hidden-dropout             0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend          flash
   --megatron-to-hf-mode        bridge
)

if [ "$OPTIMIZER_CPU_OFFLOAD" = "1" ]; then
   BACKEND_ARGS+=(
      --optimizer-cpu-offload
      --use-precision-aware-optimizer
      --overlap-cpu-optimizer-d2h-h2d
   )
fi

if [[ -n "$MODEL_ARGS_SCRIPT" ]]; then
   source "$MODEL_ARGS_SCRIPT"
else
   MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
   source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"
fi

printf '%s\0' python3 "${SLIME_DIR}/train.py" \
   --actor-num-nodes 1 --actor-num-gpus-per-node "${TRAIN_GPU_COUNT}" \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
   "${GRPO_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${SGLANG_ARGS[@]}" \
   "${BACKEND_ARGS[@]}" "${MISC_ARGS[@]}" > "${GUIRL_RUN_DIR}/train_argv.nul"
if [[ "${GUIRL_RENDER_ONLY:-0}" == "1" ]]; then
   echo "Production config rendered: ${GUIRL_RUN_DIR}"
   exit 0
fi

if [[ "${GUIRL_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
   python3 "${SLIME_DIR}/train.py" \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node "${TRAIN_GPU_COUNT}" \
      --multimodal-keys "${MULTIMODAL_KEYS}" \
      "${MODEL_ARGS[@]}" \
      "${CKPT_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${GRPO_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" \
      "${SGLANG_ARGS[@]}" \
      "${BACKEND_ARGS[@]}" \
      "${MISC_ARGS[@]}" \
      --help >/dev/null
   echo "GUIRL launcher dry-run OK: ${MODEL_NAME}"
   exit 0
fi

if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force || true
   pkill -9 ray || true

   export MASTER_ADDR="${HEAD_NODE_IP}"
   echo "Starting Ray HEAD on ${HEAD_NODE_IP} with ${GPUS_PER_NODE} GPUs..."
   ray start --head \
      --node-ip-address "${HEAD_NODE_IP}" \
      --port "${RAY_GCS_PORT}" \
      --num-gpus "${GPUS_PER_NODE}" \
      --temp-dir "${RAY_TMPDIR}" \
      --object-spilling-directory "${GUIRL_TMP_ROOT}/spill" \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0 \
      --dashboard-port="${RAY_DASHBOARD_PORT}" \
      --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_PORT}"

   echo "Waiting for Ray dashboard agent to be ready..."
   for i in $(seq 1 30); do
      if curl -sf "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/jobs/" >/dev/null 2>&1; then
         echo "Ray dashboard agent is ready."
         break
      fi
      sleep 2
   done

   echo "Waiting for worker node ${WORKER_NODE_IP} to join the Ray cluster..."
   echo "  Make sure the worker node has started Ray with:"
   echo "  ray start --address=${HEAD_NODE_IP}:${RAY_GCS_PORT} --num-gpus ${GPUS_PER_NODE} --node-ip-address ${WORKER_NODE_IP}"
   echo ""
   WORKER_WAIT_TIMEOUT="${WORKER_WAIT_TIMEOUT:-300}"
   WORKER_WAIT_POLL="${WORKER_WAIT_POLL:-5}"
   deadline=$((SECONDS + WORKER_WAIT_TIMEOUT))
   while [ $SECONDS -lt $deadline ]; do
      alive_nodes=$(python3 -c "
import ray
ray.init(address='127.0.0.1:${RAY_GCS_PORT}', ignore_reinit_error=True, logging_level='ERROR')
nodes = ray.nodes()
alive = [n for n in nodes if n.get('alive', False)]
print(len(alive))
ray.shutdown()
" 2>/dev/null || echo "0")
      if [ "$alive_nodes" -ge 2 ]; then
         echo "Worker node joined! Cluster has ${alive_nodes} alive nodes."
         break
      fi
      echo "  waiting for worker... alive_nodes=${alive_nodes}/2"
      sleep "$WORKER_WAIT_POLL"
   done

   if [ "$alive_nodes" -lt 2 ]; then
      echo "ERROR: Worker node ${WORKER_NODE_IP} did not join within ${WORKER_WAIT_TIMEOUT}s."
      echo "Check that the worker node has started Ray and can reach ${HEAD_NODE_IP}:${RAY_GCS_PORT}"
      exit 1
   fi

   echo "Ray cluster status:"
   ray status
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${TRANSFORMERS_RUNTIME_DIR:+${TRANSFORMERS_RUNTIME_DIR}:}${SLIME_DIR}:${PROJECT_ROOT}:${MEGATRON_DIR}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"WANDB_MODE\": \"${WANDB_MODE:-offline}\",
    \"TMPDIR\": \"${TMPDIR}\",
    \"TEMP\": \"${TEMP}\",
    \"TMP\": \"${TMP}\",
    \"RAY_TMPDIR\": \"${RAY_TMPDIR}\",
    \"GUIRL_V2_TRAINER_ARTIFACT_DIR\": \"${GUIRL_V2_TRAINER_ARTIFACT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
    \"TP_SOCKET_IFNAME\": \"${TP_SOCKET_IFNAME}\",
    \"SLIME_SWAP_TRAIN_ROLLOUT_GPUS\": \"${SLIME_SWAP_TRAIN_ROLLOUT_GPUS}\",
    \"SLIME_PIN_ENGINE_CVD\": \"${SLIME_PIN_ENGINE_CVD}\",
    \"SLIME_SGLANG_MODEL_PATH\": \"${SLIME_SGLANG_MODEL_PATH}\",
    \"MINICPMV46_MAX_SLICE_NUMS\": \"${MINICPMV46_MAX_SLICE_NUMS:-}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"no_proxy\": \"${no_proxy}\",
    \"NO_PROXY\": \"${NO_PROXY}\"
  }
}"

cd "${SLIME_DIR}"
echo "Ray job submit cwd: $(pwd)"
ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_DIR}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${TRAIN_GPU_COUNT}" \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${BACKEND_ARGS[@]}" \
   "${MISC_ARGS[@]}"
