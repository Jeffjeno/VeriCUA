#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUIRL_DIR="$(dirname "$SCRIPT_DIR")"
SLIME_DIR="$(cd "$GUIRL_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$SLIME_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

ACTOR_EVOCUA_API_BASE="${ACTOR_EVOCUA_API_BASE:?ERROR: set ACTOR_EVOCUA_API_BASE to the evocua actor endpoint}"
ACTOR_EVOCUA_MODEL="${ACTOR_EVOCUA_MODEL:-evocua-32b}"
ACTOR_EVOCUA_MAX_TOKENS="${ACTOR_EVOCUA_MAX_TOKENS:-32768}"

ACTOR_QWEN3VL4B_API_BASE="${ACTOR_QWEN3VL4B_API_BASE:?ERROR: set ACTOR_QWEN3VL4B_API_BASE to the qwen3vl4bthinking actor endpoint}"
ACTOR_QWEN3VL4B_MODEL="${ACTOR_QWEN3VL4B_MODEL:-Qwen3-VL-4B-Thinking}"
ACTOR_QWEN3VL4B_HARNESS="${ACTOR_QWEN3VL4B_HARNESS:-qwen3vl}"
ACTOR_QWEN3VL4B_MAX_TOKENS="${ACTOR_QWEN3VL4B_MAX_TOKENS:-32768}"
ACTOR_QWEN3VL4B_HISTORY_N="${ACTOR_QWEN3VL4B_HISTORY_N:-4}"
ACTOR_QWEN3VL4B_COORDINATE_TYPE="${ACTOR_QWEN3VL4B_COORDINATE_TYPE:-relative}"

ACTOR_HOLO3_API_BASE="${ACTOR_HOLO3_API_BASE:?ERROR: set ACTOR_HOLO3_API_BASE to the holo3 actor endpoint}"
ACTOR_HOLO3_MODEL="${ACTOR_HOLO3_MODEL:-holo3-1-35b-a3b}"
ACTOR_HOLO3_MAX_TOKENS="${ACTOR_HOLO3_MAX_TOKENS:-4096}"
ACTOR_HOLO3_MAX_IMAGES="${ACTOR_HOLO3_MAX_IMAGES:-3}"
ACTOR_HOLO3_MAX_TRAJECTORY_LENGTH="${ACTOR_HOLO3_MAX_TRAJECTORY_LENGTH:-20}"
ACTOR_HOLO3_ENABLE_THINKING="${ACTOR_HOLO3_ENABLE_THINKING:-true}"

ACTOR_QWEN36_API_BASE="${ACTOR_QWEN36_API_BASE:?ERROR: set ACTOR_QWEN36_API_BASE to the qwen3.6 actor endpoint}"
ACTOR_QWEN36_MODEL="${ACTOR_QWEN36_MODEL:-Qwen3.6-27B}"
ACTOR_QWEN36_HARNESS="${ACTOR_QWEN36_HARNESS:-qwen35vl}"
ACTOR_QWEN36_MAX_TOKENS="${ACTOR_QWEN36_MAX_TOKENS:-16384}"
ACTOR_QWEN36_COORDINATE_TYPE="${ACTOR_QWEN36_COORDINATE_TYPE:-relative}"
ACTOR_QWEN36_HISTORY_N="${ACTOR_QWEN36_HISTORY_N:-4}"
ACTOR_QWEN36_IMAGE_MAX="${ACTOR_QWEN36_IMAGE_MAX:-5}"
ACTOR_QWEN36_FOLD_SIZE="${ACTOR_QWEN36_FOLD_SIZE:-2}"

ACTOR_TEMPERATURE="${ACTOR_TEMPERATURE:-1.0}"
ACTOR_MAX_STEPS="${ACTOR_MAX_STEPS:-30}"
ACTOR_TIMEOUT_SECONDS="${ACTOR_TIMEOUT_SECONDS:-1200}"

_actor_hosts="$(python3 - "$ACTOR_EVOCUA_API_BASE" "$ACTOR_QWEN3VL4B_API_BASE" "$ACTOR_HOLO3_API_BASE" "$ACTOR_QWEN36_API_BASE" <<'PY'
import sys
from urllib.parse import urlparse
hosts = {urlparse(u).hostname for u in sys.argv[1:] if u}
print(",".join(sorted(h for h in hosts if h)))
PY
)"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,::1,${_actor_hosts}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,::1,${_actor_hosts}"

NUM_ENV_WORKERS="${NUM_ENV_WORKERS:-16}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
ENV_MASTER_PORT="${ENV_MASTER_PORT:-9900}"
ENV_WORKER_BASE_PORT="${ENV_WORKER_BASE_PORT:-9801}"
ROLLOUT_MASTER_PORT="${ROLLOUT_MASTER_PORT:-9910}"
ENV_MASTER_HEARTBEAT_TIMEOUT_SECONDS="${ENV_MASTER_HEARTBEAT_TIMEOUT_SECONDS:-90}"
ENV_WORKER_START_TIMEOUT_SECONDS="${ENV_WORKER_START_TIMEOUT_SECONDS:-180}"
ENV_WORKER_START_BATCH_SIZE="${ENV_WORKER_START_BATCH_SIZE:-4}"

GUIRL_DOCKER_NAME_PREFIX="${GUIRL_DOCKER_NAME_PREFIX:-guirl-${RUN_TAG:-judge_rl}}"

GUIRL_MANIFEST_PATH="${GUIRL_MANIFEST_PATH:-$PROJECT_ROOT/evaluation_examples/rl_tasks.json}"
GUIRL_EXAMPLES_DIR="${GUIRL_EXAMPLES_DIR:-$PROJECT_ROOT/evaluation_examples/examples}"

export OSWORLD_PATH="${OSWORLD_PATH:-$PROJECT_ROOT}"
export OSWORLD_VM_PATH="${OSWORLD_VM_PATH:?Set OSWORLD_VM_PATH}"

AGENT_RUNTIME_VENV="${AGENT_RUNTIME_VENV:?Set AGENT_RUNTIME_VENV}"
AGENT_RUNTIME_SRC="$PROJECT_ROOT"

GUIRL_RUN_DIR="${GUIRL_RUN_DIR:-${PROJECT_ROOT}/outputs/judge_rl}"
LOG_DIR="${GUIRL_LOG_DIR:-$GUIRL_RUN_DIR/logs}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/env_master.log"
ROLLOUT_MASTER_LOG="$LOG_DIR/rollout_master.log"
ACTORS_CONFIG_PATH="${ACTORS_CONFIG_PATH:-$GUIRL_RUN_DIR/actors_config.json}"

GUIRL_ROLLOUT_HOST_V2="${GUIRL_ROLLOUT_HOST_V2:-1}"
GUIRL_LOCAL_STATE_ROOT="${GUIRL_LOCAL_STATE_ROOT:-/var/tmp/guirl-$(id -un)/judge_rl}"
ROLLOUT_STATE_DB="${GUIRL_ROLLOUT_STATE_DB:-$GUIRL_LOCAL_STATE_ROOT/rollout-host/state.db}"
ROLLOUT_ARTIFACT_DIR="${GUIRL_ROLLOUT_ARTIFACT_DIR:-$GUIRL_LOCAL_STATE_ROOT/rollout-artifacts}"
ENV_MASTER_STATE_DB="${GUIRL_ENV_MASTER_STATE_DB:-$GUIRL_LOCAL_STATE_ROOT/env-master/state.db}"
RESTART_ENV_MASTER_STATE_DB=""
if [[ "$GUIRL_ROLLOUT_HOST_V2" == "1" ]]; then
    RESTART_ENV_MASTER_STATE_DB="$ENV_MASTER_STATE_DB"
fi
GUIRL_V2_MAX_ATTEMPTS="${GUIRL_V2_MAX_ATTEMPTS:-2}"
GUIRL_V2_MAX_TASK_REPLACEMENTS="${GUIRL_V2_MAX_TASK_REPLACEMENTS:-2}"
GUIRL_V2_ACTOR_CONCURRENCY="${GUIRL_V2_ACTOR_CONCURRENCY:-$(( (NUM_ENV_WORKERS + 3) / 4 ))}"
GUIRL_V2_BATCH_DEADLINE_SECONDS="${GUIRL_V2_BATCH_DEADLINE_SECONDS:-21600}"

ENV_AUTO_RESTART="${ENV_AUTO_RESTART:-1}"
ENV_RESTART_THRESHOLD_SECONDS="${ENV_RESTART_THRESHOLD_SECONDS:-300}"
ENV_RESTART_COOLDOWN_SECONDS="${ENV_RESTART_COOLDOWN_SECONDS:-600}"
ENV_RESTART_COMMAND_TIMEOUT="${ENV_RESTART_COMMAND_TIMEOUT:-900}"
ENV_RESTART_HEALTHCHECK_WAIT="${ENV_RESTART_HEALTHCHECK_WAIT:-300}"
ENV_RESTART_HEALTHY_RATIO="${ENV_RESTART_HEALTHY_RATIO:-0.75}"
RESTART_ENV_FILE="$GUIRL_RUN_DIR/restart_env_stack.env"

cat > "$ACTORS_CONFIG_PATH" <<EOF
{
  "actors": [
    {
      "name": "evocua",
      "harness": "evocua",
      "max_steps": ${ACTOR_MAX_STEPS},
      "harness_kwargs": {
        "model": "${ACTOR_EVOCUA_MODEL}",
        "api_key": "EMPTY",
        "base_url": "${ACTOR_EVOCUA_API_BASE}",
        "max_tokens": ${ACTOR_EVOCUA_MAX_TOKENS},
        "temperature": ${ACTOR_TEMPERATURE}
      }
    },
    {
      "name": "qwen3vl4bthinking",
      "harness": "${ACTOR_QWEN3VL4B_HARNESS}",
      "max_steps": ${ACTOR_MAX_STEPS},
      "harness_kwargs": {
        "model": "${ACTOR_QWEN3VL4B_MODEL}",
        "api_key": "EMPTY",
        "base_url": "${ACTOR_QWEN3VL4B_API_BASE}",
        "max_tokens": ${ACTOR_QWEN3VL4B_MAX_TOKENS},
        "temperature": ${ACTOR_TEMPERATURE},
        "enable_thinking": true,
        "coordinate_type": "${ACTOR_QWEN3VL4B_COORDINATE_TYPE}",
        "history_n": ${ACTOR_QWEN3VL4B_HISTORY_N}
      }
    },
    {
      "name": "holo3",
      "harness": "holo3",
      "max_steps": ${ACTOR_MAX_STEPS},
      "harness_kwargs": {
        "model": "${ACTOR_HOLO3_MODEL}",
        "api_key": "EMPTY",
        "base_url": "${ACTOR_HOLO3_API_BASE}",
        "max_tokens": ${ACTOR_HOLO3_MAX_TOKENS},
        "temperature": ${ACTOR_TEMPERATURE},
        "max_images": ${ACTOR_HOLO3_MAX_IMAGES},
        "max_trajectory_length": ${ACTOR_HOLO3_MAX_TRAJECTORY_LENGTH},
        "enable_thinking": ${ACTOR_HOLO3_ENABLE_THINKING}
      }
    },
    {
      "name": "qwen36",
      "harness": "${ACTOR_QWEN36_HARNESS}",
      "max_steps": ${ACTOR_MAX_STEPS},
      "harness_kwargs": {
        "model": "${ACTOR_QWEN36_MODEL}",
        "api_key": "EMPTY",
        "base_url": "${ACTOR_QWEN36_API_BASE}",
        "max_tokens": ${ACTOR_QWEN36_MAX_TOKENS},
        "temperature": ${ACTOR_TEMPERATURE},
        "coordinate_type": "${ACTOR_QWEN36_COORDINATE_TYPE}",
        "history_n": ${ACTOR_QWEN36_HISTORY_N},
        "image_max": ${ACTOR_QWEN36_IMAGE_MAX},
        "fold_size": ${ACTOR_QWEN36_FOLD_SIZE}
      }
    }
  ]
}
EOF
echo "Actor pool config written to $ACTORS_CONFIG_PATH"

cat > "$RESTART_ENV_FILE" <<EOF
NUM_ENV_WORKERS=${NUM_ENV_WORKERS}
ENV_MASTER_PORT=${ENV_MASTER_PORT}
ENV_WORKER_BASE_PORT=${ENV_WORKER_BASE_PORT}
ENV_MASTER_HEARTBEAT_TIMEOUT_SECONDS=${ENV_MASTER_HEARTBEAT_TIMEOUT_SECONDS}
ENV_WORKER_START_TIMEOUT_SECONDS=${ENV_WORKER_START_TIMEOUT_SECONDS}
ENV_WORKER_START_BATCH_SIZE=${ENV_WORKER_START_BATCH_SIZE}
DOCKER_NAME_PREFIX=${GUIRL_DOCKER_NAME_PREFIX}
OSWORLD_PATH=${OSWORLD_PATH}
OSWORLD_VM_PATH=${OSWORLD_VM_PATH}
AGENT_RUNTIME_VENV="${AGENT_RUNTIME_VENV:?Set AGENT_RUNTIME_VENV}"
PYTHONPATH_EXTRA=${SLIME_DIR}:${PROJECT_ROOT}:${AGENT_RUNTIME_SRC}
LOG_DIR=${LOG_DIR}
PROJECT_ROOT=${PROJECT_ROOT}
ENV_MASTER_STATE_DB=${RESTART_ENV_MASTER_STATE_DB}
EOF
echo "Env restart config written to $RESTART_ENV_FILE"

if [[ -f "${AGENT_RUNTIME_VENV}/bin/activate" ]]; then
    source "${AGENT_RUNTIME_VENV}/bin/activate"
else
    echo "ERROR: Cannot find venv at $AGENT_RUNTIME_VENV"
    exit 1
fi

export PYTHONPATH="${SLIME_DIR}:${PROJECT_ROOT}:${AGENT_RUNTIME_SRC}:${PYTHONPATH:-}"

python3 -c "import flask; import gymnasium; import docker" || {
    echo "ERROR: Missing deps. Run: uv pip install flask httpx docker requests"
    exit 1
}
docker info >/dev/null 2>&1 || {
    echo "ERROR: Docker not accessible. Run: newgrp docker"
    exit 1
}
if [[ "$GUIRL_ROLLOUT_HOST_V2" != "0" && "$GUIRL_ROLLOUT_HOST_V2" != "1" ]]; then
    echo "ERROR: GUIRL_ROLLOUT_HOST_V2 must be 0 or 1" >&2
    exit 1
fi
if [[ "$GUIRL_ROLLOUT_HOST_V2" == "1" ]]; then
    if [[ "$ROLLOUT_STATE_DB" == "$ENV_MASTER_STATE_DB" || "$ROLLOUT_STATE_DB" == "$ROLLOUT_ARTIFACT_DIR" || "$ENV_MASTER_STATE_DB" == "$ROLLOUT_ARTIFACT_DIR" ]]; then
        echo "ERROR: rollout state DB, Env Master state DB, and artifact directory must be distinct local paths." >&2
        exit 1
    fi
    mkdir -p "$(dirname "$ROLLOUT_STATE_DB")" "$ROLLOUT_ARTIFACT_DIR" "$(dirname "$ENV_MASTER_STATE_DB")"
    _v2_fingerprint="$(python3 - "$ACTORS_CONFIG_PATH" "$GUIRL_MANIFEST_PATH" "$NUM_ENV_WORKERS" "$GUIRL_V2_ACTOR_CONCURRENCY" "$GUIRL_V2_MAX_ATTEMPTS" "$GUIRL_V2_MAX_TASK_REPLACEMENTS" <<'PY'
import hashlib, json, pathlib, sys
actors_path, manifest_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
payload = {
    "actors": json.loads(actors_path.read_text()),
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "env_capacity": int(sys.argv[3]),
    "actor_concurrency": sys.argv[4],
    "max_attempts": int(sys.argv[5]),
    "max_task_replacements": int(sys.argv[6]),
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
)"
    _v2_fingerprint_file="$GUIRL_LOCAL_STATE_ROOT/config.sha256"
    if [[ -s "$_v2_fingerprint_file" && "$(<"$_v2_fingerprint_file")" != "$_v2_fingerprint" ]]; then
        echo "ERROR: v2 actor manifest/config fingerprint changed for $GUIRL_LOCAL_STATE_ROOT." >&2
        echo "Use a new GUIRL_LOCAL_STATE_ROOT and matching trainer run_id/state dir; do not reuse durable state across actor-order/config changes." >&2
        exit 1
    fi
    printf '%s\n' "$_v2_fingerprint" > "$_v2_fingerprint_file"
fi

PIDS=()
CLEANED_UP=0

_kill_port_listeners() {
    local port
    if ! command -v fuser >/dev/null 2>&1; then
        return 0
    fi
    for port in "$ENV_MASTER_PORT" "$ROLLOUT_MASTER_PORT"; do
        fuser -k -TERM "${port}/tcp" 2>/dev/null || true
    done
    for i in $(seq 0 $((NUM_ENV_WORKERS - 1))); do
        port=$((ENV_WORKER_BASE_PORT + i))
        fuser -k -TERM "${port}/tcp" 2>/dev/null || true
    done
    sleep 1
    for port in "$ENV_MASTER_PORT" "$ROLLOUT_MASTER_PORT"; do
        fuser -k -KILL "${port}/tcp" 2>/dev/null || true
    done
    for i in $(seq 0 $((NUM_ENV_WORKERS - 1))); do
        port=$((ENV_WORKER_BASE_PORT + i))
        fuser -k -KILL "${port}/tcp" 2>/dev/null || true
    done
}

_stop_guirl_docker_vms() {
    if ! command -v docker >/dev/null 2>&1; then
        return 0
    fi
    local cids
    cids="$(docker ps -aq --filter "name=${GUIRL_DOCKER_NAME_PREFIX}" 2>/dev/null || true)"
    if [[ -n "$cids" ]]; then
        docker stop -t 5 $cids 2>/dev/null || true
        docker rm -f $cids 2>/dev/null || true
    fi
}

cleanup() {
    if (( CLEANED_UP )); then
        return 0
    fi
    CLEANED_UP=1
    trap - INT TERM EXIT

    echo "" >&2
    echo "[$(date '+%H:%M:%S')] Shutting down env services..." >&2

    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done

    local deadline=$((SECONDS + 5))
    while (( SECONDS < deadline )); do
        local alive=0
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=1
                break
            fi
        done
        if (( ! alive )); then
            break
        fi
        sleep 0.5
    done

    for pid in "${PIDS[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true

    _kill_port_listeners || true
    _stop_guirl_docker_vms || true

    echo "[$(date '+%H:%M:%S')] Done." >&2
}

_shutdown_now() {
    cleanup
    exit 130
}

trap _shutdown_now INT TERM
trap cleanup EXIT

echo "[$(date '+%H:%M:%S')] Cleaning up leftover ${GUIRL_DOCKER_NAME_PREFIX}-* Docker containers..."
_stop_guirl_docker_vms || true

echo "============================================================"
echo "GUIRL Env + Rollout Services (HOST, 4-ACTOR)"
echo "============================================================"
echo "  OSWorld:    $OSWORLD_PATH"
echo "  VM image:   $OSWORLD_VM_PATH"
echo "  Master:     :$ENV_MASTER_PORT"
echo "  Workers:    $NUM_ENV_WORKERS  (base port: $ENV_WORKER_BASE_PORT)"
echo "  Worker init batch: $ENV_WORKER_START_BATCH_SIZE"
echo "  Rollout:    :$ROLLOUT_MASTER_PORT"
echo "  Prefill batch: $ROLLOUT_BATCH_SIZE tasks/iteration"
echo "  Actors (x4 fan-out per task):"
echo "    evocua:            $ACTOR_EVOCUA_API_BASE ($ACTOR_EVOCUA_MODEL)"
echo "    qwen3vl4bthinking: $ACTOR_QWEN3VL4B_API_BASE ($ACTOR_QWEN3VL4B_MODEL, harness=$ACTOR_QWEN3VL4B_HARNESS)"
echo "    holo3:             $ACTOR_HOLO3_API_BASE ($ACTOR_HOLO3_MODEL, harness=holo3)"
echo "    qwen36:            $ACTOR_QWEN36_API_BASE ($ACTOR_QWEN36_MODEL, harness=$ACTOR_QWEN36_HARNESS, coord=$ACTOR_QWEN36_COORDINATE_TYPE)"
echo "  Actor max:  $ACTOR_MAX_STEPS steps, timeout ${ACTOR_TIMEOUT_SECONDS}s"
echo "  Manifest:   $GUIRL_MANIFEST_PATH"
echo "  Run dir:    $GUIRL_RUN_DIR"
echo "  Rollout mode: $([[ "$GUIRL_ROLLOUT_HOST_V2" == "1" ]] && echo v2 || echo legacy)"
if [[ "$GUIRL_ROLLOUT_HOST_V2" == "1" ]]; then
    echo "  Local state: $GUIRL_LOCAL_STATE_ROOT"
fi
echo "  Logs:       $LOG_DIR/"
echo "============================================================"

echo "[$(date '+%H:%M:%S')] Starting Env Master on :$ENV_MASTER_PORT → $MASTER_LOG"
ENV_MASTER_EXTRA_ARGS=()
if [[ "$GUIRL_ROLLOUT_HOST_V2" == "1" ]]; then
    ENV_MASTER_EXTRA_ARGS+=(--state-db "$ENV_MASTER_STATE_DB")
fi
python3 -m multinode_service.env_server_master \
    --port "$ENV_MASTER_PORT" \
    --host 0.0.0.0 \
    --heartbeat-timeout-seconds "$ENV_MASTER_HEARTBEAT_TIMEOUT_SECONDS" \
    "${ENV_MASTER_EXTRA_ARGS[@]}" \
    > "$MASTER_LOG" 2>&1 &
PIDS+=($!)

echo -n "  Master health... "
deadline=$((SECONDS + ENV_WORKER_START_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
    if curl -sf "http://127.0.0.1:$ENV_MASTER_PORT/healthz" >/dev/null 2>&1; then
        echo "OK"
        break
    fi
    sleep 2
done
if ! curl -sf "http://127.0.0.1:$ENV_MASTER_PORT/healthz" >/dev/null 2>&1; then
    echo "FAILED — check $MASTER_LOG"
    tail -80 "$MASTER_LOG" || true
    exit 1
fi

worker_idx=0
while (( worker_idx < NUM_ENV_WORKERS )); do
    batch_start=$worker_idx
    batch_end=$((worker_idx + ENV_WORKER_START_BATCH_SIZE - 1))
    if (( batch_end >= NUM_ENV_WORKERS )); then
        batch_end=$((NUM_ENV_WORKERS - 1))
    fi

    batch_indices=()
    for i in $(seq "$batch_start" "$batch_end"); do
        batch_indices+=("$i")
        WORKER_PORT=$((ENV_WORKER_BASE_PORT + i))
        WORKER_LOG="$LOG_DIR/env_worker_${i}.log"
        echo "[$(date '+%H:%M:%S')] Starting Worker $i on :$WORKER_PORT → $WORKER_LOG"
        python3 -m guirl.env_service.rpc_env_server \
            --port "$WORKER_PORT" \
            --master-url "http://127.0.0.1:$ENV_MASTER_PORT" \
            --osworld-path "$OSWORLD_PATH" \
            --path-to-vm "$OSWORLD_VM_PATH" \
            --docker-tag "${GUIRL_DOCKER_NAME_PREFIX}-$i" \
            > "$WORKER_LOG" 2>&1 &
        PIDS+=($!)
    done

    for i in "${batch_indices[@]}"; do
        WORKER_PORT=$((ENV_WORKER_BASE_PORT + i))
        WORKER_LOG="$LOG_DIR/env_worker_${i}.log"
        echo -n "  Worker $i health... "
        deadline=$((SECONDS + ENV_WORKER_START_TIMEOUT_SECONDS))
        while (( SECONDS < deadline )); do
            if curl -sf "http://127.0.0.1:$WORKER_PORT/healthz" >/dev/null 2>&1; then
                echo "OK"
                break
            fi
            sleep 2
        done
        if ! curl -sf "http://127.0.0.1:$WORKER_PORT/healthz" >/dev/null 2>&1; then
            echo "FAILED — check $WORKER_LOG"
            tail -80 "$WORKER_LOG" || true
            exit 1
        fi
    done

    worker_idx=$((batch_end + 1))
done

echo -n "  Env master registered workers... "
python3 - "$ENV_MASTER_PORT" "$NUM_ENV_WORKERS" "$ENV_WORKER_START_TIMEOUT_SECONDS" <<'PY'
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
expected = int(sys.argv[2])
timeout = int(sys.argv[3])
deadline = time.time() + timeout
last = None

while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as resp:
            status = json.loads(resp.read().decode("utf-8"))
        master = status.get("master") or {}
        last = master
        if int(master.get("healthy_workers") or 0) >= expected:
            print(f"OK ({master.get('healthy_workers')}/{expected})")
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(2)

healthy = (last or {}).get("healthy_workers")
count = (last or {}).get("worker_count")
print(f"FAILED (healthy={healthy}, registered={count}, expected={expected})")
raise SystemExit(1)
PY

echo "[$(date '+%H:%M:%S')] Starting Rollout Master on :$ROLLOUT_MASTER_PORT → $ROLLOUT_MASTER_LOG"
ROLLOUT_MASTER_EXTRA_ARGS=()
if [[ "$GUIRL_ROLLOUT_HOST_V2" == "1" ]]; then
    ROLLOUT_MASTER_EXTRA_ARGS+=(
        --rollout-host-v2
        --state-db "$ROLLOUT_STATE_DB"
        --artifact-dir "$ROLLOUT_ARTIFACT_DIR"
        --max-attempts "$GUIRL_V2_MAX_ATTEMPTS"
        --max-task-replacements "$GUIRL_V2_MAX_TASK_REPLACEMENTS"
        --actor-concurrency "$GUIRL_V2_ACTOR_CONCURRENCY"
        --batch-deadline-seconds "$GUIRL_V2_BATCH_DEADLINE_SECONDS"
        --sqlite-synchronous FULL
    )
fi
if [[ "$ENV_AUTO_RESTART" == "1" ]]; then
    ENV_RESTART_MIN_HEALTHY="$(python3 -c "
import math
expected = int('$NUM_ENV_WORKERS')
ratio = float('$ENV_RESTART_HEALTHY_RATIO')
print(max(1, min(expected, math.ceil(ratio * expected))))
")"
    ROLLOUT_MASTER_EXTRA_ARGS+=(
        --env-restart-command "bash $SCRIPT_DIR/restart_env_stack.sh $RESTART_ENV_FILE"
        --env-restart-threshold-seconds "$ENV_RESTART_THRESHOLD_SECONDS"
        --env-restart-cooldown-seconds "$ENV_RESTART_COOLDOWN_SECONDS"
        --env-restart-command-timeout "$ENV_RESTART_COMMAND_TIMEOUT"
        --env-restart-healthcheck-wait "$ENV_RESTART_HEALTHCHECK_WAIT"
        --env-restart-min-healthy-workers "$ENV_RESTART_MIN_HEALTHY"
    )
    echo "  Env auto-restart: ON (threshold ${ENV_RESTART_THRESHOLD_SECONDS}s, cooldown ${ENV_RESTART_COOLDOWN_SECONDS}s, min healthy ${ENV_RESTART_MIN_HEALTHY}/${NUM_ENV_WORKERS})"
else
    echo "  Env auto-restart: OFF"
fi
python3 -m guirl.env_service.rollout_master_server \
    --host 0.0.0.0 \
    --port "$ROLLOUT_MASTER_PORT" \
    --env-master-url "http://127.0.0.1:$ENV_MASTER_PORT" \
    --manifest-path "$GUIRL_MANIFEST_PATH" \
    --examples-dir "$GUIRL_EXAMPLES_DIR" \
    --batch-size "$ROLLOUT_BATCH_SIZE" \
    --env-capacity "$NUM_ENV_WORKERS" \
    --actors-config "$ACTORS_CONFIG_PATH" \
    --actor-max-steps "$ACTOR_MAX_STEPS" \
    --actor-timeout-seconds "$ACTOR_TIMEOUT_SECONDS" \
    "${ROLLOUT_MASTER_EXTRA_ARGS[@]}" \
    > "$ROLLOUT_MASTER_LOG" 2>&1 &
PIDS+=($!)

sleep 3
echo -n "  Rollout master health... "
if curl -sf "http://127.0.0.1:$ROLLOUT_MASTER_PORT/healthz" >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAILED — check $ROLLOUT_MASTER_LOG"
fi

echo ""
echo "============================================================"
echo "All env services started. Monitor logs:"
echo "  tail -f $LOG_DIR/env_master.log $LOG_DIR/env_worker_*.log $LOG_DIR/rollout_master.log"
echo ""
echo "Then start training (in slime container):"
echo "  WORKER_NODE_IP=<node1-ip> SKIP_ENV_START=1 \\"
echo "  ENV_MASTER_HOST=127.0.0.1 ROLLOUT_MASTER_HOST=127.0.0.1 \\"
echo "  NUM_ENV_WORKERS=$NUM_ENV_WORKERS \\"
echo "    bash third_party/slime/guirl/scripts/train_judge.sh"
echo "============================================================"
echo ""

tail -f "$LOG_DIR"/env_worker_*.log "$ROLLOUT_MASTER_LOG" &
TAIL_PID=$!
PIDS+=("$TAIL_PID")

wait "$TAIL_PID" 2>/dev/null || true
