#!/usr/bin/env bash

set -euo pipefail

ENV_FILE="${1:?usage: restart_env_stack.sh <config.env>}"
source "$ENV_FILE"

: "${NUM_ENV_WORKERS:?}" "${ENV_MASTER_PORT:?}" "${ENV_WORKER_BASE_PORT:?}"
: "${DOCKER_NAME_PREFIX:?}" "${OSWORLD_PATH:?}" "${OSWORLD_VM_PATH:?}"
: "${AGENT_RUNTIME_VENV:?}" "${LOG_DIR:?}" "${PROJECT_ROOT:?}"
ENV_MASTER_HEARTBEAT_TIMEOUT_SECONDS="${ENV_MASTER_HEARTBEAT_TIMEOUT_SECONDS:-90}"
ENV_WORKER_START_TIMEOUT_SECONDS="${ENV_WORKER_START_TIMEOUT_SECONDS:-180}"
ENV_WORKER_START_BATCH_SIZE="${ENV_WORKER_START_BATCH_SIZE:-4}"
ENV_MASTER_STATE_ARGS=()
if [[ -n "${ENV_MASTER_STATE_DB:-}" ]]; then
    ENV_MASTER_STATE_ARGS+=(--state-db "$ENV_MASTER_STATE_DB")
fi

log() { echo "[restart_env_stack $(date '+%H:%M:%S')] $*"; }

cd "$PROJECT_ROOT"

source "${AGENT_RUNTIME_VENV}/bin/activate"
export PYTHONPATH="${PYTHONPATH_EXTRA:-}:${PYTHONPATH:-}"
export OSWORLD_PATH OSWORLD_VM_PATH
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,::1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,::1"

log "Killing env master (:${ENV_MASTER_PORT}) and ${NUM_ENV_WORKERS} workers..."
if command -v fuser >/dev/null 2>&1; then
    fuser -k -TERM "${ENV_MASTER_PORT}/tcp" 2>/dev/null || true
    for i in $(seq 0 $((NUM_ENV_WORKERS - 1))); do
        fuser -k -TERM "$((ENV_WORKER_BASE_PORT + i))/tcp" 2>/dev/null || true
    done
    sleep 3
    fuser -k -KILL "${ENV_MASTER_PORT}/tcp" 2>/dev/null || true
    for i in $(seq 0 $((NUM_ENV_WORKERS - 1))); do
        fuser -k -KILL "$((ENV_WORKER_BASE_PORT + i))/tcp" 2>/dev/null || true
    done
fi

cids="$(docker ps -aq --filter "name=${DOCKER_NAME_PREFIX}" 2>/dev/null || true)"
if [[ -n "$cids" ]]; then
    log "Removing $(echo "$cids" | wc -l) leftover ${DOCKER_NAME_PREFIX}-* docker container(s)..."
    docker stop -t 5 $cids 2>/dev/null || true
    docker rm -f $cids 2>/dev/null || true
fi

MASTER_LOG="$LOG_DIR/env_master.log"
log "Starting env master on :${ENV_MASTER_PORT} → ${MASTER_LOG}"
setsid python3 -m multinode_service.env_server_master \
    --port "$ENV_MASTER_PORT" \
    --host 0.0.0.0 \
    --heartbeat-timeout-seconds "$ENV_MASTER_HEARTBEAT_TIMEOUT_SECONDS" \
    "${ENV_MASTER_STATE_ARGS[@]}" \
    >> "$MASTER_LOG" 2>&1 < /dev/null &

deadline=$((SECONDS + ENV_WORKER_START_TIMEOUT_SECONDS))
until curl -sf "http://127.0.0.1:$ENV_MASTER_PORT/healthz" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
        log "ERROR: env master failed to come up — see ${MASTER_LOG}"
        exit 1
    fi
    sleep 2
done
log "Env master healthy."

worker_idx=0
while (( worker_idx < NUM_ENV_WORKERS )); do
    batch_end=$((worker_idx + ENV_WORKER_START_BATCH_SIZE - 1))
    (( batch_end >= NUM_ENV_WORKERS )) && batch_end=$((NUM_ENV_WORKERS - 1))

    for i in $(seq "$worker_idx" "$batch_end"); do
        WORKER_PORT=$((ENV_WORKER_BASE_PORT + i))
        WORKER_LOG="$LOG_DIR/env_worker_${i}.log"
        log "Starting worker $i on :${WORKER_PORT}"
        setsid python3 -m guirl.env_service.rpc_env_server \
            --port "$WORKER_PORT" \
            --master-url "http://127.0.0.1:$ENV_MASTER_PORT" \
            --osworld-path "$OSWORLD_PATH" \
            --path-to-vm "$OSWORLD_VM_PATH" \
            --docker-tag "${DOCKER_NAME_PREFIX}-$i" \
            >> "$WORKER_LOG" 2>&1 < /dev/null &
    done

    for i in $(seq "$worker_idx" "$batch_end"); do
        WORKER_PORT=$((ENV_WORKER_BASE_PORT + i))
        deadline=$((SECONDS + ENV_WORKER_START_TIMEOUT_SECONDS))
        until curl -sf "http://127.0.0.1:$WORKER_PORT/healthz" >/dev/null 2>&1; do
            if (( SECONDS >= deadline )); then
                log "ERROR: worker $i failed to come up — see $LOG_DIR/env_worker_${i}.log"
                exit 1
            fi
            sleep 2
        done
        log "Worker $i healthy."
    done

    worker_idx=$((batch_end + 1))
done

log "Waiting for env master to register ${NUM_ENV_WORKERS} healthy workers..."
python3 - "$ENV_MASTER_PORT" "$NUM_ENV_WORKERS" "$ENV_WORKER_START_TIMEOUT_SECONDS" <<'PY'
import json, sys, time, urllib.request

port, expected, timeout = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + timeout
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as resp:
            status = json.loads(resp.read().decode("utf-8"))
        last = status.get("master") or {}
        if int(last.get("healthy_workers") or 0) >= expected:
            print(f"registered OK ({last.get('healthy_workers')}/{expected})")
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass
    time.sleep(2)
print(f"ERROR: only {(last or {}).get('healthy_workers')}/{expected} healthy workers", file=sys.stderr)
raise SystemExit(1)
PY

log "Env stack restart complete."
