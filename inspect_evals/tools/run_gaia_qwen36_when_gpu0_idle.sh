#!/usr/bin/env bash
set -euo pipefail

readonly GPU_ID=0
readonly GPU_POLL_INTERVAL_SECONDS=1
readonly GPU_REQUIRED_IDLE_SECONDS=300
readonly MODEL_POLL_INTERVAL_SECONDS=10
readonly MODEL_URL=http://127.0.0.1:30000/v1/models
readonly SOURCE_CONTAINER=sglang-gemma4-12b-long
readonly TARGET_CONTAINER=sglang-qwen36-35b-a3b
readonly SERVED_MODEL_NAME=Qwen3.6-35B-A3B

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT

trap 'exit 130' INT
trap 'exit 143' TERM

for command_name in curl docker nvidia-smi python3 uv; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

wait_for_gpu_idle() {
  local gpu_utilization
  local idle_since=-1
  local last_reported_idle_seconds=-1

  echo "Waiting for GPU ${GPU_ID} to remain at 0% utilization for ${GPU_REQUIRED_IDLE_SECONDS}s..."

  while true; do
    if ! gpu_utilization="$(
      nvidia-smi \
        --id="$GPU_ID" \
        --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits
    )"; then
      echo "Failed to query GPU ${GPU_ID}; resetting idle timer." >&2
      idle_since=-1
      last_reported_idle_seconds=-1
      sleep "$GPU_POLL_INTERVAL_SECONDS"
      continue
    fi

    gpu_utilization="${gpu_utilization//[[:space:]]/}"
    case "$gpu_utilization" in
      ''|*[!0-9]*)
        echo "Unexpected GPU utilization: '${gpu_utilization}'; resetting idle timer." >&2
        idle_since=-1
        last_reported_idle_seconds=-1
        ;;
      0)
        if ((idle_since < 0)); then
          idle_since=$SECONDS
          echo "GPU ${GPU_ID} is idle (0/${GPU_REQUIRED_IDLE_SECONDS}s)."
        fi

        local idle_seconds=$((SECONDS - idle_since))
        if ((idle_seconds >= GPU_REQUIRED_IDLE_SECONDS)); then
          echo "GPU ${GPU_ID} remained idle for ${GPU_REQUIRED_IDLE_SECONDS}s."
          return
        fi
        if ((idle_seconds > 0 && idle_seconds / 30 > last_reported_idle_seconds / 30)); then
          echo "GPU ${GPU_ID} is idle (${idle_seconds}/${GPU_REQUIRED_IDLE_SECONDS}s)."
          last_reported_idle_seconds=$idle_seconds
        fi
        ;;
      *)
        if ((idle_since >= 0)); then
          echo "GPU ${GPU_ID} utilization rose to ${gpu_utilization}%; resetting idle timer."
        fi
        idle_since=-1
        last_reported_idle_seconds=-1
        ;;
    esac

    sleep "$GPU_POLL_INTERVAL_SECONDS"
  done
}

container_exists() {
  docker container inspect "$1" >/dev/null 2>&1
}

container_is_running() {
  [[ "$(docker container inspect --format '{{.State.Running}}' "$1" 2>/dev/null)" == "true" ]]
}

stop_source_container() {
  if container_exists "$SOURCE_CONTAINER"; then
    echo "Stopping ${SOURCE_CONTAINER} to release GPU ${GPU_ID}..."
    docker stop --time 120 "$SOURCE_CONTAINER"
  else
    echo "Container ${SOURCE_CONTAINER} does not exist; continuing."
  fi
}

remove_existing_target_container() {
  if ! container_exists "$TARGET_CONTAINER"; then
    return
  fi

  if container_is_running "$TARGET_CONTAINER"; then
    echo "Stopping existing ${TARGET_CONTAINER}..."
    docker stop "$TARGET_CONTAINER"
  fi

  echo "Removing existing ${TARGET_CONTAINER} so it can be recreated with the requested configuration..."
  docker rm "$TARGET_CONTAINER"
}

start_model_server() {
  echo "Starting ${TARGET_CONTAINER}..."
  docker run -d \
    --name "$TARGET_CONTAINER" \
    --restart unless-stopped \
    --init \
    --gpus '"device=0"' \
    --ipc=host \
    --shm-size 32g \
    -p 127.0.0.1:30000:30000 \
    -v /data1/models:/data1/models:ro \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
      --model-path /data1/models/Qwen3.6-35B-A3B \
      --served-model-name "$SERVED_MODEL_NAME" \
      --host 0.0.0.0 \
      --port 30000 \
      --tp-size 1 \
      --dtype bfloat16 \
      --context-length 262144 \
      --mem-fraction-static 0.65 \
      --chunked-prefill-size 8192 \
      --max-prefill-tokens 262144 \
      --max-running-requests 8 \
      --max-queued-requests 256 \
      --schedule-conservativeness 0.7 \
      --schedule-policy lpm \
      --reasoning-parser qwen3 \
      --tool-call-parser qwen3_coder \
      --speculative-algo NEXTN \
      --speculative-num-steps 3 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 4 \
      --enable-hierarchical-cache \
      --hicache-size 128 \
      --hicache-io-backend kernel \
      --hicache-mem-layout page_first_direct \
      --hicache-write-policy write_through_selective \
      --page-size 64 \
      --enable-cache-report \
      --enable-metrics
}

model_response_is_valid() {
  python3 -c '
import json
import sys

expected_model = sys.argv[1]
try:
    payload = json.load(sys.stdin)
except (json.JSONDecodeError, OSError):
    raise SystemExit(1)

models = payload.get("data")
if not isinstance(models, list):
    raise SystemExit(1)

raise SystemExit(
    0
    if any(isinstance(model, dict) and model.get("id") == expected_model for model in models)
    else 1
)
' "$SERVED_MODEL_NAME"
}

wait_for_model_server() {
  local response

  echo "Waiting for ${SERVED_MODEL_NAME} on ${MODEL_URL}..."

  while true; do
    if ! container_is_running "$TARGET_CONTAINER"; then
      echo "Container ${TARGET_CONTAINER} exited before the model server became ready." >&2
      docker logs --tail 200 "$TARGET_CONTAINER" >&2 || true
      return 1
    fi

    if response="$(curl --fail --silent --show-error --max-time 10 "$MODEL_URL" 2>/dev/null)" && \
      model_response_is_valid <<<"$response"; then
      echo "${SERVED_MODEL_NAME} is ready and listening on port 30000."
      return
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') Model server is not ready; retrying in ${MODEL_POLL_INTERVAL_SECONDS}s."
    sleep "$MODEL_POLL_INTERVAL_SECONDS"
  done
}

run_benchmark() {
  cd "$REPO_ROOT"

  echo "Running the GAIA high-confidence benchmark with ${SERVED_MODEL_NAME}..."
  uv run inspect eval inspect_evals/gaia \
    --model openai/Qwen3.6-35B-A3B \
    --model-base-url http://localhost:30000/v1 \
    --solver inspect_evals/gaia_high_confidence_recursive_agent \
    -S planning_tools=false \
    --max-samples 8 \
    --max-connections 8 \
    --max-sandboxes 8 \
    --continue-on-fail \
    --score-on-error \
    --message-limit 250 \
    --sandbox docker:src/inspect_evals/gaia/compose.proxy.yaml \
    --log-dir logs/gaia-high-confidence-full-qwen36-26b
}

cd "$REPO_ROOT"
# wait_for_gpu_idle
# stop_source_container
# remove_existing_target_container
# start_model_server
# wait_for_model_server
run_benchmark

echo "GAIA benchmark completed successfully."
