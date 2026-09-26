#!/usr/bin/env bash
set -euo pipefail

readonly GPU_ID=1
readonly GPU_POLL_INTERVAL_SECONDS=1
readonly GPU_REQUIRED_IDLE_SECONDS=120
readonly MODEL_NAME=Qwen3.5-4B
readonly MODEL_BASE_URL=http://localhost:30000/v1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT

trap 'exit 130' INT
trap 'exit 143' TERM

for command_name in nvidia-smi uv; do
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
    if ! gpu_utilization="$(nvidia-smi \
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

cd "$REPO_ROOT"
# wait_for_gpu_idle

echo "Starting GAIA high-confidence evaluation with ${MODEL_NAME}..."
exec uv run inspect eval inspect_evals/gaia \
  --model "openai/${MODEL_NAME}" \
  --model-base-url "$MODEL_BASE_URL" \
  --solver inspect_evals/gaia_high_confidence_recursive_agent \
  --max-samples 4 \
  --max-connections 4 \
  --max-sandboxes 4 \
  --continue-on-fail \
  --message-limit 250 \
  --score-on-error \
  --sandbox docker:src/inspect_evals/gaia/compose.proxy.yaml \
  --log-dir logs/gaia-high-confidence-qwen35-4b
