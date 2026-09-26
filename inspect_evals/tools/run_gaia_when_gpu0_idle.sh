#!/usr/bin/env bash
set -euo pipefail

readonly GPU_ID=0
readonly CHECK_INTERVAL_SECONDS=30
readonly REQUIRED_IDLE_SECONDS=30

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for command_name in nvidia-smi uv; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

idle_since=-1

echo "Waiting for GPU $GPU_ID to remain at 0% utilization for ${REQUIRED_IDLE_SECONDS}s..."

while true; do
  if ! gpu_utilization="$(
    nvidia-smi \
      --id="$GPU_ID" \
      --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits
  )"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Failed to query GPU $GPU_ID; retrying in ${CHECK_INTERVAL_SECONDS}s." >&2
    idle_since=-1
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  gpu_utilization="${gpu_utilization//[[:space:]]/}"
  case "$gpu_utilization" in
    ''|*[!0-9]*)
      echo "$(date '+%Y-%m-%d %H:%M:%S') Unexpected GPU utilization: '$gpu_utilization'; retrying in ${CHECK_INTERVAL_SECONDS}s." >&2
      idle_since=-1
      ;;
    0)
      if [ "$idle_since" -lt 0 ]; then
        idle_since=$SECONDS
      fi

      idle_seconds=$((SECONDS - idle_since))
      echo "$(date '+%Y-%m-%d %H:%M:%S') GPU $GPU_ID is idle (${idle_seconds}/${REQUIRED_IDLE_SECONDS}s)."

      if [ "$idle_seconds" -ge "$REQUIRED_IDLE_SECONDS" ]; then
        break
      fi
      ;;
    *)
      echo "$(date '+%Y-%m-%d %H:%M:%S') GPU $GPU_ID utilization is ${gpu_utilization}%; resetting idle timer."
      idle_since=-1
      ;;
  esac

  sleep "$CHECK_INTERVAL_SECONDS"
done

echo "GPU $GPU_ID has been idle for ${REQUIRED_IDLE_SECONDS}s; starting GAIA evaluation."
cd "$REPO_ROOT"

exec uv run inspect eval inspect_evals/gaia \
  --model openai/gemma-4-26B-A4B-it \
  --model-base-url http://localhost:30000/v1 \
  --max-samples 8 \
  --max-connections 8 \
  --max-sandboxes 8 \
  --continue-on-fail \
  --message-limit 250 \
  --score-on-error \
  --sandbox docker:src/inspect_evals/gaia/compose.proxy.yaml \
  --log-dir logs/gaia-react-gemma-4-full
