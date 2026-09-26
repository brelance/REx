#!/usr/bin/env bash
set -euo pipefail

readonly POLL_INTERVAL_SECONDS=10
readonly EVAL_PROCESS_PATTERN='[i]nspect[[:space:]]+eval[[:space:]]+inspect_evals/gaia([[:space:]]|$)'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT

trap 'exit 130' INT
trap 'exit 143' TERM

for command_name in pgrep uv; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

wait_for_gaia_evals() {
  local -a processes
  local first_poll=true

  while true; do
    mapfile -t processes < <(pgrep -af -- "$EVAL_PROCESS_PATTERN" || true)
    if ((${#processes[@]} == 0)); then
      if [[ "$first_poll" == true ]]; then
        echo "No running GAIA eval process found; starting the queued evaluation now."
      else
        echo "$(date '+%Y-%m-%d %H:%M:%S') All monitored GAIA eval processes have finished."
      fi
      return
    fi

    if [[ "$first_poll" == true ]]; then
      echo "Waiting for ${#processes[@]} running GAIA eval process(es) to finish:"
      printf '  %s\n' "${processes[@]}"
      first_poll=false
    else
      echo "$(date '+%Y-%m-%d %H:%M:%S') ${#processes[@]} GAIA eval process(es) still running."
    fi

    sleep "$POLL_INTERVAL_SECONDS"
  done
}

cd "$REPO_ROOT"
wait_for_gaia_evals

echo "Starting scoped GAIA high-confidence evaluation with Qwen3.5-4B..."
exec uv run inspect eval inspect_evals/gaia \
  --model openai/Qwen3.5-4B \
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
  --log-dir logs/gaia-high-confidence-qwen35-4b
