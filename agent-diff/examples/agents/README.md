# Agent-Diff Example Agents

The reusable controllers in this directory include the REx runner and
three Claw-Eval baselines adapted to Agent-Diff's Bash executor:

- `explicit-plan-execute`: create a complete plan, execute one ReAct loop per
  step, and replace the unexecuted suffix after a failed step.
- `recap`: recursively decompose and revisit tasks with the ReCAP controller.
- `reflection`: run ReAct and reflect once after every Bash observation.

The baseline runner uses the same dataset selection, Agent-Diff lifecycle,
checkpoint format, and evaluation summaries as the REx runner.

## Run REx

The controller is in `agents/rex_runner.py`, with `RExRunner`, `RExConfig`, and
`run_rex_runner` exported from `agents`.

The default maximum recursion depth is 4, counting the root frame as depth 1.

REx has no global model-call limit. Model calls are still counted in the trace;
the tool-call budget and per-frame/per-step limits remain configurable.

A frame finishes when its complete plan has executed successfully, without a
separate completion-review call. On a failed step or child frame, it discards
the unexecuted batch suffix and replans once. If that recovery batch fails, the
failure returns to the parent. A fully successful batch resets the recovery
allowance, so a later failed batch can replan locally again. Success of an
individual step does not reset the allowance. Dropped steps still count toward
the frame's cumulative planning budget.

Compression supports only `tree`, which is enabled by default. A returning child
frame is eligible for compression once its history reaches
`--compression-frame-trigger-tokens` (default: 6144); root frames and individual
direct steps are not compressed. Compressor requests include task context and
the full filtered history, without an overall input-length cap. Tool calls and
observations remain in that history. The resulting checkpoint includes a separate
evidence ledger; its outputs and task-context fields use compact representations.
The model sees only Bash, Output, and Exit in each evidence record. Full records,
including call IDs, status, errors, and output hashes, are stored in trace.evidence.
The optional `--compression-mode` argument accepts only `tree`.

```bash
python examples/rex_runner_benchmark.py --model your-model --test-id slack_1
python examples/rex_runner_benchmark_parallel.py --model your-model --workers 8
```

The parallel entry point also reads `REX_WORKERS` (default: 8). New checkpoints
use the agent identifier `rex_runner` and default to
`examples/evaluation_outputs/rex_runner/checkpoints/<timestamp>`.
Existing checkpoints with the old agent identifier are not accepted for resume.

Export checkpoint contexts and tool results with:

```bash
python examples/export_rex_contexts.py --help
```

## Run a baseline

Install the Python SDK plus `datasets` and `httpx`, then configure the model and
Agent-Diff endpoints:

```bash
export OPENAI_MODEL="your-model"
export OPENAI_BASE_URL="http://127.0.0.1:8001/v1"
export OPENAI_API_KEY="..."
export AGENT_DIFF_BASE_URL="http://127.0.0.1:8000"
export AGENT_DIFF_API_KEY="..."
```

Run one task serially:

```bash
python examples/baseline_agent_benchmark.py \
  --agent explicit-plan-execute \
  --test-id slack_1 \
  --workers 1
```

Run ReCAP or reflection in parallel:

```bash
python examples/baseline_agent_benchmark.py --agent recap --workers 8
python examples/baseline_agent_benchmark.py --agent reflection --workers 8
```

Run all 224 tasks once with each baseline using the configured Gemma endpoint:

```bash
examples/run_all_baselines_gemma_4_e4b_it.sh
```

Set `WORKERS`, `RUN_ROOT`, `PYTHON_BIN`, or `DRY_RUN=1` to override the batch
defaults or inspect the generated commands without starting tasks.

Each task is stored in its own JSON checkpoint. Resume a run by passing the same
directory and configuration:

```bash
python examples/baseline_agent_benchmark.py \
  --agent recap \
  --workers 8 \
  --checkpoint-dir examples/evaluation_outputs/baselines/recap/checkpoints/<run>
```

All baseline modes default to 100 model calls and 40 Bash calls per task. Structural
limits can be changed with the `--explicit-*` and `--recap-*` options shown by
`--help`. A checkpoint rejects resume attempts made with another agent, model,
dataset, endpoint, or controller configuration.
