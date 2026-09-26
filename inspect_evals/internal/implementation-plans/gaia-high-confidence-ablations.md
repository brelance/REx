# GAIA High-Confidence Agent: Eight Ablation Implementations

## Status

- Draft

## Owner

- GAIA evaluation maintainers

## Linked decisions

- `src/inspect_evals/gaia/high_confidence_agent.py` is the reference implementation.
- Ablations must preserve the GAIA task, dataset, tools, scorer, model, and outer Inspect limits.

## Background and constraints

The goal is to reproduce the seven ReCAP-style ablations on GAIA so that the
comparison tests whether the method generalizes across benchmarks. The
reference agent currently combines rolling planning, optional planning tools,
recursive decomposition, frame/node compression, and a reduce/review phase.

The implementation must change only the mechanism under test. Do not copy
benchmark-specific values from another runner: task IDs, prompt wrappers,
token thresholds, model names, concurrency, or result-processing conventions
are not part of an ablation. All seven variants use the same GAIA task and
scorer, the same tool set, the same model, and the same externally configured
Inspect message limit.

## Scope

### In scope

- Add a small, explicit ablation configuration surface to
  `high_confidence_agent.py`.
- Implement the seven variants below with the smallest behavioral change from
  the reference agent.
- Record the selected variant and effective settings in Inspect transcript
  metadata.
- Provide a single repeatable command/config convention for three runs per
  variant.

### Out of scope

- Changes to `gaia.py`, the GAIA dataset, `scorer.py`, sandbox tools, or answer
  normalization.
- New benchmark-specific prompts or hard-coded task lists.
- Changes to model sampling, temperature, concurrency, or retry policy.
- Copying token budgets or frame limits from another benchmark. Use the
  existing GAIA defaults unless a controlled experiment explicitly changes a
  shared limit for every variant.

## Common experimental contract

All runs invoke `inspect_evals/gaia` with the same dataset subset, model,
provider, task-level `message_limit`, tools, and scorer. Only the agent factory
arguments listed below vary. Each variant is run three times with independent
run directories and identical random/provider settings. Report per-run and
per-run accuracy/pass rate and average score; do not pool or
silently discard failed samples.

The reference configuration is the current factory behavior:

```text
planning_schedule=progressive
planning_tools=true
max_depth=3
compression_mode=none (or the selected reference compression setting)
reduce_enabled=true
execution_modes=execute|decompose
```

`max_steps_per_frame`, `max_turns_per_step`, `max_tool_calls`, and observation limits remain fixed across variants. There is no global frame-count limit.

## Ablation definitions

The names correspond to the seven existing ablation groups. Unless stated,
every other reference behavior remains enabled.

| Variant | Minimal change | Intended isolation |
| --- | --- | --- |
| `one_shot` | Set `planning_schedule="one_shot"`; use one planning prompt that requires the complete remaining plan and `planning_complete=true`; do not issue a successful-batch planning call afterward | Rolling/replanning benefit |
| `no_compression` | Set `compression_mode="none"`; bypass node/tree compression calls and retain the original messages | Compression benefit |
| `no_audit` | Set `reduce_enabled=false`; skip `_reduce_frame()` and any repair-step loop | Review/reduction benefit |
| `single_pass_audit` | Keep one reduce/review call but disallow returned `additional_steps`; an incomplete review fails the frame rather than starting repair steps | Iterative repair benefit |
| `execute_only` | Keep planning, but force every parsed step to `execution_mode="execute"`; reject or coerce `decompose` at the runner boundary | Recursive execution benefit |
| `recursive_core_only` | Keep recursive planning/execution and tools, but disable audit/reduction and compression orchestration; use the minimal core plan/execute loop | Added systems benefit |
| `single_item` | Restrict each planning patch to one step (`plan_batch_limit=1`), while retaining rolling replanning after each completed step | Batch planning benefit |

The unmodified reference agent is the shared control and is not counted as one
of the seven ablations. `one_shot` must still permit `decompose` steps and child frames; it removes only successful
horizontal replanning. `single_pass_audit` differs from `no_audit` because its
single review still validates the completed frame. `recursive_core_only` is
defined as the smallest common core and must not inherit unrelated settings
from another benchmark.

## Required code changes

### Configuration

Add typed options near `AgentConfig`:

```python
PlanningSchedule = Literal["progressive", "one_shot"]
Ablation = Literal[
    "one_shot", "no_compression", "no_audit",
    "single_pass_audit", "execute_only",
    "recursive_core_only", "single_item",
]
```

Expose `planning_schedule`, `plan_batch_limit`, `reduce_enabled`, and an
optional `ablation` label in `gaia_high_confidence_recursive_agent()`. Keep
backward-compatible defaults. Validate positive limits and reject incompatible
combinations rather than silently changing them. Resolve an ablation into an
immutable effective config once, then pass that config to
`HighConfidenceRecursiveRunner`.

### Planning prompt and loop

Factor the current planning text into `_progressive_plan_prompt()` and add
`_one_shot_plan_prompt()`. The one-shot prompt must say:

```text
Return the complete remaining executable plan for this frame now.
Set planning_complete=true. No successful-batch planning call will follow.
```

In `_run_frame_body()`, progressive mode keeps the existing `while` loop.
One-shot mode accepts one complete patch, executes it, and may still enter the
existing recovery path after a failed step. `parse_plan()` and its caller must
enforce `planning_complete=true` for one-shot mode and the remaining step
budget. Do not remove recursion or tools as a side effect.

### Reduction and compression switches

Guard the existing `_reduce_frame()` call with `reduce_enabled`. For
`single_pass_audit`, call it once and reject/ignore repair steps with an
explicit failed status; for `no_audit` and `recursive_core_only`, do not call
it. Route compression through the existing `compression_mode` switch. The
`no_compression` and core variants must not alter tool-result content or task
prompts merely to compensate for compression being disabled.

### Execution-mode switches

Apply `execute_only` immediately after plan parsing, in one place, so the
prompt schema and runtime agree. A validation error must use the existing
recovery/error protocol and be visible in the transcript.

### Metadata and observability

Add `ablation`, `planning_schedule`, `plan_batch_limit`, `reduce_enabled`,
`compression_mode`, and effective recursion mode to the existing agent event
artifacts. Do not add benchmark-specific output files. The run directory and
Inspect log remain the source of truth for per-sample outcomes.

## Milestones

### Milestone 1: Shared switches and control

#### Goal

- Make the reference behavior selectable without changing its default output.

#### Tasks

- [ ] Add typed config fields, validation, and effective-config resolution.
- [ ] Refactor progressive planning prompt without changing its rendered text.
- [ ] Add transcript metadata and unit tests for config validation.

#### Dependencies

- Existing `high_confidence_agent.py` tests and Inspect agent factory.

#### Exit criteria

- Existing tests pass and default factory behavior is unchanged.
- A smoke run records the effective config in the transcript.

### Milestone 2: Implement seven variants

#### Goal

- Run every named ablation through the same agent factory and GAIA task.

#### Tasks

- [ ] Implement one-shot planning and its strict parser/loop constraint.
- [ ] Implement compression, audit, batch-size, execution-mode, and child-only switches.
- [ ] Add focused tests for each switch and their interaction with failures.

#### Dependencies

- Milestone 1.

#### Exit criteria

- Each variant can be constructed by configuration only.
- One-shot never performs a successful follow-up planning call.
- No-audit/core skip reduction; single-pass performs at most one review.

### Milestone 3: Benchmark evaluation

#### Goal

- Produce comparable runs for all seven variants.

#### Tasks

- [ ] Run a small GAIA smoke subset for all variants.
- [ ] Run one full evaluation per variant with fixed shared settings.
- [ ] Extract per-run accuracy/pass rate, average score, errors, and budget usage.

#### Dependencies

- Milestone 2; GAIA dataset access; model provider; Docker sandbox.

#### Exit criteria

- 7 labeled run directories exist (7 variants x 1 run by default).
- All runs use the same dataset/scorer/model settings and preserve errors.

## Risks and mitigations

- Risk: one-shot planning can emit an incomplete plan.
  - Mitigation: enforce `planning_complete=true`, budget bounds, and explicit recovery behavior.
- Risk: child-only changes the task difficulty by forbidding useful direct root work.
  - Mitigation: require only the root decomposition constraint; preserve child-frame behavior and all shared budgets.
- Risk: disabling compression changes context length and causes unrelated failures.
  - Mitigation: keep shared Inspect limits fixed and report budget/context failures separately.
- Risk: coercing execution modes hides model behavior.
  - Mitigation: record original and effective modes; prefer strict validation in diagnostic runs.

## Rollout and validation

Run unit tests for parsing, config resolution, planning-call counts, reduction
calls, and execution-mode constraints. Run a deterministic smoke subset before
full evaluation. Inspect transcripts to verify that only the intended switch
changed. Record single-run results clearly; additional replicates can be enabled
with `GAIA_REPLICATES` when variance estimates are required.

## Issue tracker links

- Parent issue/epic: TBD
- Child issues: one implementation issue and one evaluation issue per milestone

## Success criteria

- The seven variants are implemented with configuration-level selection and no
  benchmark-specific constants copied from the prior implementation.
- The default reference agent remains behaviorally compatible.
- Three comparable runs per variant produce auditable pass rate and average
  score results.

## Changelog

- 2026-09-06: Initial draft.
