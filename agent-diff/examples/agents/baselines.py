"""Claw-Eval baseline controllers adapted for Agent-Diff.

The source runners use model-native tool calls and Claw-Eval's message/trace
types. These adaptations preserve the controller protocols while routing Bash
commands through ``BashExecutorProxy``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Literal

from .rex_runner import (
    BudgetExceeded,
    CodeExecutor,
    ModelClient,
    ModelError,
    ModelTurn,
    ProtocolError,
    extract_json_dict,
    normalize_usage,
    parse_action,
    utc_now,
)

BaselineMode = Literal["explicit-plan-execute", "recap", "reflection"]
StepStatus = Literal["success", "failed"]


@dataclass(frozen=True)
class BaselineBudgetConfig:
    max_model_calls: int = 100
    max_tool_calls: int = 40

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1")


@dataclass(frozen=True)
class ExplicitPlanExecuteConfig(BaselineBudgetConfig):
    max_steps: int = 16
    max_step_turns: int = 20


@dataclass(frozen=True)
class RecapConfig(BaselineBudgetConfig):
    max_depth: int = 3
    max_subtasks: int = 16
    max_obs_chars: int = 6000
    max_tree_chars: int = 20_000


@dataclass(frozen=True)
class ReflectionConfig(BaselineBudgetConfig):
    pass


BaselineConfig = ExplicitPlanExecuteConfig | RecapConfig | ReflectionConfig


@dataclass
class AgentBudget:
    model_calls: int = 0
    tool_calls: int = 0


@dataclass(frozen=True)
class RunResult:
    status: StepStatus
    summary: str


def _protocol_text_candidates(turn: ModelTurn) -> tuple[str, ...]:
    """Return visible and provider reasoning text for internal JSON protocols."""
    candidates: list[str] = []
    if turn.content.strip():
        candidates.append(turn.content)
    raw = turn.raw_response
    if isinstance(raw, dict):
        try:
            reasoning = raw["choices"][0]["message"].get("reasoning_content")
        except (KeyError, IndexError, TypeError):
            reasoning = None
        if isinstance(reasoning, str) and reasoning.strip() and reasoning not in candidates:
            candidates.append(reasoning)
    return tuple(candidates)


class _BaselineAgent:
    agent_name = "baseline"

    def __init__(
        self,
        *,
        model_client: ModelClient,
        executor: CodeExecutor,
        config: BaselineBudgetConfig,
        on_trace_update: Any | None = None,
    ) -> None:
        self.model_client = model_client
        self.executor = executor
        self.config = config
        self.on_trace_update = on_trace_update
        self.messages: list[dict[str, str]] = []
        self.budget = AgentBudget()
        self.trace: dict[str, Any] = {}

    def run(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        self.budget = AgentBudget()
        self.trace = {
            "schema_version": 1,
            "agent": self.agent_name,
            "completed": False,
            "stage": "agent_started",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "config": asdict(self.config),
            "usage": normalize_usage(None),
            "budget": asdict(self.budget),
            "messages": self.messages,
            "events": [],
            "root_result": None,
            "final_answer": None,
        }
        self._emit("agent_started")
        try:
            result, final_answer = self._run(prompt)
        except (BudgetExceeded, ModelError) as exc:
            result = RunResult("failed", str(exc))
            final_answer = None
            self.trace["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            self._emit(
                "budget_exhausted" if isinstance(exc, BudgetExceeded) else "model_error"
            )
        except Exception as exc:  # noqa: BLE001 - preserve cleanup in benchmark runner
            result = RunResult("failed", str(exc))
            final_answer = None
            self.trace["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            self._emit("agent_error")

        self.trace["root_result"] = asdict(result)
        self.trace["completed"] = result.status == "success"
        self.trace["final_answer"] = final_answer
        self.trace["finished_at"] = utc_now()
        self._emit("agent_completed", artifact=asdict(result))
        return self.trace

    def _run(self, prompt: str) -> tuple[RunResult, str | None]:
        raise NotImplementedError

    def _append_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def _call_model(self, phase: str, **context: Any) -> ModelTurn:
        if self.budget.model_calls >= self.config.max_model_calls:
            raise BudgetExceeded("Agent exceeded the global model-call budget.")
        turn = self.model_client.generate(tuple(self.messages))
        self.budget.model_calls += 1
        self._append_message("assistant", turn.content)
        self._add_usage(turn.usage)
        self._sync_budget()
        self._emit(
            "model_response",
            artifact={
                "phase": phase,
                "content": turn.content,
                "usage": normalize_usage(turn.usage),
                "attempts": list(turn.attempts),
                "raw_response": turn.raw_response,
                **context,
            },
        )
        return turn

    def _execute_action(self, action: str, **context: Any) -> str:
        if self.budget.tool_calls >= self.config.max_tool_calls:
            raise BudgetExceeded("Agent exceeded the global tool-call budget.")
        self.budget.tool_calls += 1
        self._sync_budget()
        try:
            result = self.executor.execute(action)
            observation = self._format_observation(result)
        except Exception as exc:  # noqa: BLE001 - executor implementations vary
            result = {"status": "error", "error": str(exc)}
            observation = f"[executor error]\n{type(exc).__name__}: {exc}"
        self._append_message("user", f"<observation>\n{observation}\n</observation>")
        self._emit(
            "tool_result",
            artifact={
                "action": action,
                "result": result,
                "observation": observation,
                **context,
            },
        )
        return observation

    def _add_usage(self, usage: dict[str, Any]) -> None:
        normalized = normalize_usage(usage)
        for key, value in normalized.items():
            self.trace["usage"][key] += value

    def _sync_budget(self) -> None:
        if self.trace:
            self.trace["budget"] = asdict(self.budget)

    def _emit(self, event: str, *, artifact: dict[str, Any] | None = None) -> None:
        if not self.trace:
            return
        entry = {
            "event": event,
            "timestamp": utc_now(),
            "artifact": artifact or {},
        }
        self.trace["events"].append(entry)
        self.trace["stage"] = event
        self.trace["updated_at"] = entry["timestamp"]
        self._sync_budget()
        if self.on_trace_update:
            self.on_trace_update(self.trace, event)

    @staticmethod
    def _format_observation(result: dict[str, Any]) -> str:
        stdout = str(result.get("stdout", "") or "")
        stderr = str(result.get("stderr", "") or "")
        status = result.get("status", "success")
        exit_code = result.get("exit_code", 0)
        error = result.get("error")
        if status == "success" and exit_code == 0:
            return stdout.strip() or "(empty output)"
        parts = [stdout.strip()]
        if stderr.strip():
            parts.append(f"[stderr]: {stderr.strip()}")
        if error:
            parts.append(f"[error]: {error}")
        parts.append(f"[exit_code]: {exit_code}")
        return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class ExplicitPlanStep:
    id: str
    revision: int
    task: str


@dataclass(frozen=True)
class ExplicitPlan:
    revision: int
    steps: tuple[ExplicitPlanStep, ...]


def parse_explicit_plan(text: str, *, revision: int, max_steps: int) -> ExplicitPlan:
    raw = extract_json_dict(text)
    if raw is None:
        raise ProtocolError("response did not contain a JSON object")
    if set(raw) != {"steps", "planning_complete"}:
        raise ProtocolError("plan must contain exactly steps and planning_complete")
    if raw["planning_complete"] is not True:
        raise ProtocolError("planning_complete must be true")
    raw_steps = raw["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ProtocolError("plan must contain at least one step")
    if len(raw_steps) > max_steps:
        raise ProtocolError(f"plan contains {len(raw_steps)} steps, max is {max_steps}")
    steps: list[ExplicitPlanStep] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict) or set(raw_step) != {"step_goal"}:
            raise ProtocolError("each plan step must contain exactly step_goal")
        task = raw_step["step_goal"]
        if not isinstance(task, str) or not task.strip():
            raise ProtocolError("step_goal must be a non-empty string")
        steps.append(
            ExplicitPlanStep(
                id=f"plan_{revision}_step_{index}",
                revision=revision,
                task=task.strip(),
            )
        )
    return ExplicitPlan(revision=revision, steps=tuple(steps))


def parse_explicit_step_terminal(text: str) -> StepStatus | None:
    raw = extract_json_dict(text)
    if raw is None or set(raw) != {"complete", "status"}:
        return None
    if raw["complete"] is not True or raw["status"] not in {"success", "failed"}:
        return None
    return raw["status"]


class ExplicitPlanExecuteAgent(_BaselineAgent):
    agent_name = "explicit_plan_execute"

    def __init__(
        self,
        *,
        model_client: ModelClient,
        executor: CodeExecutor,
        config: ExplicitPlanExecuteConfig | None = None,
        on_trace_update: Any | None = None,
    ) -> None:
        super().__init__(
            model_client=model_client,
            executor=executor,
            config=config or ExplicitPlanExecuteConfig(),
            on_trace_update=on_trace_update,
        )
        self.config: ExplicitPlanExecuteConfig

    def _run(self, prompt: str) -> tuple[RunResult, str | None]:
        protocol_result = self._run_protocol(prompt)
        final_answer = self._request_final_answer(protocol_result)
        self._emit("done", artifact=asdict(protocol_result))
        return protocol_result, final_answer

    def _run_protocol(self, goal: str) -> RunResult:
        plan = self._request_plan(
            goal=goal,
            revision=0,
            max_steps=self.config.max_steps,
            failed_step=None,
            discarded_steps=(),
            execution_records=(),
        )
        if plan is None:
            return RunResult(
                "failed", "Initial planning did not produce a valid executable plan."
            )

        active_plan = plan
        active_index = 0
        executed_count = 0
        records: list[dict[str, Any]] = []
        while active_index < len(active_plan.steps):
            step = active_plan.steps[active_index]
            executed_count += 1
            status, failure_kind = self._execute_step(
                active_plan, step, records, executed_count
            )
            record: dict[str, Any] = {
                "id": step.id,
                "revision": step.revision,
                "step_goal": step.task,
                "complete": failure_kind in {None, "model_reported"},
                "status": status,
            }
            if status == "failed":
                record["failure_kind"] = failure_kind
            records.append(record)
            if status == "success":
                active_index += 1
                continue

            discarded = active_plan.steps[active_index + 1 :]
            remaining_budget = self.config.max_steps - executed_count
            if remaining_budget <= 0:
                return RunResult(
                    "failed",
                    "A step failed and the execution step budget was exhausted.",
                )
            replacement = self._request_plan(
                goal=goal,
                revision=active_plan.revision + 1,
                max_steps=remaining_budget,
                failed_step=step,
                discarded_steps=discarded,
                execution_records=tuple(records),
            )
            if replacement is None:
                return RunResult(
                    "failed",
                    "Failure replanning did not produce a valid replacement plan.",
                )
            active_plan = replacement
            active_index = 0
        return RunResult(
            "success", "All steps in the active plan completed successfully."
        )

    def _request_plan(
        self,
        *,
        goal: str,
        revision: int,
        max_steps: int,
        failed_step: ExplicitPlanStep | None,
        discarded_steps: tuple[ExplicitPlanStep, ...],
        execution_records: tuple[dict[str, Any], ...],
    ) -> ExplicitPlan | None:
        event = "replan" if failed_step is not None else "plan"
        last_error = "response did not contain a valid complete plan"
        for attempt in range(2):
            if attempt == 0:
                text = (
                    self._replan_prompt(
                        failed_step,
                        discarded_steps,
                        execution_records,
                        max_steps,
                    )
                    if failed_step is not None
                    else self._plan_prompt(goal)
                )
            else:
                text = self._plan_format_feedback(last_error, max_steps)
            self._append_message("user", text)
            turn = self._call_model(event, revision=revision)
            plan = None
            for candidate in _protocol_text_candidates(turn):
                try:
                    plan = parse_explicit_plan(
                        candidate, revision=revision, max_steps=max_steps
                    )
                    break
                except ProtocolError as exc:
                    last_error = str(exc)
            if plan is None:
                self._emit(
                    event,
                    artifact={
                        "revision": revision,
                        "valid": False,
                        "error": last_error,
                        "attempt": attempt + 1,
                    },
                )
                continue
            self._emit(
                event,
                artifact={
                    "revision": revision,
                    "valid": True,
                    "steps": [self._dump_step(item) for item in plan.steps],
                    "remaining_step_budget": max_steps,
                    "discarded_steps": [
                        self._dump_step(item) for item in discarded_steps
                    ],
                },
            )
            return plan
        return None

    def _execute_step(
        self,
        plan: ExplicitPlan,
        step: ExplicitPlanStep,
        records: list[dict[str, Any]],
        executed_count: int,
    ) -> tuple[StepStatus, str | None]:
        self._emit(
            "step_start",
            artifact={
                "step": self._dump_step(step),
                "executed_step_count": executed_count,
                "remaining_step_budget": self.config.max_steps - executed_count,
            },
        )
        self._append_message("user", self._execute_prompt(plan, step, records))
        format_errors = 0
        for _ in range(self.config.max_step_turns):
            turn = self._call_model(
                "execute_step", step_id=step.id, step_goal=step.task
            )
            action, _ = parse_action(turn.content)
            if action:
                self._execute_action(action, step_id=step.id, step_goal=step.task)
                self._append_message(
                    "user", self._observation_checkpoint_prompt(step)
                )
                continue
            terminal = next(
                (
                    parsed
                    for candidate in _protocol_text_candidates(turn)
                    if (parsed := parse_explicit_step_terminal(candidate)) is not None
                ),
                None,
            )
            if terminal is not None:
                artifact: dict[str, Any] = {
                    "step": self._dump_step(step),
                    "complete": True,
                    "status": terminal,
                }
                if terminal == "failed":
                    artifact.update(
                        {"failure_kind": "model_reported", "recoverable": True}
                    )
                self._emit("step_complete", artifact=artifact)
                return terminal, "model_reported" if terminal == "failed" else None
            if format_errors == 0:
                format_errors += 1
                self._append_message("user", self._step_format_feedback())
                continue
            self._emit(
                "step_complete",
                artifact={
                    "step": self._dump_step(step),
                    "complete": False,
                    "status": "failed",
                    "failure_kind": "invalid_terminal",
                    "recoverable": True,
                },
            )
            return "failed", "invalid_terminal"
        self._emit(
            "step_complete",
            artifact={
                "step": self._dump_step(step),
                "complete": False,
                "status": "failed",
                "failure_kind": "turn_limit",
                "recoverable": True,
            },
        )
        return "failed", "turn_limit"

    def _request_final_answer(self, result: RunResult) -> str | None:
        self._append_message(
            "user",
            f"""Final answer phase.

Use the complete conversation and execution history to answer the original user request.

Overall execution status:
{result.status}

Do not call tools. Do not return planning or step-status JSON. Provide only the final user-facing answer.
Report only results and actions that the execution history verifies. Do not claim that a failed or unverified action succeeded.
If missing or ambiguous information prevented safe completion, ask the user for the specific clarification needed.""",
        )
        answer = self._call_model("final_answer").content.strip()
        if answer:
            return answer
        self._append_message(
            "user",
            "No visible final answer was returned. Provide only the final user-facing answer in assistant content.",
        )
        return self._call_model("final_answer_correction").content.strip() or None

    @staticmethod
    def _plan_prompt(goal: str) -> str:
        return f"""Planning phase.

Create the complete ordered execution plan for the user task. Do not call tools and do not execute the task.

Original user task:
{goal}

Return ONLY JSON:
{{"steps": [{{"step_goal": string}}], "planning_complete": true}}"""

    @classmethod
    def _replan_prompt(
        cls,
        failed_step: ExplicitPlanStep,
        discarded_steps: tuple[ExplicitPlanStep, ...],
        records: tuple[dict[str, Any], ...],
        max_steps: int,
    ) -> str:
        return f"""Failure replanning phase.

The current step failed. Replace every unexecuted step from the old plan with a complete new suffix. The failed step remains part of execution history; include a repair step if recovery requires one. Do not call tools or execute the task.

Failed step:
{json.dumps(cls._dump_step(failed_step), ensure_ascii=False, indent=2)}

Discarded unexecuted suffix:
{json.dumps([cls._dump_step(item) for item in discarded_steps], ensure_ascii=False, indent=2)}

Execution records:
{json.dumps(records, ensure_ascii=False, indent=2)}

Return ONLY JSON:
{{"steps": [{{"step_goal": string}}], "planning_complete": true}}

Rules:
- Return the complete replacement suffix, not a patch to the old plan.
- Return between 1 and {max_steps} steps.
- Use the smallest practical number of outcome-oriented steps; do not create one step per tool call.
- Do not add external side effects or write actions that the user did not request.
- Preserve ambiguity instead of guessing, and do not include a final-answer presentation step.
- The new steps must recover from the failure and finish the original task."""

    @staticmethod
    def _plan_format_feedback(error: str, max_steps: int) -> str:
        return f"""Plan format correction required.

The previous planning response was invalid: {error}
Do not call tools, execute the task, or explain the error.
Return ONLY JSON with between 1 and {max_steps} steps:
{{"steps": [{{"step_goal": string}}], "planning_complete": true}}"""

    @classmethod
    def _execute_prompt(
        cls,
        plan: ExplicitPlan,
        step: ExplicitPlanStep,
        records: list[dict[str, Any]],
    ) -> str:
        return f"""Step execution phase.

Use ReAct to complete only the current step. Execute at most one Bash command per response using:
<action>one Bash command</action>

Active plan:
{json.dumps([cls._dump_step(item) for item in plan.steps], ensure_ascii=False, indent=2)}

Prior execution records:
{json.dumps(records, ensure_ascii=False, indent=2)}

Current step:
{json.dumps(cls._dump_step(step), ensure_ascii=False, indent=2)}

When the current step reaches a terminal outcome, make no tool call and return ONLY JSON:
{{"complete": true, "status": "success|failed"}}

Rules:
- Work only on the current step. Do not execute later plan steps early.
- Do not perform external side effects or write actions outside the current step or original user request.
- Return success only when this step is complete and verified.
- Return failed when the step cannot be completed or verified.
- A response containing an action continues this step and is not terminal."""

    @classmethod
    def _observation_checkpoint_prompt(cls, step: ExplicitPlanStep) -> str:
        return f"""Step observation checkpoint.

The immediately preceding tool result belongs to the current step:
{json.dumps(cls._dump_step(step), ensure_ascii=False, indent=2)}

Decide whether that observation completes and verifies the current step.
- If complete, return ONLY {{"complete": true, "status": "success"}}.
- If it cannot be completed or verified, return ONLY {{"complete": true, "status": "failed"}}.
- Otherwise, return only the next <action>one Bash command</action> for this step.

Do not execute a later plan step and do not perform actions outside the original user request."""

    @staticmethod
    def _step_format_feedback() -> str:
        return """Step terminal format correction required.

If more tool work is required, return only <action>one Bash command</action>.
Otherwise return ONLY JSON:
{"complete": true, "status": "success|failed"}"""

    @staticmethod
    def _dump_step(step: ExplicitPlanStep) -> dict[str, Any]:
        return {"id": step.id, "revision": step.revision, "step_goal": step.task}


class RecapState(str, Enum):
    INIT = "init"
    DOWN = "down"
    ACTION_TAKEN = "action_taken"
    UP = "up"
    FINALIZE = "finalize"


@dataclass
class RecapInfo:
    think: str
    subtasks: list[str]


@dataclass(frozen=True)
class RecapDecompose:
    think: str
    subtasks: list[str]


@dataclass(frozen=True)
class RecapAction:
    command: str


@dataclass(frozen=True)
class RecapComplete:
    summary: str


RecapResponse = RecapDecompose | RecapAction | RecapComplete


class RecapNode:
    def __init__(self, task_name: str, parent: RecapNode | None = None) -> None:
        self.task_name = task_name
        self.parent = parent
        self.children: list[RecapNode] = []
        self.info_list: list[RecapInfo] = []
        self.obs_list: list[str] = []
        self.completion_summary: str | None = None

    def latest_info(self) -> RecapInfo:
        return self.info_list[-1] if self.info_list else RecapInfo("", [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "children": [child.to_dict() for child in self.children],
            "info_list": [asdict(info) for info in self.info_list],
            "obs_list": self.obs_list,
            "completion_summary": self.completion_summary,
        }


@dataclass(frozen=True)
class RecapStep:
    continue_loop: bool
    state: RecapState
    prompt: str | None = None
    done_task_name: str | None = None
    remaining_subtasks: tuple[str, ...] = ()
    note: str = ""
    done: bool = False
    final_answer: str | None = None


def parse_recap_json(text: str) -> RecapResponse | None:
    raw = extract_json_dict(text)
    if raw is None:
        return None

    response_type = raw.get("type")
    if response_type == "decompose":
        if set(raw) != {"type", "think", "subtasks"}:
            return None
        if not isinstance(raw["think"], str) or not isinstance(
            raw["subtasks"], list
        ):
            return None
        if not raw["subtasks"] or not all(
            isinstance(item, str) and item.strip() for item in raw["subtasks"]
        ):
            return None
        subtasks = [item.strip() for item in raw["subtasks"]]
        return RecapDecompose(raw["think"], subtasks)

    if response_type == "action":
        if set(raw) != {"type", "command"}:
            return None
        command = raw["command"]
        if not isinstance(command, str) or not command.strip():
            return None
        return RecapAction(command.strip())

    if response_type == "complete":
        if set(raw) != {"type", "summary"}:
            return None
        summary = raw["summary"]
        if not isinstance(summary, str) or not summary.strip():
            return None
        return RecapComplete(summary.strip())

    return None


def _parse_recap_turn(turn: ModelTurn) -> RecapResponse | None:
    for candidate in _protocol_text_candidates(turn):
        response = parse_recap_json(candidate)
        if response is not None:
            return response
    return None


class RecapController:
    def __init__(self, root_task: str, config: RecapConfig) -> None:
        self.config = config
        self.root = RecapNode(root_task)
        self.node = self.root
        self.depth = 1
        self.state = RecapState.INIT
        self._format_retry_pending = False

    @property
    def current_task(self) -> str:
        return self.node.task_name

    def initial_prompt(self) -> str:
        return self._down_prompt(self.root.task_name)

    def after_tool_action(self, observation: str) -> RecapStep:
        self._format_retry_pending = False
        fitted = observation
        if len(fitted) > self.config.max_obs_chars:
            fitted = fitted[: self.config.max_obs_chars] + "\n[observation truncated]"
        self.node.obs_list.append(fitted)
        info = self.node.latest_info()
        self.state = RecapState.ACTION_TAKEN
        return RecapStep(
            True,
            self.state,
            self._action_taken_prompt(fitted, self.node.task_name, info.subtasks),
            remaining_subtasks=tuple(info.subtasks),
        )

    def process_assistant_text(self, text: str) -> RecapStep:
        return self.process_response(parse_recap_json(text))

    def process_response(self, response: RecapResponse | None) -> RecapStep:
        if self.state == RecapState.FINALIZE:
            return RecapStep(False, self.state, note="final_answer", done=True)
        if response is None or isinstance(response, RecapAction):
            if not self._format_retry_pending:
                self._format_retry_pending = True
                return RecapStep(
                    True,
                    self.state,
                    self._format_feedback_prompt(),
                    note="invalid_recap_json_retry",
                )
            self._format_retry_pending = False
            return RecapStep(False, self.state, note="non_recap_final")

        self._format_retry_pending = False
        if isinstance(response, RecapDecompose):
            if self.depth >= self.config.max_depth:
                return RecapStep(
                    True,
                    self.state,
                    self._depth_limit_prompt(),
                    note="max_depth_reached",
                )
            subtasks = response.subtasks[: self.config.max_subtasks]
            info = RecapInfo(response.think, subtasks)
            self.node.info_list.append(info)
            child = RecapNode(subtasks[0], self.node)
            self.node.children.append(child)
            self.node = child
            self.depth += 1
            self.state = RecapState.DOWN
            return RecapStep(
                True,
                self.state,
                self._down_prompt(child.task_name),
                remaining_subtasks=tuple(subtasks[1:]),
            )

        self.node.completion_summary = response.summary
        done_task = self.node.task_name
        self.state = RecapState.UP
        if self.node.parent is None:
            self.state = RecapState.FINALIZE
            return RecapStep(
                False,
                self.state,
                done_task_name=done_task,
                note="task_tree_complete",
                done=True,
                final_answer=response.summary,
            )
        self.node = self.node.parent
        self.depth = max(1, self.depth - 1)
        parent_info = self.node.latest_info()
        remaining = parent_info.subtasks[1:]
        return RecapStep(
            True,
            self.state,
            self._up_prompt(
                done_task,
                response.summary,
                self.node.task_name,
                parent_info.think,
                remaining,
            ),
            done_task_name=done_task,
            remaining_subtasks=tuple(remaining),
        )

    def tree_snapshot(self) -> dict[str, Any]:
        data = self.root.to_dict()
        if len(json.dumps(data, ensure_ascii=False)) <= self.config.max_tree_chars:
            return data
        return {
            "task_name": self.root.task_name,
            "truncated": True,
            "current_task": self.node.task_name,
            "depth": self.depth,
        }

    @staticmethod
    def _response_instruction(*, allow_decompose: bool) -> str:
        variants = []
        if allow_decompose:
            variants.append(
                '- Decompose unfinished work: {"type":"decompose","think":string,'
                '"subtasks":[non-empty strings]}'
            )
        variants.extend(
            [
                '- Execute exactly one Bash command: {"type":"action","command":string}',
                '- Complete the current task: {"type":"complete","summary":string}',
            ]
        )
        return (
            "Return ONLY one JSON object matching exactly one allowed variant. "
            "Do not mix fields from different variants and do not add fields.\n"
            + "\n".join(variants)
            + "\nUse complete only when the current task is actually finished based "
            "on available information or tool observations. Never claim an "
            "unexecuted command succeeded."
        )

    def _at_max_depth(self) -> bool:
        return self.depth >= self.config.max_depth

    def _depth_limit_guidance(self) -> str:
        return (
            f"Maximum recursion depth ({self.config.max_depth}) has been reached. "
            "The decompose response is not allowed. Execute the necessary Bash "
            "commands one at a time with action responses, then use complete only "
            "after this task is finished."
        )

    def _format_feedback_prompt(self) -> str:
        if self._at_max_depth():
            return (
                "Your previous response did not match one ReCAP response variant. "
                f"{self._depth_limit_guidance()}\n\n"
                f"{self._response_instruction(allow_decompose=False)}"
            )
        return (
            "Your previous response did not match one ReCAP response variant.\n\n"
            f"{self._response_instruction(allow_decompose=True)}"
        )

    def _depth_limit_prompt(self) -> str:
        return f"""Your current task:
{self.node.task_name}

{self._depth_limit_guidance()}

{self._response_instruction(allow_decompose=False)}"""

    def _down_prompt(self, task_name: str) -> str:
        guidance = (
            self._depth_limit_guidance()
            if self._at_max_depth()
            else "Decompose complex unfinished work, execute one Bash command, or complete the current task."
        )
        return f"""OK.

Your current task:
{task_name}

{guidance}

{self._response_instruction(allow_decompose=not self._at_max_depth())}"""

    def _action_taken_prompt(
        self, observation: str, task_name: str, remaining_subtasks: list[str]
    ) -> str:
        remaining = "\n".join(remaining_subtasks) or "No remaining subtasks."
        guidance = (
            self._depth_limit_guidance()
            if self._at_max_depth()
            else "Refine unfinished work by decomposing, execute the next Bash command, or complete the current task."
        )
        return f"""Latest observation:
{observation or "[no textual observation]"}

Your current task:
{task_name}

Your previously proposed subtasks:
{remaining}

{guidance}

{self._response_instruction(allow_decompose=not self._at_max_depth())}"""

    def _up_prompt(
        self,
        done_task: str,
        completion_summary: str,
        parent_task: str,
        parent_think: str,
        remaining_subtasks: list[str],
    ) -> str:
        remaining = "\n".join(remaining_subtasks) or "No remaining subtasks."
        guidance = (
            self._depth_limit_guidance()
            if self._at_max_depth()
            else "Continue unfinished work by decomposing, execute the next Bash command, or complete the parent task."
        )
        return f"""The child task {done_task} reported completion.
Completion summary: {completion_summary}

Now, you return to the parent task.
Your current task: {parent_task}

Your previous think: {parent_think}

Your remaining subtasks:
{remaining}

{guidance}

{self._response_instruction(allow_decompose=not self._at_max_depth())}"""


class RecapAgent(_BaselineAgent):
    agent_name = "recap"

    def __init__(
        self,
        *,
        model_client: ModelClient,
        executor: CodeExecutor,
        config: RecapConfig | None = None,
        on_trace_update: Any | None = None,
    ) -> None:
        super().__init__(
            model_client=model_client,
            executor=executor,
            config=config or RecapConfig(),
            on_trace_update=on_trace_update,
        )
        self.config: RecapConfig

    def _run(self, prompt: str) -> tuple[RunResult, str | None]:
        controller = RecapController(prompt, self.config)
        self._append_message("user", controller.initial_prompt())
        self._emit_recap(controller, "init")
        while True:
            turn = self._call_model(
                "recap", state=controller.state.value, current_task=controller.current_task
            )
            response = _parse_recap_turn(turn)
            if isinstance(response, RecapAction):
                observation = self._execute_action(
                    response.command,
                    state=controller.state.value,
                    current_task=controller.current_task,
                )
                step = controller.after_tool_action(observation)
            else:
                step = controller.process_response(response)

            self._emit_recap(controller, step.note or step.state.value, step)
            if step.prompt:
                self._append_message("user", step.prompt)
            if step.done:
                return (
                    RunResult("success", "Recursive task tree completed."),
                    step.final_answer,
                )
            if not step.continue_loop:
                return (
                    RunResult(
                        "failed",
                        "ReCAP stopped after repeated invalid controller output.",
                    ),
                    turn.content.strip() or None,
                )

    def _emit_recap(
        self,
        controller: RecapController,
        note: str,
        step: RecapStep | None = None,
    ) -> None:
        self._emit(
            "recap_state",
            artifact={
                "state": controller.state.value,
                "depth": controller.depth,
                "current_task": controller.current_task,
                "done_task": step.done_task_name if step else None,
                "remaining_subtasks": list(step.remaining_subtasks) if step else [],
                "note": note,
                "tree": controller.tree_snapshot(),
            },
        )


class ReflectionAgent(_BaselineAgent):
    agent_name = "reflection"

    def __init__(
        self,
        *,
        model_client: ModelClient,
        executor: CodeExecutor,
        config: ReflectionConfig | None = None,
        on_trace_update: Any | None = None,
    ) -> None:
        super().__init__(
            model_client=model_client,
            executor=executor,
            config=config or ReflectionConfig(),
            on_trace_update=on_trace_update,
        )

    def _run(self, prompt: str) -> tuple[RunResult, str | None]:
        del prompt
        self._append_message(
            "user",
            """Use Bash commands, primarily curl, to complete the task. Return at most one command per response as <action>one Bash command</action>. Inspect every observation, preserve successful side effects, and verify ambiguous mutations before retrying. When complete, return <done>the verified user-facing result</done> or a plain final answer.""",
        )
        reflection_pending = False
        empty_correction_used = False
        while True:
            if reflection_pending:
                reflection_pending = False
                self._append_message(
                    "user",
                    "Reflect briefly on the latest tool result, then continue the task.\nCall tools if more work is needed. If the task is complete, answer the user.",
                )
                self._emit("reflection", artifact={"phase": "reflect"})
                phase = "reflection"
            else:
                phase = "react"
            turn = self._call_model(phase)
            action, done = parse_action(turn.content)
            if action:
                self._execute_action(action, phase=phase)
                reflection_pending = True
                empty_correction_used = False
                continue
            answer = done or turn.content.strip()
            if answer:
                return RunResult("success", "Agent returned a final answer."), answer
            if empty_correction_used:
                return RunResult("failed", "Agent returned no visible action or answer."), None
            empty_correction_used = True
            self._append_message(
                "user",
                "Your response produced no visible output. Return the required <action>, or provide the final user-facing answer.",
            )
            self._emit("protocol_warning", artifact={"phase": phase})


def run_explicit_plan_execute_agent(
    *,
    model_client: ModelClient,
    prompt: str,
    executor: CodeExecutor,
    system_prompt: str,
    config: ExplicitPlanExecuteConfig | None = None,
    on_trace_update: Any | None = None,
) -> dict[str, Any]:
    return ExplicitPlanExecuteAgent(
        model_client=model_client,
        executor=executor,
        config=config,
        on_trace_update=on_trace_update,
    ).run(prompt, system_prompt)


def run_recap_agent(
    *,
    model_client: ModelClient,
    prompt: str,
    executor: CodeExecutor,
    system_prompt: str,
    config: RecapConfig | None = None,
    on_trace_update: Any | None = None,
) -> dict[str, Any]:
    return RecapAgent(
        model_client=model_client,
        executor=executor,
        config=config,
        on_trace_update=on_trace_update,
    ).run(prompt, system_prompt)


def run_reflection_agent(
    *,
    model_client: ModelClient,
    prompt: str,
    executor: CodeExecutor,
    system_prompt: str,
    config: ReflectionConfig | None = None,
    on_trace_update: Any | None = None,
) -> dict[str, Any]:
    return ReflectionAgent(
        model_client=model_client,
        executor=executor,
        config=config,
        on_trace_update=on_trace_update,
    ).run(prompt, system_prompt)


def run_baseline_agent(
    *,
    mode: BaselineMode,
    model_client: ModelClient,
    prompt: str,
    executor: CodeExecutor,
    system_prompt: str,
    config: BaselineConfig | None = None,
    on_trace_update: Any | None = None,
) -> dict[str, Any]:
    runners = {
        "explicit-plan-execute": (
            ExplicitPlanExecuteConfig,
            run_explicit_plan_execute_agent,
        ),
        "recap": (RecapConfig, run_recap_agent),
        "reflection": (ReflectionConfig, run_reflection_agent),
    }
    try:
        config_type, runner = runners[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported baseline mode: {mode!r}") from exc
    if config is not None and not isinstance(config, config_type):
        raise TypeError(
            f"{mode} requires {config_type.__name__}, got {type(config).__name__}"
        )
    return runner(
        model_client=model_client,
        prompt=prompt,
        executor=executor,
        system_prompt=system_prompt,
        config=config,
        on_trace_update=on_trace_update,
    )


__all__ = [
    "BaselineConfig",
    "BaselineMode",
    "ExplicitPlanExecuteAgent",
    "ExplicitPlanExecuteConfig",
    "RecapAction",
    "RecapAgent",
    "RecapComplete",
    "RecapConfig",
    "RecapController",
    "RecapDecompose",
    "RecapResponse",
    "RecapState",
    "ReflectionAgent",
    "ReflectionConfig",
    "parse_explicit_plan",
    "parse_explicit_step_terminal",
    "parse_recap_json",
    "run_baseline_agent",
    "run_explicit_plan_execute_agent",
    "run_recap_agent",
    "run_reflection_agent",
]
