"""Full-plan execution with per-step ReAct and failure replanning."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..models.content import TextBlock, ToolResultBlock, ToolUseBlock
from ..models.message import Message
from ..models.task import TaskDefinition
from ..models.tool import ToolEndpoint, ToolSpec
from ..models.trace import PlanExecuteEvent, TokenUsage, TraceMessage
from ..trace.writer import TraceWriter
from .compact import do_auto_compact, micro_compact, should_auto_compact
from .dispatcher import ToolDispatcher
from .protocol import is_reasoning_only_response, protocol_response_text
from .rex_runner import (
    RExRunResult,
    RExStepResult,
    _json_loads,
    _provider_chat_with_timeout,
    _safe_tool_result,
)
from .todo import TodoManager


class PlanExecuteError(ValueError):
    """Invalid plan-execute protocol output."""


class PlanPromptStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_goal: str


class PlanPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[PlanPromptStep] = Field(default_factory=list)
    planning_complete: bool = False


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    revision: int
    task: str


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int
    steps: list[PlanStep] = Field(default_factory=list)


class StepTerminal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    complete: Literal[True]
    status: Literal["success", "failed"]


@dataclass
class _AggregateResult:
    result: RExStepResult
    usage: TokenUsage = field(default_factory=TokenUsage)
    turns: int = 0
    model_time_s: float = 0.0
    tool_time_s: float = 0.0
    timed_out: bool = False
    timeout_type: str | None = None
    timeout_seconds: int | None = None


@dataclass
class _StepResult(_AggregateResult):
    terminal: StepTerminal | None = None
    failure_kind: str | None = None


PLAN_EXECUTE_INTERNAL_MESSAGE_EXTRA = {
    "internal": True,
    "source": "plan_execute",
}


def parse_plan(
    text: str, *, revision: int, max_steps: int
) -> Plan | None:
    raw = _json_loads(text, required_keys=("steps", "planning_complete"))
    if raw is None or not all(key in raw for key in ("steps", "planning_complete")):
        return None
    try:
        prompt_plan = PlanPrompt.model_validate(raw)
    except ValidationError as exc:
        raise PlanExecuteError(str(exc)) from exc
    if not prompt_plan.planning_complete:
        raise PlanExecuteError("planning_complete must be true")
    if not prompt_plan.steps:
        raise PlanExecuteError("plan must contain at least one step")
    if len(prompt_plan.steps) > max_steps:
        raise PlanExecuteError(
            f"plan contains {len(prompt_plan.steps)} steps, max is {max_steps}"
        )

    steps: list[PlanStep] = []
    for index, step in enumerate(prompt_plan.steps, start=1):
        task = step.step_goal.strip()
        if not task:
            raise PlanExecuteError("step_goal cannot be empty")
        steps.append(
            PlanStep(
                id=f"plan_{revision}_step_{index}",
                revision=revision,
                task=task,
            )
        )
    return Plan(revision=revision, steps=steps)


def parse_step_terminal(text: str) -> StepTerminal | None:
    raw = _json_loads(text, required_keys=("complete", "status"))
    if raw is None or not all(key in raw for key in ("complete", "status")):
        return None
    try:
        return StepTerminal.model_validate(raw)
    except ValidationError:
        return None


class PlanExecuteRunner:
    """Run an explicit plan, one ReAct loop per step, replanning on failure."""

    def __init__(
        self,
        *,
        task: TaskDefinition,
        provider: Any,
        trace_id: str,
        writer: TraceWriter,
        base_messages: list[Message],
        tools: list[ToolSpec],
        endpoints: dict[str, ToolEndpoint],
        todo_mgr: TodoManager | None,
        writer_lock: threading.Lock | None = None,
        chat_timeout_s: int = 300,
        task_timeout_s: int | None = None,
        context_window: int = 128_000,
        allow_needs_user: bool = False,
    ) -> None:
        self.task = task
        self.provider = provider
        self.trace_id = trace_id
        self.writer = writer
        self.messages = base_messages
        self.tools = tools
        self.endpoints = endpoints
        self.todo_mgr = todo_mgr
        self.writer_lock = writer_lock or threading.Lock()
        self.chat_timeout_s = chat_timeout_s
        self.task_timeout_s = max(
            1,
            task_timeout_s
            if task_timeout_s is not None
            else task.environment.timeout_seconds,
        )
        self.task_deadline = time.monotonic() + self.task_timeout_s
        self.context_window = max(1, context_window)
        self.max_steps = max(1, task.environment.plan_execute_max_steps)
        self.max_step_turns = max(
            1, task.environment.plan_execute_max_step_turns
        )
        self.todo_lock = threading.Lock()
        self.auto_compacts = 0

    def run(
        self, *, root_goal: str | None = None, frame_id: str = "root"
    ) -> RExRunResult:
        result = RExRunResult()
        goal = root_goal or self.task.prompt.text
        protocol = self._run_protocol(goal=goal)
        self._accumulate(result, protocol)
        result.root_status = protocol.result.status

        final = self._final_answer(protocol.result)
        self._accumulate(result, final)
        result.final_message = final.final_message
        self._write_event("done", artifact=protocol.result.model_dump())
        return result

    def _run_protocol(self, *, goal: str) -> _AggregateResult:
        aggregate = _AggregateResult(
            result=RExStepResult(
                status="failed", summary="Plan-execute did not complete."
            )
        )
        plan_run, plan = self._request_plan(
            goal=goal,
            revision=0,
            max_steps=self.max_steps,
            failed_step=None,
            discarded_steps=[],
            execution_records=[],
        )
        self._accumulate(aggregate, plan_run)
        if plan is None:
            aggregate.result = RExStepResult(
                status="failed",
                summary="Initial planning did not produce a valid executable plan.",
            )
            return aggregate

        active_plan = plan
        active_index = 0
        executed_count = 0
        execution_records: list[dict[str, Any]] = []

        while active_index < len(active_plan.steps):
            step = active_plan.steps[active_index]
            executed_count += 1
            step_run = self._execute_step(
                goal=goal,
                plan=active_plan,
                step=step,
                execution_records=execution_records,
                executed_count=executed_count,
            )
            self._accumulate(aggregate, step_run)
            if step_run.timed_out:
                aggregate.result = step_run.result
                return aggregate

            terminal = step_run.terminal
            record = {
                "id": step.id,
                "revision": step.revision,
                "step_goal": step.task,
                "complete": terminal.complete if terminal is not None else False,
                "status": terminal.status if terminal is not None else "failed",
            }
            if record["status"] == "failed":
                record["failure_kind"] = step_run.failure_kind or "model_reported"
            execution_records.append(record)

            if terminal is not None and terminal.status == "success":
                active_index += 1
                continue

            discarded_steps = active_plan.steps[active_index + 1 :]
            remaining_budget = self.max_steps - executed_count
            if remaining_budget <= 0:
                aggregate.result = RExStepResult(
                    status="failed",
                    summary="A step failed and the execution step budget was exhausted.",
                )
                return aggregate

            replan_run, replacement = self._request_plan(
                goal=goal,
                revision=active_plan.revision + 1,
                max_steps=remaining_budget,
                failed_step=step,
                discarded_steps=discarded_steps,
                execution_records=execution_records,
            )
            self._accumulate(aggregate, replan_run)
            if replacement is None:
                aggregate.result = RExStepResult(
                    status="failed",
                    summary="Failure replanning did not produce a valid replacement plan.",
                )
                return aggregate
            active_plan = replacement
            active_index = 0

        aggregate.result = RExStepResult(
            status="success",
            summary="All steps in the active plan completed successfully.",
        )
        return aggregate

    def _request_plan(
        self,
        *,
        goal: str,
        revision: int,
        max_steps: int,
        failed_step: PlanStep | None,
        discarded_steps: list[PlanStep],
        execution_records: list[dict[str, Any]],
    ) -> tuple[RExRunResult, Plan | None]:
        aggregate = RExRunResult()
        event = "replan" if failed_step is not None else "plan"
        last_error = "response did not contain a valid complete plan"

        for attempt in range(2):
            if attempt == 0:
                prompt_text = (
                    self._replan_prompt(
                        failed_step=failed_step,
                        discarded_steps=discarded_steps,
                        execution_records=execution_records,
                        max_steps=max_steps,
                    )
                    if failed_step is not None
                    else self._plan_prompt(goal=goal)
                )
            else:
                prompt_text = self._plan_format_feedback(last_error, max_steps=max_steps)
            self._append_message(
                Message(role="user", content=[TextBlock(text=prompt_text)])
            )
            chat = self._chat_current(tools=None, focus=f"{event} phase")
            self._accumulate(aggregate, chat)
            if chat.timed_out:
                self._write_event(
                    event,
                    artifact={
                        "revision": revision,
                        "remaining_step_budget": max_steps,
                    },
                    note=f"{chat.timeout_type or 'model'} timeout",
                )
                return aggregate, None

            response = chat.final_message
            tool_uses = (
                [block for block in response.content if block.type == "tool_use"]
                if response is not None
                else []
            )
            try:
                if response is None:
                    raise PlanExecuteError("model returned no response")
                if tool_uses:
                    raise PlanExecuteError(
                        "planning returned a tool call while tools were disabled"
                    )
                response_text = protocol_response_text(
                    response,
                    lambda text: parse_plan(
                        text, revision=revision, max_steps=max_steps
                    ),
                )
                parsed = parse_plan(
                    response_text, revision=revision, max_steps=max_steps
                )
                if parsed is None:
                    raise PlanExecuteError(last_error)
            except PlanExecuteError as exc:
                last_error = str(exc)
                self._write_event(
                    event,
                    artifact={
                        "revision": revision,
                        "remaining_step_budget": max_steps,
                        "discarded_steps": [
                            self._dump_step(item) for item in discarded_steps
                        ],
                    },
                    note=f"{last_error}; retrying" if attempt == 0 else last_error,
                )
                continue

            self._write_event(
                event,
                artifact={
                    "revision": revision,
                    "steps": [self._dump_step(item) for item in parsed.steps],
                    "planning_complete": True,
                    "remaining_step_budget": max_steps,
                    "failed_step": (
                        self._dump_step(failed_step) if failed_step is not None else None
                    ),
                    "discarded_steps": [
                        self._dump_step(item) for item in discarded_steps
                    ],
                },
                execution_snapshot=self._execution_snapshot(parsed, execution_records),
            )
            return aggregate, parsed

        return aggregate, None

    def _execute_step(
        self,
        *,
        goal: str,
        plan: Plan,
        step: PlanStep,
        execution_records: list[dict[str, Any]],
        executed_count: int,
    ) -> _StepResult:
        self._write_event(
            "step_start",
            step_id=step.id,
            step_task=step.task,
            artifact={
                "executed_step_count": executed_count,
                "remaining_step_budget": self.max_steps - executed_count,
            },
            execution_snapshot=self._execution_snapshot(plan, execution_records),
        )
        self._append_message(
            Message(
                role="user",
                content=[
                    TextBlock(
                        text=self._execute_prompt(
                            plan=plan,
                            step=step,
                            execution_records=execution_records,
                        )
                    )
                ],
            )
        )
        aggregate = _StepResult(
            result=RExStepResult(
                status="failed", summary="Step did not reach a valid terminal response."
            )
        )
        format_errors = 0

        for _ in range(self.max_step_turns):
            chat = self._chat_current(
                tools=self.tools,
                focus=f"execute step: {step.task[:120]}",
            )
            self._accumulate(aggregate, chat)
            if chat.timed_out:
                aggregate.result = RExStepResult(
                    status="failed", summary="Step timed out while calling the model."
                )
                aggregate.failure_kind = "timeout"
                self._write_event(
                    "step_complete",
                    step_id=step.id,
                    step_task=step.task,
                    artifact={
                        "complete": False,
                        "status": "failed",
                        "failure_kind": "timeout",
                        "recoverable": False,
                    },
                    note=f"{chat.timeout_type or 'model'} timeout",
                )
                return aggregate

            response = chat.final_message
            if response is None:
                continue
            tool_uses = [block for block in response.content if block.type == "tool_use"]
            if tool_uses:
                tool_message, tool_time = self._dispatch_tools(tool_uses, step=step)
                aggregate.tool_time_s += tool_time
                self._append_message(tool_message)
                self._append_message(
                    Message(
                        role="user",
                        content=[
                            TextBlock(
                                text=self._observation_checkpoint_prompt(step=step)
                            )
                        ],
                    )
                )
                continue

            terminal = parse_step_terminal(
                protocol_response_text(response, parse_step_terminal)
            )
            if terminal is not None:
                aggregate.terminal = terminal
                aggregate.result = RExStepResult(
                    status=terminal.status,
                    summary=f"Step returned terminal status {terminal.status}.",
                )
                self._write_event(
                    "step_complete",
                    step_id=step.id,
                    step_task=step.task,
                    artifact=(
                        {
                            **terminal.model_dump(),
                            "failure_kind": "model_reported",
                            "recoverable": True,
                        }
                        if terminal.status == "failed"
                        else terminal.model_dump()
                    ),
                )
                if terminal.status == "failed":
                    aggregate.failure_kind = "model_reported"
                return aggregate

            if format_errors == 0:
                format_errors += 1
                self._append_message(
                    Message(
                        role="user",
                        content=[TextBlock(text=self._step_format_feedback())],
                    )
                )
                continue

            aggregate.result = RExStepResult(
                status="failed",
                summary="Step terminal response remained invalid after one correction.",
            )
            aggregate.failure_kind = "invalid_terminal"
            self._write_event(
                "step_complete",
                step_id=step.id,
                step_task=step.task,
                artifact={
                    "complete": False,
                    "status": "failed",
                    "failure_kind": "invalid_terminal",
                    "recoverable": True,
                },
                note=aggregate.result.summary,
            )
            return aggregate

        aggregate.result = RExStepResult(
            status="failed",
            summary=(
                "Step exceeded the plan-execute per-step model turn limit."
            ),
        )
        aggregate.failure_kind = "turn_limit"
        self._write_event(
            "step_complete",
            step_id=step.id,
            step_task=step.task,
            artifact={
                "complete": False,
                "status": "failed",
                "failure_kind": "turn_limit",
                "recoverable": True,
            },
            note=aggregate.result.summary,
        )
        return aggregate

    def _final_answer(
        self, root_result: RExStepResult
    ) -> RExRunResult:
        self._append_message(
            Message(
                role="user",
                content=[
                    TextBlock(
                        text=f"""Final answer phase.

Use the complete conversation and execution history to answer the original user request.

Overall execution status:
{root_result.status}

Do not call tools. Do not return planning or step-status JSON. Provide only the final user-facing answer.
Report only results and actions that the execution history verifies. Do not claim that a failed or unverified action succeeded.
If missing or ambiguous information prevented safe completion, ask the user for the specific clarification needed."""
                    )
                ],
            )
        )
        aggregate = RExRunResult()
        first = self._chat_current(
            tools=None,
            focus="final answer",
            internal_response=lambda response: not bool(response.text.strip()),
        )
        self._accumulate(aggregate, first)
        aggregate.final_message = first.final_message
        if first.timed_out or not is_reasoning_only_response(first.final_message):
            return aggregate

        self._append_message(
            Message(
                role="user",
                content=[
                    TextBlock(
                        text=(
                            "No visible final answer was returned. Provide only the "
                            "final user-facing answer in assistant content."
                        )
                    )
                ],
            )
        )
        second = self._chat_current(
            tools=None,
            focus="final answer visibility correction",
            internal_response=lambda response: not bool(response.text.strip()),
        )
        self._accumulate(aggregate, second)
        aggregate.final_message = second.final_message
        return aggregate

    def _dispatch_tools(
        self, tool_uses: list[ToolUseBlock], *, step: PlanStep
    ) -> tuple[Message, float]:
        result_blocks: list[ToolResultBlock] = []
        total_tool_time = 0.0
        dispatcher = ToolDispatcher(self.endpoints)
        try:
            for tool_use in tool_uses:
                if tool_use.name == "todo" and self.todo_mgr is not None:
                    with self.todo_lock:
                        text = self.todo_mgr.update(tool_use.input.get("items", []))
                    result_blocks.append(_safe_tool_result(tool_use, text))
                    self._write_event(
                        "step_tool",
                        step_id=step.id,
                        step_task=step.task,
                        artifact={
                            "tool_name": tool_use.name,
                            "request_body": tool_use.input,
                            "response_body": {"result": text},
                            "response_status": 200,
                        },
                    )
                    continue

                result, dispatch_event = dispatcher.dispatch(tool_use, self.trace_id)
                total_tool_time += dispatch_event.latency_ms / 1000.0
                result_blocks.append(result)
                with self.writer_lock:
                    self.writer.write_event(dispatch_event)
                self._write_event(
                    "step_tool",
                    step_id=step.id,
                    step_task=step.task,
                    artifact={
                        "tool_name": dispatch_event.tool_name,
                        "request_body": dispatch_event.request_body,
                        "response_body": dispatch_event.response_body,
                        "response_status": dispatch_event.response_status,
                        "is_error": result.is_error,
                    },
                )
        finally:
            dispatcher.close()

        return Message(role="user", content=result_blocks), total_tool_time

    def _chat_current(
        self,
        *,
        tools: list[ToolSpec] | None,
        focus: str,
        internal_response: bool | Callable[[Message], bool] = True,
    ) -> RExRunResult:
        out = RExRunResult()
        remaining_seconds = self.task_deadline - time.monotonic()
        if remaining_seconds <= 0:
            out.timed_out = True
            out.timeout_type = "task"
            out.timeout_seconds = self.task_timeout_s
            return out
        self._compact_messages_if_needed(focus=focus)
        remaining_seconds = self.task_deadline - time.monotonic()
        if remaining_seconds <= 0:
            out.timed_out = True
            out.timeout_type = "task"
            out.timeout_seconds = self.task_timeout_s
            return out
        deadline_limited = remaining_seconds < self.chat_timeout_s
        call_timeout_s = min(
            self.chat_timeout_s,
            max(1, math.ceil(remaining_seconds)),
        )
        started = time.monotonic()
        chat_result = _provider_chat_with_timeout(
            self.provider, self.messages, tools, timeout_s=call_timeout_s
        )
        out.model_time_s += time.monotonic() - started
        if chat_result is None:
            out.timed_out = True
            out.timeout_type = "task" if deadline_limited else "provider_chat"
            out.timeout_seconds = (
                self.task_timeout_s if deadline_limited else self.chat_timeout_s
            )
            return out
        response, usage = chat_result
        out.final_message = response
        out.usage = usage
        out.turns = 1
        internal = (
            internal_response(response)
            if callable(internal_response)
            else internal_response
        )
        self._append_message(response, usage=usage, internal=internal)
        return out

    def _compact_messages_if_needed(self, *, focus: str) -> None:
        if len(self.messages) <= 2:
            return
        if self.task.environment.plan_execute_enable_micro_compact:
            micro_compact(
                self.messages,
                keep_recent=self.task.environment.compact_keep_recent,
                min_chars=self.task.environment.compact_min_chars,
            )
        if not self.task.environment.enable_compact:
            return
        if self.auto_compacts >= self.task.environment.compact_max_auto_compacts:
            return
        if not should_auto_compact(
            self.messages,
            self.context_window,
            self.task.environment.compact_threshold_pct,
        ):
            return

        before = len(self.messages)
        compacted = do_auto_compact(
            self.messages,
            self.provider,
            keep_recent_on_summary=self.task.environment.compact_keep_recent_on_summary,
            protect_tokens=self.task.environment.compact_protect_tokens,
            todo_mgr=self.todo_mgr,
            focus=f"plan execute: {focus}",
        )
        if len(compacted) != before:
            self.messages[:] = compacted
            self.auto_compacts += 1
            self._write_event(
                "context_compact",
                artifact={
                    "message_count_before": before,
                    "message_count_after": len(self.messages),
                    "auto_compacts": self.auto_compacts,
                },
                note=focus,
            )

    def _append_message(
        self,
        message: Message,
        usage: TokenUsage | None = None,
        *,
        internal: bool = True,
    ) -> None:
        self.messages.append(message)
        event = TraceMessage(
            trace_id=self.trace_id,
            message=message,
            usage=usage or TokenUsage(),
            extra=dict(PLAN_EXECUTE_INTERNAL_MESSAGE_EXTRA) if internal else {},
        )
        with self.writer_lock:
            self.writer.write_event(event)

    def _write_event(
        self,
        event: Any,
        *,
        step_id: str | None = None,
        step_task: str | None = None,
        artifact: dict[str, Any] | None = None,
        execution_snapshot: dict[str, Any] | None = None,
        note: str = "",
    ) -> None:
        trace_event = PlanExecuteEvent(
            trace_id=self.trace_id,
            event=event,
            step_id=step_id,
            step_task=step_task,
            artifact=artifact or {},
            execution_snapshot=execution_snapshot or {},
            note=note,
        )
        with self.writer_lock:
            self.writer.write_event(trace_event)

    def _accumulate(
        self,
        target: RExRunResult | _AggregateResult,
        source: RExRunResult | _AggregateResult,
    ) -> None:
        target.usage.input_tokens += source.usage.input_tokens
        target.usage.output_tokens += source.usage.output_tokens
        target.turns += source.turns
        target.model_time_s += source.model_time_s
        target.tool_time_s += source.tool_time_s
        if source.timed_out:
            target.timed_out = True
            target.timeout_type = source.timeout_type
            target.timeout_seconds = source.timeout_seconds

    def _plan_prompt(self, *, goal: str) -> str:
        return f"""Planning phase.

Create the complete ordered execution plan for the user task. Do not call tools and do not execute the task.

Original user task:
{goal}

Return ONLY JSON:
{{"steps": [{{"step_goal": string}}], "planning_complete": true}}"""

    def _replan_prompt(
        self,
        *,
        failed_step: PlanStep,
        discarded_steps: list[PlanStep],
        execution_records: list[dict[str, Any]],
        max_steps: int,
    ) -> str:
        return f"""Failure replanning phase.

The current step failed. Replace every unexecuted step from the old plan with a complete new suffix. The failed step remains part of execution history; include a repair step if recovery requires one. Do not call tools or execute the task.

Failed step:
{json.dumps(self._dump_step(failed_step), ensure_ascii=False, indent=2)}

Discarded unexecuted suffix:
{json.dumps([self._dump_step(item) for item in discarded_steps], ensure_ascii=False, indent=2)}

Execution records:
{json.dumps(execution_records, ensure_ascii=False, indent=2)}

Return ONLY JSON:
{{"steps": [{{"step_goal": string}}], "planning_complete": true}}

Rules:
- Return the complete replacement suffix, not a patch to the old plan.
- Return between 1 and {max_steps} steps.
- Use the smallest practical number of outcome-oriented steps; do not create one step per tool call.
- Do not add external side effects or write actions that the user did not request.
- Preserve ambiguity instead of guessing, and do not include a final-answer presentation step.
- The new steps must recover from the failure and finish the original task."""

    def _plan_format_feedback(self, error: str, *, max_steps: int) -> str:
        return f"""Plan format correction required.

The previous planning response was invalid: {error}
Do not call tools, execute the task, or explain the error.
Return ONLY JSON with between 1 and {max_steps} steps:
{{"steps": [{{"step_goal": string}}], "planning_complete": true}}"""

    def _execute_prompt(
        self,
        *,
        plan: Plan,
        step: PlanStep,
        execution_records: list[dict[str, Any]],
    ) -> str:
        return f"""Step execution phase.

Use ReAct to complete only the current step. Call tools when needed and use their observations before deciding the terminal status.

Active plan:
{json.dumps([self._dump_step(item) for item in plan.steps], ensure_ascii=False, indent=2)}

Prior execution records:
{json.dumps(execution_records, ensure_ascii=False, indent=2)}

Current step:
{json.dumps(self._dump_step(step), ensure_ascii=False, indent=2)}

When the current step reaches a terminal outcome, make no tool call and return ONLY JSON:
{{"complete": true, "status": "success|failed"}}

Rules:
- Work only on the current step. Do not execute later plan steps early.
- Do not perform external side effects or write actions outside the current step or original user request.
- Return success only when this step is complete and verified.
- Return failed when the step cannot be completed or verified.
- A response containing any tool call continues this step and is not terminal."""

    def _observation_checkpoint_prompt(self, *, step: PlanStep) -> str:
        return f"""Step observation checkpoint.

The immediately preceding tool results belong to the current step:
{json.dumps(self._dump_step(step), ensure_ascii=False, indent=2)}

Decide whether those observations complete and verify the current step.
- If the step is complete, make no tool call and return ONLY JSON:
  {{"complete": true, "status": "success"}}
- If the step cannot be completed or verified, make no tool call and return ONLY JSON:
  {{"complete": true, "status": "failed"}}
- Otherwise, call only the next tool needed for this current step.

Do not execute a later plan step and do not perform actions outside the original user request."""

    def _step_format_feedback(self) -> str:
        return """Step terminal format correction required.

The previous response did not contain a valid terminal result. Do not explain.
If more tool work is required, call the tool now. Otherwise return ONLY JSON:
{"complete": true, "status": "success|failed"}

The complete field must be true and status must be exactly success or failed."""

    def _dump_step(self, step: PlanStep) -> dict[str, Any]:
        return {
            "id": step.id,
            "revision": step.revision,
            "step_goal": step.task,
        }

    def _execution_snapshot(
        self,
        plan: Plan,
        execution_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "active_revision": plan.revision,
            "steps": [self._dump_step(item) for item in plan.steps],
            "execution_records": list(execution_records),
        }
