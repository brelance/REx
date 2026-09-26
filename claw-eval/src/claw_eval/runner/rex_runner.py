"""Implicit full-context planning with high-confidence rolling execution."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict

from .dispatcher import ToolDispatcher
from .hierarchical_compact import (
    CompressionScope,
    CompressionSibling,
    PreparedCompression,
    deterministic_handoff,
    estimate_request_tokens,
    extract_compression_evidence,
    extract_inherited_evidence_ledgers,
    filter_runner_control_messages,
    handoff_is_valid,
    normalize_handoff_text,
    prepare_compression_request,
    render_compressed_history,
)
from .protocol import protocol_response_text
from .rex_core.batched_engine import RExBatchFrameEngine
from .rex_core.compression import RExCompressionController
from .rex_core.models import (
    RExObserveResult,
    RExPlanPatch,
    RExPromptPlanPatch,
    RExPromptStep,
    RExRunResult,
    RExStatus,
    RExStep,
    RExStepKind,
    RExStepResult,
)
from .rex_core.policies import ResolvedRExPolicy
from .rex_core.prompting import (
    depth_context as _depth_context,
    execution_mode_guidance as _execution_mode_guidance,
)
from .rex_core.runtime import RExRuntime, provider_chat_with_timeout, safe_tool_result
from .rex_core.types import (
    RExFrameResult,
    RExGroundPlanResult,
    RExPlanBatchResult,
    RExReplanContext,
    RExStepExecutionResult,
)
from .todo import TodoManager
from ..models.content import TextBlock, ToolResultBlock, ToolUseBlock
from ..models.message import Message
from ..models.task import TaskDefinition
from ..models.tool import ToolEndpoint, ToolSpec
from ..models.trace import TokenUsage
from ..trace.writer import TraceWriter

_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL
)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


class RExPlanError(ValueError):
    """Invalid REx plan patch."""


def build_rex_plan_correction_prompt(
    error: str, *, accepted_format: str | None = None
) -> str:
    feedback = (
        "Correction required:\n"
        f"The previous plan response could not be parsed: {error}.\n"
        "In this plan phase, do not call tools, do not answer the task, and do not explain.\n"
        'Return only JSON in this shape: {"steps": [{"step_goal": string, '
        '"execution_mode": "direct|recursive"}], "planning_complete": true|false}'
    )
    if accepted_format is not None:
        feedback = f"{feedback}\n\nAccepted format:\n{accepted_format}"
    return feedback


REX_INTERNAL_MESSAGE_EXTRA = {
    "internal": True,
    "source": "rex",
}

_RExFrameResult = RExFrameResult

_RExGroundPlanResult = RExGroundPlanResult

_RExStepExecutionResult = RExStepExecutionResult


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = _JSON_FENCE_RE.match(text)
    return m.group(1).strip() if m else text


def _json_dict_matches(
    raw: object, *, required_keys: tuple[str, ...] = ()
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    if required_keys and not any(key in raw for key in required_keys):
        return None
    return raw


def _json_loads(
    text: str, *, required_keys: tuple[str, ...] = ()
) -> dict[str, Any] | None:
    def load_candidate(candidate: str) -> dict[str, Any] | None:
        try:
            raw = json.loads(candidate.strip())
        except json.JSONDecodeError:
            return None
        return _json_dict_matches(raw, required_keys=required_keys)

    direct = load_candidate(_strip_json_fence(text))
    if direct is not None:
        return direct

    seen: set[str] = set()
    for match in _JSON_BLOCK_RE.finditer(text):
        candidate = match.group(1).strip()
        if candidate in seen:
            continue
        seen.add(candidate)
        raw = load_candidate(candidate)
        if raw is not None:
            return raw

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            raw, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        matched = _json_dict_matches(raw, required_keys=required_keys)
        if matched is not None:
            return matched
    return None


def _normalize_status(value: object) -> RExStatus:
    return "success" if value == "success" else "failed"


def parse_observe_result(text: str) -> RExObserveResult | None:
    raw = _json_loads(text, required_keys=("ready_for_planning",))
    if raw is None or "ready_for_planning" not in raw:
        return None
    return RExObserveResult.model_validate(raw)


def parse_rex_plan_patch(text: str, *, max_steps: int) -> RExPlanPatch | None:
    raw = _json_loads(text, required_keys=("steps", "planning_complete"))
    if raw is None or "steps" not in raw or "planning_complete" not in raw:
        return None
    prompt_patch = RExPromptPlanPatch.model_validate(raw)
    patch = RExPlanPatch(
        steps=_assign_step_ids(prompt_patch.steps, start_index=1),
        complete=prompt_patch.planning_complete,
    )
    validate_rex_plan_patch([], patch, max_steps=max_steps)
    return patch


def parse_step_result(text: str) -> RExStepResult | None:
    raw = _json_loads(text, required_keys=("status",))
    if raw is None:
        return None
    if "status" not in raw:
        return None
    raw["status"] = _normalize_status(raw.get("status"))
    return RExStepResult.model_validate(raw)


def _without_summary(raw: dict[str, Any]) -> dict[str, Any]:
    clean = dict(raw)
    clean.pop("summary", None)
    return clean


def _normalize_observe_message(message: Message) -> Message:
    raw = _json_loads(message.text, required_keys=("ready_for_planning",))
    if raw is None or "ready_for_planning" not in raw:
        return message
    return Message(
        role=message.role,
        content=[TextBlock(text=json.dumps(_without_summary(raw), ensure_ascii=False))],
        reasoning_content=message.reasoning_content,
    )


def _normalize_step_message(message: Message) -> Message:
    raw = _json_loads(message.text, required_keys=("status",))
    if raw is None or "status" not in raw:
        return message
    return Message(
        role=message.role,
        content=[TextBlock(text=json.dumps(_without_summary(raw), ensure_ascii=False))],
        reasoning_content=message.reasoning_content,
    )


def _assign_step_ids(steps: list[RExPromptStep], *, start_index: int) -> list[RExStep]:
    return [
        RExStep(
            id=f"step_{start_index + index}",
            task=step.step_goal,
            kind=step.execution_mode,
        )
        for index, step in enumerate(steps)
    ]


def validate_rex_plan_patch(
    existing_steps: list[RExStep],
    patch: RExPlanPatch,
    *,
    max_steps: int,
) -> None:
    if len(existing_steps) + len(patch.steps) > max_steps:
        raise RExPlanError(
            f"REx has {len(existing_steps) + len(patch.steps)} steps, max is {max_steps}"
        )

    seen = {step.id for step in existing_steps}
    patch_ids: set[str] = set()
    for step in patch.steps:
        if not step.id.strip():
            raise RExPlanError("step id cannot be empty")
        if step.id in seen or step.id in patch_ids:
            raise RExPlanError(f"duplicate step id: {step.id}")
        patch_ids.add(step.id)


_provider_chat_with_timeout = provider_chat_with_timeout

_safe_tool_result = safe_tool_result


REX_MODEL_TO_INTERNAL_EXECUTION_MODE = {
    "execute": "direct",
    "decompose": "recursive",
    # Accept previously recorded responses during migrations and replay.
    "direct": "direct",
    "recursive": "recursive",
}

_GROUND_PLAN_TOOL_REMINDER = """If more evidence is required for planning, call a tool.
Otherwise return ONLY:
{"thinking": string, "steps": [...], "planning_complete": boolean}"""


def _normalize_model_execution_modes(
    raw: dict[str, Any], *, steps_key: str
) -> dict[str, Any]:
    normalized = dict(raw)
    steps = raw.get(steps_key)
    if not isinstance(steps, list):
        return normalized

    normalized_steps: list[Any] = []
    for step in steps:
        if not isinstance(step, dict):
            normalized_steps.append(step)
            continue
        normalized_step = dict(step)
        mode = normalized_step.get("execution_mode")
        if mode in REX_MODEL_TO_INTERNAL_EXECUTION_MODE:
            normalized_step["execution_mode"] = REX_MODEL_TO_INTERNAL_EXECUTION_MODE[
                mode
            ]
        normalized_steps.append(normalized_step)
    normalized[steps_key] = normalized_steps
    return normalized


class RExBatchPromptPlanPatch(BaseModel):
    """Wire protocol for a rolling-horizon planning response."""

    model_config = ConfigDict(extra="forbid")

    # Keep the default for compatibility with previously recorded responses.
    thinking: str = ""
    steps: list[RExPromptStep]
    planning_complete: bool


@dataclass
class _ParsedRExBatchPlan:
    patch: RExPlanPatch
    thinking: str


_RExPlanBatchResult = RExPlanBatchResult

_RExReplanContext = RExReplanContext


def parse_rex_batch_plan_patch(
    text: str, *, max_steps: int
) -> _ParsedRExBatchPlan | None:
    """Parse a batched plan and its frame-local planning memory."""
    raw = _json_loads(
        text,
        required_keys=("thinking", "steps", "planning_complete"),
    )
    if raw is None or "steps" not in raw or "planning_complete" not in raw:
        return None
    raw = _normalize_model_execution_modes(raw, steps_key="steps")
    prompt_patch = RExBatchPromptPlanPatch.model_validate(raw)
    patch = RExPlanPatch(
        steps=_assign_step_ids(prompt_patch.steps, start_index=1),
        complete=prompt_patch.planning_complete,
    )
    validate_rex_plan_patch([], patch, max_steps=max_steps)
    return _ParsedRExBatchPlan(
        patch=patch,
        thinking=prompt_patch.thinking.strip(),
    )


def build_rex_batch_plan_correction_prompt(
    error: str,
    *,
    max_batch_steps: int,
    remaining_step_budget: int,
    depth: int,
    max_depth: int,
) -> str:
    available = min(max_batch_steps, remaining_step_budget)
    return (
        "Correction required:\n"
        f"The previous plan response could not be accepted: {error}.\n"
        "In this planning phase, do not answer the task and do not explain.\n"
        'Return only JSON in this shape: {"thinking": string, "steps": '
        '[{"step_goal": string, '
        '"execution_mode": "execute|decompose"}], "planning_complete": true|false}\n'
        f"{_depth_context(depth=depth, max_depth=max_depth)}\n"
        f"{_execution_mode_guidance(depth=depth, max_depth=max_depth)}\n"
        'Set "thinking" to a concise, self-contained planning memo for the '
        "next batch.\n"
        "If planning_complete is false, return at least one step."
    )


class RExHighConfidencePlanError(RExPlanError):
    """Invalid high-confidence planning response."""


REX_DIRECT_STEP_FEEDBACK = """\
This is the latest tool result.
If the step is complete, return ONLY {"status":"success"}.
If it cannot proceed, return ONLY {"status":"failed"}.
Otherwise, call only the tool(s) needed for specific missing evidence."""

# A tiny configured value would compress every recursive return, even when a
# deeper subtree contains only a few planning messages.  Keep such subtrees
# intact so an ancestor can coalesce them into one compression request.
_MIN_TREE_COMPRESSION_TRIGGER_TOKENS = 3072


@dataclass(frozen=True)
class _RExFrameCompressionState:
    frame_id: str
    depth: int
    goal: str


@dataclass(frozen=True)
class _RExStepCompressionState:
    frame_id: str
    frame_goal: str
    depth: int
    step: RExStep
    steps: list[RExStep]
    completed: dict[str, RExStepResult]


@dataclass(frozen=True)
class _RExCompressionResult:
    checkpoint_text: str
    call: RExRunResult
    diagnostics: dict[str, Any]
    fallback_reason: str | None
    fallback_code: str | None
    prepared: PreparedCompression
    raw_response: Message | None
    started_at: str
    completed_at: str
    duration_s: float
    error_type: str | None = None
    error_message: str | None = None


def validate_rex_high_confidence_plan_patch(
    existing_steps: list[RExStep],
    patch: RExPlanPatch,
    *,
    max_steps: int,
    planned_step_count: int,
    planning_schedule: str | None = None,
) -> None:
    """Validate a planning response against the remaining frame budget."""
    if planning_schedule == "one_shot" and not patch.complete:
        raise RExHighConfidencePlanError(
            "one-shot planning requires planning_complete=true"
        )
    if planning_schedule == "single-plan" and len(patch.steps) > 1:
        raise RExHighConfidencePlanError(
            "single-plan planning permits at most one step per batch"
        )
    if not patch.complete and not patch.steps:
        raise RExHighConfidencePlanError(
            "planning_complete=false requires at least one step"
        )
    if planned_step_count + len(patch.steps) > max_steps:
        raise RExHighConfidencePlanError(
            "planning response exceeds the remaining frame step budget"
        )
    validate_rex_plan_patch(existing_steps, patch, max_steps=max_steps)
    if any(not step.task.strip() for step in patch.steps):
        raise RExHighConfidencePlanError("step_goal cannot be empty")


def build_rex_high_confidence_plan_correction_prompt(
    error: str,
    *,
    remaining_step_budget: int,
    depth: int,
    max_depth: int,
    recursive_decomposition_enabled: bool = True,
    planning_schedule: str | None = None,
) -> str:
    """Request a valid plan without exposing execution batching metadata."""
    del error
    effective_max_depth = max_depth if recursive_decomposition_enabled else depth
    execution_schema = (
        "execute|decompose" if recursive_decomposition_enabled else "execute"
    )
    execution_guidance = (
        _execution_mode_guidance(depth=depth, max_depth=effective_max_depth)
        if recursive_decomposition_enabled
        else (
            '- Use "execute" for every step. Complete each step directly in this '
            "frame.\n- Do not request decomposition or child frames."
        )
    )
    selection_rule = (
        "- Return the next useful planning batch of high-confidence.\n"
        f"The response must fit within the remaining total frame budget of "
        f"{remaining_step_budget} steps. If planning_complete is false, return "
        "at least one step."
    )
    if planning_schedule == "one_shot":
        selection_rule = (
            "- Return the complete remaining plan for this frame now.\n"
            "- Set planning_complete true; no successful-batch planning call will follow.\n"
            f"The plan must fit within the remaining budget of {remaining_step_budget} steps."
        )
    elif planning_schedule == "single-plan":
        selection_rule = (
            "- Return at most one next step. If work remains, return exactly one step.\n"
            "- Set planning_complete false if more work will remain after that step; "
            "set it true only if the step covers all remaining work or no work remains.\n"
            f"The remaining total frame budget is {remaining_step_budget} steps."
        )
    return (
        "Correction required:\n"
        "The previous planning response could not be accepted.\n"
        "In this planning phase, do not answer the task and do not explain.\n"
        'Return only JSON in this shape: {"thinking": string, "steps": '
        '[{"step_goal": string, '
        f'"execution_mode": "{execution_schema}"}}], '
        '"planning_complete": true|false}\n'
        f"{execution_guidance}\n"
        'Set "thinking" to a concise, self-contained planning memo. Consider '
        "the complete remaining work before selecting steps.\n"
        f"{selection_rule}"
    )


class RExRunner:
    """Re-ground after high-confidence plan patches chosen with full context."""

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
        self.runtime = RExRuntime(
            provider=provider,
            messages=self.messages,
            writer=writer,
            trace_id=trace_id,
            writer_lock=self.writer_lock,
            chat_timeout_s=chat_timeout_s,
        )
        self.context_window = max(1, context_window)
        self.allow_needs_user = allow_needs_user
        self.max_steps = max(1, task.environment.rex_max_steps)
        self.max_depth = max(1, task.environment.rex_max_depth)
        self.max_tool_calls = task.environment.rex_max_tool_calls
        self.tool_calls = 0
        self._tool_budget_exhausted = False
        self.max_obs_chars = max(500, task.environment.rex_observation_max_chars)
        self.observe_max_turns = min(
            10,
            max(1, task.environment.rex_observation_max_turns),
        )
        self.todo_lock = threading.Lock()
        self.frame_engine = RExBatchFrameEngine()
        environment = self.task.environment
        self.execution_policy = ResolvedRExPolicy.from_environment(environment)
        self.recursive_decomposition_enabled = (
            environment.rex_recursive_decomposition_enabled
        )
        self.compression_mode = environment.rex_compression_mode
        self.frame_compression_trigger_tokens = max(
            _MIN_TREE_COMPRESSION_TRIGGER_TOKENS,
            environment.rex_frame_compression_trigger_tokens,
        )
        self._frame_compression_stack: list[_RExFrameCompressionState] = []
        self._step_compression_stack: list[_RExStepCompressionState] = []
        self._compression_sequence_number = 0
        self.compression_controller = RExCompressionController(self)
        # This mode has no separate model-visible batch-size constraint.
        self.max_batch_steps = self.max_steps

    def _protocol_response_text(
        self,
        response: Message | None,
        *validators: Callable[[str], Any | None],
    ) -> str:
        return protocol_response_text(response, *validators)

    def run(
        self, *, root_goal: str | None = None, frame_id: str = "root"
    ) -> RExRunResult:
        result = RExRunResult()
        goal = root_goal or self.task.prompt.text
        frame = self._execute_frame(
            frame_id=frame_id,
            goal=goal,
            depth=1,
        )
        self._accumulate(result, frame)
        result.root_status = frame.result.status
        if frame.result.status == "needs_user":
            result.needs_user = True
            result.user_prompt = frame.result.summary
            self._write_event(
                "needs_user",
                frame_id=frame_id,
                artifact={"clarifying_question": frame.result.summary},
                note=frame.result.summary,
            )
            return result
        final = self._final_answer(frame.result, root_goal=goal)
        self._accumulate(result, final)
        result.final_message = final.final_message
        self._write_event(
            "done",
            frame_id=frame_id,
            artifact={
                **frame.result.model_dump(),
                "tool_calls": self.tool_calls,
                "max_tool_calls": self.max_tool_calls,
            },
            note=frame.result.status,
        )
        return result

    def _execute_frame(
        self, *, frame_id: str, goal: str, depth: int
    ) -> _RExFrameResult:
        engine = getattr(self, "frame_engine", RExBatchFrameEngine())
        return engine.run_frame(self, frame_id=frame_id, goal=goal, depth=depth)

    def _execute_step_inner(
        self,
        *,
        frame_id: str,
        steps: list[RExStep],
        step: RExStep,
        goal: str,
        completed: dict[str, RExStepResult],
        depth: int,
    ) -> _RExStepExecutionResult:
        self._write_event(
            "step_start",
            frame_id=frame_id,
            step_id=step.id,
            step_kind=step.kind,
            step_task=step.task,
        )
        if step.kind == "recursive" and depth < self.max_depth:
            child_frame_id = f"{frame_id}.{step.id}"
            child = self._execute_frame(
                frame_id=child_frame_id,
                goal=step.task,
                depth=depth + 1,
            )
            result = _RExStepExecutionResult(result=child.result)
            self._accumulate(result, child)
            self._write_event(
                "step_expand",
                frame_id=frame_id,
                step_id=step.id,
                step_kind=step.kind,
                step_task=step.task,
                artifact=child.result.model_dump(),
            )
            self._write_event(
                "step_done",
                frame_id=frame_id,
                step_id=step.id,
                step_kind=step.kind,
                step_task=step.task,
                artifact=child.result.model_dump(),
            )
            return result

        direct = self._run_direct_step(
            frame_id=frame_id,
            steps=steps,
            step=step,
            goal=goal,
            completed=completed,
            forced_direct=step.kind == "recursive",
        )
        self._write_event(
            "step_done",
            frame_id=frame_id,
            step_id=step.id,
            step_kind=step.kind,
            step_task=step.task,
            artifact=direct.result.model_dump(),
        )
        return direct

    def _execute_step(
        self,
        *,
        frame_id: str,
        steps: list[RExStep],
        step: RExStep,
        goal: str,
        completed: dict[str, RExStepResult],
        depth: int,
    ) -> _RExStepExecutionResult:
        step_state = _RExStepCompressionState(
            frame_id=frame_id,
            frame_goal=goal,
            depth=depth,
            step=step,
            steps=steps,
            completed=completed,
        )
        self._step_compression_stack.append(step_state)
        controller = getattr(self, "compression_controller", None)
        if controller is not None:
            controller.step_started(
                frame_id=frame_id, step=step, goal=goal, depth=depth
            )
        result: _RExStepExecutionResult | None = None
        try:
            result = self._execute_step_inner(
                frame_id=frame_id,
                steps=steps,
                step=step,
                goal=goal,
                completed=completed,
                depth=depth,
            )
            return result
        finally:
            if controller is not None:
                controller.step_finished(
                    frame_id=frame_id,
                    step=step,
                    result=result.result if result is not None else None,
                )
            self._step_compression_stack.pop()

    def _run_direct_step(
        self,
        *,
        frame_id: str,
        steps: list[RExStep],
        step: RExStep,
        goal: str,
        completed: dict[str, RExStepResult],
        forced_direct: bool,
    ) -> _RExStepExecutionResult:
        prompt = Message(
            role="user",
            content=[
                TextBlock(
                    text=self._step_prompt(
                        goal=goal,
                        steps=steps,
                        step=step,
                        completed=completed,
                        forced_direct=forced_direct,
                    )
                )
            ],
        )
        self._append_message(prompt)
        aggregate = _RExStepExecutionResult(
            result=RExStepResult(status="failed", summary="Step did not complete.")
        )
        reasoning_format_retried = False
        # origin is 4 steps
        for _ in range(20):
            chat = self._chat_current(
                tools=self.tools,
                focus=f"step task: {step.task[:120]}",
                normalize_response=_normalize_step_message,
            )
            self._accumulate(aggregate, chat)
            if chat.timed_out:
                aggregate.timed_out = True
                aggregate.result = RExStepResult(
                    status="failed",
                    summary="Step timed out while calling the model.",
                )
                return aggregate
            response = chat.final_message
            if response is None:
                continue
            parsed = parse_step_result(
                self._protocol_response_text(response, parse_step_result)
            )
            if parsed is not None:
                aggregate.result = parsed
                return aggregate
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                if reasoning_format_retried:
                    aggregate.result = RExStepResult(
                        status="failed",
                        summary=(
                            "Step returned no visible tool call or valid status "
                            "JSON after one correction."
                        ),
                    )
                    return aggregate
                reasoning_format_retried = True
                self._append_message(
                    Message(
                        role="user",
                        content=[
                            TextBlock(
                                text=(
                                    "No valid step response was returned. Call a "
                                    "required tool, or return only status JSON in "
                                    'assistant content: {"status":"success"} or '
                                    '{"status":"failed"}.'
                                )
                            )
                        ],
                    )
                )
                continue
            tool_msg, tool_time, _ = self._dispatch_tools(
                tool_uses, frame_id=frame_id, step=step
            )
            aggregate.tool_time_s += tool_time
            self._append_message(tool_msg)
            if self._tool_budget_exhausted:
                aggregate.result = self._tool_budget_failure()
                return aggregate

        aggregate.result = RExStepResult(
            status="failed",
            summary="Step exceeded the REx step turn limit.",
        )
        return aggregate

    def _tool_budget_failure(self) -> RExStepResult:
        return RExStepResult(
            status="failed",
            summary=f"Global tool-call budget exhausted ({self.tool_calls}/{self.max_tool_calls}).",
        )

    def _dispatch_step_tools(
        self,
        tool_uses: list[ToolUseBlock],
        *,
        frame_id: str,
        step: RExStep | None = None,
    ) -> tuple[Message, float, str]:
        result_blocks: list[ToolResultBlock] = []
        total_tool_time = 0.0
        observation_parts: list[str] = []
        dispatcher = ToolDispatcher(self.endpoints)
        try:
            for tool_use in tool_uses:
                if self.tool_calls >= self.max_tool_calls:
                    text = self._tool_budget_failure().summary + " This tool was not executed."
                    result_blocks.append(_safe_tool_result(tool_use, text, is_error=True))
                    observation_parts.append(text)
                    if not self._tool_budget_exhausted:
                        self._write_event(
                            "tool_budget_exhausted",
                            frame_id=frame_id,
                            step_id=step.id if step is not None else None,
                            artifact={
                                "tool_calls": self.tool_calls,
                                "max_tool_calls": self.max_tool_calls,
                                "blocked_tool_name": tool_use.name,
                                "blocked_tool_use_id": tool_use.id,
                            },
                            note=text,
                        )
                    self._tool_budget_exhausted = True
                    continue
                self.tool_calls += 1
                if tool_use.name == "todo" and self.todo_mgr is not None:
                    with self.todo_lock:
                        text = self.todo_mgr.update(tool_use.input.get("items", []))
                    result_blocks.append(_safe_tool_result(tool_use, text))
                    observation_parts.append(text)
                    self._write_event(
                        "observe_tool",
                        frame_id=frame_id,
                        step_id=step.id if step is not None else None,
                        step_kind=step.kind if step is not None else None,
                        step_task=step.task if step is not None else None,
                        artifact={
                            "tool_name": tool_use.name,
                            "request_body": tool_use.input,
                            "response_body": {"result": text},
                            "response_status": 200,
                        },
                    )
                    continue
                result, event = dispatcher.dispatch(tool_use, self.trace_id)[:2]
                total_tool_time += event.latency_ms / 1000.0
                result_blocks.append(result)
                with self.writer_lock:
                    self.writer.write_event(event)
                observation_parts.append(self._tool_result_text(result))
                self._write_event(
                    "observe_tool",
                    frame_id=frame_id,
                    step_id=step.id if step is not None else None,
                    step_kind=step.kind if step is not None else None,
                    step_task=step.task if step is not None else None,
                    artifact={
                        "tool_name": event.tool_name,
                        "request_body": event.request_body,
                        "response_body": event.response_body,
                        "response_status": event.response_status,
                    },
                )
        finally:
            dispatcher.close()
        observation = "\n\n".join(observation_parts)
        if len(observation) > self.max_obs_chars:
            observation = (
                observation[: self.max_obs_chars] + "\n[observation truncated]"
            )
        return Message(role="user", content=result_blocks), total_tool_time, observation

    def _dispatch_tools(
        self,
        tool_uses: list[ToolUseBlock],
        *,
        frame_id: str,
        step: RExStep | None = None,
    ) -> tuple[Message, float, str]:
        tool_message, tool_time, observation = self._dispatch_step_tools(
            tool_uses,
            frame_id=frame_id,
            step=step,
        )
        if step is not None:
            tool_message.content.append(TextBlock(text=REX_DIRECT_STEP_FEEDBACK))
        return tool_message, tool_time, observation

    def _chat_current(
        self,
        *,
        tools: list[ToolSpec] | None,
        focus: str | None = None,
        internal_response: bool = True,
        normalize_response: Any | None = None,
    ) -> RExRunResult:
        out = RExRunResult()
        controller = getattr(self, "compression_controller", None)
        if controller is not None:
            controller.before_model_call(focus=focus)
        self._compact_messages_if_needed(focus=focus)
        t0 = time.monotonic()
        chat_result = self.runtime.chat(self.messages, tools)
        out.model_time_s += time.monotonic() - t0
        if chat_result is None:
            out.timed_out = True
            out.timeout_type = "provider_chat"
            out.timeout_seconds = self.chat_timeout_s
            return out
        response, usage = chat_result
        if normalize_response is not None:
            response = normalize_response(response)
        has_content = any(
            block.type != "text" or bool(block.text.strip())
            for block in response.content
        )
        has_reasoning = bool(
            response.reasoning_content and response.reasoning_content.strip()
        )
        if response.role == "assistant" and not has_content and not has_reasoning:
            raise RuntimeError(
                "Provider returned empty assistant completion; refusing to append it "
                "to conversation history"
            )
        out.final_message = response
        out.usage = usage
        out.turns = 1
        self._append_message(response, usage=usage, internal=internal_response)
        return out

    def _final_answer(
        self, root_result: RExStepResult, *, root_goal: str
    ) -> RExRunResult:
        prompt = Message(
            role="user",
            content=[
                TextBlock(text=self._final_prompt(root_result, root_goal=root_goal))
            ],
        )
        self._append_message(prompt)
        return self._chat_current(
            tools=None,
            focus="final answer",
            internal_response=False,
        )

    def _derive_frame_status(self, completed: dict[str, RExStepResult]) -> RExStatus:
        statuses = [result.status for result in completed.values()]
        for status in ("failed", "needs_user"):
            if status in statuses:
                return status
        return "success"

    def _fallback_observation(self, observations: list[str]) -> str:
        text = "\n\n".join(obs for obs in observations if obs).strip()
        if not text:
            text = "No additional environment observations were required."
        if len(text) > self.max_obs_chars:
            text = text[: self.max_obs_chars] + "\n[observation truncated]"
        return text

    def _tool_result_text(self, message: ToolResultBlock) -> str:
        text = "\n".join(block.text for block in message.content)
        return f"[tool error]\n{text}" if message.is_error else text

    def _render_results(self, completed: dict[str, RExStepResult]) -> str:
        data = {step_id: result.model_dump() for step_id, result in completed.items()}
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _render_prompt_steps(
        self,
        steps: list[RExStep],
        completed: dict[str, RExStepResult],
    ) -> str:
        data = [
            {
                "step_goal": step.task,
                "status": (
                    completed[step.id].status if step.id in completed else "pending"
                ),
            }
            for step in steps
        ]
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _render_step_results_for_prompt(
        self,
        steps: list[RExStep],
        completed: dict[str, RExStepResult],
    ) -> str:
        rows = []
        for step in steps:
            result = completed.get(step.id)
            if result is None:
                continue
            row: dict[str, str] = {
                "step_goal": step.task,
                "status": result.status,
            }
            if result.summary:
                row["summary"] = result.summary
            rows.append(row)
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def _renumber_steps(self, steps: list[RExStep], *, start_index: int) -> None:
        for index, step in enumerate(steps):
            step.id = f"step_{start_index + index}"

    def _execution_snapshot(
        self,
        steps: list[RExStep],
        completed: dict[str, RExStepResult],
    ) -> dict[str, Any]:
        return {
            "steps": [self._dump_step_protocol(step) for step in steps],
            "results": {
                step_id: {"status": result.status, "summary": result.summary[:200]}
                for step_id, result in completed.items()
            },
        }

    def _dump_step_protocol(self, step: RExStep) -> dict[str, Any]:
        return {
            "id": step.id,
            "step_goal": step.task,
            "execution_mode": step.kind,
        }

    def _dump_plan_patch_protocol(self, patch: RExPlanPatch) -> dict[str, Any]:
        return {
            "steps": [self._dump_step_protocol(step) for step in patch.steps],
            "planning_complete": patch.complete,
        }

    def _append_message(
        self,
        message: Message,
        usage: TokenUsage | None = None,
        *,
        internal: bool = True,
    ) -> None:
        self.runtime.append_message(message, usage=usage, internal=internal)

    def _write_message(
        self,
        message: Message,
        usage: TokenUsage | None = None,
        *,
        internal: bool = True,
    ) -> None:
        self.runtime.write_message(message, usage=usage, internal=internal)

    def _compact_messages_if_needed(self, *, focus: str | None = None) -> None:
        del focus

    def _write_event(
        self,
        event: Any,
        *,
        frame_id: str = "root",
        step_id: str | None = None,
        step_kind: str | None = None,
        step_task: str | None = None,
        artifact: dict[str, Any] | None = None,
        execution_snapshot: dict[str, Any] | None = None,
        note: str = "",
    ) -> None:
        self.runtime.write_event(
            event,
            frame_id=frame_id,
            step_id=step_id,
            step_kind=step_kind,
            step_task=step_task,
            artifact=artifact,
            execution_snapshot=execution_snapshot,
            note=note,
        )

    def _accumulate(
        self,
        target: RExRunResult | _RExFrameResult | _RExStepExecutionResult,
        source: RExRunResult | _RExFrameResult | _RExStepExecutionResult,
    ) -> None:
        self.runtime.accumulate(target, source)

    def _step_prompt(
        self,
        *,
        goal: str,
        steps: list[RExStep],
        step: RExStep,
        completed: dict[str, RExStepResult],
        forced_direct: bool,
    ) -> str:
        forced = (
            "\nThis recursive step reached the maximum recursion depth, so execute it directly."
            if forced_direct
            else ""
        )
        return f"""Direct subtask execution.{forced}

Use the root goal as background. Complete only the current step.

Root goal:
{goal}

Current step:
{step.task}

Prior completed step results:
{self._render_step_results_for_prompt(steps, completed)}

If a tool call is needed, call the appropriate tool.
When complete, return ONLY JSON:
{{"status": "success|failed"}}

Rules:
- Return "success" only when the current step is complete.
- Return "failed" when the step is incomplete, cannot proceed, or cannot be verified."""

    def _final_prompt(self, root_result: RExStepResult, *, root_goal: str) -> str:
        return f"""Use the conversation history and task status to answer the original user request.

Original user request:
{root_goal}

Task status:
{root_result.status}

Do not return planning JSON. Provide the final answer requested by the user."""

    def _horizontal_rolling_enabled(self) -> bool:
        return True

    def _normalize_planned_steps(self, steps: list[RExStep], *, depth: int) -> None:
        del depth
        if self._vertical_recursion_enabled():
            return
        for step in steps:
            step.kind = "direct"

    def _planning_policy_metadata(self) -> dict[str, Any]:
        return {
            "planning_schedule": self.execution_policy.planning_schedule,
            "horizontal_policy": ("rolling"),
            "vertical_policy": (
                "recursive" if self._vertical_recursion_enabled() else "flat"
            ),
        }

    def _depth_context_for_prompt(self, *, depth: int) -> str:
        return _depth_context(
            depth=depth,
            max_depth=self._effective_max_depth(depth=depth),
        )

    def _execution_mode_guidance_for_prompt(self, *, depth: int) -> str:
        if not self._vertical_recursion_enabled():
            return (
                '- Use "execute" for every step. Complete each step directly in '
                "this frame.\n- Do not request decomposition or child frames."
            )
        return _execution_mode_guidance(
            depth=depth,
            max_depth=self._effective_max_depth(depth=depth),
        )

    def _execution_mode_schema_for_prompt(self) -> str:
        return "execute|decompose" if self._vertical_recursion_enabled() else "execute"

    def _run_batched_frame(
        self,
        *,
        frame_id: str,
        goal: str,
        depth: int,
    ) -> _RExFrameResult:
        aggregate = _RExFrameResult(
            result=RExStepResult(
                status="failed",
                summary="Frame did not complete.",
            )
        )
        steps: list[RExStep] = []
        completed: dict[str, RExStepResult] = {}
        planned_step_count = 0
        next_step_index = 1
        batch_index = 0
        planning_thinking = ""
        replan_context: _RExReplanContext | None = None
        # A fully successful batch restores the local recovery allowance.
        recovery_attempted = False

        while True:
            if self._tool_budget_exhausted:
                aggregate.result = self._tool_budget_failure()
                return aggregate
            if planned_step_count >= self.max_steps:
                aggregate.result = RExStepResult(
                    status="failed",
                    summary=(
                        "Batched planning exhausted the frame step budget before "
                        "planning completed."
                    ),
                )
                return aggregate

            batch_index += 1
            active_replan_context = replan_context
            plan_outcome = self._plan_next_batch(
                frame_id=frame_id,
                goal=goal,
                steps=steps,
                completed=completed,
                batch_index=batch_index,
                planned_step_count=planned_step_count,
                next_step_index=next_step_index,
                depth=depth,
                previous_thinking=planning_thinking,
                replan_context=active_replan_context,
            )
            self._accumulate(aggregate, plan_outcome.result)
            if plan_outcome.result.needs_user:
                aggregate.result = RExStepResult(
                    status="needs_user",
                    summary=plan_outcome.result.user_prompt
                    or "Please provide the missing information.",
                )
                return aggregate
            if plan_outcome.result.timed_out:
                aggregate.result = RExStepResult(
                    status="failed",
                    summary="Batched planning timed out while calling the model.",
                )
                return aggregate

            patch = plan_outcome.patch
            if patch is None:
                if planned_step_count > 0:
                    aggregate.result = RExStepResult(
                        status="failed",
                        summary="Batched replanning did not produce a valid batch.",
                    )
                    return aggregate
                fallback = RExStep(
                    id=f"step_{next_step_index}", task=goal, kind="direct"
                )
                patch = RExPlanPatch(steps=[fallback], complete=True)
                self._write_event(
                    "plan_patch",
                    frame_id=frame_id,
                    artifact={
                        **self._dump_plan_patch_protocol(patch),
                        "batch_index": batch_index,
                        "batch_size": 1,
                        "planning_mode": "rolling",
                        **self._planning_policy_metadata(),
                    },
                    execution_snapshot=self._execution_snapshot(
                        [*steps, fallback], completed
                    ),
                    note="fallback single direct step",
                )
            else:
                # Legacy responses may omit thinking; keep the last useful memo.
                planning_thinking = plan_outcome.thinking or planning_thinking
                if active_replan_context is not None:
                    # A valid recovery patch consumes the one-shot context.
                    replan_context = None

            steps.extend(patch.steps)
            planned_step_count += len(patch.steps)
            next_step_index += len(patch.steps)

            batch_failed = False
            for offset, step in enumerate(patch.steps):
                step_result = self._execute_step(
                    frame_id=frame_id,
                    steps=steps,
                    step=step,
                    goal=goal,
                    completed=completed,
                    depth=depth,
                )
                self._accumulate(aggregate, step_result)
                completed[step.id] = step_result.result
                if self._tool_budget_exhausted:
                    aggregate.result = self._tool_budget_failure()
                    return aggregate
                if step_result.result.status == "success":
                    continue

                dropped = patch.steps[offset + 1 :]
                dropped_ids = {item.id for item in dropped}
                if dropped_ids:
                    steps[:] = [item for item in steps if item.id not in dropped_ids]
                self._write_event(
                    "plan_batch_abort",
                    frame_id=frame_id,
                    step_id=step.id,
                    step_kind=step.kind,
                    step_task=step.task,
                    artifact={
                        "batch_index": batch_index,
                        "failed_step_id": step.id,
                        "failed_status": step_result.result.status,
                        "dropped_steps": [
                            self._dump_step_protocol(item) for item in dropped
                        ],
                        "remaining_step_budget": self.max_steps - planned_step_count,
                        "recovery_exhausted": recovery_attempted,
                    },
                    execution_snapshot=self._execution_snapshot(steps, completed),
                )
                if step_result.result.status == "needs_user":
                    aggregate.result = step_result.result
                    return aggregate
                if recovery_attempted:
                    # Return the failure to the parent instead of retrying here.
                    aggregate.result = step_result.result
                    return aggregate
                recovery_attempted = True
                replan_context = _RExReplanContext(
                    failed_step_id=step.id,
                    failed_step_goal=step.task,
                    failed_status=step_result.result.status,
                    failure_summary=step_result.result.summary,
                )
                batch_failed = True
                break

            if batch_failed:
                continue
            recovery_attempted = False
            if not patch.complete:
                continue

            aggregate.result = RExStepResult(
                # Rolling recovery supersedes the original failed attempt.
                # Keep it in completed for the planner and execution trace.
                status=(
                    "success"
                    if self._horizontal_rolling_enabled()
                    else self._derive_frame_status(completed)
                ),
                summary="",
            )
            return aggregate

    def _execute_batched_frame(
        self,
        *,
        frame_id: str,
        goal: str,
        depth: int,
    ) -> _RExFrameResult:
        start_index = len(self.messages)
        self._frame_compression_stack.append(
            _RExFrameCompressionState(frame_id=frame_id, depth=depth, goal=goal)
        )
        controller = getattr(self, "compression_controller", None)
        if controller is not None:
            controller.frame_started(frame_id=frame_id, goal=goal, depth=depth)
        result: _RExFrameResult | None = None
        try:
            result = self._run_batched_frame(frame_id=frame_id, goal=goal, depth=depth)
            if (
                depth > 1
                and self.compression_mode == "tree"
                and result.result.status == "success"
            ):
                root_goal, ancestor_goals, consumer_goal = self._compression_ancestry()
                parent_step_state = (
                    self._step_compression_stack[-1]
                    if self._step_compression_stack
                    else None
                )
                if parent_step_state is None:
                    completed_siblings: tuple[CompressionSibling, ...] = ()
                    pending_siblings: tuple[CompressionSibling, ...] = ()
                else:
                    completed_siblings, pending_siblings = self._compression_siblings(
                        parent_step_state
                    )
                scope = CompressionScope(
                    kind="frame",
                    frame_id=frame_id,
                    goal=goal,
                    status=result.result.status,
                    root_goal=root_goal,
                    ancestor_goals=ancestor_goals,
                    consumer_goal=consumer_goal,
                    depth=depth,
                    execution_mode="decomposed_frame",
                    completed_siblings=completed_siblings,
                    pending_siblings=pending_siblings,
                )
                compression = self._compress_execution_scope(
                    scope=scope,
                    start_index=start_index,
                    trigger_tokens=self.frame_compression_trigger_tokens,
                    result=result.result,
                )
                self._accumulate(result, compression)
            return result
        finally:
            if controller is not None:
                controller.frame_finished(
                    frame_id=frame_id, goal=goal, depth=depth, result=result
                )
            self._frame_compression_stack.pop()

    def _plan_next_batch(
        self,
        *,
        frame_id: str,
        goal: str,
        steps: list[RExStep],
        completed: dict[str, RExStepResult],
        batch_index: int,
        planned_step_count: int,
        next_step_index: int,
        depth: int,
        previous_thinking: str = "",
        replan_context: _RExReplanContext | None = None,
    ) -> _RExPlanBatchResult:
        aggregate = RExRunResult()
        remaining_budget = self.max_steps - planned_step_count
        planning_mode = (
            "recovery"
            if replan_context is not None
            else self.execution_policy.planning_schedule
        )
        plan_trace_context: dict[str, Any] = {
            "planning_mode": planning_mode,
            **self._planning_policy_metadata(),
        }
        if replan_context is not None:
            plan_trace_context["recovery_for_step_id"] = replan_context.failed_step_id
        self._write_event(
            "observe_start",
            frame_id=frame_id,
            artifact={
                "batch_index": batch_index,
                "remaining_step_budget": remaining_budget,
                **self._planning_policy_metadata(),
            },
            note=goal,
        )
        if replan_context is None:
            prompt_text = self._build_plan_batch_prompt(
                goal=goal,
                steps=steps,
                completed=completed,
                batch_index=batch_index,
                remaining_step_budget=remaining_budget,
                depth=depth,
                previous_thinking=previous_thinking,
            )
        else:
            prompt_text = self._build_replan_prompt(
                goal=goal,
                steps=steps,
                completed=completed,
                batch_index=batch_index,
                remaining_step_budget=remaining_budget,
                depth=depth,
                previous_thinking=previous_thinking,
                replan_context=replan_context,
            )

        self._append_message(
            Message(role="user", content=[TextBlock(text=prompt_text)])
        )
        observations: list[str] = []
        last_error = "planning response did not contain valid JSON"

        for _ in range(self.observe_max_turns):
            chat = self._chat_current(
                # Ground-and-plan is intentionally planning-only.  Do not
                # expose execution tools here: tool observations belong to the
                # step execution phase, after a plan patch has been accepted.
                tools=None,
                focus=f"{planning_mode} planning call {batch_index}",
            )
            self._accumulate(aggregate, chat)
            if chat.timed_out:
                return _RExPlanBatchResult(result=aggregate)
            response = chat.final_message
            if response is None:
                continue
            response_text = self._protocol_response_text(
                response,
                lambda value: parse_rex_batch_plan_patch(
                    value, max_steps=self.max_steps
                ),
            )

            try:
                parsed_plan = parse_rex_batch_plan_patch(
                    response_text, max_steps=self.max_steps
                )
                if parsed_plan is not None:
                    patch = parsed_plan.patch
                    self._renumber_steps(patch.steps, start_index=next_step_index)
                    self._normalize_planned_steps(patch.steps, depth=depth)
                    validate_rex_high_confidence_plan_patch(
                        steps,
                        patch,
                        max_steps=self.max_steps,
                        planned_step_count=planned_step_count,
                        planning_schedule=self.execution_policy.planning_schedule,
                    )
            except (RExPlanError, ValueError) as exc:
                last_error = str(exc)
                break
            if parsed_plan is not None:
                self._write_event(
                    "observe_done",
                    frame_id=frame_id,
                    artifact={
                        "ready_for_planning": True,
                        "batch_index": batch_index,
                    },
                )
                self._write_event(
                    "plan_patch",
                    frame_id=frame_id,
                    artifact={
                        **self._dump_plan_patch_protocol(patch),
                        "thinking": parsed_plan.thinking,
                        "batch_index": batch_index,
                        "batch_size": len(patch.steps),
                        **plan_trace_context,
                    },
                    execution_snapshot=self._execution_snapshot(
                        [*steps, *patch.steps], completed
                    ),
                )
                return _RExPlanBatchResult(
                    result=aggregate,
                    patch=patch,
                    thinking=parsed_plan.thinking,
                )

            # Tools are not available in this phase, so a non-JSON response is
            # handled as a planning-format failure and repaired below.
            observations.append(response.text)
            break

        self._write_event(
            "observe_done",
            frame_id=frame_id,
            artifact={
                "ready_for_planning": True,
                "observation": self._fallback_observation(observations),
                "batch_index": batch_index,
            },
            note="fallback observation",
        )
        self._append_message(
            Message(
                role="user",
                content=[
                    TextBlock(
                        text=build_rex_high_confidence_plan_correction_prompt(
                            last_error,
                            remaining_step_budget=remaining_budget,
                            depth=depth,
                            max_depth=self.max_depth,
                            recursive_decomposition_enabled=self._vertical_recursion_enabled(),
                            planning_schedule=self.execution_policy.planning_schedule,
                        )
                    )
                ],
            )
        )
        repair = self._chat_current(
            tools=None,
            focus=f"planning format repair {batch_index}",
        )
        self._accumulate(aggregate, repair)
        text = self._protocol_response_text(
            repair.final_message,
            lambda value: parse_rex_batch_plan_patch(value, max_steps=self.max_steps),
        )
        try:
            parsed_plan = parse_rex_batch_plan_patch(text, max_steps=self.max_steps)
            if parsed_plan is None:
                raise RExPlanError("plan response did not contain JSON steps")
            patch = parsed_plan.patch
            self._renumber_steps(patch.steps, start_index=next_step_index)
            self._normalize_planned_steps(patch.steps, depth=depth)
            validate_rex_high_confidence_plan_patch(
                steps,
                patch,
                max_steps=self.max_steps,
                planned_step_count=planned_step_count,
                planning_schedule=self.execution_policy.planning_schedule,
            )
        except (RExPlanError, ValueError) as exc:
            self._write_event(
                "plan_patch",
                frame_id=frame_id,
                artifact={
                    "batch_index": batch_index,
                    "batch_size": 0,
                    **plan_trace_context,
                },
                execution_snapshot=self._execution_snapshot(steps, completed),
                note=str(exc),
            )
            return _RExPlanBatchResult(result=aggregate)

        self._write_event(
            "plan_patch",
            frame_id=frame_id,
            artifact={
                **self._dump_plan_patch_protocol(patch),
                "thinking": parsed_plan.thinking,
                "batch_index": batch_index,
                "batch_size": len(patch.steps),
                **plan_trace_context,
            },
            execution_snapshot=self._execution_snapshot(
                [*steps, *patch.steps], completed
            ),
        )
        return _RExPlanBatchResult(
            result=aggregate,
            patch=patch,
            thinking=parsed_plan.thinking,
        )

    def _build_plan_batch_prompt(
        self,
        *,
        goal: str,
        steps: list[RExStep],
        completed: dict[str, RExStepResult],
        batch_index: int,
        remaining_step_budget: int,
        depth: int,
        previous_thinking: str = "",
    ) -> str:
        del batch_index
        if self._planning_schedule() == "one_shot":
            return self._build_one_shot_plan_prompt(
                goal=goal,
                remaining_step_budget=remaining_step_budget,
                depth=depth,
            )
        horizon_rule = (
            "- Return at most one next step. If work remains, return exactly one step."
            if self._planning_schedule() == "single-plan"
            else "- Plan as far ahead as the current information reliably supports. Return one\n"
            "  step, a partial batch, or the full remaining plan as appropriate. Stop before work whose specification materially depends on future observations."
        )
        completion_rule = (
            "- Set planning_complete false if more work will remain after that step; "
            "set it true only if the step covers all remaining work or no work remains."
            if self._planning_schedule() == "single-plan"
            else "- Set planning_complete true only when every Must finish item is in Done and no further work is expected."
        )
        return f"""planning and task decomposition phase.

Use the existing conversation history and recent tool observations to produce a reliable plan.
be called.

Goal:
{goal}

{self._depth_context_for_prompt(depth=depth)}

Return ONLY JSON:
{{"thinking": string, "steps": [{{"step_goal": string,
"execution_mode": "{self._execution_mode_schema_for_prompt()}"}}], "planning_complete": boolean}}

Rules:
{horizon_rule}
{self._execution_mode_guidance_for_prompt(depth=depth)}
- If more work may be needed after the returned steps, set planning_complete false and return at least one step.
{completion_rule}
- Every proposed must make new progress.
{self._decomposition_rule_for_prompt()}"""

    def _build_replan_prompt(
        self,
        *,
        goal: str,
        steps: list[RExStep],
        completed: dict[str, RExStepResult],
        batch_index: int,
        remaining_step_budget: int,
        depth: int,
        previous_thinking: str,
        replan_context: _RExReplanContext,
    ) -> str:
        del batch_index
        failed_step = json.dumps(
            {
                "step_goal": replan_context.failed_step_goal,
                "status": replan_context.failed_status,
                "summary": replan_context.failure_summary,
            },
            ensure_ascii=False,
        )
        selection_rule = (
            "- Return at most one next recovery step. If work remains, return exactly one step.\n"
            "- Set planning_complete false if more work will remain after that step; "
            "set it true only if the step covers all remaining work or no work remains."
            if self._planning_schedule() == "single-plan"
            else "- Return either the complete recovery plan (`planning_complete=true`) for one-shot mode, or the next plan batch for progressive mode."
        )
        return f"""Recovery replanning phase.

Use the conversation's failure evidence to choose a changed, executable
approach. Do not perform task work or answer the user.

Goal:
{goal}

Failed step:
{failed_step}

Return ONLY JSON:
{{"thinking": string, "steps": [{{"step_goal": string,
"execution_mode": "{self._execution_mode_schema_for_prompt()}"}}], "planning_complete": boolean}}

Rules:
- Before selecting steps, consider the complete remaining work for this Goal and how the failure changes it.
- Repair or bypass the failure with a materially changed executable step.
- Do not repeat the failed approach unchanged; name the changed tool, input, or strategy.
{selection_rule}
- If planning_complete is false, return at least one step.
{self._replan_execution_rule_for_prompt()}"""

    def _vertical_recursion_enabled(self) -> bool:
        return getattr(self, "recursive_decomposition_enabled", True)

    def _compression_trace_context(self) -> tuple[str, str | None, str | None]:
        frame_id = (
            self._frame_compression_stack[-1].frame_id
            if self._frame_compression_stack
            else "root"
        )
        if not self._step_compression_stack:
            return frame_id, None, None
        state = self._step_compression_stack[-1]
        return frame_id, state.step.id, state.step.task

    def _compression_ancestry(
        self,
    ) -> tuple[str | None, tuple[str, ...], str | None]:
        if not self._frame_compression_stack:
            return None, (), None
        root_goal = self._frame_compression_stack[0].goal
        consumer_index = len(self._frame_compression_stack) - 2
        if consumer_index < 0:
            return root_goal, (), None
        consumer_goal = self._frame_compression_stack[consumer_index].goal
        ancestor_goals = tuple(
            state.goal for state in self._frame_compression_stack[1:consumer_index]
        )
        return root_goal, ancestor_goals, consumer_goal

    def _compression_siblings(
        self, state: _RExStepCompressionState
    ) -> tuple[tuple[CompressionSibling, ...], tuple[CompressionSibling, ...]]:
        current_index = next(
            (
                index
                for index, candidate in enumerate(state.steps)
                if candidate.id == state.step.id
            ),
            len(state.steps),
        )
        completed_siblings = []
        for candidate in state.steps[:current_index]:
            result = state.completed.get(candidate.id)
            if result is None:
                continue
            completed_siblings.append(
                CompressionSibling(
                    goal=candidate.task,
                    kind=candidate.kind,
                    status=result.status,
                    summary=result.summary or None,
                )
            )
        pending_siblings = tuple(
            CompressionSibling(goal=candidate.task, kind=candidate.kind)
            for candidate in state.steps[current_index + 1 : current_index + 2]
            if candidate.id not in state.completed
        )
        return tuple(completed_siblings), pending_siblings

    def _request_compression_checkpoint(
        self,
        scope: CompressionScope,
        source_context_messages: list[Message],
    ) -> _RExCompressionResult:
        call = RExRunResult()
        max_input_tokens = max(512, int(self.context_window * 0.60))
        compressor_source_messages = filter_runner_control_messages(
            source_context_messages
        )
        prepared = prepare_compression_request(
            scope,
            compressor_source_messages,
            max_input_tokens=max_input_tokens,
            include_task_context=(
                self.task.environment.rex_compression_include_task_context
            ),
        )
        diagnostics: dict[str, Any] = {
            "compressor_input_tokens_estimated": prepared.estimated_tokens,
            "compressor_history_truncated": prepared.history_truncated,
            "compressor_task_context_tokens_estimated": (prepared.task_context_tokens),
            "compressor_task_context_truncated": (prepared.task_context_truncated),
            "compressor_task_context_enabled": (
                self.task.environment.rex_compression_include_task_context
            ),
        }
        started_at = self._utc_now()
        started = time.monotonic()
        try:
            chat_result = provider_chat_with_timeout(
                self.provider,
                prepared.messages,
                None,
                timeout_s=self.chat_timeout_s,
            )
        except Exception as exc:
            duration_s = time.monotonic() - started
            call.model_time_s += duration_s
            reason = f"compressor error: {type(exc).__name__}: {exc}"
            diagnostics["compressor_format_valid"] = False
            return _RExCompressionResult(
                checkpoint_text=deterministic_handoff(reason=reason),
                call=call,
                diagnostics=diagnostics,
                fallback_reason=reason,
                fallback_code="compressor_error",
                prepared=prepared,
                raw_response=None,
                started_at=started_at,
                completed_at=self._utc_now(),
                duration_s=duration_s,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        duration_s = time.monotonic() - started
        call.model_time_s += duration_s
        if chat_result is None:
            reason = "compressor timeout"
            diagnostics["compressor_format_valid"] = False
            return _RExCompressionResult(
                checkpoint_text=deterministic_handoff(reason=reason),
                call=call,
                diagnostics=diagnostics,
                fallback_reason=reason,
                fallback_code="compressor_timeout",
                prepared=prepared,
                raw_response=None,
                started_at=started_at,
                completed_at=self._utc_now(),
                duration_s=duration_s,
                error_type="TimeoutError",
                error_message=reason,
            )
        response, usage = chat_result
        call.usage.input_tokens += usage.input_tokens
        call.usage.output_tokens += usage.output_tokens
        call.turns += 1
        diagnostics.update(
            {
                "compressor_input_tokens_actual": usage.input_tokens,
                "compressor_output_tokens_actual": usage.output_tokens,
            }
        )
        if not response.text.strip():
            reason = "compressor returned no visible text"
            diagnostics["compressor_format_valid"] = False
            return _RExCompressionResult(
                checkpoint_text=deterministic_handoff(reason=reason),
                call=call,
                diagnostics=diagnostics,
                fallback_reason=reason,
                fallback_code="empty_response",
                prepared=prepared,
                raw_response=response,
                started_at=started_at,
                completed_at=self._utc_now(),
                duration_s=duration_s,
                error_type="EmptyResponseError",
                error_message=reason,
            )
        normalized_handoff = normalize_handoff_text(response.text)
        diagnostics["compressor_format_valid"] = handoff_is_valid(normalized_handoff)
        if not diagnostics["compressor_format_valid"]:
            reason = "compressor returned an invalid or oversized handoff"
            return _RExCompressionResult(
                checkpoint_text=deterministic_handoff(reason=reason),
                call=call,
                diagnostics=diagnostics,
                fallback_reason=reason,
                fallback_code="invalid_handoff",
                prepared=prepared,
                raw_response=response,
                started_at=started_at,
                completed_at=self._utc_now(),
                duration_s=duration_s,
                error_type="InvalidHandoffError",
                error_message=reason,
            )
        return _RExCompressionResult(
            checkpoint_text=normalized_handoff,
            call=call,
            diagnostics=diagnostics,
            fallback_reason=None,
            fallback_code=None,
            prepared=prepared,
            raw_response=response,
            started_at=started_at,
            completed_at=self._utc_now(),
            duration_s=duration_s,
        )

    def _next_compression_identity(self) -> tuple[int, str]:
        self._compression_sequence_number += 1
        return self._compression_sequence_number, (
            f"{self.trace_id}-compression-{self._compression_sequence_number}"
        )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _compress_execution_scope(
        self,
        *,
        scope: CompressionScope,
        start_index: int,
        trigger_tokens: int,
        result: RExStepResult,
    ) -> RExRunResult:
        call = RExRunResult()
        if self.compression_mode == "none" or result.status != "success":
            return call
        compression_sequence, compression_id = self._next_compression_identity()
        source_context_messages = [
            message.model_copy(deep=True) for message in self.messages[start_index:]
        ]
        source_context_tokens = estimate_request_tokens(source_context_messages)
        evidence = extract_compression_evidence(source_context_messages)
        inherited = extract_inherited_evidence_ledgers(source_context_messages)
        event = "tree_compress"
        event_context = {
            "frame_id": scope.frame_id,
            "step_id": scope.step_id,
            "step_kind": scope.execution_mode,
            "step_task": None,
        }
        if not source_context_messages or source_context_tokens <= trigger_tokens:
            self._write_event(
                event,
                **event_context,
                artifact={
                    "compression_id": compression_id,
                    "compression_sequence": compression_sequence,
                    "triggered": False,
                    "applied": False,
                    "reason": "below_threshold",
                    "estimated_tokens_before": source_context_tokens,
                    "trigger_tokens": trigger_tokens,
                },
            )
            return call

        compressor = self._request_compression_checkpoint(
            scope, source_context_messages
        )
        call = compressor.call
        checkpoint_text = compressor.checkpoint_text
        compressed_history = render_compressed_history(
            scope,
            checkpoint_text,
            evidence,
            inherited_evidence_ledgers=inherited,
        )
        checkpoint = Message(role="user", content=[TextBlock(text=compressed_history)])
        compressed_context_tokens = estimate_request_tokens([checkpoint])
        applied = compressed_context_tokens < source_context_tokens
        decision_reason = (
            compressor.fallback_code
            if compressor.fallback_code is not None
            else "applied"
            if applied
            else "not_smaller"
        )
        if applied:
            self.messages[start_index:] = [checkpoint]
            result.summary = checkpoint_text

        artifact = {
            "compression_id": compression_id,
            "compression_sequence": compression_sequence,
            "triggered": True,
            "applied": applied,
            "reason": decision_reason,
            "status": scope.status,
            "estimated_tokens_before": source_context_tokens,
            "estimated_tokens_after": compressed_context_tokens,
            "trigger_tokens": trigger_tokens,
            "message_count_before": len(source_context_messages),
            **compressor.diagnostics,
        }
        if compressor.fallback_reason is not None:
            artifact["fallback_reason"] = compressor.fallback_reason
            artifact["fallback_code"] = compressor.fallback_code
            self._write_event(
                "compression_fallback",
                **event_context,
                artifact={"scope": scope.kind, **artifact},
                note=compressor.fallback_reason,
            )
        self._write_event(event, **event_context, artifact=artifact)
        return call

    def _effective_max_depth(self, *, depth: int) -> int:
        return self.max_depth if self._vertical_recursion_enabled() else depth

    def _decomposition_rule_for_prompt(self) -> str:
        if not self._vertical_recursion_enabled():
            return "- Every step must execute directly in this frame."
        return (
            "- A decomposed step must be a strictly narrower child goal; never "
            "delegate a restatement of this frame's Goal."
        )

    def _replan_execution_rule_for_prompt(self) -> str:
        if not self._vertical_recursion_enabled():
            return self._execution_mode_guidance_for_prompt(depth=1)
        return (
            "- Use decompose only for a strictly narrower multi-stage goal; use "
            "execute when no decomposition level remains."
        )

    def _ground_plan_tool_reminder(self) -> str:
        schema = self._execution_mode_schema_for_prompt()
        return (
            "If more evidence is required for planning, call a tool.\n"
            "Otherwise return ONLY:\n"
            '{"thinking": string, "steps": [{"step_goal": string, '
            f'"execution_mode": "{schema}"}}], "planning_complete": boolean}}'
        )

    def _planning_schedule(self) -> str:
        policy = getattr(self, "execution_policy", None)
        if policy is not None:
            return policy.planning_schedule
        return "progressive"

    def _build_one_shot_plan_prompt(
        self,
        *,
        goal: str,
        remaining_step_budget: int,
        depth: int,
    ) -> str:
        return f"""Planning and task decomposition phase.

Use the existing conversation history to produce the complete remaining plan
for this frame. This is a planning-only phase: tools are unavailable and must
not be called.

Goal:
{goal}

{self._depth_context_for_prompt(depth=depth)}

Return ONLY JSON:
{{"thinking": string, "steps": [{{"step_goal": string,
"execution_mode": "{self._execution_mode_schema_for_prompt()}"}}], "planning_complete": true}}

Rules:
- Return the complete remaining executable plan now.
- Set planning_complete true. No successful-batch planning call will follow.
- Make every step useful progress.
- Keep thinking concise and do not answer the task.
{self._execution_mode_guidance_for_prompt(depth=depth)}
{self._decomposition_rule_for_prompt()}"""
