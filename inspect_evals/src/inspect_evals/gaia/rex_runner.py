"""High-confidence recursive planning agent for GAIA."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    Model,
    ModelOutput,
    execute_tools,
    get_model,
)
from inspect_ai.tool import Tool, bash, python, web_browser
from inspect_ai.util import span
from pydantic import BaseModel, ConfigDict, ValidationError

from inspect_evals.gaia.dataset import (
    DEFAULT_INPUT_PROMPT,
    GAIA_FINAL_ANSWER_FORMAT,
)
from inspect_evals.gaia.gaia import DEFAULT_AGENT_INSTRUCTIONS
from inspect_evals.gaia.hierarchical_compact import (
    CompressionBackend,
    CompressionMode,
    CompressionScope,
    extract_compression_evidence,
    extract_inherited_evidence_ledgers,
    handoff_is_valid,
    normalize_handoff_text,
    prepare_compression_request,
    render_compressed_history,
)
from inspect_evals.gaia.hierarchical_compact_audit import (
    COMPRESSION_TEXT_LOG_SCHEMA_VERSION,
    CompressionLogContext,
    dump_messages,
    dump_model_output,
    make_compression_id,
    resolve_compression_log_context,
    write_compressor_text_log,
)

RExStepKind = Literal["execute", "decompose"]
RExStatus = Literal["success", "failed"]
PlanningSchedule = Literal["progressive", "one_shot"]
Ablation = Literal["one_shot", "execute_only", "single_item"]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
REX_AGENT_INSTRUCTIONS = DEFAULT_AGENT_INSTRUCTIONS.replace(
    "use the submit tool to provide your final answer",
    "provide the shortest answer that satisfies the user's requested format",
)
DEFAULT_GAIA_AGENT_INSTRUCTIONS = DEFAULT_AGENT_INSTRUCTIONS.replace(
    "use the submit tool to provide your final answer",
    f"""follow these rules when producing the final answer:

{GAIA_FINAL_ANSWER_FORMAT}

These rules apply only to the final answer. During planning, execution, and
review, follow the current phase's JSON response protocol instead.""",
)


class RExPlanError(ValueError):
    """Raised when a planning response violates the agent protocol."""


class RExPromptStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_goal: str
    execution_mode: RExStepKind


class RExPromptPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thinking: str
    steps: list[RExPromptStep]
    planning_complete: bool


class RExPromptStepResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: RExStatus
    summary: str = ""


@dataclass
class RExStep:
    id: str
    task: str
    kind: RExStepKind


@dataclass
class RExStepResult:
    status: RExStatus
    summary: str = ""


@dataclass
class RExPlanPatch:
    steps: list[RExStep]
    complete: bool
    thinking: str = ""


@dataclass
class RExReplanContext:
    failed_step_id: str
    failed_step_goal: str
    failure_summary: str


@dataclass(frozen=True)
class RExRootPrompt:
    goal: str
    final_answer_format: str | None


@dataclass
class RExBudget:
    model_calls: int = 0
    tool_calls: int = 0
    frames: int = 0


@dataclass(frozen=True)
class RExConfig:
    max_steps_per_frame: int
    max_depth: int
    max_turns_per_step: int
    max_planning_turns: int
    max_tool_calls: int
    max_observation_chars: int
    planning_tools: bool
    compression_mode: CompressionMode = "tree"
    compression_backend: CompressionBackend = "standalone"
    frame_compression_trigger_tokens: int = 6144
    compression_text_logs: bool = True
    compression_log_root: str | None = None
    planning_schedule: PlanningSchedule = "progressive"
    plan_batch_limit: int | None = None
    ablation: str | None = None


@dataclass
class RExChatResult:
    output: ModelOutput
    model_calls: int
    tool_calls: int


class RExBudgetExceeded(RuntimeError):
    """Raised when a sample reaches an agent-level execution budget."""


def _json_dict(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a plain, fenced, or wrapped response."""
    candidates = [text.strip()]
    candidates.extend(match.group(1).strip() for match in _JSON_FENCE_RE.finditer(text))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_plan(
    text: str,
    *,
    next_step_index: int,
    planned_step_count: int,
    max_steps: int,
    planning_schedule: PlanningSchedule = "progressive",
    batch_limit: int | None = None,
) -> RExPlanPatch:
    raw = _json_dict(text)
    if raw is None:
        raise RExPlanError("response did not contain a JSON object")

    try:
        parsed = RExPromptPlan.model_validate(raw)
    except ValidationError as exc:
        raise RExPlanError(str(exc)) from exc

    if not parsed.planning_complete and not parsed.steps:
        raise RExPlanError("planning_complete=false requires at least one step")
    if planned_step_count + len(parsed.steps) > max_steps:
        raise RExPlanError("planning response exceeds the remaining frame step budget")
    if planning_schedule == "one_shot" and not parsed.planning_complete:
        raise RExPlanError("one-shot planning requires planning_complete=true")
    if batch_limit is not None and len(parsed.steps) > batch_limit:
        raise RExPlanError(f"planning response exceeds batch limit of {batch_limit}")
    if any(not step.step_goal.strip() for step in parsed.steps):
        raise RExPlanError("step_goal cannot be empty")

    steps = [
        RExStep(
            id=f"step_{next_step_index + index}",
            task=step.step_goal.strip(),
            kind=step.execution_mode,
        )
        for index, step in enumerate(parsed.steps)
    ]
    return RExPlanPatch(
        steps=steps,
        complete=parsed.planning_complete,
        thinking=parsed.thinking.strip(),
    )


def parse_step_result(text: str) -> RExStepResult:
    """Parse a step control response, treating missing status as failure."""
    raw = _json_dict(text)
    if raw is None or "status" not in raw:
        return RExStepResult(
            status="failed",
            summary="Step returned no valid status JSON.",
        )
    try:
        parsed = RExPromptStepResult.model_validate(raw)
    except ValidationError:
        return RExStepResult(status="failed", summary="Invalid step result JSON.")
    return RExStepResult(status=parsed.status, summary=parsed.summary)


class RExRunner:
    def __init__(
        self,
        *,
        model: Model,
        messages: list[ChatMessage],
        tools: Sequence[Tool],
        config: RExConfig,
        compression_log_context: CompressionLogContext | None = None,
        compression_log_context_error: str | None = None,
    ) -> None:
        self.model = model
        self.messages = messages
        self.tools = list(tools)
        self.config = config
        self.budget = RExBudget()
        self._compression_sequence = 0
        self._compression_log_context = compression_log_context
        self._compression_log_context_error = compression_log_context_error

    async def run(
        self, goal: str, *, final_answer_format: str | None = None
    ) -> ModelOutput:
        """Execute the recursive plan and return a short final answer."""
        root_result = RExStepResult("failed", "The root frame did not complete.")
        try:
            root_result = await self._run_frame("root", goal, depth=1)
        except RExBudgetExceeded as exc:
            root_result = RExStepResult("failed", str(exc))
            self._event("budget_exhausted", note=str(exc))

        self.messages.append(
            ChatMessageUser(
                content=_final_prompt(goal, root_result, final_answer_format)
            )
        )
        async with span("gaia high-confidence final", type="agent_final"):
            output = await self.model.generate(self.messages, tools=[])
        self.messages.append(output.message)
        self.budget.model_calls += 1
        self._event(
            "done",
            artifact={
                "status": root_result.status,
                "summary": self._truncate(root_result.summary),
                "budget": self.budget.__dict__,
            },
        )
        return output

    async def _run_frame(
        self,
        frame_id: str,
        goal: str,
        *,
        depth: int,
        parent_goal: str | None = None,
        successor: RExStep | None = None,
    ) -> RExStepResult:
        start_index = len(self.messages)
        result = await self._run_frame_body(frame_id, goal, depth=depth)
        if (
            depth > 1
            and self.config.compression_mode == "tree"
            and result.status == "success"
        ):
            scope = CompressionScope(
                kind="frame",
                frame_id=frame_id,
                goal=goal,
                status=result.status,
                consumer_goal=parent_goal,
                depth=depth,
                execution_mode="decomposed_frame",
                successor_goal=successor.task if successor is not None else None,
                successor_execution_mode=(
                    successor.kind if successor is not None else None
                ),
            )
            await self._compact_completed_scope(
                scope=scope,
                start_index=start_index,
                trigger_tokens=self.config.frame_compression_trigger_tokens,
                result=result,
            )
        return result

    async def _run_frame_body(
        self, frame_id: str, goal: str, *, depth: int
    ) -> RExStepResult:
        self.budget.frames += 1
        self._check_budget()

        async with span(f"frame {frame_id}", type="agent_frame"):
            steps: list[RExStep] = []
            completed: dict[str, RExStepResult] = {}
            planned_step_count = 0
            next_step_index = 1
            batch_index = 0
            recovery: RExReplanContext | None = None
            recovery_attempted = False

            while True:
                if planned_step_count >= self.config.max_steps_per_frame:
                    return RExStepResult(
                        "failed", "Frame exhausted its step budget before completion."
                    )

                batch_index += 1
                patch = await self._plan_batch(
                    frame_id=frame_id,
                    goal=goal,
                    batch_index=batch_index,
                    planned_step_count=planned_step_count,
                    next_step_index=next_step_index,
                    depth=depth,
                    recovery=recovery,
                )
                if patch is None:
                    if self.config.planning_schedule == "one_shot":
                        return RExStepResult(
                            "failed",
                            "One-shot planning did not produce a valid complete plan.",
                        )
                    if planned_step_count:
                        return RExStepResult(
                            "failed", "Replanning did not produce a valid plan."
                        )
                    patch = RExPlanPatch(
                        steps=[RExStep(f"step_{next_step_index}", goal, "execute")],
                        complete=True,
                    )
                    self._event(
                        "plan_patch",
                        frame_id=frame_id,
                        artifact={
                            "fallback": True,
                            "steps": self._dump_steps(patch.steps),
                        },
                    )
                else:
                    recovery = None

                if self.config.ablation == "execute_only":
                    patch.steps = [
                        RExStep(step.id, step.task, "execute") for step in patch.steps
                    ]

                steps.extend(patch.steps)
                planned_step_count += len(patch.steps)
                next_step_index += len(patch.steps)

                batch_failed = False
                for offset, step in enumerate(patch.steps):
                    result = await self._run_step(
                        frame_id=frame_id,
                        frame_goal=goal,
                        steps=steps,
                        completed=completed,
                        step=step,
                        depth=depth,
                    )
                    completed[step.id] = result
                    if result.status == "success":
                        continue

                    dropped = patch.steps[offset + 1 :]
                    dropped_ids = {item.id for item in dropped}
                    steps = [item for item in steps if item.id not in dropped_ids]
                    self._event(
                        "plan_batch_abort",
                        frame_id=frame_id,
                        step=step,
                        artifact={
                            "batch_index": batch_index,
                            "failed_step_id": step.id,
                            "failed_status": result.status,
                            "dropped_steps": self._dump_steps(dropped),
                            "recovery_exhausted": recovery_attempted,
                        },
                    )
                    if recovery_attempted:
                        # Return the failure to the parent instead of retrying here.
                        return result
                    recovery_attempted = True
                    recovery = RExReplanContext(step.id, step.task, result.summary)
                    batch_failed = True
                    break

                if batch_failed:
                    continue
                # Reset only after the entire batch succeeds, not individual steps.
                recovery_attempted = False
                if not patch.complete:
                    if self.config.planning_schedule == "one_shot":
                        return RExStepResult("failed", "One-shot plan was incomplete.")
                    continue

                return RExStepResult("success", "All planned steps completed.")

    async def _plan_batch(
        self,
        *,
        frame_id: str,
        goal: str,
        batch_index: int,
        planned_step_count: int,
        next_step_index: int,
        depth: int,
        recovery: RExReplanContext | None,
    ) -> RExPlanPatch | None:
        remaining = self.config.max_steps_per_frame - planned_step_count
        if recovery is None:
            prompt = _planning_prompt(
                goal=goal,
                depth=depth,
                max_depth=self.config.max_depth,
                remaining_steps=remaining,
                one_shot=self.config.planning_schedule == "one_shot",
                batch_limit=self.config.plan_batch_limit,
            )
            planning_mode = "rolling"
        else:
            prompt = _recovery_prompt(
                goal=goal,
                depth=depth,
                max_depth=self.config.max_depth,
                recovery=recovery,
                remaining_steps=remaining,
                batch_limit=self.config.plan_batch_limit,
                planning_schedule=self.config.planning_schedule,
            )
            planning_mode = "recovery"

        self.messages.append(ChatMessageUser(content=prompt))
        result = await self._chat(
            self.tools if self.config.planning_tools else [],
            phase_turn_limit=self.config.max_planning_turns,
        )
        try:
            patch = parse_plan(
                result.output.completion,
                next_step_index=next_step_index,
                planned_step_count=planned_step_count,
                max_steps=self.config.max_steps_per_frame,
                planning_schedule=self.config.planning_schedule,
                batch_limit=self.config.plan_batch_limit,
            )
        except RExPlanError as exc:
            self.messages.append(
                ChatMessageUser(
                    content=_plan_format_feedback(
                        str(exc),
                        remaining,
                        depth,
                        self.config.max_depth,
                        planning_schedule=self.config.planning_schedule,
                        batch_limit=self.config.plan_batch_limit,
                    )
                )
            )
            repair = await self._chat([], phase_turn_limit=1)
            try:
                patch = parse_plan(
                    repair.output.completion,
                    next_step_index=next_step_index,
                    planned_step_count=planned_step_count,
                    max_steps=self.config.max_steps_per_frame,
                    planning_schedule=self.config.planning_schedule,
                    batch_limit=self.config.plan_batch_limit,
                )
            except RExPlanError as repair_exc:
                self._event(
                    "plan_patch",
                    frame_id=frame_id,
                    artifact={
                        "batch_index": batch_index,
                        "planning_mode": planning_mode,
                    },
                    note=str(repair_exc),
                )
                return None

        self._event(
            "plan_patch",
            frame_id=frame_id,
            artifact={
                "batch_index": batch_index,
                "planning_mode": planning_mode,
                "recovery_for_step_id": (
                    recovery.failed_step_id if recovery is not None else None
                ),
                "thinking": patch.thinking,
                "steps": self._dump_steps(patch.steps),
                "planning_complete": patch.complete,
                "planning_schedule": self.config.planning_schedule,
                "plan_batch_limit": self.config.plan_batch_limit,
                "batch_size": len(patch.steps),
            },
        )
        return patch

    async def _run_step(
        self,
        *,
        frame_id: str,
        frame_goal: str,
        steps: list[RExStep],
        completed: dict[str, RExStepResult],
        step: RExStep,
        depth: int,
    ) -> RExStepResult:
        return await self._run_step_body(
            frame_id=frame_id,
            frame_goal=frame_goal,
            steps=steps,
            completed=completed,
            step=step,
            depth=depth,
        )

    async def _run_step_body(
        self,
        *,
        frame_id: str,
        frame_goal: str,
        steps: list[RExStep],
        completed: dict[str, RExStepResult],
        step: RExStep,
        depth: int,
    ) -> RExStepResult:
        self._event("step_start", frame_id=frame_id, step=step)
        async with span(f"{frame_id}.{step.id}", type="agent_step"):
            if step.kind == "decompose" and depth < self.config.max_depth:
                # Only this batch's next planned sibling is the continuation.
                # Previous batches precede this step; future batches do not exist yet.
                next_index = steps.index(step) + 1
                successor = steps[next_index] if next_index < len(steps) else None
                result = await self._run_frame(
                    f"{frame_id}.{step.id}",
                    step.task,
                    depth=depth + 1,
                    parent_goal=frame_goal,
                    successor=successor,
                )
            else:
                forced_direct = step.kind == "decompose"
                step_prompt = ChatMessageUser(
                    content=_step_prompt(
                        frame_goal,
                        step,
                        self._render_completed(steps, completed),
                        forced_direct,
                    )
                )
                self.messages.append(step_prompt)
                chat = await self._chat_with_turn_limit(
                    self.tools,
                    turn_limit=self.config.max_turns_per_step,
                )
                result = parse_step_result(chat.output.completion)

        self._event(
            "step_done",
            frame_id=frame_id,
            step=step,
            artifact={
                "status": result.status,
                "summary": self._truncate(result.summary),
            },
        )
        return result

    def _next_compression_identity(self) -> tuple[int, str]:
        self._compression_sequence += 1
        sequence = self._compression_sequence
        compression_id = (
            make_compression_id(self._compression_log_context, sequence)
            if self._compression_log_context is not None
            else f"cmp_{sequence:06d}"
        )
        return sequence, compression_id

    def _write_compression_attempt_log(
        self,
        *,
        event: str,
        scope: CompressionScope,
        sequence: int,
        compression_id: str,
        started_at: str,
        started_monotonic: float,
        source_messages: list[ChatMessage],
        request_kind: str,
        request_messages: list[ChatMessage] | None,
        task_context: dict[str, Any] | None,
        evidence: Sequence[Any],
        inherited_evidence_ledgers: Sequence[str],
        output: ModelOutput | None,
        raw_text: str,
        normalized_handoff: str,
        applied: bool,
        reason: str,
        estimated_tokens_before: int,
        estimated_tokens_after: int | None = None,
        error: Exception | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        if not self.config.compression_text_logs:
            return {"status": "disabled"}
        context = self._compression_log_context
        if context is None:
            return {
                "status": "write_error",
                "error_type": "CompressionLogContextError",
                "error_message": self._compression_log_context_error
                or "compression log context is unavailable",
            }

        try:
            if error is not None:
                error_type = type(error).__name__
                error_message = str(error)
            response_message = output.message if output is not None else None
            completed_at = datetime.now(timezone.utc).isoformat()
            payload: dict[str, Any] = {
                "schema_version": COMPRESSION_TEXT_LOG_SCHEMA_VERSION,
                "task_id": context.task_id,
                "epoch": context.epoch,
                "sample_uuid": context.sample_uuid,
                "compression_id": compression_id,
                "compression_sequence": sequence,
                "event": event,
                "backend": self.config.compression_backend,
                "model": getattr(self.model, "name", None),
                "scope": asdict(scope),
                "request": {
                    "kind": request_kind,
                    "messages": dump_messages(request_messages or []),
                    "source_messages": dump_messages(source_messages),
                    "task_context": task_context,
                    "evidence": [asdict(item) for item in evidence],
                    "inherited_evidence_ledgers": list(inherited_evidence_ledgers),
                },
                "response": {
                    "message": (
                        response_message.model_dump(mode="json")
                        if response_message is not None
                        else None
                    ),
                    "model_output": dump_model_output(output),
                    "raw_text": raw_text,
                    "normalized_handoff": normalized_handoff,
                    "error_type": error_type,
                    "error_message": error_message,
                },
                "decision": {
                    "triggered": True,
                    "applied": applied,
                    "reason": reason,
                    "estimated_tokens_before": estimated_tokens_before,
                    "estimated_tokens_after": estimated_tokens_after,
                },
                "timing": {
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_s": time.monotonic() - started_monotonic,
                },
            }
            return write_compressor_text_log(
                context,
                sequence=sequence,
                scope_kind="tree",
                compression_id=compression_id,
                payload=payload,
            )
        except Exception as exc:
            return {
                "status": "write_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

    async def _compact_completed_scope(
        self,
        *,
        scope: CompressionScope,
        start_index: int,
        trigger_tokens: int,
        result: RExStepResult,
        step: RExStep | None = None,
    ) -> None:
        if self.config.compression_mode == "none" or result.status != "success":
            return
        event = "tree_compress"
        source_messages = list(self.messages[start_index:])
        base_artifact: dict[str, Any] = {
            "backend": self.config.compression_backend,
            "scope": scope.kind,
            "status": scope.status,
            "trigger_tokens": trigger_tokens,
            "message_count_before": len(source_messages),
        }

        try:
            before_tokens = (
                await self.model.count_tokens(source_messages) if source_messages else 0
            )
        except Exception as exc:
            self._compression_fallback(
                event=event,
                scope=scope,
                step=step,
                artifact=base_artifact,
                reason=f"token_count_error: {type(exc).__name__}: {exc}",
            )
            return

        base_artifact["estimated_tokens_before"] = before_tokens
        if not source_messages or before_tokens <= trigger_tokens:
            self._event(
                event,
                frame_id=scope.frame_id,
                step=step,
                artifact={
                    **base_artifact,
                    "triggered": False,
                    "applied": False,
                    "reason": "below_threshold",
                },
            )
            return

        sequence, compression_id = self._next_compression_identity()
        started_at = datetime.now(timezone.utc).isoformat()
        started_monotonic = time.monotonic()
        base_artifact.update(
            {
                "compression_id": compression_id,
                "compression_sequence": sequence,
            }
        )
        evidence: Sequence[Any] = ()
        inherited: Sequence[str] = ()
        request_kind = "standalone"
        request_messages: list[ChatMessage] | None = None
        task_context: dict[str, Any] | None = None
        output: ModelOutput | None = None
        raw_text = ""
        normalized_handoff = ""

        def write_text_log(
            *,
            applied: bool,
            reason: str,
            estimated_tokens_after: int | None = None,
            error: Exception | None = None,
            error_type: str | None = None,
            error_message: str | None = None,
        ) -> dict[str, Any]:
            return self._write_compression_attempt_log(
                event=event,
                scope=scope,
                sequence=sequence,
                compression_id=compression_id,
                started_at=started_at,
                started_monotonic=started_monotonic,
                source_messages=source_messages,
                request_kind=request_kind,
                request_messages=request_messages,
                task_context=task_context,
                evidence=evidence,
                inherited_evidence_ledgers=inherited,
                output=output,
                raw_text=raw_text,
                normalized_handoff=normalized_handoff,
                applied=applied,
                reason=reason,
                estimated_tokens_before=before_tokens,
                estimated_tokens_after=estimated_tokens_after,
                error=error,
                error_type=error_type,
                error_message=error_message,
            )

        try:
            evidence = extract_compression_evidence(source_messages)
            inherited = extract_inherited_evidence_ledgers(source_messages)
        except Exception as exc:
            text_log = write_text_log(
                applied=False,
                reason="evidence_error",
                error=exc,
            )
            self._compression_fallback(
                event=event,
                scope=scope,
                step=step,
                artifact={**base_artifact, "text_log": text_log},
                reason=f"evidence_error: {type(exc).__name__}: {exc}",
            )
            return

        try:
            prepared = prepare_compression_request(scope, source_messages)
        except Exception as exc:
            text_log = write_text_log(applied=False, reason="request_error", error=exc)
            self._compression_fallback(
                event=event,
                scope=scope,
                step=step,
                artifact={**base_artifact, "text_log": text_log},
                reason=f"request_error: {type(exc).__name__}: {exc}",
            )
            return
        evidence = prepared.evidence
        inherited = prepared.inherited_evidence_ledgers
        request_messages = prepared.messages
        task_context = prepared.task_context
        try:
            output = await self.model.generate(prepared.messages, tools=[])
        except Exception as exc:
            text_log = write_text_log(
                applied=False, reason="compressor_error", error=exc
            )
            self._compression_fallback(
                event=event,
                scope=scope,
                step=step,
                artifact={**base_artifact, "text_log": text_log},
                reason=f"compressor_error: {type(exc).__name__}: {exc}",
            )
            return
        self.budget.model_calls += 1
        raw_text = output.completion
        handoff = normalize_handoff_text(output.completion)
        normalized_handoff = handoff
        if not handoff_is_valid(handoff):
            text_log = write_text_log(
                applied=False,
                reason="invalid_compressor_handoff",
                error_type="InvalidHandoff",
                error_message="compressor returned an empty or oversized handoff",
            )
            self._compression_fallback(
                event=event,
                scope=scope,
                step=step,
                artifact={**base_artifact, "text_log": text_log},
                reason="compressor returned an empty or oversized handoff",
            )
            return

        try:
            checkpoint = ChatMessageUser(
                content=render_compressed_history(scope, handoff, evidence, inherited),
                metadata={
                    "gaia_compression": {
                        "scope": scope.kind,
                        "frame_id": scope.frame_id,
                        "step_id": scope.step_id,
                    }
                },
            )
        except Exception as exc:
            text_log = write_text_log(
                applied=False,
                reason="checkpoint_error",
                error=exc,
            )
            self._compression_fallback(
                event=event,
                scope=scope,
                step=step,
                artifact={**base_artifact, "text_log": text_log},
                reason=f"checkpoint_error: {type(exc).__name__}: {exc}",
            )
            return
        try:
            after_tokens = await self.model.count_tokens([checkpoint])
        except Exception as exc:
            text_log = write_text_log(
                applied=False,
                reason="checkpoint_token_count_error",
                error=exc,
            )
            self._compression_fallback(
                event=event,
                scope=scope,
                step=step,
                artifact={**base_artifact, "text_log": text_log},
                reason=f"checkpoint_token_count_error: {type(exc).__name__}: {exc}",
            )
            return

        applied = after_tokens < before_tokens
        if applied:
            self.messages[start_index:] = [checkpoint]
            result.summary = handoff
        decision_reason = "applied" if applied else "not_smaller"
        text_log = write_text_log(
            applied=applied,
            reason=decision_reason,
            estimated_tokens_after=after_tokens,
        )
        self._event(
            event,
            frame_id=scope.frame_id,
            step=step,
            artifact={
                **base_artifact,
                "triggered": True,
                "applied": applied,
                "reason": decision_reason,
                "estimated_tokens_after": after_tokens,
                "evidence_count": len(evidence),
                "inherited_evidence_count": len(inherited),
                "text_log": text_log,
            },
        )

    def _compression_fallback(
        self,
        *,
        event: str,
        scope: CompressionScope,
        step: RExStep | None,
        artifact: dict[str, Any],
        reason: str,
    ) -> None:
        fallback_artifact = {
            **artifact,
            "triggered": True,
            "applied": False,
            "reason": "fallback",
            "fallback_reason": reason,
        }
        self._event(
            "compression_fallback",
            frame_id=scope.frame_id,
            step=step,
            artifact=fallback_artifact,
            note=reason,
        )
        self._event(
            event,
            frame_id=scope.frame_id,
            step=step,
            artifact=fallback_artifact,
        )

    async def _chat(
        self, tools: Sequence[Tool], *, phase_turn_limit: int
    ) -> RExChatResult:
        self._check_budget()
        new_messages, output = await self.model.generate_loop(
            self.messages, tools=tools
        )
        self.messages.extend(new_messages)
        model_calls = sum(
            isinstance(message, ChatMessageAssistant) for message in new_messages
        )
        tool_calls = sum(
            len(message.tool_calls or [])
            for message in new_messages
            if isinstance(message, ChatMessageAssistant)
        )
        self.budget.model_calls += model_calls
        self.budget.tool_calls += tool_calls
        if model_calls > phase_turn_limit:
            raise RExBudgetExceeded(
                f"Phase used {model_calls} model turns; limit is {phase_turn_limit}."
            )
        self._check_budget()
        return RExChatResult(output, model_calls, tool_calls)

    async def _chat_with_turn_limit(
        self, tools: Sequence[Tool], *, turn_limit: int
    ) -> RExChatResult:
        """Generate and execute tools until completion or the per-step turn limit."""
        new_messages: list[ChatMessage] = []
        model_calls = 0
        tool_calls = 0
        output: ModelOutput | None = None

        for _ in range(turn_limit):
            self._check_budget()
            output = await self.model.generate(
                self.messages + new_messages, tools=tools
            )
            new_messages.append(output.message)
            model_calls += 1
            self.budget.model_calls += 1

            calls = len(output.message.tool_calls or [])
            tool_calls += calls
            self.budget.tool_calls += calls
            self._check_budget()
            if not calls:
                self.messages.extend(new_messages)
                return RExChatResult(output, model_calls, tool_calls)

            tool_messages, tools_output = await execute_tools(
                self.messages + new_messages, tools
            )
            new_messages.extend(tool_messages)
            if tools_output is not None:
                output = tools_output

        assert output is not None
        self.messages.extend(new_messages)
        return RExChatResult(
            output,
            model_calls,
            tool_calls,
        )

    def _check_budget(self) -> None:
        if self.budget.tool_calls > self.config.max_tool_calls:
            raise RExBudgetExceeded("Agent exceeded the global tool-call budget.")

    def _render_completed(
        self, steps: list[RExStep], completed: dict[str, RExStepResult]
    ) -> str:
        rows = [
            {"step_goal": step.task, "status": completed[step.id].status}
            for step in steps
            if step.id in completed
        ]
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def _dump_steps(self, steps: Sequence[RExStep]) -> list[dict[str, str]]:
        return [
            {"id": step.id, "step_goal": step.task, "execution_mode": step.kind}
            for step in steps
        ]

    def _truncate(self, text: str) -> str:
        if len(text) <= self.config.max_observation_chars:
            return text
        return text[: self.config.max_observation_chars] + "\n[truncated]"

    def _event(
        self,
        event: str,
        *,
        frame_id: str = "root",
        step: RExStep | None = None,
        artifact: dict[str, Any] | None = None,
        note: str = "",
    ) -> None:
        transcript().info(
            {
                "event": event,
                "frame_id": frame_id,
                "step_id": step.id if step else None,
                "step_goal": step.task if step else None,
                "artifact": artifact or {},
                "note": note,
            },
            source="gaia.high_confidence_recursive",
        )


@agent
def gaia_high_confidence_recursive_agent(
    *,
    max_steps_per_frame: int = 16,
    max_depth: int = 4,
    max_turns_per_step: int = 20,
    max_planning_turns: int = 1,
    max_tool_calls: int = 100,
    max_observation_chars: int = 6000,
    planning_tools: bool = False,
    compression_mode: CompressionMode = "tree",
    compression_backend: CompressionBackend = "standalone",
    frame_compression_trigger_tokens: int = 6144,
    compression_text_logs: bool = True,
    compression_log_root: str | None = None,
    planning_schedule: PlanningSchedule = "progressive",
    plan_batch_limit: int | None = None,
    ablation: Ablation | None = None,
    tool_timeout: int = 180,
    model: str | Model | None = None,
    tools: Sequence[Tool] | None = None,
) -> Agent:
    """Create a recursive rolling-plan agent for GAIA.

    The assessed model plans, executes, reviews, and produces the final answer.
    Inspect remains responsible for model retries, sandbox tools, limits, and logs.
    """
    positive_values = {
        "max_steps_per_frame": max_steps_per_frame,
        "max_depth": max_depth,
        "max_turns_per_step": max_turns_per_step,
        "max_planning_turns": max_planning_turns,
        "max_tool_calls": max_tool_calls,
        "max_observation_chars": max_observation_chars,
        "frame_compression_trigger_tokens": frame_compression_trigger_tokens,
    }
    for name, value in positive_values.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if compression_mode not in ("none", "tree"):
        raise ValueError('compression_mode must be "none" or "tree"')
    if compression_backend != "standalone":
        raise ValueError('compression_backend must be "standalone"')
    if compression_log_root is not None and not compression_log_root.strip():
        raise ValueError("compression_log_root cannot be empty")
    if planning_schedule not in ("progressive", "one_shot"):
        raise ValueError("planning_schedule must be 'progressive' or 'one_shot'")
    if plan_batch_limit is not None and plan_batch_limit < 1:
        raise ValueError("plan_batch_limit must be at least 1")
    if ablation is not None:
        if ablation == "one_shot":
            planning_schedule = "one_shot"
        elif ablation == "single_item":
            plan_batch_limit = 1
        elif ablation == "execute_only":
            pass

    resolved_tools = (
        list(tools)
        if tools is not None
        else [bash(tool_timeout), python(tool_timeout), *web_browser()]
    )
    config = RExConfig(
        max_steps_per_frame=max_steps_per_frame,
        max_depth=max_depth,
        max_turns_per_step=max_turns_per_step,
        max_planning_turns=max_planning_turns,
        max_tool_calls=max_tool_calls,
        max_observation_chars=max_observation_chars,
        planning_tools=planning_tools,
        compression_mode=compression_mode,
        compression_backend=compression_backend,
        frame_compression_trigger_tokens=frame_compression_trigger_tokens,
        compression_text_logs=compression_text_logs,
        compression_log_root=compression_log_root,
        planning_schedule=planning_schedule,
        plan_batch_limit=plan_batch_limit,
        ablation=ablation,
    )

    async def execute(state: AgentState) -> AgentState:
        root_message_index = next(
            (
                index
                for index, message in enumerate(state.messages)
                if isinstance(message, ChatMessageUser)
            ),
            None,
        )
        root_text = (
            state.messages[root_message_index].text
            if root_message_index is not None
            else ""
        )
        root_prompt = _split_root_prompt(root_text)
        if root_message_index is not None:
            del state.messages[root_message_index]

        instructions = (
            DEFAULT_GAIA_AGENT_INSTRUCTIONS
            if root_prompt.final_answer_format is not None
            else REX_AGENT_INSTRUCTIONS
        )
        state.messages.insert(0, ChatMessageSystem(content=instructions))
        resolved_model = get_model(model)
        compression_log_context: CompressionLogContext | None = None
        compression_log_context_error: str | None = None
        if config.compression_mode == "tree" and config.compression_text_logs:
            compression_log_context, compression_log_context_error = (
                resolve_compression_log_context(config.compression_log_root)
            )
        runner = RExRunner(
            model=resolved_model,
            messages=state.messages,
            tools=resolved_tools,
            config=config,
            compression_log_context=compression_log_context,
            compression_log_context_error=compression_log_context_error,
        )
        state.output = await runner.run(
            root_prompt.goal,
            final_answer_format=root_prompt.final_answer_format,
        )
        return state

    return execute


def _split_root_prompt(text: str) -> RExRootPrompt:
    before_file, remainder = DEFAULT_INPUT_PROMPT.split("{file}", maxsplit=1)
    between_file_and_question, after_question = remainder.split(
        "{question}", maxsplit=1
    )
    if not text.startswith(before_file) or (
        after_question and not text.endswith(after_question)
    ):
        return RExRootPrompt(text, None)

    end = len(text) - len(after_question) if after_question else len(text)
    rendered_fields = text[len(before_file) : end]
    file_context, separator, question = rendered_fields.partition(
        between_file_and_question
    )
    if not separator or not question.strip():
        return RExRootPrompt(text, None)

    question = question.strip()
    file_context = file_context.strip()
    goal = f"{file_context}\n\nQuestion:\n{question}" if file_context else question
    return RExRootPrompt(goal, GAIA_FINAL_ANSWER_FORMAT)


def _depth_context(depth: int, max_depth: int) -> str:
    remaining = max(0, max_depth - depth)
    if remaining == 0:
        detail = "No decomposition levels remain."
    elif remaining == 1:
        detail = "One decomposition level remains."
    else:
        detail = f"{remaining} decomposition levels remain."
    return f"Current depth: {depth}. Maximum depth: {max_depth}. {detail}"


def _mode_guidance(
    depth: int,
    max_depth: int,
) -> str:
    if depth >= max_depth:
        return (
            '- No decomposition levels remain. Use "execute" for every step.\n'
            "- Split composite remaining work into multiple narrower execute steps."
        )

    guidance = (
        '- Use "execute" for atomic work \n'
        '- Use "decompose" for work requiring multiple retrieval, verification, '
        "transformation, or action stages."
    )
    return (
        guidance
        + '\n- Prefer "decompose" for multi-source or gather-then-synthesize work.'
    )


def _planning_prompt(
    *,
    goal: str,
    depth: int,
    max_depth: int,
    remaining_steps: int,
    one_shot: bool = False,
    batch_limit: int | None = None,
) -> str:
    schedule_rules = (
        "- Return the complete remaining executable plan now.\n"
        "- Set planning_complete true; no successful-batch planning call will follow."
        if one_shot
        else "- Plan as far ahead as the current information reliably supports. Return one\n"
        "  step, a partial batch, or the full remaining plan as appropriate. Stop before work whose specification materially depends on future observations.\n"
        "- Set planning_complete true only when the returned batch covers all remaining work."
    )
    limit_rule = (
        f"\n- Return at most {batch_limit} step(s) in this batch."
        if batch_limit is not None
        else ""
    )
    return f"""
Use the conversation and recent tool observations to decompose goal.

Goal:
{goal}

{_depth_context(depth, max_depth)}
Return ONLY JSON:
{{"thinking": string, "steps": [{{"step_goal": string, "execution_mode": "execute|decompose"}}], "planning_complete": boolean}}

Rules:
- Consider the complete remaining work before selecting steps.
- Keep thinking concise and self-contained; do not answer the task there.
{schedule_rules}
{_mode_guidance(depth, max_depth)}
- Return at least one step when planning_complete is false.
- Do not repeat completed work or retry a failed approach unchanged.
{limit_rule}
- A decomposed step must be strictly narrower than this Goal."""


def _recovery_prompt(
    *,
    goal: str,
    depth: int,
    max_depth: int,
    recovery: RExReplanContext,
    remaining_steps: int,
    batch_limit: int | None = None,
    planning_schedule: PlanningSchedule = "progressive",
) -> str:
    failure = json.dumps(
        {
            "step_goal": recovery.failed_step_goal,
            "status": "failed",
            "summary": recovery.failure_summary,
        },
        ensure_ascii=False,
    )
    return f"""
Use the failure evidence to choose a materially changed executable approach.

Goal:
{goal}

Failed step:
{failure}

{_depth_context(depth, max_depth)}

Return ONLY JSON:
{{"thinking": string, "steps": [{{"step_goal": string, "execution_mode": "execute|decompose"}}], "planning_complete": boolean}}

Rules:
- Repair or bypass the failure with a changed tool, input, or strategy.
- Return at least one step when planning_complete is false.
{"- Return the complete remaining plan and set planning_complete true." if planning_schedule == "one_shot" else ""}
{_mode_guidance(depth, max_depth)}
- A decomposed step must be strictly narrower than this Goal."""


def _plan_format_feedback(
    error: str,
    remaining_steps: int,
    depth: int,
    max_depth: int,
    *,
    planning_schedule: PlanningSchedule = "progressive",
    batch_limit: int | None = None,
) -> str:
    base = f"""Correction required: the planning response was invalid: {error}

Do not call tools, answer the task, or explain. Return only JSON:
{{"thinking": string, "steps": [{{"step_goal": string, "execution_mode": "execute|decompose"}}], "planning_complete": boolean}}

{_depth_context(depth, max_depth)}
The response must fit within the remaining budget of {remaining_steps} steps."""
    suffix = ""
    if planning_schedule == "one_shot":
        suffix += "\nReturn the complete remaining executable plan now and set planning_complete true."
    if batch_limit is not None:
        suffix += f"\nReturn at most {batch_limit} step(s) in this batch."
    return base + suffix


def _step_prompt(
    frame_goal: str, step: RExStep, completed: str, forced_direct: bool
) -> str:
    forced = (
        " This decomposed step reached maximum depth and must execute directly."
        if forced_direct
        else ""
    )
    return f"""Direct subtask execution.{forced}

Use the frame goal as background and complete only the current step.

Frame goal:
{frame_goal}

Current step:
{step.task}

Call tools if needed. When complete, return ONLY JSON:
{{"status": "success|failed", "summary": string}}

Return success only when the step is complete and verified."""


def _final_prompt(
    goal: str, result: RExStepResult, final_answer_format: str | None
) -> str:
    if final_answer_format is None:
        answer_instructions = """Return only the requested number, short phrase, or comma-separated list. Do not
return planning JSON, reasoning, labels, units unless requested, or commentary."""
    else:
        answer_instructions = f"""Final answer rules:
{final_answer_format}

Do not return planning JSON, reasoning, labels, or commentary."""
    return f"""Answer the original GAIA question using the conversation evidence.

Original question:
{goal}

Execution status: {result.status}
Execution summary: {result.summary}

{answer_instructions}"""


__all__ = [
    "RExRunner",
    "gaia_high_confidence_recursive_agent",
    "parse_plan",
    "parse_step_result",
]
