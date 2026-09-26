"""REx runner for Agent-Diff benchmarks.

The controller is intentionally independent of Inspect and model-native tool
calling. Planning uses strict JSON responses, while direct execution
uses the benchmark's existing XML action protocol and Bash executor proxy.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

from .hierarchical_compact import (
    CompressionEvidence,
    CompressionScope,
    CompressionSibling,
    deterministic_handoff,
    estimate_messages_tokens,
    handoff_is_valid,
    normalize_handoff,
    prepare_compression_request,
    render_compressed_history,
)

ExecutionMode = Literal["execute", "decompose"]
StepStatus = Literal["success", "failed"]
CompressionMode = Literal["tree"]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_DONE_RE = re.compile(r"<done>(.*?)</done>", re.IGNORECASE | re.DOTALL)


class ModelError(RuntimeError):
    """Raised when the model endpoint cannot produce a response."""


class ProtocolError(ValueError):
    """Raised when a controller response violates its JSON protocol."""


class BudgetExceeded(RuntimeError):
    """Raised when the agent reaches a global execution budget."""


@dataclass(frozen=True)
class ModelTurn:
    content: str
    usage: dict[str, float | int]
    raw_response: dict[str, Any] | None = None
    attempts: tuple[dict[str, Any], ...] = ()


class ModelClient(Protocol):
    def generate(self, messages: Sequence[dict[str, str]]) -> ModelTurn:
        """Generate the next assistant message."""


class CodeExecutor(Protocol):
    def execute(self, code: str) -> dict[str, Any]:
        """Execute one Bash action and return an Agent-Diff execution result."""


@dataclass(frozen=True)
class RExConfig:
    max_depth: int = 4
    max_steps_per_frame: int = 8
    max_turns_per_step: int = 8
    max_tool_calls: int = 40
    compression_mode: CompressionMode = "tree"
    compression_frame_trigger_tokens: int = 6144

    def __post_init__(self) -> None:
        numeric = {
            name: value
            for name, value in asdict(self).items()
            if name != "compression_mode"
        }
        for name, value in numeric.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.compression_mode != "tree":
            raise ValueError("compression_mode must be tree")


@dataclass(frozen=True)
class Step:
    id: str
    goal: str
    mode: ExecutionMode


@dataclass(frozen=True)
class StepResult:
    status: StepStatus
    summary: str


@dataclass(frozen=True)
class PlanPatch:
    steps: tuple[Step, ...]
    planning_complete: bool
    thinking: str = ""


@dataclass
class AgentBudget:
    model_calls: int = 0
    tool_calls: int = 0
    frames: int = 0


@dataclass(frozen=True)
class RecoveryContext:
    step_goal: str
    failure_summary: str


@dataclass
class _CompressionFrameState:
    frame_id: str
    goal: str
    depth: int
    start_index: int
    evidence: list[CompressionEvidence]


@dataclass
class _CompressionStepState:
    step: Step
    all_steps: list[Step]
    completed: dict[str, StepResult]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, float | int]:
    usage = usage or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "cost": float(usage.get("cost", 0.0) or 0.0),
    }


class OpenAICompatibleModelClient:
    """Minimal text-only client for OpenAI-compatible chat completion servers."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "",
        timeout: float = 120,
        max_retries: int = 3,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.model = model
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, messages: Sequence[dict[str, str]]) -> ModelTurn:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        attempts: list[dict[str, Any]] = []
        for attempt_index in range(self.max_retries):
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json={"model": self.model, "messages": list(messages)},
                    )
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"].get("content") or ""
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "success",
                        "http_status": response.status_code,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                )
                return ModelTurn(
                    content=content,
                    usage=normalize_usage(payload.get("usage")),
                    raw_response=payload,
                    attempts=tuple(attempts),
                )
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                attempt: dict[str, Any] = {
                    "attempt": attempt_index + 1,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
                if isinstance(exc, httpx.HTTPStatusError):
                    attempt["http_status"] = exc.response.status_code
                    attempt["response_body"] = exc.response.text
                attempts.append(attempt)
                if attempt_index < self.max_retries - 1:
                    time.sleep(2 * (2**attempt_index) + random.uniform(0, 1))
                    continue
                raise ModelError(str(exc)) from exc

        raise ModelError("model request failed")


def extract_json_dict(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from plain, fenced, or wrapped text."""
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


def _parse_steps(
    raw_steps: Any,
    *,
    next_step_index: int,
    existing_step_count: int,
    max_steps: int,
) -> tuple[Step, ...]:
    if not isinstance(raw_steps, list):
        raise ProtocolError("steps must be a list")
    if existing_step_count + len(raw_steps) > max_steps:
        raise ProtocolError("response exceeds the frame step budget")

    steps: list[Step] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ProtocolError("each step must be an object")
        goal = raw_step.get("step_goal")
        mode = raw_step.get("execution_mode")
        if not isinstance(goal, str) or not goal.strip():
            raise ProtocolError("step_goal must be a non-empty string")
        if mode not in ("execute", "decompose"):
            raise ProtocolError("execution_mode must be execute or decompose")
        steps.append(Step(f"step_{next_step_index + index}", goal.strip(), mode))
    return tuple(steps)


def parse_plan(
    text: str,
    *,
    next_step_index: int,
    existing_step_count: int,
    max_steps: int,
) -> PlanPatch:
    raw = extract_json_dict(text)
    if raw is None:
        raise ProtocolError("response did not contain a JSON object")
    complete = raw.get("planning_complete")
    if not isinstance(complete, bool):
        raise ProtocolError("planning_complete must be boolean")
    thinking = raw.get("thinking", "")
    if not isinstance(thinking, str):
        raise ProtocolError("thinking must be a string")
    steps = _parse_steps(
        raw.get("steps"),
        next_step_index=next_step_index,
        existing_step_count=existing_step_count,
        max_steps=max_steps,
    )
    if not complete and not steps:
        raise ProtocolError("planning_complete=false requires at least one step")
    return PlanPatch(steps, complete, thinking.strip())


def parse_action(text: str) -> tuple[str | None, str | None]:
    action_match = _ACTION_RE.search(text)
    done_match = _DONE_RE.search(text)
    action = action_match.group(1).strip() if action_match else None
    done = done_match.group(1).strip() if done_match else None
    return action or None, done or None


class RExRunner:
    """Recursive rolling-plan controller specialized for Agent-Diff actions."""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        executor: CodeExecutor,
        config: RExConfig | None = None,
        on_trace_update: Any | None = None,
        compression_log_dir: str | Path | None = None,
    ) -> None:
        self.model_client = model_client
        self.executor = executor
        self.config = config or RExConfig()
        self.on_trace_update = on_trace_update
        self.messages: list[dict[str, str]] = []
        self.full_messages: list[dict[str, str]] = []
        self.budget = AgentBudget()
        self.trace: dict[str, Any] = {}
        self.compression_log_dir = (
            Path(compression_log_dir) if compression_log_dir is not None else None
        )
        self._compression_frames: list[_CompressionFrameState] = []
        self._compression_steps: list[_CompressionStepState] = []
        self._compression_sequence = 0
        self._tool_sequence = 0

    def run(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        self.full_messages = [dict(message) for message in self.messages]
        self.budget = AgentBudget()
        self._compression_frames = []
        self._compression_steps = []
        self._compression_sequence = 0
        self._tool_sequence = 0
        self.trace = {
            "schema_version": 1,
            "agent": "rex_runner",
            "completed": False,
            "stage": "agent_started",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "config": asdict(self.config),
            "usage": normalize_usage(None),
            "agent_usage": normalize_usage(None),
            "budget": asdict(self.budget),
            "compression": {
                "mode": self.config.compression_mode,
                "checks": 0,
                "calls": 0,
                "applied": 0,
                "estimated_tokens_before": 0,
                "estimated_tokens_after": 0,
                "usage": normalize_usage(None),
            },
            "events": [],
            "evidence": [],
            "messages": self.full_messages,
            "active_messages": self.messages,
            "root_result": None,
        }
        self._emit("agent_started")

        try:
            root_result = self._run_frame("root", prompt, depth=1)
        except (BudgetExceeded, ModelError) as exc:
            root_result = StepResult("failed", str(exc))
            self.trace["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            self._emit(
                "budget_exhausted" if isinstance(exc, BudgetExceeded) else "model_error"
            )
        except Exception as exc:  # noqa: BLE001 - keep benchmark cleanup reachable
            root_result = StepResult("failed", str(exc))
            self.trace["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            self._emit("agent_error")

        self.trace["root_result"] = asdict(root_result)
        self.trace["completed"] = root_result.status == "success"
        self.trace["finished_at"] = utc_now()
        self._emit("agent_completed")
        return self.trace

    def _run_frame(self, frame_id: str, goal: str, *, depth: int) -> StepResult:
        state = _CompressionFrameState(
            frame_id=frame_id,
            goal=goal,
            depth=depth,
            start_index=len(self.messages),
            evidence=[],
        )
        self._compression_frames.append(state)
        try:
            result = self._run_frame_inner(frame_id, goal, depth=depth)
            if depth > 1 and result.status == "success":
                parent_goal = (
                    self._compression_frames[-2].goal
                    if len(self._compression_frames) > 1
                    else goal
                )
                parent_step = (
                    self._compression_steps[-1] if self._compression_steps else None
                )
                pending = (
                    self._pending_siblings(parent_step)
                    if parent_step is not None
                    else ()
                )
                scope = CompressionScope(
                    kind="frame",
                    frame_id=frame_id,
                    goal=goal,
                    status=result.status,
                    parent_goal=parent_goal,
                    depth=depth,
                    execution_mode="decomposed_frame",
                    pending_siblings=pending,
                )
                result = self._compress_completed_scope(
                    scope=scope,
                    start_index=state.start_index,
                    trigger_tokens=self.config.compression_frame_trigger_tokens,
                    evidence=state.evidence,
                    result=result,
                )
            return result
        finally:
            self._compression_frames.pop()

    def _run_frame_inner(self, frame_id: str, goal: str, *, depth: int) -> StepResult:
        self.budget.frames += 1
        self._sync_budget()
        self._emit(
            "frame_started", frame_id=frame_id, artifact={"goal": goal, "depth": depth}
        )

        steps: list[Step] = []
        completed: dict[str, StepResult] = {}
        planned_step_count = 0
        next_step_index = 1
        recovery: RecoveryContext | None = None
        # A fully successful batch restores the local recovery allowance.
        recovery_attempted = False

        while True:
            if planned_step_count >= self.config.max_steps_per_frame:
                result = StepResult("failed", "Frame exhausted its step budget.")
                self._emit(
                    "frame_completed", frame_id=frame_id, artifact=asdict(result)
                )
                return result

            patch = self._request_plan(
                frame_id=frame_id,
                goal=goal,
                depth=depth,
                planned_step_count=planned_step_count,
                next_step_index=next_step_index,
                recovery=recovery,
            )
            if patch is None:
                if planned_step_count == 0:
                    patch = PlanPatch(
                        (Step(f"step_{next_step_index}", goal, "execute"),),
                        True,
                        "",
                    )
                    self._emit(
                        "plan_created",
                        frame_id=frame_id,
                        artifact={
                            "fallback": True,
                            "steps": self._dump_steps(patch.steps),
                        },
                    )
                else:
                    result = StepResult(
                        "failed", "Planning did not produce a valid recovery plan."
                    )
                    self._emit(
                        "frame_completed", frame_id=frame_id, artifact=asdict(result)
                    )
                    return result

            recovery = None
            steps.extend(patch.steps)
            planned_step_count += len(patch.steps)
            next_step_index += len(patch.steps)
            batch_failed = False
            for offset, step in enumerate(patch.steps):
                result = self._run_step(
                    frame_id=frame_id,
                    frame_goal=goal,
                    depth=depth,
                    all_steps=steps,
                    completed=completed,
                    step=step,
                )
                completed[step.id] = result
                if result.status == "success":
                    continue

                dropped = patch.steps[offset + 1 :]
                dropped_ids = {item.id for item in dropped}
                steps[:] = [item for item in steps if item.id not in dropped_ids]
                self._emit(
                    "plan_batch_abort",
                    frame_id=frame_id,
                    step=step,
                    artifact={
                        "failed_step_id": step.id,
                        "failed_status": result.status,
                        "dropped_steps": self._dump_steps(dropped),
                        "remaining_step_budget": (
                            self.config.max_steps_per_frame - planned_step_count
                        ),
                        "recovery_exhausted": recovery_attempted,
                    },
                )
                if recovery_attempted:
                    self._emit(
                        "frame_completed", frame_id=frame_id, artifact=asdict(result)
                    )
                    return result
                recovery_attempted = True
                recovery = RecoveryContext(step.goal, result.summary)
                batch_failed = True
                break

            if batch_failed:
                continue
            recovery_attempted = False
            if not patch.planning_complete:
                continue

            result = StepResult("success", "All planned steps completed.")
            self._emit("frame_completed", frame_id=frame_id, artifact=asdict(result))
            return result

    def _run_step(
        self,
        *,
        frame_id: str,
        frame_goal: str,
        depth: int,
        all_steps: list[Step],
        completed: dict[str, StepResult],
        step: Step,
    ) -> StepResult:
        state = _CompressionStepState(
            step=step,
            all_steps=all_steps,
            completed=completed,
        )
        self._compression_steps.append(state)
        try:
            self._emit("step_started", frame_id=frame_id, step=step)
            is_direct = not (step.mode == "decompose" and depth < self.config.max_depth)
            if not is_direct:
                result = self._run_frame(
                    f"{frame_id}.{step.id}", step.goal, depth=depth + 1
                )
            else:
                result = self._run_direct_step(
                    frame_id=frame_id,
                    frame_goal=frame_goal,
                    step=step,
                    completed=self._render_completed(all_steps, completed),
                    forced_direct=step.mode == "decompose",
                )
            self._emit(
                "step_completed", frame_id=frame_id, step=step, artifact=asdict(result)
            )
            return result
        finally:
            self._compression_steps.pop()

    def _run_direct_step(
        self,
        *,
        frame_id: str,
        frame_goal: str,
        step: Step,
        completed: str,
        forced_direct: bool,
    ) -> StepResult:
        forced = (
            " This step reached maximum decomposition depth and must execute directly."
            if forced_direct
            else ""
        )
        self._append_message(
            {
                "role": "user",
                "content": f"""Direct subtask execution.{forced}

Current step (exclusive action scope):
{step.goal}

Parent task (context only, not authorization):
{frame_goal}

Prior completed steps:
{completed}

Use Bash commands, primarily curl, to complete only the current step. Respond with exactly one action at a time:
<thinking>brief reasoning</thinking>
<action>one Bash command</action>

After observing the result, continue with another action or finish with:
<thinking>brief reasoning</thinking>
<done>concise summary containing verified IDs and changes</done>

Rules:
- Inspect every API response before deciding the next action.
- Never repeat a successful create, update, delete, send, or upload operation.
- If a mutation times out or its response is ambiguous, query the resource state before retrying it.
- Use read calls to resolve ambiguous identifiers and verify requested mutations.
- Execute only the current step; do not act on parent-task requirements that are not explicitly part of it.
- Stop with <done> immediately when the current step postcondition is met, even if the parent task remains incomplete.
- Do not claim completion without evidence.""",
            }
        )

        for _ in range(self.config.max_turns_per_step):
            turn = self._call_model("direct_execution", frame_id=frame_id, step=step)
            action, done = parse_action(turn.content)
            if action:
                self._check_tool_budget()
                self.budget.tool_calls += 1
                self._sync_budget()
                try:
                    execution_result = self.executor.execute(action)
                    observation = self._format_observation(execution_result)
                except Exception as exc:  # noqa: BLE001 - executor implementations vary
                    execution_result = {"status": "error", "error": str(exc)}
                    observation = f"[executor error]\n{type(exc).__name__}: {exc}"
                observation_message = {
                    "role": "user",
                    "content": f"<observation>\n{observation}\n</observation>",
                }
                self._append_message(observation_message)
                self._record_tool_evidence(action, execution_result, observation)
                self._emit(
                    "tool_result",
                    frame_id=frame_id,
                    step=step,
                    artifact={
                        "action": action,
                        "result": execution_result,
                        "observation": observation,
                    },
                )
                continue
            if done:
                return StepResult("success", done)

            self._append_message(
                {
                    "role": "user",
                    "content": "Protocol correction: return either <action>...</action> or <done>...</done>.",
                }
            )
            self._emit(
                "protocol_warning",
                frame_id=frame_id,
                step=step,
                artifact={"response": turn.content},
            )

        self._append_message(
            {
                "role": "user",
                "content": "The direct step reached its turn limit. Summarize verified progress, blockers, and what must not be repeated in one concise paragraph. Do not call tools.",
            }
        )
        summary = self._call_model(
            "step_checkpoint", frame_id=frame_id, step=step
        ).content.strip()
        return StepResult("failed", summary or "Direct step reached its turn limit.")

    def _request_plan(
        self,
        *,
        frame_id: str,
        goal: str,
        depth: int,
        planned_step_count: int,
        next_step_index: int,
        recovery: RecoveryContext | None,
    ) -> PlanPatch | None:
        if recovery is None:
            prompt = f"""Planning phase. Do not execute Bash or answer the task.

Goal:
{goal}

Depth: {depth}/{self.config.max_depth}

Return ONLY JSON:
{{"thinking": string, "steps": [{{"step_goal": string, "execution_mode": "execute|decompose"}}], "planning_complete": boolean}}

Rules:
- Plan as far ahead as the current information reliably supports. Return one
  step, a partial batch, or the full remaining plan as appropriate. Stop before work whose specification materially depends on future observations.
- Use execute for a focused API objective and decompose for multi-stage work with independently verifiable children.
- Plan by observable postcondition, not API call. One step exclusively owns one mutation plus its prerequisite reads and read-only verification.
- Do not repeat completed work.
- planning_complete means plan coverage, not execution status. If this batch covers all unmet postconditions, planning_complete MUST be true now.
- A decomposed goal must be narrower than the current Goal."""
        else:
            prompt = f"""Recovery planning phase. Do not execute Bash or answer the task.

Goal:
{goal}

Failed step: {recovery.step_goal}
Failure evidence: {recovery.failure_summary}

Return ONLY JSON:
{{"thinking": string, "steps": [{{"step_goal": string, "execution_mode": "execute|decompose"}}], "planning_complete": boolean}}

Rules:
- Plan by observable postcondition, not API call. One step exclusively owns one mutation plus its prerequisite reads and read-only verification.
- Preserve successful side effects and never repeat a successful mutation.
- If mutation outcome was ambiguous, plan a read-only verification before any retry.
- planning_complete means plan coverage, not execution status. If this batch covers all unmet postconditions, planning_complete MUST be true now.
- A decomposed goal must be narrower than the current Goal."""

        self._append_message({"role": "user", "content": prompt})
        last_error = ""
        for attempt in range(2):
            turn = self._call_model("planning", frame_id=frame_id)
            try:
                patch = parse_plan(
                    turn.content,
                    next_step_index=next_step_index,
                    existing_step_count=planned_step_count,
                    max_steps=self.config.max_steps_per_frame,
                )
                self._emit(
                    "plan_created",
                    frame_id=frame_id,
                    artifact={
                        "thinking": patch.thinking,
                        "steps": self._dump_steps(patch.steps),
                        "planning_complete": patch.planning_complete,
                        "recovery": recovery is not None,
                    },
                )
                return patch
            except ProtocolError as exc:
                last_error = str(exc)
                self._emit(
                    "protocol_warning",
                    frame_id=frame_id,
                    artifact={"phase": "planning", "error": last_error},
                )
                if attempt == 0:
                    self._append_message(
                        {
                            "role": "user",
                            "content": f"Planning JSON was invalid: {last_error}. Return only a corrected JSON object and do not execute tools.",
                        }
                    )
        return None


    def _append_message(self, message: dict[str, str]) -> None:
        self.messages.append(dict(message))
        self.full_messages.append(dict(message))

    @staticmethod
    def _pending_siblings(
        state: _CompressionStepState,
    ) -> tuple[CompressionSibling, ...]:
        try:
            current_index = next(
                index
                for index, candidate in enumerate(state.all_steps)
                if candidate.id == state.step.id
            )
        except StopIteration:
            return ()
        return tuple(
            CompressionSibling(goal=candidate.goal, mode=candidate.mode)
            for candidate in state.all_steps[current_index + 1 : current_index + 2]
            if candidate.id not in state.completed
        )

    def _record_tool_evidence(
        self,
        action: str,
        execution_result: dict[str, Any],
        observation: str,
    ) -> None:
        self._tool_sequence += 1
        raw_exit_code = execution_result.get("exit_code")
        try:
            exit_code = int(raw_exit_code) if raw_exit_code is not None else None
        except (TypeError, ValueError):
            exit_code = None
        status = str(execution_result.get("status", "success"))
        if exit_code not in (None, 0):
            status = "error"
        error = execution_result.get("error")
        evidence = CompressionEvidence(
            call_id=f"bash-{self._tool_sequence:06d}",
            action=action,
            status=status,
            exit_code=exit_code,
            error=str(error) if error else None,
            observation=observation,
        )
        self.trace["evidence"].append(
            {
                **asdict(evidence),
                "output_sha256": hashlib.sha256(observation.encode("utf-8")).hexdigest(),
            }
        )
        for frame in self._compression_frames:
            frame.evidence.append(evidence)

    @staticmethod
    def _add_usage(target: dict[str, float | int], usage: dict[str, Any]) -> None:
        normalized = normalize_usage(usage)
        for key, value in normalized.items():
            target[key] += value

    def _write_compression_log(
        self,
        *,
        sequence: int,
        scope: CompressionScope,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if self.compression_log_dir is None:
            return {"status": "disabled"}
        safe_scope = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            f"{scope.kind}_{scope.frame_id}_{scope.step_id or 'frame'}",
        )
        path = self.compression_log_dir / f"{sequence:04d}_{safe_scope}.txt"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            rendered = (
                "REX COMPRESSION CALL\n"
                "="
                * 80
                + "\n"
                + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
                + "\n"
            )
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return {
                "status": "written",
                "path": str(path),
                "bytes": path.stat().st_size,
            }
        except OSError as exc:
            return {
                "status": "write_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _compress_completed_scope(
        self,
        *,
        scope: CompressionScope,
        start_index: int,
        trigger_tokens: int,
        evidence: Sequence[CompressionEvidence],
        result: StepResult,
    ) -> StepResult:
        if result.status != "success":
            return result
        compression = self.trace["compression"]
        compression["checks"] += 1
        source_messages = [dict(message) for message in self.messages[start_index:]]
        before_tokens = estimate_messages_tokens(source_messages)
        event = "frame_compress"
        if not source_messages or before_tokens <= trigger_tokens:
            self._emit(
                event,
                frame_id=scope.frame_id,
                artifact={
                    "triggered": False,
                    "applied": False,
                    "reason": "below_threshold",
                    "estimated_tokens_before": before_tokens,
                    "trigger_tokens": trigger_tokens,
                },
            )
            return result

        self._compression_sequence += 1
        sequence = self._compression_sequence
        prepared = prepare_compression_request(
            scope,
            source_messages,
            evidence,
        )
        compression["calls"] += 1
        fallback_reason: str | None = None
        turn: ModelTurn | None = None
        try:
            turn = self.model_client.generate(prepared.messages)
            self._add_usage(compression["usage"], turn.usage)
            self._add_usage(self.trace["usage"], turn.usage)
            if handoff_is_valid(turn.content):
                handoff = normalize_handoff(turn.content)
            else:
                fallback_reason = "the compressor returned an invalid handoff"
                handoff = deterministic_handoff(
                    scope, summary=result.summary, reason=fallback_reason
                )
        except Exception as exc:  # noqa: BLE001 - compression must not fail the task
            fallback_reason = f"{type(exc).__name__}: {exc}"
            handoff = deterministic_handoff(
                scope, summary=result.summary, reason=fallback_reason
            )

        checkpoint_text = render_compressed_history(scope, handoff, evidence)
        checkpoint = {"role": "user", "content": checkpoint_text}
        after_tokens = estimate_messages_tokens([checkpoint])
        applied = after_tokens < before_tokens
        if applied:
            self.messages[start_index:] = [checkpoint]
            self.full_messages.append(dict(checkpoint))
            result = StepResult(result.status, handoff)
            compression["applied"] += 1
            compression["estimated_tokens_before"] += before_tokens
            compression["estimated_tokens_after"] += after_tokens

        reason = (
            "fallback_applied"
            if fallback_reason and applied
            else "fallback_not_smaller"
            if fallback_reason
            else "applied"
            if applied
            else "not_smaller"
        )
        log_payload = {
            "sequence": sequence,
            "scope": asdict(scope),
            "trigger_tokens": trigger_tokens,
            "estimated_tokens_before": before_tokens,
            "estimated_tokens_after": after_tokens,
            "applied": applied,
            "reason": reason,
            "fallback_reason": fallback_reason,
            "request": {
                "messages": list(prepared.messages),
                "estimated_tokens": prepared.estimated_tokens,
                "history_sha256": prepared.history_sha256,
                "task_context": prepared.task_context,
                "evidence_count": prepared.evidence_count,
            },
            "response": {
                "content": turn.content if turn else None,
                "usage": normalize_usage(turn.usage) if turn else normalize_usage(None),
                "attempts": list(turn.attempts) if turn else [],
                "raw_response": turn.raw_response if turn else None,
            },
            "checkpoint": checkpoint_text,
        }
        log = self._write_compression_log(
            sequence=sequence, scope=scope, payload=log_payload
        )
        artifact = {
            "compression_id": f"compression-{sequence:04d}",
            "triggered": True,
            "applied": applied,
            "reason": reason,
            "fallback_reason": fallback_reason,
            "estimated_tokens_before": before_tokens,
            "estimated_tokens_after": after_tokens,
            "trigger_tokens": trigger_tokens,
            "compressor_input_tokens": prepared.estimated_tokens,
            "evidence_count": prepared.evidence_count,
            "log": log,
        }
        if fallback_reason:
            self._emit(
                "compression_fallback",
                frame_id=scope.frame_id,
                artifact=artifact,
            )
        if log.get("status") == "write_error":
            self._emit(
                "compression_log_error",
                frame_id=scope.frame_id,
                artifact={"compression_id": artifact["compression_id"], **log},
            )
        self._emit(event, frame_id=scope.frame_id, artifact=artifact)
        return result

    def _call_model(
        self,
        phase: str,
        *,
        frame_id: str,
        step: Step | None = None,
    ) -> ModelTurn:
        turn = self.model_client.generate(tuple(self.messages))
        self.budget.model_calls += 1
        self._append_message({"role": "assistant", "content": turn.content})
        self._add_usage(self.trace["agent_usage"], turn.usage)
        self._add_usage(self.trace["usage"], turn.usage)
        self._sync_budget()
        self._emit(
            "model_response",
            frame_id=frame_id,
            step=step,
            artifact={
                "phase": phase,
                "content": turn.content,
                "usage": normalize_usage(turn.usage),
                "attempts": list(turn.attempts),
                "raw_response": turn.raw_response,
            },
        )
        return turn

    def _check_tool_budget(self) -> None:
        if self.budget.tool_calls >= self.config.max_tool_calls:
            raise BudgetExceeded("Agent exceeded the global tool-call budget.")

    def _sync_budget(self) -> None:
        if self.trace:
            self.trace["budget"] = asdict(self.budget)

    def _emit(
        self,
        event: str,
        *,
        frame_id: str = "root",
        step: Step | None = None,
        artifact: dict[str, Any] | None = None,
    ) -> None:
        if not self.trace:
            return
        entry = {
            "event": event,
            "timestamp": utc_now(),
            "frame_id": frame_id,
            "step_id": step.id if step else None,
            "step_goal": step.goal if step else None,
            "artifact": artifact or {},
        }
        self.trace["events"].append(entry)
        self.trace["stage"] = event
        self.trace["updated_at"] = entry["timestamp"]
        self._sync_budget()
        if self.on_trace_update:
            self.on_trace_update(self.trace, event)

    @staticmethod
    def _render_completed(
        steps: Sequence[Step], completed: dict[str, StepResult]
    ) -> str:
        rows = [
            {
                "step_goal": step.goal,
                "status": completed[step.id].status,
                "summary": completed[step.id].summary,
            }
            for step in steps
            if step.id in completed
        ]
        return json.dumps(rows, ensure_ascii=False, indent=2)

    @staticmethod
    def _dump_steps(steps: Sequence[Step]) -> list[dict[str, str]]:
        return [
            {"id": step.id, "step_goal": step.goal, "execution_mode": step.mode}
            for step in steps
        ]

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


def run_rex_runner(
    *,
    model_client: ModelClient,
    prompt: str,
    executor: CodeExecutor,
    system_prompt: str,
    config: RExConfig | None = None,
    on_trace_update: Any | None = None,
    compression_log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the REx runner and return its full trace."""
    return RExRunner(
        model_client=model_client,
        executor=executor,
        config=config,
        on_trace_update=on_trace_update,
        compression_log_dir=compression_log_dir,
    ).run(prompt, system_prompt)


__all__ = [
    "RExRunner",
    "RExConfig",
    "ModelTurn",
    "OpenAICompatibleModelClient",
    "ProtocolError",
    "extract_json_dict",
    "parse_action",
    "parse_plan",
    "run_rex_runner",
]
