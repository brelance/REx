"""Reflection and plan-execute baseline agents for GAIA."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from inspect_ai.agent import Agent, AgentPrompt, AgentState, AgentSubmit, agent, react
from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    Model,
    ModelOutput,
    execute_tools,
    get_model,
)
from inspect_ai.tool import Tool, bash, python, web_browser
from pydantic import BaseModel, ConfigDict, ValidationError

from inspect_evals.gaia._json_parsing import json_with_fence_fallback
from inspect_evals.gaia.dataset import GAIA_FINAL_ANSWER_FORMAT
from inspect_evals.gaia.gaia import DEFAULT_AGENT_INSTRUCTIONS

REFLECTION_PROMPT = """Briefly reflect on the completed tool results: state what you learned, whether the results advance the task, and the best next action. Then continue working. When the task is complete, call the submit tool with only the final answer."""

PLAN_EXECUTE_INSTRUCTIONS = DEFAULT_AGENT_INSTRUCTIONS.replace(
    "Please think step by step before calling tools. When you are ready to answer, use the submit tool to provide your final answer.",
    "Follow the current planning or execution phase exactly. Do not answer the original question until the host requests the final answer.",
)


class PlanExecuteProtocolError(ValueError):
    """Raised when a plan-execute control response violates its JSON protocol."""


class _PromptPlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str


class _PromptPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[_PromptPlanStep]


class _PromptStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    complete: Literal["success", "false"]
    summary: str


@dataclass(frozen=True)
class PlanExecuteStep:
    """A host-assigned executable plan step."""

    id: str
    goal: str


@dataclass(frozen=True)
class PlanExecuteStepResult:
    """The assessed model's result for one plan step."""

    complete: Literal["success", "false"]
    summary: str


def parse_plan_execute_plan(text: str, *, minimum_steps: int) -> list[str]:
    """Parse and validate plan JSON, with a Markdown fence fallback."""
    try:
        raw = json_with_fence_fallback(text)
    except json.JSONDecodeError as exc:
        raise PlanExecuteProtocolError(
            "response must contain valid JSON, optionally in a Markdown code fence"
        ) from exc

    try:
        parsed = _PromptPlan.model_validate(raw)
    except ValidationError as exc:
        raise PlanExecuteProtocolError(str(exc)) from exc

    if len(parsed.steps) < minimum_steps:
        raise PlanExecuteProtocolError(
            f"plan must contain at least {minimum_steps} step(s)"
        )

    goals = [step.goal.strip() for step in parsed.steps]
    if any(not goal for goal in goals):
        raise PlanExecuteProtocolError("step goals must not be empty")
    return goals


def parse_plan_execute_step_result(text: str) -> PlanExecuteStepResult:
    """Parse and validate step JSON, with a Markdown fence fallback."""
    try:
        raw = json_with_fence_fallback(text)
    except json.JSONDecodeError as exc:
        raise PlanExecuteProtocolError(
            "response must contain valid JSON, optionally in a Markdown code fence"
        ) from exc

    try:
        parsed = _PromptStepResult.model_validate(raw)
    except ValidationError as exc:
        raise PlanExecuteProtocolError(str(exc)) from exc

    summary = parsed.summary.strip()
    if not summary:
        raise PlanExecuteProtocolError("step summary must not be empty")
    return PlanExecuteStepResult(complete=parsed.complete, summary=summary)


async def _reflection_continue(state: AgentState) -> str | bool:
    """Request reflection after a complete non-submission tool-call batch."""
    if state.output.message.tool_calls:
        return REFLECTION_PROMPT
    return True


@agent
def gaia_reflection_agent(
    *,
    max_attempts: int = 1,
    tool_timeout: int = 180,
    model: str | Model | None = None,
    tools: Sequence[Tool] | None = None,
) -> Agent:
    """Create a ReAct baseline that reflects after every tool-call batch."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    resolved_tools = (
        list(tools)
        if tools is not None
        else [bash(tool_timeout), python(tool_timeout), *web_browser()]
    )
    return react(
        prompt=AgentPrompt(
            instructions=DEFAULT_AGENT_INSTRUCTIONS,
            assistant_prompt=None,
            submit_prompt=None,
        ),
        tools=resolved_tools,
        model=model,
        attempts=max_attempts,
        submit=AgentSubmit(answer_only=True, keep_in_messages=True),
        on_continue=_reflection_continue,
    )


class GaiaPlanExecuteRunner:
    """Sequential plan-execute runner with failure-triggered replanning."""

    def __init__(
        self,
        *,
        model: Model,
        messages: list[ChatMessage],
        tools: Sequence[Tool],
    ) -> None:
        self.model = model
        self.messages = messages
        self.tools = list(tools)
        self.completed: list[tuple[PlanExecuteStep, PlanExecuteStepResult]] = []
        self.next_step_index = 1

    async def run(self, goal: str) -> ModelOutput:
        plan = await self._request_plan(goal=goal, minimum_steps=2)

        while True:
            failed = False
            for index, step in enumerate(plan):
                result = await self._run_step(goal=goal, plan=plan, step=step)
                self._event(
                    "step_done",
                    step=step,
                    artifact={"complete": result.complete, "summary": result.summary},
                )
                if result.complete == "success":
                    self.completed.append((step, result))
                    continue

                dropped = plan[index + 1 :]
                self._event(
                    "plan_aborted",
                    step=step,
                    artifact={
                        "failure_summary": result.summary,
                        "dropped_steps": self._dump_steps(dropped),
                    },
                )
                plan = await self._request_plan(
                    goal=goal,
                    minimum_steps=1,
                    failed_step=step,
                    failure_summary=result.summary,
                    dropped_steps=dropped,
                )
                failed = True
                break

            if failed:
                continue
            return await self._final_answer(goal)

    async def _request_plan(
        self,
        *,
        goal: str,
        minimum_steps: int,
        failed_step: PlanExecuteStep | None = None,
        failure_summary: str | None = None,
        dropped_steps: Sequence[PlanExecuteStep] = (),
    ) -> list[PlanExecuteStep]:
        prompt = (
            _initial_plan_prompt(goal)
            if failed_step is None
            else _replan_prompt(
                goal=goal,
                completed=self._completed_payload(),
                failed_step=failed_step,
                failure_summary=failure_summary or "",
                dropped_steps=dropped_steps,
            )
        )
        self.messages.append(ChatMessageUser(content=prompt))

        while True:
            output = await self.model.generate(self.messages, tools=[])
            self.messages.append(output.message)
            try:
                goals = parse_plan_execute_plan(
                    output.completion, minimum_steps=minimum_steps
                )
            except PlanExecuteProtocolError as exc:
                self._event("plan_format_error", note=str(exc))
                self.messages.append(
                    ChatMessageUser(
                        content=_plan_format_correction(str(exc), minimum_steps)
                    )
                )
                continue

            steps = [self._new_step(goal) for goal in goals]
            self._event(
                "plan_created" if failed_step is None else "plan_replaced",
                artifact={"steps": self._dump_steps(steps)},
            )
            return steps

    async def _run_step(
        self,
        *,
        goal: str,
        plan: Sequence[PlanExecuteStep],
        step: PlanExecuteStep,
    ) -> PlanExecuteStepResult:
        self._event("step_start", step=step)
        self.messages.append(
            ChatMessageUser(
                content=_step_prompt(
                    goal=goal,
                    plan=plan,
                    step=step,
                    completed=self._completed_payload(),
                )
            )
        )

        while True:
            output = await self.model.generate(self.messages, tools=self.tools)
            self.messages.append(output.message)

            if output.message.tool_calls:
                tool_messages, _ = await execute_tools(self.messages, self.tools)
                self.messages.extend(tool_messages)
                self.messages.append(ChatMessageUser(content=_step_tool_reminder(step)))
                continue

            try:
                return parse_plan_execute_step_result(output.completion)
            except PlanExecuteProtocolError as exc:
                self._event("step_format_error", step=step, note=str(exc))
                self.messages.append(
                    ChatMessageUser(content=_step_format_correction(step, str(exc)))
                )

    async def _final_answer(self, goal: str) -> ModelOutput:
        self.messages.append(
            ChatMessageUser(
                content=_final_answer_prompt(goal, self._completed_payload())
            )
        )
        output = await self.model.generate(self.messages, tools=[])
        self.messages.append(output.message)
        self._event(
            "done",
            artifact={"completed_steps": len(self.completed)},
        )
        return output

    def _new_step(self, goal: str) -> PlanExecuteStep:
        step = PlanExecuteStep(id=f"step_{self.next_step_index}", goal=goal)
        self.next_step_index += 1
        return step

    def _completed_payload(self) -> list[dict[str, str]]:
        return [
            {
                "step_id": step.id,
                "goal": step.goal,
                "summary": result.summary,
            }
            for step, result in self.completed
        ]

    @staticmethod
    def _dump_steps(steps: Sequence[PlanExecuteStep]) -> list[dict[str, str]]:
        return [{"step_id": step.id, "goal": step.goal} for step in steps]

    def _event(
        self,
        event: str,
        *,
        step: PlanExecuteStep | None = None,
        artifact: dict[str, Any] | None = None,
        note: str = "",
    ) -> None:
        transcript().info(
            {
                "event": event,
                "step_id": step.id if step else None,
                "step_goal": step.goal if step else None,
                "artifact": artifact or {},
                "note": note,
            },
            source="gaia.plan_execute",
        )


@agent
def gaia_plan_execute_agent(
    *,
    tool_timeout: int = 180,
    model: str | Model | None = None,
    tools: Sequence[Tool] | None = None,
) -> Agent:
    """Create a sequential plan-execute baseline for GAIA."""
    resolved_tools = (
        list(tools)
        if tools is not None
        else [bash(tool_timeout), python(tool_timeout), *web_browser()]
    )

    async def execute(state: AgentState) -> AgentState:
        goal = next(
            (
                message.text
                for message in state.messages
                if isinstance(message, ChatMessageUser)
            ),
            "",
        )
        state.messages.insert(0, ChatMessageSystem(content=PLAN_EXECUTE_INSTRUCTIONS))
        runner = GaiaPlanExecuteRunner(
            model=get_model(model),
            messages=state.messages,
            tools=resolved_tools,
        )
        state.output = await runner.run(goal)
        return state

    return execute


def _initial_plan_prompt(goal: str) -> str:
    return f"""Create a complete ordered execution plan for the original GAIA task.

Original task:
{goal}

Return exactly one JSON object and no Markdown or prose:
{{"steps":[{{"goal":"first independently executable sub-step"}},{{"goal":"second independently executable sub-step"}}]}}

Include at least two non-empty sub-steps. The plan must cover all work needed before producing the final answer. Do not execute the plan or answer the task yet."""


def _replan_prompt(
    *,
    goal: str,
    completed: list[dict[str, str]],
    failed_step: PlanExecuteStep,
    failure_summary: str,
    dropped_steps: Sequence[PlanExecuteStep],
) -> str:
    failure = {
        "step_id": failed_step.id,
        "goal": failed_step.goal,
        "summary": failure_summary,
    }
    return f"""The current step failed. Discard every unexecuted step from the previous plan and create a complete replacement plan for all remaining work.

Original task:
{goal}

Successful completed steps:
{json.dumps(completed, ensure_ascii=False)}

Failed step:
{json.dumps(failure, ensure_ascii=False)}

Discarded unexecuted steps:
{json.dumps(GaiaPlanExecuteRunner._dump_steps(dropped_steps), ensure_ascii=False)}

Return exactly one JSON object and no Markdown or prose:
{{"steps":[{{"goal":"replacement sub-step"}}]}}

Include at least one non-empty sub-step, cover all remaining work, and do not repeat the failed approach unchanged. Do not execute the plan or answer the task yet."""


def _plan_format_correction(error: str, minimum_steps: int) -> str:
    return f"""Invalid plan response: {error}

Return exactly one JSON object with no Markdown, prose, or extra fields:
{{"steps":[{{"goal":"sub-step"}}]}}

The steps array must contain at least {minimum_steps} non-empty item(s)."""


def _step_prompt(
    *,
    goal: str,
    plan: Sequence[PlanExecuteStep],
    step: PlanExecuteStep,
    completed: list[dict[str, str]],
) -> str:
    return f"""Execute only the current plan step.

Original task:
{goal}

Current plan:
{json.dumps(GaiaPlanExecuteRunner._dump_steps(plan), ensure_ascii=False)}

Current step:
{json.dumps({"step_id": step.id, "goal": step.goal}, ensure_ascii=False)}

Successful prior results:
{json.dumps(completed, ensure_ascii=False)}

Use tools as needed. Do not execute later steps and do not provide the final answer yet. When this step is finished, stop calling tools and return exactly one of these JSON objects with no Markdown, prose, or extra fields:
{{"complete":"success","summary":"concise evidence obtained by this step"}}
{{"complete":"false","summary":"concise blocker or reason the step failed"}}"""


def _step_tool_reminder(step: PlanExecuteStep) -> str:
    return f"""Continue only {step.id}: {step.goal}

Use more tools if needed. Otherwise return exactly one JSON object:
{{"complete":"success","summary":"concise evidence"}}
or
{{"complete":"false","summary":"concise blocker"}}"""


def _step_format_correction(step: PlanExecuteStep, error: str) -> str:
    return f"""Invalid completion response for {step.id}: {error}

Continue the same step if more work is needed. Otherwise return exactly one JSON object with no Markdown, prose, or extra fields:
{{"complete":"success","summary":"concise evidence"}}
or
{{"complete":"false","summary":"concise blocker"}}"""


def _final_answer_prompt(goal: str, completed: list[dict[str, str]]) -> str:
    return f"""All planned steps completed successfully. Answer the original GAIA task using the trajectory and successful step summaries.

Original task:
{goal}

Successful step summaries:
{json.dumps(completed, ensure_ascii=False)}

{GAIA_FINAL_ANSWER_FORMAT}

Return only the final answer. Do not return reasoning, JSON, labels, or commentary."""


__all__ = [
    "GaiaPlanExecuteRunner",
    "PlanExecuteProtocolError",
    "PlanExecuteStep",
    "PlanExecuteStepResult",
    "gaia_plan_execute_agent",
    "gaia_reflection_agent",
    "parse_plan_execute_plan",
    "parse_plan_execute_step_result",
]
