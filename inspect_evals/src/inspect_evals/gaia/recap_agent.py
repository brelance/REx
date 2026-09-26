"""Recursive context-aware planning baseline agent for GAIA."""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import Enum
from typing import Any, Literal

from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Model,
    ModelOutput,
    execute_tools,
    get_model,
)
from inspect_ai.tool import Tool, bash, python, web_browser
from pydantic import BaseModel, Field

from inspect_evals.gaia._json_parsing import json_with_fence_fallback
from inspect_evals.gaia.dataset import GAIA_FINAL_ANSWER_FORMAT
from inspect_evals.gaia.gaia import DEFAULT_AGENT_INSTRUCTIONS

RECAP_INSTRUCTIONS = DEFAULT_AGENT_INSTRUCTIONS.replace(
    "Please think step by step before calling tools. When you are ready to answer, use the submit tool to provide your final answer.",
    "Follow the ReCAP planning prompts exactly. Use tools when they help the current task, and provide a final answer only when the host requests it.",
)


class RecapState(str, Enum):
    """High-level state of the ReCAP controller."""

    INIT = "init"
    DOWN = "down"
    ACTION_TAKEN = "action_taken"
    UP = "up"
    FINALIZE = "finalize"


class RecapInfo(BaseModel):
    """Model-produced reasoning and child tasks for one ReCAP node."""

    think: str = ""
    subtasks: list[str] = Field(default_factory=list)


class RecapNode:
    """A task node in the recursive ReCAP tree."""

    def __init__(self, task_name: str, parent: RecapNode | None = None) -> None:
        self.task_name = task_name
        self.parent = parent
        self.children: list[RecapNode] = []
        self.info_list: list[RecapInfo] = []
        self.obs_list: list[str] = []

    def add_child(self, child: RecapNode) -> None:
        self.children.append(child)
        child.parent = self

    def set_info(self, info: RecapInfo) -> None:
        self.info_list.append(info)

    def latest_info(self) -> RecapInfo:
        return self.info_list[-1] if self.info_list else RecapInfo()

    def set_obs(self, observation: str) -> None:
        if observation:
            self.obs_list.append(observation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "children": [child.to_dict() for child in self.children],
            "info_list": [info.model_dump() for info in self.info_list],
            "obs_list": self.obs_list,
        }


class RecapStep(BaseModel):
    """Controller output after processing one assistant response."""

    continue_loop: bool
    state: RecapState
    prompt: str | None = None
    done_task_name: str | None = None
    remaining_subtasks: list[str] = Field(default_factory=list)
    note: str = ""
    done: bool = False


def _validate_recap_config(
    *,
    max_depth: int,
    max_subtasks: int,
    max_obs_chars: int,
    max_tree_chars: int,
    action_taken_prompt_variant: str,
) -> None:
    positive_values = {
        "max_depth": max_depth,
        "max_subtasks": max_subtasks,
        "max_obs_chars": max_obs_chars,
        "max_tree_chars": max_tree_chars,
    }
    for name, value in positive_values.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if action_taken_prompt_variant not in {"baseline", "sibling_guard"}:
        raise ValueError(
            "action_taken_prompt_variant must be 'baseline' or 'sibling_guard'"
        )


def parse_recap_json(text: str) -> RecapInfo | None:
    """Parse ReCAP JSON with a fence fallback, returning ``None`` on mismatch."""
    try:
        raw = json_with_fence_fallback(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict) or "think" not in raw or "subtasks" not in raw:
        return None
    subtasks = raw.get("subtasks")
    if not isinstance(subtasks, list):
        return None
    return RecapInfo(
        think=str(raw.get("think", "")),
        subtasks=[str(item).strip() for item in subtasks if str(item).strip()],
    )


def extract_tool_observation(messages: Sequence[ChatMessage], max_chars: int) -> str:
    """Render one compact observation from a completed tool-call batch."""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, ChatMessageTool):
            continue
        text = message.text.strip()
        if message.error is not None:
            error_text = message.error.message
            text = f"[tool error]\n{text or error_text}"
        if text:
            parts.append(text)
    rendered = "\n\n".join(parts)
    if len(rendered) > max_chars:
        return rendered[:max_chars] + "\n[observation truncated]"
    return rendered


class RecapController:
    """State machine for recursive context-aware planning."""

    def __init__(
        self,
        root_task: str,
        *,
        max_depth: int = 4,
        max_subtasks: int = 12,
        max_obs_chars: int = 6000,
        max_tree_chars: int = 20000,
        force_final_answer: bool = True,
        action_taken_prompt_variant: Literal["baseline", "sibling_guard"] = (
            "baseline"
        ),
    ) -> None:
        _validate_recap_config(
            max_depth=max_depth,
            max_subtasks=max_subtasks,
            max_obs_chars=max_obs_chars,
            max_tree_chars=max_tree_chars,
            action_taken_prompt_variant=action_taken_prompt_variant,
        )

        self.force_final_answer = force_final_answer
        self.action_taken_prompt_variant = action_taken_prompt_variant
        self.root = RecapNode(root_task)
        self.node = self.root
        self.depth = 1
        self.max_depth = max_depth
        self.max_subtasks = max_subtasks
        self.max_obs_chars = max_obs_chars
        self.max_tree_chars = max_tree_chars
        self.state = RecapState.INIT
        self._format_retry_pending = False

    @property
    def current_task(self) -> str:
        return self.node.task_name

    def initial_prompt(self) -> str:
        return self._down_prompt(self.root.task_name)

    def observe_tool_results(self, messages: Sequence[ChatMessage]) -> str:
        observation = extract_tool_observation(messages, self.max_obs_chars)
        self.node.set_obs(observation)
        return observation

    def after_tool_action(self, observation: str) -> RecapStep:
        self._format_retry_pending = False
        info = self.node.latest_info()
        self.state = RecapState.ACTION_TAKEN
        return RecapStep(
            continue_loop=True,
            state=self.state,
            prompt=self._action_taken_prompt(
                observation=observation,
                task_name=self.node.task_name,
                remaining_subtasks=info.subtasks,
            ),
            remaining_subtasks=info.subtasks,
        )

    def process_assistant_text(self, text: str) -> RecapStep:
        """Advance the task tree from one assistant text response."""
        if self.state == RecapState.FINALIZE:
            return RecapStep(
                continue_loop=False,
                state=self.state,
                note="final_answer",
                done=True,
            )

        info = parse_recap_json(text)
        if info is None:
            if not self._format_retry_pending:
                self._format_retry_pending = True
                return RecapStep(
                    continue_loop=True,
                    state=self.state,
                    prompt=self._format_feedback_prompt(),
                    note="invalid_recap_json_retry",
                )
            self._format_retry_pending = False
            return RecapStep(
                continue_loop=False,
                state=self.state,
                note="non_recap_final",
            )

        self._format_retry_pending = False
        info.subtasks = info.subtasks[: self.max_subtasks]
        self.node.set_info(info)

        if info.subtasks:
            if self.depth >= self.max_depth:
                self.state = RecapState.DOWN
                return RecapStep(
                    continue_loop=True,
                    state=self.state,
                    prompt=self._max_depth_prompt(),
                    remaining_subtasks=info.subtasks,
                    note="max_depth_reached",
                )
            next_task = info.subtasks[0]
            child = RecapNode(next_task, parent=self.node)
            self.node.add_child(child)
            self.node = child
            self.depth += 1
            self.state = RecapState.DOWN
            return RecapStep(
                continue_loop=True,
                state=self.state,
                prompt=self._down_prompt(next_task),
                remaining_subtasks=info.subtasks[1:],
            )

        done_task = self.node.task_name
        self.state = RecapState.UP
        if self.node.parent is None:
            if self.force_final_answer:
                self.state = RecapState.FINALIZE
                return RecapStep(
                    continue_loop=True,
                    state=self.state,
                    prompt=self._finalize_prompt(),
                    done_task_name=done_task,
                    note="task_tree_complete",
                )
            return RecapStep(
                continue_loop=False,
                state=self.state,
                done_task_name=done_task,
                note="complete",
                done=True,
            )

        self.node = self.node.parent
        self.depth -= 1
        parent_info = self.node.latest_info()
        remaining = parent_info.subtasks[1:]
        return RecapStep(
            continue_loop=True,
            state=self.state,
            prompt=self._up_prompt(
                done_task_name=done_task,
                previous_stage_task_name=self.node.task_name,
                previous_stage_think=parent_info.think,
                remaining_subtasks=remaining,
            ),
            done_task_name=done_task,
            remaining_subtasks=remaining,
        )

    def tree_snapshot(self) -> dict[str, Any]:
        data = self.root.to_dict()
        if len(json.dumps(data, ensure_ascii=False)) <= self.max_tree_chars:
            return data
        return {
            "task_name": self.root.task_name,
            "truncated": True,
            "current_task": self.node.task_name,
            "depth": self.depth,
        }

    @staticmethod
    def _json_instruction() -> str:
        return (
            'Return ONLY a JSON object with this schema: {"think": string, '
            '"subtasks": string[]}.\n'
            "The subtasks are internal planning state, not the final answer to the user."
        )

    def _format_feedback_prompt(self) -> str:
        return (
            "Your previous response was not valid ReCAP JSON. Return ONLY a JSON "
            'object like {"think": "brief reasoning", "subtasks": ["next task"]}. '
            "Use an empty subtasks list when the current task is complete."
        )

    def _max_depth_prompt(self) -> str:
        return f"""The maximum ReCAP depth ({self.max_depth}) has been reached. Do not create another subtask.

Your current task:
{self.node.task_name}

Call an available tool directly if more evidence is needed. If the current task is complete, return an empty subtasks list.

{self._json_instruction()}"""

    def _down_prompt(self, task_name: str) -> str:
        return f"""OK.

Your current task:
{task_name}

You can decompose the task if it is too complex, use an empty list if your current task is done, further decompose the subtask by not calling any tool, and/or make tool calls to make progress.

{self._json_instruction()}"""

    def _action_taken_prompt(
        self,
        *,
        observation: str,
        task_name: str,
        remaining_subtasks: list[str],
    ) -> str:
        remaining = (
            "\n".join(remaining_subtasks)
            if remaining_subtasks
            else "No remaining subtasks."
        )
        if self.action_taken_prompt_variant == "sibling_guard":
            return f"""Latest observation:
{observation or "[no textual observation]"}

Your current task:
{task_name}

Your previously proposed subtasks:
{remaining}

Update the plan based on the latest observation. If the current task is complete, return an empty subtask list. If more work is needed, return only the remaining subtasks required for this current task. If a tool call is the right next step, call the tool. Do not repeat successful irreversible actions such as sending, creating, deleting, publishing, or updating records.

Remember that sibling tasks from a parent plan are not child subtasks of the current task.

{self._json_instruction()}"""
        return f"""Latest observation:
{observation or "[no textual observation]"}

Your current task:
{task_name}

Your previously proposed subtasks:
{remaining}

You can refine the subtasks based on the latest observation, use an empty list if your current task is done, further decompose the first subtask by not calling any tool, and/or make tool calls to make progress.

{self._json_instruction()}"""

    def _up_prompt(
        self,
        *,
        done_task_name: str,
        previous_stage_task_name: str,
        previous_stage_think: str,
        remaining_subtasks: list[str],
    ) -> str:
        remaining = (
            "\n".join(remaining_subtasks)
            if remaining_subtasks
            else "No remaining subtasks."
        )
        return f"""You have determined that the task {done_task_name} has been completed.

Now, return to the parent task.
Your current task: {previous_stage_task_name}

Your previous think: {previous_stage_think}

Your remaining subtasks:
{remaining}

You can refine the subtasks based on the latest observation, use an empty list if your current task is done, further decompose the first subtask by not calling any tool, and/or make tool calls to make progress.

{self._json_instruction()}"""

    def _finalize_prompt(self) -> str:
        return f"""The recursive task tree is complete. Answer the original GAIA task using the completed work and tool results in the conversation.

Original task:
{self.root.task_name}

{GAIA_FINAL_ANSWER_FORMAT}

Return only the final answer. Do not return ReCAP planning JSON, reasoning, labels, or commentary. If essential verification is still needed, call an available tool; otherwise respond with the final answer now."""


class GaiaRecapRunner:
    """Run a ReCAP controller using Inspect model and tool interfaces."""

    def __init__(
        self,
        *,
        model: Model,
        messages: list[ChatMessage],
        tools: Sequence[Tool],
        controller: RecapController,
    ) -> None:
        self.model = model
        self.messages = messages
        self.tools = list(tools)
        self.controller = controller

    async def run(self) -> ModelOutput:
        self.messages.append(ChatMessageUser(content=self.controller.initial_prompt()))
        self._event(note="init")

        while True:
            output = await self.model.generate(self.messages, tools=self.tools)
            self.messages.append(output.message)

            if output.message.tool_calls:
                tool_messages, tools_output = await execute_tools(
                    self.messages, self.tools
                )
                self.messages.extend(tool_messages)
                if tools_output is not None:
                    output = tools_output
                observation = self.controller.observe_tool_results(tool_messages)
                step = self.controller.after_tool_action(observation)
            else:
                step = self.controller.process_assistant_text(output.completion)

            self._event(step=step)
            if step.continue_loop and step.prompt:
                self.messages.append(ChatMessageUser(content=step.prompt))
                continue
            return output

    def _event(self, *, step: RecapStep | None = None, note: str = "") -> None:
        event: dict[str, Any] = {
            "state": (step.state if step else self.controller.state).value,
            "depth": self.controller.depth,
            "current_task": self.controller.current_task,
            "done_task": step.done_task_name if step else None,
            "remaining_subtasks": step.remaining_subtasks if step else [],
            "tree": self.controller.tree_snapshot(),
            "note": step.note if step else note,
        }
        transcript().info(event, source="gaia.recap")


@agent
def gaia_recap_agent(
    *,
    max_depth: int = 4,
    max_subtasks: int = 12,
    max_obs_chars: int = 6000,
    max_tree_chars: int = 20000,
    force_final_answer: bool = True,
    action_taken_prompt_variant: Literal["baseline", "sibling_guard"] = "baseline",
    tool_timeout: int = 180,
    model: str | Model | None = None,
    tools: Sequence[Tool] | None = None,
) -> Agent:
    """Create the recursive context-aware planning baseline for GAIA."""
    _validate_recap_config(
        max_depth=max_depth,
        max_subtasks=max_subtasks,
        max_obs_chars=max_obs_chars,
        max_tree_chars=max_tree_chars,
        action_taken_prompt_variant=action_taken_prompt_variant,
    )
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
        state.messages.insert(0, ChatMessageSystem(content=RECAP_INSTRUCTIONS))
        controller = RecapController(
            goal,
            max_depth=max_depth,
            max_subtasks=max_subtasks,
            max_obs_chars=max_obs_chars,
            max_tree_chars=max_tree_chars,
            force_final_answer=force_final_answer,
            action_taken_prompt_variant=action_taken_prompt_variant,
        )
        runner = GaiaRecapRunner(
            model=get_model(model),
            messages=state.messages,
            tools=resolved_tools,
            controller=controller,
        )
        state.output = await runner.run()
        return state

    return execute


__all__ = [
    "GaiaRecapRunner",
    "RecapController",
    "RecapInfo",
    "RecapNode",
    "RecapState",
    "RecapStep",
    "extract_tool_observation",
    "gaia_recap_agent",
    "parse_recap_json",
]
