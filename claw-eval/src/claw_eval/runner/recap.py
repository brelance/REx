"""Recursive context-aware planning helpers for the agent loop."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ..models.content import TextBlock, ToolResultBlock
from ..models.message import Message

class RecapState(str, Enum):
    """High-level ReCAP state for trace/debug output."""

    INIT = "init"
    DOWN = "down"
    ACTION_TAKEN = "action_taken"
    UP = "up"
    FINALIZE = "finalize"


class RecapInfo(BaseModel):
    """Model-produced task reasoning and subtask list."""

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
        if not self.info_list:
            return RecapInfo()
        return self.info_list[-1]

    def set_obs(self, obs: str) -> None:
        if obs:
            self.obs_list.append(obs)

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


_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL
)


def parse_recap_json(text: str) -> RecapInfo | None:
    """Parse a ReCAP JSON object from assistant text, returning None on mismatch."""
    text = text.strip()
    if not text:
        return None
    m = _JSON_FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    if "think" not in raw or "subtasks" not in raw:
        return None
    subtasks = raw.get("subtasks")
    if not isinstance(subtasks, list):
        return None
    return RecapInfo(
        think=str(raw.get("think", "")),
        subtasks=[str(item) for item in subtasks if str(item).strip()],
    )


def extract_tool_observation(message: Message, max_chars: int) -> str:
    """Render a compact observation string from a tool-result/media message."""
    parts: list[str] = []
    for block in message.content:
        if block.type == "tool_result":
            block = (
                block
                if isinstance(block, ToolResultBlock)
                else ToolResultBlock.model_validate(block)
            )
            text = "\n".join(t.text for t in block.content)
            if block.is_error:
                text = f"[tool error]\n{text}"
            parts.append(text)
        elif block.type == "text":
            parts.append(block.text)
        elif block.type in {"image", "audio", "video"}:
            parts.append(f"[{block.type} content attached]")
    rendered = "\n\n".join(p for p in parts if p).strip()
    if len(rendered) > max_chars:
        return rendered[:max_chars] + "\n[observation truncated]"
    return rendered


class RecapController:
    """State machine for SWEbench-style recursive planning in claw-eval."""

    def __init__(
        self,
        root_task: str,
        *,
        max_depth: int = 6,
        max_subtasks: int = 12,
        max_obs_chars: int = 6000,
        max_tree_chars: int = 20000,
        force_final_answer: bool = True,
        action_taken_prompt_variant: str = "baseline",
    ) -> None:
        if action_taken_prompt_variant not in {"baseline", "sibling_guard"}:
            raise ValueError(
                "recap_action_taken_prompt_variant must be 'baseline' or "
                f"'sibling_guard', got {action_taken_prompt_variant!r}"
            )
        self.force_final_answer = force_final_answer
        self.action_taken_prompt_variant = action_taken_prompt_variant
        self.root = RecapNode(root_task)
        self.node = self.root
        self.depth = 1
        self.max_depth = max(1, max_depth)
        self.max_subtasks = max_subtasks
        self.max_obs_chars = max_obs_chars
        self.max_tree_chars = max_tree_chars
        self.state = RecapState.INIT
        self._format_retry_pending = False

    @property
    def current_task(self) -> str:
        return self.node.task_name

    def initial_prompt(self) -> Message:
        return Message(
            role="user",
            content=[TextBlock(text=self._down_prompt(self.root.task_name))],
        )

    def observe_tool_result(self, message: Message) -> str:
        obs = extract_tool_observation(message, self.max_obs_chars)
        self.node.set_obs(obs)
        return obs

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
        """Advance tree from assistant text."""
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
                continue_loop=False, state=self.state, note="non_recap_final"
            )

        self._format_retry_pending = False
        info.subtasks = info.subtasks[: self.max_subtasks]

        if info.subtasks and self.depth >= self.max_depth:
            info.subtasks = []
            self.node.set_info(info)
            return RecapStep(
                continue_loop=True,
                state=self.state,
                prompt=self._depth_limit_prompt(),
                note="max_depth_reached",
            )

        self.node.set_info(info)

        if info.subtasks:
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
        if self.depth == 0 or self.node.parent is None:
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
        self.depth = max(0, self.depth - 1)
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
        snapshot_root = self.root
        data = snapshot_root.to_dict()
        rendered = json.dumps(data, ensure_ascii=False)
        if len(rendered) <= self.max_tree_chars:
            return data
        return {
            "task_name": snapshot_root.task_name,
            "truncated": True,
            "current_task": self.node.task_name,
            "depth": self.depth,
        }

    def _json_instruction(self) -> str:
        return (
            'Return ONLY a JSON object with this schema: {"think": string, "subtasks": string[]}.\n'
            "The subtasks are internal planning state, not the final answer to the user."
        )

    def _format_feedback_prompt(self) -> str:
        if self._at_max_depth():
            return (
                "Your previous response was not valid ReCAP JSON. "
                f"{self._depth_limit_guidance()} "
                'When complete, return ONLY {"think": "brief reasoning", '
                '"subtasks": []}.'
            )
        return (
            "Your previous response was not valid ReCAP JSON. Return ONLY a JSON "
            'object like {"think": "brief reasoning", "subtasks": ["next task"]}. '
            "Use an empty subtasks list when the current task is complete."
        )

    def _at_max_depth(self) -> bool:
        return self.depth >= self.max_depth

    def _depth_limit_guidance(self) -> str:
        return (
            f"Maximum recursion depth ({self.max_depth}) has been reached. "
            "Do not decompose this task or return non-empty subtasks. If work "
            "remains, call the appropriate tools and complete it directly at "
            "this depth. Return an empty subtask list only when the current "
            "task is complete."
        )

    def _depth_limit_prompt(self) -> str:
        return f"""Your current task:
{self.node.task_name}

{self._depth_limit_guidance()}

{self._json_instruction()}"""

    def _down_prompt(self, task_name: str) -> str:
        guidance = (
            self._depth_limit_guidance()
            if self._at_max_depth()
            else """You can decompose the task if it is too complex, empty list if you think your
current task is done, further decompose the subtask by not calling any tool,
and/or make tool calls to make progress."""
        )
        return f"""OK.

Your current task:
{task_name}

{guidance}

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
            if self._at_max_depth():
                update_guidance = self._depth_limit_guidance()
            else:
                update_guidance = """Update the plan based on the latest observation.
If the current task is complete, return an empty subtask list.
If more work is needed, return only the remaining subtasks required for
this current task.
If a tool call is the right next step, call the tool."""
            return f"""Latest observation:
{observation or "[no textual observation]"}

Your current task:
{task_name}

Your previously proposed subtasks:
{remaining}

{update_guidance}
Do not repeat successful irreversible actions such as sending, creating,
deleting, publishing, or updating records.

Positive example:
  Earlier plan for a parent task included these sibling tasks:
  "Check user's calendar", "Check Mike's calendar", "Find a free slot",
  "Create event"

  Current task:
  Check user's calendar

  Latest observation:
  The user's calendar has been retrieved successfully.

  Correct response:
  {{"think": "The user's calendar has been retrieved successfully, so the current task is complete. The remaining meeting-scheduling steps are siblings, not child subtasks of checking the user's calendar.", "subtasks": []}}

{self._json_instruction()}"""
        guidance = (
            self._depth_limit_guidance()
            if self._at_max_depth()
            else """You can refine the subtasks based on the latest observation, empty list if you
think your current task is done, further decompose the first subtask by not
calling any tool, and/or make tool calls to make progress."""
        )
        return f"""Latest observation:
{observation or "[no textual observation]"}

Your current task:
{task_name}

Your previously proposed subtasks:
{remaining}

{guidance}

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
        guidance = (
            self._depth_limit_guidance()
            if self._at_max_depth()
            else """You can refine the subtasks based on the latest observation, empty list if you
think your current task is done, further decompose the first subtask by not
calling any tool, and/or make tool calls to make progress."""
        )
        return f"""You have determined that the task {done_task_name} has been completed.

Now, you return to the parent task.
Your current task: {previous_stage_task_name}

Your previous think: {previous_stage_think}

Your remaining subtasks:
{remaining}

{guidance}

{self._json_instruction()}"""

    def _finalize_prompt(self) -> str:
        return f"""The recursive task tree is complete.

Original user request:
{self.root.task_name}

Now provide the final answer to the user using the completed work and tool
results in the conversation. Do not return ReCAP planning JSON. Do not expose
the internal task tree, planning states, or hidden reasoning. Give a clear,
direct account of the result and include any deliverables, important findings,
or limitations the user needs to know.

If final verification reveals that essential work is still unfinished, you may
call an available tool. Otherwise, respond with the final answer now."""
