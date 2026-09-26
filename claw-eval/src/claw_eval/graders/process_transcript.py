"""Build runner-neutral conversations for LLM judges.

Formal judge views expose the same observable information for native ReAct,
ReCAP, and PER runners:

- non-internal user/assistant text
- business-tool dispatches when the grader explicitly requests them

Runner control-plane metadata is deliberately excluded from formal judge
views.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from ..models.content import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from ..models.trace import ToolDispatch, TraceMessage

CONTROL_TOOL_NAMES = frozenset({"read_plan", "propose_plan_patch"})


@dataclass
class GradingViewContext:
    """Runner-neutral evidence available to existing grader formatters."""

    dispatches: list[ToolDispatch] = field(default_factory=list)
    include_tool_args: bool = True
    include_tool_results: bool = True


_GRADING_VIEW: ContextVar[GradingViewContext | None] = ContextVar(
    "claw_eval_grading_view", default=None
)


def get_grading_view_context() -> GradingViewContext | None:
    return _GRADING_VIEW.get()


@contextmanager
def grading_view_context(ctx: GradingViewContext | None) -> Iterator[None]:
    """Install a grading view for the duration of ``grader.grade``."""
    token = _GRADING_VIEW.set(ctx)
    try:
        yield
    finally:
        _GRADING_VIEW.reset(token)


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _format_tool_args(body: Any, *, include: bool) -> str:
    if not include:
        return ""
    return _json_dumps(body if body is not None else {})


def _format_tool_result(body: Any, *, include: bool) -> str:
    if not include:
        return "(omitted)"
    if body is None:
        return ""
    if isinstance(body, str):
        raw = body
    else:
        raw = _json_dumps(body)
    return raw


def _canonical_message_lines(
    message: TraceMessage,
    *,
    include_user_text: bool,
    include_assistant_text: bool,
    include_reasoning: bool,
    include_image: bool,
) -> list[str]:
    """Render the runner-independent, public portion of one message."""
    if message.extra.get("internal") or message.message.role == "system":
        return []

    role = message.message.role
    assistant_tool_turn = role == "assistant" and any(
        isinstance(block, ToolUseBlock) for block in message.message.content
    )
    lines: list[str] = []
    if (
        role == "assistant"
        and not assistant_tool_turn
        and include_reasoning
        and message.message.reasoning_content
    ):
        lines.append(f"[ASSISTANT THINKING]: {message.message.reasoning_content}")

    include_text = not assistant_tool_turn and (
        (role == "user" and include_user_text)
        or (role == "assistant" and include_assistant_text)
    )
    for block in message.message.content:
        if isinstance(block, TextBlock) and include_text and block.text.strip():
            lines.append(f"[{role.upper()}]: {block.text}")
        elif (
            isinstance(block, ImageBlock) and not assistant_tool_turn and include_image
        ):
            source = block.source_path or "inline image"
            lines.append(f"[IMAGE]: {source} ({block.mime_type})")
    return lines


def _canonical_dispatch_lines(
    dispatch: ToolDispatch,
    *,
    include_tool_use: bool,
    include_tool_result: bool,
    include_tool_args: bool,
    include_tool_results: bool,
) -> list[str]:
    """Render a business-tool dispatch without runner-specific metadata."""
    if dispatch.tool_name in CONTROL_TOOL_NAMES:
        return []

    lines: list[str] = []
    if include_tool_use:
        args = _format_tool_args(dispatch.request_body, include=include_tool_args)
        if args:
            lines.append(f"[TOOL CALL]: {dispatch.tool_name}({args})")
        else:
            lines.append(f"[TOOL CALL]: {dispatch.tool_name}()")

    if include_tool_result:
        result = _format_tool_result(
            dispatch.response_body,
            include=include_tool_results,
        )
        tag = "TOOL RESULT ERROR" if dispatch.response_status >= 400 else "TOOL RESULT"
        lines.append(f"[{tag}]: status={dispatch.response_status} {result}".rstrip())
    return lines


def build_canonical_transcript(
    messages: Sequence[TraceMessage],
    dispatches: Sequence[ToolDispatch] | None = None,
    *,
    include_user_text: bool = True,
    include_assistant_text: bool = True,
    include_tool_use: bool = False,
    include_tool_result: bool = False,
    include_tool_args: bool = True,
    include_tool_results: bool = True,
    include_reasoning: bool = False,
    include_image: bool = False,
) -> str:
    """Build one evidence-equivalent transcript for every runner.

    Tool evidence comes from ``ToolDispatch`` whenever it is available. This
    avoids comparing native message blocks with ReCAP/PER control-plane events.
    Old traces without dispatch records fall back to public message tool blocks.
    """
    dispatches = list(dispatches or [])
    if not dispatches and (include_tool_use or include_tool_result):
        return _build_legacy_message_transcript(
            messages,
            include_user_text=include_user_text,
            include_assistant_text=include_assistant_text,
            include_tool_use=include_tool_use,
            include_tool_result=include_tool_result,
            include_tool_args=include_tool_args,
            include_tool_results=include_tool_results,
            include_reasoning=include_reasoning,
            include_image=include_image,
        )

    events: list[tuple[str, int, list[str]]] = []
    ordinal = 0
    for message in messages:
        lines = _canonical_message_lines(
            message,
            include_user_text=include_user_text,
            include_assistant_text=include_assistant_text,
            include_reasoning=include_reasoning,
            include_image=include_image,
        )
        if lines:
            events.append((message.timestamp or "", ordinal, lines))
            ordinal += 1

    if include_tool_use or include_tool_result:
        for dispatch in dispatches:
            lines = _canonical_dispatch_lines(
                dispatch,
                include_tool_use=include_tool_use,
                include_tool_result=include_tool_result,
                include_tool_args=include_tool_args,
                include_tool_results=include_tool_results,
            )
            if lines:
                events.append((dispatch.timestamp or "", ordinal, lines))
                ordinal += 1

    events.sort(key=lambda item: (item[0], item[1]))
    return "\n".join(line for _, _, lines in events for line in lines)


def _build_legacy_message_transcript(
    messages: Sequence[TraceMessage],
    *,
    include_user_text: bool = True,
    include_assistant_text: bool = True,
    include_tool_use: bool = True,
    include_tool_result: bool = True,
    include_tool_args: bool = True,
    include_tool_results: bool = True,
    include_reasoning: bool = False,
    include_image: bool = False,
) -> str:
    """Legacy fallback for traces that have no dispatch records."""
    lines: list[str] = []
    control_tool_use_ids: set[str] = set()
    for m in messages:
        if m.extra.get("internal"):
            continue
        role = m.message.role
        if role == "system":
            continue
        assistant_tool_turn = role == "assistant" and any(
            isinstance(block, ToolUseBlock) for block in m.message.content
        )

        if (
            role == "assistant"
            and not assistant_tool_turn
            and include_reasoning
            and m.message.reasoning_content
        ):
            lines.append(f"[ASSISTANT THINKING]: {m.message.reasoning_content}")

        for block in m.message.content:
            if isinstance(block, TextBlock):
                if not block.text.strip():
                    continue
                if role == "user" and include_user_text:
                    lines.append(f"[USER]: {block.text}")
                elif (
                    role == "assistant"
                    and not assistant_tool_turn
                    and include_assistant_text
                ):
                    lines.append(f"[ASSISTANT]: {block.text}")
            elif isinstance(block, ToolUseBlock) and role == "assistant":
                if block.name in CONTROL_TOOL_NAMES:
                    control_tool_use_ids.add(block.id)
                    continue
                if include_tool_use:
                    args = _format_tool_args(
                        block.input,
                        include=include_tool_args,
                    )
                    if args:
                        lines.append(f"[TOOL CALL]: {block.name}({args})")
                    else:
                        lines.append(f"[TOOL CALL]: {block.name}()")
            elif isinstance(block, ToolResultBlock):
                if block.tool_use_id in control_tool_use_ids:
                    continue
                if not include_tool_result:
                    continue
                text_parts = [
                    tb.text for tb in block.content if isinstance(tb, TextBlock)
                ]
                result_text = "\n".join(text_parts)
                if not include_tool_results:
                    result_text = "(omitted)"
                tag = "TOOL RESULT ERROR" if block.is_error else "TOOL RESULT"
                lines.append(f"[{tag}]: {result_text}")
            elif isinstance(block, ImageBlock) and include_image:
                source = block.source_path or "inline image"
                lines.append(f"[IMAGE]: {source} ({block.mime_type})")

    return "\n".join(lines)


def conversation_for_judge(
    messages: Sequence[TraceMessage],
    *,
    detailed: bool = False,
    include_tool_use: bool = False,
    include_tool_result: bool = False,
    include_reasoning: bool = False,
    include_image: bool = False,
    include_internal: bool = False,
    include_user_text: bool = True,
    include_assistant_text: bool = True,
) -> str | None:
    """Return a canonical conversation when grading context is installed.

    Returns ``None`` when the caller should use the legacy formatter path.
    """
    ctx = get_grading_view_context()
    if ctx is None:
        return None

    # Explicit include_internal means the caller wants raw messages; do not override.
    if include_internal:
        return None

    return build_canonical_transcript(
        messages,
        ctx.dispatches,
        include_user_text=include_user_text,
        include_assistant_text=include_assistant_text,
        include_tool_use=detailed and include_tool_use,
        include_tool_result=detailed and include_tool_result,
        include_tool_args=ctx.include_tool_args,
        include_tool_results=ctx.include_tool_results,
        include_reasoning=detailed and include_reasoning,
        include_image=detailed and include_image,
    )
