"""Shared provider/runtime primitives for REx runners."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any, cast

from ...models.content import TextBlock, ToolResultBlock, ToolUseBlock
from ...models.message import Message
from ...models.tool import ToolSpec
from ...models.trace import RExEvent, TokenUsage, TraceMessage
from ...trace.writer import TraceWriter
from .models import RExRunResult


class RExContextOverflowError(RuntimeError):
    """Provider rejected a request because it exceeded the context window."""


def is_context_overflow_error(exc: BaseException) -> bool:
    """Recognize context-length failures without swallowing ordinary 400s."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    if status is not None and status not in {400, 413}:
        return False
    text = str(exc).lower()
    markers = (
        "context length",
        "context window",
        "maximum context",
        "max context",
        "token limit",
        "too many tokens",
        "prompt is too long",
    )
    return any(marker in text for marker in markers)


def provider_chat_with_timeout(
    provider: Any,
    messages: list[Message],
    tools: list[ToolSpec] | None,
    *,
    timeout_s: int,
) -> tuple[Message, TokenUsage] | None:
    """Call a provider without allowing a stuck request to block the runner."""
    result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            result_queue.put(("ok", provider.chat(messages, tools=tools)), block=False)
        except BaseException as exc:
            result_queue.put(("err", exc), block=False)

    thread = threading.Thread(target=run, name="REx-chat-timeout", daemon=True)
    thread.start()
    try:
        status, payload = result_queue.get(timeout=timeout_s)
    except queue.Empty:
        return None
    if status == "err":
        raise cast(BaseException, payload)
    return cast(tuple[Message, TokenUsage], payload)


def safe_tool_result(
    tool_use: ToolUseBlock, text: str, *, is_error: bool = False
) -> ToolResultBlock:
    return ToolResultBlock(
        tool_use_id=tool_use.id,
        content=[TextBlock(text=text)],
        is_error=is_error,
    )


class RExRuntime:
    """Small, explicit runtime shared by migrated runners.

    The runner still owns its state machine and policy methods; this object owns
    the side-effectful protocol operations so future strategies do not need to
    inherit the legacy runner implementation.
    """

    def __init__(
        self,
        *,
        provider: Any,
        messages: list[Message],
        writer: TraceWriter,
        trace_id: str,
        writer_lock: threading.Lock,
        chat_timeout_s: int,
    ) -> None:
        self.provider = provider
        self.messages = messages
        self.writer = writer
        self.trace_id = trace_id
        self.writer_lock = writer_lock
        self.chat_timeout_s = chat_timeout_s

    def append_message(
        self,
        message: Message,
        usage: TokenUsage | None = None,
        *,
        internal: bool = True,
    ) -> None:
        self.messages.append(message)
        self.write_message(message, usage=usage, internal=internal)

    def write_message(
        self,
        message: Message,
        usage: TokenUsage | None = None,
        *,
        internal: bool = True,
    ) -> None:
        extra = {"internal": True, "source": "rex"} if internal else {}
        event = TraceMessage(
            trace_id=self.trace_id,
            message=message,
            usage=usage or TokenUsage(),
            extra=extra,
        )
        with self.writer_lock:
            self.writer.write_event(event)

    def write_event(
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
        trace_event = RExEvent(
            trace_id=self.trace_id,
            event=event,
            frame_id=frame_id,
            step_id=step_id,
            step_kind=step_kind,
            step_task=step_task,
            artifact=artifact or {},
            execution_snapshot=execution_snapshot or {},
            note=note,
        )
        with self.writer_lock:
            self.writer.write_event(trace_event)

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None) -> tuple[Message, TokenUsage] | None:
        return provider_chat_with_timeout(
            self.provider,
            messages,
            tools,
            timeout_s=self.chat_timeout_s,
        )

    @staticmethod
    def accumulate(target: Any, source: Any) -> None:
        target.usage.input_tokens += source.usage.input_tokens
        target.usage.output_tokens += source.usage.output_tokens
        target.turns += source.turns
        target.model_time_s += source.model_time_s
        target.tool_time_s += source.tool_time_s
        if source.timed_out:
            target.timed_out = True
            if isinstance(target, RExRunResult) and isinstance(source, RExRunResult):
                target.timeout_type = source.timeout_type
                target.timeout_seconds = source.timeout_seconds
