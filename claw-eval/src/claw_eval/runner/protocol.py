"""Helpers for parsing model-internal control protocols."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..models.message import Message


def is_reasoning_only_response(response: Message | None) -> bool:
    """Return whether a response has reasoning but no user-visible output."""
    if response is None or response.text.strip():
        return False
    if any(block.type != "text" for block in response.content):
        return False
    reasoning = response.reasoning_content
    return isinstance(reasoning, str) and bool(reasoning.strip())


def protocol_response_text(
    response: Message | None,
    *validators: Callable[[str], Any | None],
) -> str:
    """Use reasoning text only when it is a valid, isolated protocol payload."""
    if response is None:
        return ""

    visible_text = response.text
    if visible_text.strip():
        return visible_text
    if any(block.type != "text" for block in response.content):
        return visible_text

    reasoning = response.reasoning_content
    if not isinstance(reasoning, str) or not reasoning.strip():
        return visible_text

    for validator in validators:
        try:
            if validator(reasoning) is not None:
                return reasoning
        except ValueError:
            continue
    return visible_text
