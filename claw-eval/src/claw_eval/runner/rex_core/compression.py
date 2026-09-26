"""Compression lifecycle adapter used during the high-confidence migration."""

from __future__ import annotations

from typing import Any


class RExCompressionController:
    """Own lifecycle notifications while legacy compression code is retained.

    The callbacks deliberately receive explicit context and return no data.  A
    later migration can move the compression implementation behind this seam
    without changing the frame engine or prompt strategies.
    """

    def __init__(self, runner: Any) -> None:
        self.runner = runner

    def before_model_call(self, *, focus: str | None = None) -> None:
        del focus

    def frame_started(self, *, frame_id: str, goal: str, depth: int) -> None:
        del frame_id, goal, depth

    def frame_finished(self, **context: Any) -> None:
        del context

    def step_started(self, **context: Any) -> None:
        del context

    def step_finished(self, **context: Any) -> None:
        del context
