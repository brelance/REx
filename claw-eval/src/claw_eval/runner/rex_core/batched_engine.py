"""Entry point for the shared REx batched state machine."""

from __future__ import annotations

from typing import Any


class RExBatchFrameEngine:
    """Own the batched frame invocation while REx execution bodies are composed here.

    The callback boundary is intentional: it lets policy and compression
    implementations be composed without making the core package import a
    runner module.  The callback is removed once the remaining state-machine
    body has moved into this package.
    """

    def run_frame(self, runner: Any, **context: Any) -> Any:
        return runner._execute_batched_frame(**context)
