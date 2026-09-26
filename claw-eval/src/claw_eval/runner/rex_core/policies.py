"""Resolve REx planning controls from environment settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ResolvedRExPolicy:
    """Resolved controls shared by the paper ablation runner."""

    planning_schedule: Literal["progressive", "one_shot", "single-plan"]

    @classmethod
    def from_environment(cls, environment: Any) -> "ResolvedRExPolicy":
        configured_schedule = getattr(
            environment, "rex_planning_schedule", None
        )
        schedule = configured_schedule or "progressive"
        if schedule not in {"progressive", "one_shot", "single-plan"}:
            raise ValueError("planning_schedule must be progressive, one_shot or single-plan")

        return cls(planning_schedule=schedule)
