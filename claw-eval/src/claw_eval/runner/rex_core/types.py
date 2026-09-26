"""Data-only contracts shared by REx execution paths."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import (
    RExPlanPatch,
    RExRunResult,
    RExStep,
    RExStepResult,
)
from ...models.trace import TokenUsage


@dataclass
class RExFrameResult:
    result: RExStepResult
    usage: TokenUsage = field(default_factory=TokenUsage)
    turns: int = 0
    model_time_s: float = 0.0
    tool_time_s: float = 0.0
    timed_out: bool = False


@dataclass
class RExGroundPlanResult:
    result: RExRunResult
    patch: RExPlanPatch | None = None


@dataclass
class RExStepExecutionResult:
    result: RExStepResult
    usage: TokenUsage = field(default_factory=TokenUsage)
    turns: int = 0
    model_time_s: float = 0.0
    tool_time_s: float = 0.0
    timed_out: bool = False


@dataclass(frozen=True)
class RExReplanContext:
    failed_step_id: str
    failed_step_goal: str
    failed_status: str
    failure_summary: str


@dataclass
class RExPlanBatchResult(RExGroundPlanResult):
    thinking: str = ""
