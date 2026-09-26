"""Protocol models used by REx runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ...models.message import Message
from ...models.trace import TokenUsage

RExStepKind = Literal["direct", "recursive"]
RExStatus = Literal["success", "failed", "needs_user"]


@dataclass
class RExRunResult:
    final_message: Message | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    turns: int = 0
    model_time_s: float = 0.0
    tool_time_s: float = 0.0
    timed_out: bool = False
    timeout_type: str | None = None
    timeout_seconds: int | None = None
    needs_user: bool = False
    user_prompt: str | None = None
    root_status: str | None = None


class RExObserveResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str = ""
    ready_for_planning: bool = True
    clarifying_question: str = ""


class RExStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    task: str
    kind: RExStepKind = "direct"


class RExPromptStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_goal: str
    execution_mode: RExStepKind = "direct"


class RExPromptPlanPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[RExPromptStep] = Field(default_factory=list)
    planning_complete: bool = False


class RExPlanPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[RExStep] = Field(default_factory=list)
    complete: bool = False


class RExStepResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: RExStatus = "success"
    summary: str = ""
