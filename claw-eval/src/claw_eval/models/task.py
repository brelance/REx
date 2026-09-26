"""TaskDefinition — loaded from YAML task files (v3 aligned)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .tool import ToolEndpoint, ToolSpec


class Prompt(BaseModel):
    text: str
    language: str = "zh"
    attachments: list[str] = Field(default_factory=list)


class DeterministicCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    field: str | None = None
    tool_name: str | None = None
    min_calls: int | None = None
    categories: list[str] | None = None
    min_length: int | None = None
    patterns: list[str] | None = None
    keywords: list[str] | None = None
    description: str | None = None
    rubric: str | None = None

    @field_validator("keywords", mode="before")
    @classmethod
    def _coerce_keywords_to_str(cls, v: Any) -> list[str] | None:
        """YAML parses unquoted numbers as ints; coerce to str."""
        if v is None:
            return v
        return [str(item) for item in v]


class ScoringComponent(BaseModel):
    name: str
    weight: float
    check: DeterministicCheck


class SafetyCheck(BaseModel):
    type: str
    tool_name: str | None = None
    patterns: list[str] | None = None
    description: str = ""


class Environment(BaseModel):
    timeout_seconds: int = 300
    max_turns: int = 20
    mock_today: str | None = None  # "YYYY-MM-DD" — disables date offset in services, injected into system prompt
    fixtures: list[str] = Field(default_factory=list)
    env_snapshot_timeout: int = 10
    # TodoWrite settings
    enable_todo: bool = False
    todo_nag_rounds: int = 3  # 0 = no reminder
    # Context Compact settings
    enable_compact: bool = False
    compact_keep_recent: int = 20         # Layer 1: keep recent N turns intact
    compact_min_chars: int = 500          # Layer 1: don't truncate short results
    compact_threshold_pct: float = 0.70   # Layer 2: trigger at this % of context window
    compact_keep_recent_on_summary: int = 4  # Layer 2: keep N messages after summary
    compact_protect_tokens: int = 40_000  # Layer 2: protect recent token budget
    compact_max_auto_compacts: int = 2    # Layer 2: max auto-compacts per task run
    # REx execution settings
    enable_recap: bool = False
    recap_max_depth: int = 6
    recap_max_subtasks: int = 12
    recap_max_obs_chars: int = 6000
    recap_max_tree_chars: int = 20_000
    recap_force_final_answer: bool = True
    recap_action_taken_prompt_variant: str = "baseline"
    recap_mode: str = "recursive"
    recap_dag_max_workers: int = 4
    recap_dag_max_nodes: int = 16
    recap_dag_max_artifact_chars: int = 4000
    recap_dag_dynamic_replan: bool = False
    recap_dag_two_phase_commit: bool = True
    recap_dag_enable_observation_memory: bool = False
    recap_dag_injected_observation_max_chars: int = 16000
    recap_dag_enable_llm_observation_summary: bool = False
    recap_dag_observation_summary_max_observations: int = 16
    recap_dag_llm_observation_summary_chars: int = 1600
    rex_max_steps: int = 16
    rex_max_depth: int = 4
    rex_max_tool_calls: int = Field(default=100, ge=1)
    rex_observation_max_chars: int = 6000
    rex_observation_max_turns: int = 3
    rex_micro_compact_enabled: bool = True
    rex_max_plan_batch_steps: int = 3
    rex_plan_tools_enabled: bool = False
    rex_recursive_decomposition_enabled: bool = True
    rex_planning_schedule: Literal[
        "progressive", "one_shot", "single-plan"
    ] | None = None
    rex_compression_mode: Literal["none", "tree"] = "tree"
    rex_compression_include_task_context: bool = True
    rex_frame_compression_trigger_tokens: int = 3072
    # Plan -> Execute -> Reflection baseline settings
    plan_execute_reflection_max_steps: int = 16
    plan_execute_reflection_max_tool_chars: int = 6000
    plan_execute_reflection_enable_micro_compact: bool = True
    # Plan -> per-step ReAct -> failure replan baseline settings
    plan_execute_max_steps: int = 16
    plan_execute_max_step_turns: int = 20
    plan_execute_enable_micro_compact: bool = True


class ServiceDef(BaseModel):
    """A mock service that must be running for a task."""

    name: str
    command: str
    port: int
    health_check: str
    health_check_method: str = "POST"
    ready_timeout: int = 10
    reset_endpoint: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class ExpectedAction(BaseModel):
    """Describes an action the agent is expected to perform."""

    service: str  # "gmail", "calendar", etc.
    action_key: str  # key in /audit response: "drafts", "created_events", etc.
    required: bool = True


class UserAgentTaskConfig(BaseModel):
    """Per-task user agent simulation settings."""
    enabled: bool = False
    persona: str = ""
    max_rounds: int = 3
    system_prompt_suffix: str = ""


class TaskDefinition(BaseModel):
    task_id: str
    task_name: str
    version: str = "1.0"
    category: str = ""
    difficulty: str = "simple"
    prompt: Prompt
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_endpoints: list[ToolEndpoint] = Field(default_factory=list)
    environment: Environment = Field(default_factory=Environment)
    scoring_components: list[ScoringComponent] = Field(default_factory=list)
    safety_checks: list[SafetyCheck] = Field(default_factory=list)
    services: list[ServiceDef] = Field(default_factory=list)
    expected_actions: list[ExpectedAction] = Field(default_factory=list)
    judge_rubric: str = ""
    reference_solution: str = ""
    primary_dimensions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    user_agent: UserAgentTaskConfig = Field(default_factory=UserAgentTaskConfig)
    sandbox_files: list[str] = Field(default_factory=list)
    sandbox_grader_files: list[str] = Field(default_factory=list)
    env_snapshot_files: list[str] = Field(default_factory=list)
    env_snapshot_commands: list[str] = Field(default_factory=list)
    local_grader_files: list[str] = Field(default_factory=list)
    task_file: str | None = Field(default=None, exclude=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskDefinition:
        with open(path) as f:
            data = yaml.safe_load(f)
        data["task_file"] = str(Path(path).resolve())
        return cls.model_validate(data)

    def apply_port_offset(self, offset: int) -> None:
        """Shift all service ports and endpoint URLs by *offset*.

        This allows multiple task instances to run in parallel on
        non-overlapping port ranges.
        """
        if offset == 0:
            return

        def _shift_url(url: str) -> str:
            """Replace localhost:<port> with localhost:<port+offset>."""
            return re.sub(
                r"localhost:(\d+)",
                lambda m: f"localhost:{int(m.group(1)) + offset}",
                url,
            )

        for svc in self.services:
            svc.port += offset
            svc.health_check = _shift_url(svc.health_check)
            if svc.reset_endpoint:
                svc.reset_endpoint = _shift_url(svc.reset_endpoint)
            # Tell the subprocess which port to bind
            svc.env = {**(svc.env or {}), "PORT": str(svc.port)}

        for ep in self.tool_endpoints:
            ep.url = _shift_url(ep.url)

    def get_endpoint_map(self) -> dict[str, ToolEndpoint]:
        """Return {tool_name: ToolEndpoint} for dispatcher lookup."""
        return {ep.tool_name: ep for ep in self.tool_endpoints}
