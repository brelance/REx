"""Load config.yaml with env-var expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .env import load_dotenv_files

load_dotenv_files()


_ENV_RE = re.compile(r"\$\{(\w+)\}")

# Search order: CWD -> project root (where pyproject.toml lives)
_SEARCH_PATHS = [
    Path.cwd() / "config.yaml",
    Path(__file__).resolve().parent.parent.parent / "config.yaml",
]


def _expand_env(value: str) -> str | None:
    """Replace ${VAR} with os.environ[VAR]. Returns None if var is unset."""
    m = _ENV_RE.fullmatch(value.strip())
    if m:
        return os.environ.get(m.group(1))
    return value


def _walk_expand(obj):
    """Recursively expand ${ENV} references in string values."""
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _walk_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_expand(v) for v in obj]
    return obj


class ModelConfig(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model_id: str = "anthropic/claude-opus-4-6"
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    system_prompt_prefix: str | None = None
    extra_body: dict | None = None
    reasoning_effort: str | None = None
    context_window: int = 262144
    temperature: float | None = 0.0  # None = don't send temperature param


class JudgeConfig(BaseModel):
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    model_id: str = "google/gemini-3-flash-preview"
    enabled: bool = True
    # Additional HTTP headers required by some OpenAI-compatible gateways.
    extra_headers: dict[str, str] = Field(default_factory=dict)


class DefaultsConfig(BaseModel):
    trace_dir: str = "traces"
    tasks_dir: str = "tasks"


class SandboxConfig(BaseModel):
    """Configuration for Docker sandbox execution."""

    enabled: bool = False
    image: str = "claw-eval-agent:latest"
    docker_host: str | None = None
    memory_limit: str = "4g"
    cpu_limit: float = 2.0
    sandbox_port: int = 8080
    container_timeout: int = 900
    max_concurrent: int = 10
    enable_browser: bool = True
    enable_shell: bool = True
    enable_file: bool = True


class PromptFilesConfig(BaseModel):
    """Workspace markdown files to inject into system prompt."""

    agents_md: str | None = None
    soul_md: str | None = None
    user_md: str | None = None
    tools_md: str | None = None


class SkillEntry(BaseModel):
    """A skill descriptor shown in the default skills list."""

    name: str
    description: str
    path: str


class SkillsConfig(BaseModel):
    """Skills configuration for prompt composition."""

    default: list[SkillEntry] = Field(default_factory=list)
    load_via_tool_call: bool = True
    read_tool_name: str = "read"


class BehaviorRulesConfig(BaseModel):
    """Behavior-policy text included in system prompt."""

    safety: str = "No independent objective; do not pursue self-preservation, replication, or resource acquisition."
    tool_call_style: str = "For low-risk actions, call tools directly without narration; narrate only for complex tasks."
    reply_tags: str = "Use [[reply_to_current]] to control reply relationship when needed."
    silent_reply: str = "If no reply is needed, output NO_REPLY."
    heartbeat: str = "Heartbeat checks should return HEARTBEAT_OK when no action is needed."


class PromptConfig(BaseModel):
    """Configuration for dynamic system prompt construction."""

    enabled: bool = True
    strict_file_check: bool = False
    include_tool_schema: bool = True
    files: PromptFilesConfig = PromptFilesConfig()
    behavior_rules: BehaviorRulesConfig = BehaviorRulesConfig()
    skills: SkillsConfig = SkillsConfig()


class MediaConfig(BaseModel):
    """Configuration for media detection and loading from prompts."""

    enabled: bool = True
    strict_mode: bool = False
    max_files: int = 6
    max_bytes_per_file: int = 8 * 1024 * 1024
    image_max_dimension: int = 2048
    # Tool-media injection settings (for ReadMedia / Read with image/PDF)
    inject_tool_media: bool = True
    max_images_per_turn: int = 64
    max_tool_images_total: int = 64
    video_frame_budget: int = 8
    tool_image_quality: int = 60
    tool_image_max_dimension: int = 1280
    max_conversation_images: int = 256
    image_keep_recent_turns: int = 3


class UserAgentModelConfig(BaseModel):
    """LLM configuration for simulated user agent."""
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    model_id: str = "google/gemini-3-flash-preview"


class GradingConfig(BaseModel):
    """How conversation transcripts are built for LLM judges.

    All runners use the same canonical transcript builder. Existing grader
    formatter arguments decide whether business-tool calls/results are shown;
    runner control-plane metadata is never shown.
    """

    include_tool_args: bool = True
    include_tool_results: bool = True


class EnvironmentOverrideConfig(BaseModel):
    """Optional task.environment overrides applied at run time.

    Only fields explicitly present in the YAML are applied, so omitted values
    never overwrite per-task environment settings.
    """

    enable_compact: bool | None = None
    compact_keep_recent: int | None = None
    compact_min_chars: int | None = None
    compact_threshold_pct: float | None = None
    compact_keep_recent_on_summary: int | None = None
    compact_protect_tokens: int | None = None
    compact_max_auto_compacts: int | None = None
    enable_recap: bool | None = None
    recap_max_depth: int | None = None
    recap_max_subtasks: int | None = None
    recap_max_obs_chars: int | None = None
    recap_max_tree_chars: int | None = None
    recap_force_final_answer: bool | None = None
    recap_action_taken_prompt_variant: str | None = None
    recap_mode: str | None = None
    recap_dag_max_workers: int | None = None
    recap_dag_max_nodes: int | None = None
    recap_dag_max_artifact_chars: int | None = None
    recap_dag_dynamic_replan: bool | None = None
    recap_dag_two_phase_commit: bool | None = None
    recap_dag_enable_observation_memory: bool | None = None
    recap_dag_injected_observation_max_chars: int | None = None
    recap_dag_enable_llm_observation_summary: bool | None = None
    recap_dag_observation_summary_max_observations: int | None = None
    recap_dag_llm_observation_summary_chars: int | None = None
    rex_max_steps: int | None = None
    rex_max_depth: int | None = None
    rex_observation_max_chars: int | None = None
    rex_observation_max_turns: int | None = None
    rex_micro_compact_enabled: bool | None = None
    rex_max_plan_batch_steps: int | None = None
    rex_plan_tools_enabled: bool | None = None
    rex_recursive_decomposition_enabled: bool | None = None
    rex_planning_schedule: Literal[
        "progressive", "one_shot", "single-plan"
    ] | None = None
    rex_compression_mode: Literal["none", "tree"] | None = None
    rex_compression_include_task_context: bool | None = None
    rex_frame_compression_trigger_tokens: int | None = None
    plan_execute_reflection_max_steps: int | None = None
    plan_execute_reflection_max_tool_chars: int | None = None
    plan_execute_reflection_enable_micro_compact: bool | None = None
    plan_execute_max_steps: int | None = None
    plan_execute_max_step_turns: int | None = None
    plan_execute_enable_micro_compact: bool | None = None


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    judge: JudgeConfig = JudgeConfig()
    defaults: DefaultsConfig = DefaultsConfig()
    sandbox: SandboxConfig = SandboxConfig()
    prompt: PromptConfig = PromptConfig()
    media: MediaConfig = MediaConfig()
    user_agent_model: UserAgentModelConfig = UserAgentModelConfig()
    environment: EnvironmentOverrideConfig = EnvironmentOverrideConfig()
    grading: GradingConfig = GradingConfig()


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML file with ${ENV} expansion.

    Searches config.yaml in CWD then project root if path is not given.
    Returns defaults if no file is found.
    """
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = _SEARCH_PATHS

    for p in candidates:
        if p.exists():
            with open(p) as f:
                raw = yaml.safe_load(f) or {}
            expanded = _walk_expand(raw)
            return Config.model_validate(expanded)

    return Config()
