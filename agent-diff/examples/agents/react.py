"""ReAct agent with XML actions and checkpoint-friendly traces."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .rex_runner import CodeExecutor, ModelClient, normalize_usage, utc_now

REACT_CONTROL_PROMPT = """Complete the task using a ReAct loop.

Return exactly one of these response forms per turn:

<thinking>Brief reasoning for the next action</thinking>
<action>one Bash command</action>

<thinking>Brief reasoning that the task is complete</thinking>
<done>verified user-facing result</done>

Execute at most one command per response. Inspect every observation, preserve
successful side effects, and verify ambiguous mutations before retrying. Use
<done> only after the requested work is complete."""


@dataclass(frozen=True)
class ReactConfig:
    max_iterations: int = 40

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_iterations, int)
            or isinstance(self.max_iterations, bool)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be an integer of at least 1")


def parse_react_response(text: str) -> tuple[str | None, str | None, str | None]:
    thinking = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    action = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    done = re.search(r"<done>(.*?)</done>", text, re.DOTALL)
    return (
        thinking.group(1).strip() if thinking else None,
        action.group(1).strip() if action else None,
        done.group(1).strip() if done else None,
    )


def _format_observation(result: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(result, dict):
        result = {
            "status": "success",
            "stdout": str(result or ""),
            "stderr": "",
            "exit_code": 0,
        }
    stdout = str(result.get("stdout", ""))
    stderr = str(result.get("stderr", ""))
    exit_code = int(result.get("exit_code", 0) or 0)
    observation = stdout.strip() or "(empty output)"
    if exit_code != 0:
        observation = (
            f"{stdout}\n[stderr]: {stderr}\n[exit_code]: {exit_code}"
        ).strip()
    return result, observation


def run_react_agent(
    *,
    model_client: ModelClient,
    prompt: str,
    executor: CodeExecutor,
    system_prompt: str,
    config: ReactConfig | None = None,
    on_trace_update: Any | None = None,
) -> dict[str, Any]:
    config = config or ReactConfig()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "user", "content": REACT_CONTROL_PROMPT},
    ]
    trace: dict[str, Any] = {
        "schema_version": 1,
        "agent": "react",
        "config": asdict(config),
        "steps": [],
        "final": None,
        "final_answer": None,
        "iterations": 0,
        "completed": False,
        "usage": normalize_usage(None),
        "budget": {"model_calls": 0, "tool_calls": 0},
        "messages": messages,
        "started_at": utc_now(),
    }

    def emit(stage: str) -> None:
        trace["stage"] = stage
        trace["updated_at"] = utc_now()
        if on_trace_update:
            on_trace_update(trace, stage)

    def add_usage(usage: dict[str, Any]) -> None:
        for key, value in normalize_usage(usage).items():
            trace["usage"][key] += value

    emit("agent_started")
    for iteration in range(1, config.max_iterations + 1):
        step: dict[str, Any] = {
            "iteration": iteration,
            "status": "requesting_model",
            "request": {"message_count": len(messages)},
        }
        trace["steps"].append(step)
        trace["iterations"] = iteration
        emit("model_request_started")

        try:
            turn = model_client.generate(tuple(messages))
        except Exception as exc:  # noqa: BLE001 - preserve partial task trace
            step.update(
                {
                    "status": "model_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            trace["error"] = {"type": type(exc).__name__, "message": str(exc)}
            emit("model_error")
            break

        trace["budget"]["model_calls"] += 1
        add_usage(turn.usage)
        thinking, action, done = parse_react_response(turn.content)
        step.update(
            {
                "status": "model_response_received",
                "thinking": thinking,
                "action": action,
                "done": done,
                "raw_response": turn.content,
                "api_response": turn.raw_response,
                "attempts": list(turn.attempts),
                "usage": normalize_usage(turn.usage),
            }
        )
        messages.append({"role": "assistant", "content": turn.content})
        emit("model_response_received")

        if action:
            trace["budget"]["tool_calls"] += 1
            try:
                executor_result = executor.execute(action)
            except Exception as exc:  # noqa: BLE001 - return errors to the agent
                executor_result = {
                    "status": "error",
                    "stdout": "",
                    "stderr": str(exc),
                    "exit_code": 1,
                    "error_type": type(exc).__name__,
                }
            executor_result, observation = _format_observation(executor_result)
            step["executor_result"] = executor_result
            step["observation"] = observation
            step["status"] = "observation_recorded"
            messages.append(
                {
                    "role": "user",
                    "content": f"<observation>\n{observation}\n</observation>",
                }
            )
            emit("observation_recorded")
            continue

        if done:
            trace["final"] = {
                "iteration": iteration,
                "thinking": thinking,
                "summary": done,
                "raw_response": turn.content,
                "api_response": turn.raw_response,
                "usage": normalize_usage(turn.usage),
            }
            trace["final_answer"] = done
            trace["completed"] = True
            trace["finished_at"] = utc_now()
            emit("agent_completed")
            return trace

        step["status"] = "protocol_warning"
        step["warning"] = "No <action> or <done> tag found"
        messages.append(
            {
                "role": "user",
                "content": "Return exactly one <action> or <done> response.",
            }
        )
        emit("protocol_warning")

    if "error" not in trace:
        trace["error"] = {
            "type": "MaxIterationsExceeded",
            "message": f"Reached {config.max_iterations} iterations",
        }
        emit("max_iterations_reached")
    trace["finished_at"] = utc_now()
    return trace


__all__ = [
    "REACT_CONTROL_PROMPT",
    "ReactConfig",
    "parse_react_response",
    "run_react_agent",
]
