from __future__ import annotations

import json
import sys
import threading
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

if "agent_diff" not in sys.modules:
    agent_diff_stub = types.ModuleType("agent_diff")
    agent_diff_stub.AgentDiff = object
    agent_diff_stub.BashExecutorProxy = object
    sys.modules["agent_diff"] = agent_diff_stub
if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = datasets_stub

import baseline_agent_benchmark as benchmark  # noqa: E402
from agents.baselines import (  # noqa: E402
    ExplicitPlanExecuteConfig,
    RecapAction,
    RecapComplete,
    RecapConfig,
    RecapController,
    RecapDecompose,
    RecapState,
    ReflectionConfig,
    parse_explicit_plan,
    parse_explicit_step_terminal,
    parse_recap_json,
    run_baseline_agent,
    run_explicit_plan_execute_agent,
    run_recap_agent,
    run_reflection_agent,
)
from agents.rex_runner import ModelTurn, ProtocolError  # noqa: E402
from agents.react import (  # noqa: E402
    ReactConfig,
    parse_react_response,
    run_react_agent,
)


class ScriptedModel:
    def __init__(self, responses: Sequence[str | ModelTurn]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict[str, str]]] = []

    def generate(self, messages: Sequence[dict[str, str]]) -> ModelTurn:
        self.requests.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("Scripted model ran out of responses")
        response = self.responses.pop(0)
        if isinstance(response, ModelTurn):
            return response
        return ModelTurn(
            response,
            {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cost": 0.01,
            },
        )


class FakeExecutor:
    def __init__(self, results: Sequence[dict[str, Any]] | None = None) -> None:
        self.results = list(results or [])
        self.actions: list[str] = []

    def execute(self, code: str) -> dict[str, Any]:
        self.actions.append(code)
        if self.results:
            return self.results.pop(0)
        return {
            "status": "success",
            "stdout": '{"ok":true}',
            "stderr": "",
            "exit_code": 0,
        }


def run_explicit(
    responses: Sequence[str | ModelTurn],
    *,
    config: ExplicitPlanExecuteConfig | None = None,
    executor: FakeExecutor | None = None,
) -> tuple[dict[str, Any], ScriptedModel, FakeExecutor]:
    model = ScriptedModel(responses)
    executor = executor or FakeExecutor()
    trace = run_explicit_plan_execute_agent(
        model_client=model,
        prompt="Complete the SaaS task",
        executor=executor,
        system_prompt="Use the current controller phase.",
        config=config,
    )
    return trace, model, executor


def test_baseline_protocol_parsers() -> None:
    plan = parse_explicit_plan(
        'prefix ```json\n{"steps":[{"step_goal":"One"}],"planning_complete":true}\n```',
        revision=2,
        max_steps=1,
    )
    assert plan.steps[0].id == "plan_2_step_1"
    assert (
        parse_explicit_step_terminal('{"complete":true,"status":"success"}')
        == "success"
    )
    assert (
        parse_explicit_step_terminal(
            '{"complete":true,"status":"success","summary":"extra"}'
        )
        is None
    )
    decompose = parse_recap_json(
        '{"type":"decompose","think":"split","subtasks":["child"]}'
    )
    assert decompose == RecapDecompose("split", ["child"])
    assert parse_recap_json('{"type":"action","command":" curl /items "}') == (
        RecapAction("curl /items")
    )
    assert parse_recap_json('{"type":"complete","summary":"done"}') == (
        RecapComplete("done")
    )
    invalid_recap_responses = [
        '{"think":"split","subtasks":["child"]}',
        '{"type":"decompose","think":"split","subtasks":[]}',
        '{"type":"decompose","think":"split","subtasks":[1]}',
        '{"type":"action","command":""}',
        '{"type":"action","command":"curl /items","subtasks":["child"]}',
        '{"type":"complete","summary":"done","command":"curl /items"}',
    ]
    assert all(parse_recap_json(response) is None for response in invalid_recap_responses)
    with pytest.raises(ProtocolError, match="exactly"):
        parse_explicit_plan(
            '{"steps":[],"planning_complete":true,"extra":1}',
            revision=0,
            max_steps=2,
        )


def test_explicit_plan_executes_step_and_records_usage() -> None:
    trace, model, executor = run_explicit(
        [
            '{"steps":[{"step_goal":"Send the message"}],"planning_complete":true}',
            "<action>curl https://slack.com/api/chat.postMessage</action>",
            '{"complete":true,"status":"success"}',
            "Message sent and verified.",
        ]
    )

    assert trace["completed"] is True
    assert trace["final_answer"] == "Message sent and verified."
    assert trace["budget"] == {"model_calls": 4, "tool_calls": 1}
    assert trace["usage"]["total_tokens"] == 48
    assert executor.actions == ["curl https://slack.com/api/chat.postMessage"]
    assert any(event["event"] == "step_complete" for event in trace["events"])
    assert not model.responses


def reasoning_turn(text: str) -> ModelTurn:
    return ModelTurn(
        "",
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cost": 0.01,
        },
        raw_response={
            "choices": [{"message": {"content": "", "reasoning_content": text}}]
        },
    )


def test_explicit_accepts_control_json_from_reasoning_content() -> None:
    trace, _, _ = run_explicit(
        [
            reasoning_turn(
                '{"steps":[{"step_goal":"Verify"}],"planning_complete":true}'
            ),
            reasoning_turn('{"complete":true,"status":"success"}'),
            "Visible final answer.",
        ]
    )

    assert trace["completed"] is True
    assert trace["final_answer"] == "Visible final answer."
    assert trace["budget"] == {"model_calls": 3, "tool_calls": 0}


def test_explicit_replaces_failed_suffix_without_reexecuting_old_step() -> None:
    trace, _, executor = run_explicit(
        [
            json.dumps(
                {
                    "steps": [
                        {"step_goal": "Inspect target"},
                        {"step_goal": "Obsolete mutation"},
                    ],
                    "planning_complete": True,
                }
            ),
            '{"complete":true,"status":"failed"}',
            '{"steps":[{"step_goal":"Recover safely"}],"planning_complete":true}',
            "<action>curl https://example.test/verify</action>",
            '{"complete":true,"status":"success"}',
            "Recovered and verified.",
        ]
    )

    assert trace["completed"] is True
    assert executor.actions == ["curl https://example.test/verify"]
    replans = [event for event in trace["events"] if event["event"] == "replan"]
    assert len(replans) == 1
    assert replans[0]["artifact"]["discarded_steps"][0]["step_goal"] == (
        "Obsolete mutation"
    )
    starts = [
        event["artifact"]["step"]["step_goal"]
        for event in trace["events"]
        if event["event"] == "step_start"
    ]
    assert starts == ["Inspect target", "Recover safely"]


def test_explicit_invalid_terminal_is_corrected_then_replanned() -> None:
    trace, model, _ = run_explicit(
        [
            '{"steps":[{"step_goal":"Try once"}],"planning_complete":true}',
            "not terminal",
            "still invalid",
            '{"steps":[{"step_goal":"Recover"}],"planning_complete":true}',
            '{"complete":true,"status":"success"}',
            "Done.",
        ],
        config=ExplicitPlanExecuteConfig(max_steps=2),
    )

    assert trace["completed"] is True
    assert sum(
        "Step terminal format correction required" in request[-1]["content"]
        for request in model.requests
    ) == 1
    invalid = next(
        event
        for event in trace["events"]
        if event["event"] == "step_complete"
        and event["artifact"].get("failure_kind") == "invalid_terminal"
    )
    assert invalid["artifact"]["recoverable"] is True


def test_recap_controller_descends_returns_and_finalizes() -> None:
    controller = RecapController("Root", RecapConfig())
    initial_prompt = controller.initial_prompt()
    assert "Your current task:\nRoot" in initial_prompt
    assert '"type":"decompose"' in initial_prompt
    assert '"type":"action"' in initial_prompt
    assert '"type":"complete"' in initial_prompt
    assert "<action>" not in initial_prompt
    step = controller.process_assistant_text(
        '{"type":"decompose","think":"split","subtasks":["Child","Wrap up"]}'
    )
    assert step.state == RecapState.DOWN
    assert controller.current_task == "Child"
    step = controller.process_assistant_text(
        '{"type":"complete","summary":"Child finished"}'
    )
    assert step.state == RecapState.UP
    assert step.remaining_subtasks == ("Wrap up",)
    step = controller.process_assistant_text(
        '{"type":"complete","summary":"Root finished"}'
    )
    assert step.state == RecapState.FINALIZE
    assert step.done is True
    assert step.final_answer == "Root finished"


def test_recap_accepts_controller_json_from_reasoning_content() -> None:
    trace = run_recap_agent(
        model_client=ScriptedModel(
            [
                reasoning_turn(
                    '{"type":"complete","summary":"Visible final answer."}'
                ),
            ]
        ),
        prompt="Complete the task",
        executor=FakeExecutor(),
        system_prompt="Use ReCAP.",
    )

    assert trace["completed"] is True
    assert trace["final_answer"] == "Visible final answer."


def test_recap_executes_recursive_task_and_preserves_full_observation() -> None:
    long_observation = "ABCDEFGHIJ"
    model = ScriptedModel(
        [
            '{"type":"decompose","think":"split","subtasks":["Inspect"]}',
            '{"type":"action","command":"curl https://example.test/item"}',
            '{"type":"complete","summary":"Inspection finished"}',
            '{"type":"complete","summary":"Verified final answer."}',
        ]
    )
    executor = FakeExecutor(
        [
            {
                "status": "success",
                "stdout": long_observation,
                "stderr": "",
                "exit_code": 0,
            }
        ]
    )
    trace = run_recap_agent(
        model_client=model,
        prompt="Complete the task",
        executor=executor,
        system_prompt="Use ReCAP.",
        config=RecapConfig(max_obs_chars=5),
    )

    assert trace["completed"] is True
    assert trace["budget"] == {"model_calls": 4, "tool_calls": 1}
    assert long_observation in json.dumps(trace["messages"])
    action_followup = model.requests[2][-1]["content"]
    assert "ABCDE\n[observation truncated]" in action_followup
    assert long_observation not in action_followup


def test_recap_enforces_depth_and_fails_after_two_invalid_outputs() -> None:
    controller = RecapController("Root", RecapConfig(max_depth=1))
    initial_prompt = controller.initial_prompt()
    assert '"type":"decompose"' not in initial_prompt
    assert '"type":"action"' in initial_prompt
    assert '"type":"complete"' in initial_prompt
    step = controller.process_assistant_text(
        '{"type":"decompose","think":"split","subtasks":["Too deep"]}'
    )
    assert step.note == "max_depth_reached"
    assert controller.current_task == "Root"

    model = ScriptedModel(["invalid", "still invalid"])
    trace = run_recap_agent(
        model_client=model,
        prompt="Complete the task",
        executor=FakeExecutor(),
        system_prompt="Use ReCAP.",
    )
    assert trace["completed"] is False
    assert trace["root_result"]["status"] == "failed"


def test_reflection_runs_once_after_each_observation() -> None:
    model = ScriptedModel(
        [
            "<action>curl https://example.test/one</action>",
            "<action>curl https://example.test/two</action>",
            "<done>Both operations verified.</done>",
        ]
    )
    executor = FakeExecutor()
    trace = run_reflection_agent(
        model_client=model,
        prompt="Complete the task",
        executor=executor,
        system_prompt="Use reflection.",
    )

    assert trace["completed"] is True
    assert trace["budget"] == {"model_calls": 3, "tool_calls": 2}
    assert len(
        [event for event in trace["events"] if event["event"] == "reflection"]
    ) == 2
    assert sum(
        "Reflect briefly on the latest tool result" in request[-1]["content"]
        for request in model.requests
    ) == 2


def test_reflection_tool_budget_stops_second_action() -> None:
    executor = FakeExecutor()
    trace = run_reflection_agent(
        model_client=ScriptedModel(
            [
                "<action>curl https://example.test/one</action>",
                "<action>curl https://example.test/two</action>",
            ]
        ),
        prompt="Complete the task",
        executor=executor,
        system_prompt="Use reflection.",
        config=ReflectionConfig(max_tool_calls=1),
    )

    assert trace["completed"] is False
    assert trace["error"]["type"] == "BudgetExceeded"
    assert executor.actions == ["curl https://example.test/one"]


def test_react_parses_and_executes_one_action_per_turn() -> None:
    assert parse_react_response(
        "<thinking>inspect</thinking><action>curl /items</action>"
    ) == ("inspect", "curl /items", None)
    assert parse_react_response(
        "<thinking>finished</thinking><done>Verified.</done>"
    ) == ("finished", None, "Verified.")

    model = ScriptedModel(
        [
            "<thinking>inspect</thinking><action>curl /items</action>",
            "<thinking>finished</thinking><done>Verified.</done>",
        ]
    )
    executor = FakeExecutor()
    trace = run_react_agent(
        model_client=model,
        prompt="Complete the task",
        executor=executor,
        system_prompt="Use the API.",
    )

    assert trace["completed"] is True
    assert trace["final_answer"] == "Verified."
    assert trace["budget"] == {"model_calls": 2, "tool_calls": 1}
    assert executor.actions == ["curl /items"]
    assert "<observation>" in model.requests[1][-1]["content"]


def test_react_records_protocol_warning_and_iteration_limit() -> None:
    trace = run_react_agent(
        model_client=ScriptedModel(["invalid"]),
        prompt="Complete the task",
        executor=FakeExecutor(),
        system_prompt="Use the API.",
        config=ReactConfig(max_iterations=1),
    )

    assert trace["completed"] is False
    assert trace["error"]["type"] == "MaxIterationsExceeded"
    assert trace["budget"] == {"model_calls": 1, "tool_calls": 0}
    assert trace["steps"][0]["status"] == "protocol_warning"


def test_baseline_dispatch_validates_mode_config() -> None:
    trace = run_baseline_agent(
        mode="reflection",
        model_client=ScriptedModel(["Finished."]),
        prompt="Complete the task",
        executor=FakeExecutor(),
        system_prompt="Use reflection.",
        config=ReflectionConfig(),
    )
    assert trace["agent"] == "reflection"
    with pytest.raises(TypeError, match="ReflectionConfig"):
        run_baseline_agent(
            mode="reflection",
            model_client=ScriptedModel(["Finished."]),
            prompt="Complete the task",
            executor=FakeExecutor(),
            system_prompt="Use reflection.",
            config=RecapConfig(),
        )


def test_runner_builds_mode_specific_configs() -> None:
    parser = benchmark.build_parser()
    explicit_args = parser.parse_args(
        [
            "--agent",
            "explicit-plan-execute",
            "--model",
            "model",
            "--explicit-max-steps",
            "7",
        ]
    )
    benchmark.validate_args(parser, explicit_args)
    explicit = benchmark.build_agent_config(explicit_args)
    assert explicit == ExplicitPlanExecuteConfig(max_steps=7)

    recap_args = parser.parse_args(
        ["--agent", "recap", "--model", "model", "--recap-max-depth", "2"]
    )
    benchmark.validate_args(parser, recap_args)
    recap = benchmark.build_agent_config(recap_args)
    assert recap == RecapConfig(max_depth=2)


def test_baseline_checkpoint_rejects_cross_mode_resume(tmp_path: Path) -> None:
    metadata = {
        "schema_version": 2,
        "agent": "recap",
        "model": "model",
        "openai_base_url": "http://model",
        "agent_diff_base_url": "http://agent-diff",
        "dataset_name": "dataset",
        "dataset_split": "test",
        "dataset_fingerprint": "abc",
        "config": {"max_depth": 3},
    }
    store = benchmark.CheckpointStore(tmp_path, metadata)
    key = store.task_key("model", "slack/one")
    checkpoint = store.for_task(test_id="slack/one", task_key=key)
    checkpoint.complete_task(key, {"passed": True})
    assert checkpoint.path.name == "slack%2Fone.json"

    changed_store = benchmark.CheckpointStore(
        tmp_path,
        {**metadata, "agent": "reflection", "config": {}},
    )
    with pytest.raises(ValueError, match="Checkpoint configuration mismatch"):
        changed_store.for_task(test_id="slack/one", task_key=key)


def test_parallel_runner_uses_one_checkpoint_per_task(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dataset = [
        {"test_id": "slack_1", "test_name": "One", "task_horizon": 2},
        {"test_id": "box_2", "test_name": "Two", "task_horizon": 5},
    ]
    barrier = threading.Barrier(2)
    worker_threads: set[int] = set()
    lock = threading.Lock()

    def fake_execute_task(
        *,
        example: dict[str, Any],
        checkpoint: Any,
        key: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        with lock:
            worker_threads.add(threading.get_ident())
        barrier.wait(timeout=2)
        record = {
            "test_id": example["test_id"],
            "task_horizon": example["task_horizon"],
            "passed": True,
            "score": 1.0,
            "time": 0.1,
            "trace": {"usage": {}, "budget": {}},
        }
        checkpoint.complete_task(key, record)
        return record

    monkeypatch.setattr(benchmark, "load_dataset", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(benchmark, "execute_task", fake_execute_task)
    exit_code = benchmark.main(
        [
            "--agent",
            "reflection",
            "--model",
            "model",
            "--workers",
            "2",
            "--checkpoint-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(worker_threads) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2
