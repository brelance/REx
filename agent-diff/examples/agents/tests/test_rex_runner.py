from __future__ import annotations

import hashlib
import json
import sys
import threading
import types
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

# The controller has no SDK/datasets dependency. Lightweight stubs let this test
# module exercise the CLI lifecycle in an isolated unit-test environment too.
if "agent_diff" not in sys.modules:
    agent_diff_stub = types.ModuleType("agent_diff")
    agent_diff_stub.AgentDiff = object
    agent_diff_stub.BashExecutorProxy = object
    sys.modules["agent_diff"] = agent_diff_stub
if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = datasets_stub

import rex_runner_benchmark as benchmark
import rex_runner_benchmark_parallel as parallel_benchmark
from agents.rex_runner import (
    RExConfig,
    RExRunner,
    ModelTurn,
    ProtocolError,
    extract_json_dict,
    parse_action,
    parse_plan,
    run_rex_runner,
)


class ScriptedModel:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict[str, str]]] = []

    def generate(self, messages: Sequence[dict[str, str]]) -> ModelTurn:
        self.requests.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("Scripted model ran out of responses")
        return ModelTurn(
            self.responses.pop(0),
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


def plan_response(*steps: tuple[str, str], planning_complete: bool = True) -> str:
    return json.dumps(
        {
            "thinking": "plan",
            "steps": [
                {"step_goal": goal, "execution_mode": mode} for goal, mode in steps
            ],
            "planning_complete": planning_complete,
        }
    )


def run_script(
    responses: Sequence[str],
    *,
    executor: FakeExecutor | None = None,
    config: RExConfig | None = None,
) -> tuple[dict[str, Any], ScriptedModel, FakeExecutor]:
    model = ScriptedModel(responses)
    executor = executor or FakeExecutor()
    trace = run_rex_runner(
        model_client=model,
        prompt="Complete the SaaS task",
        executor=executor,
        system_prompt="Use the current phase protocol.",
        config=config,
    )
    return trace, model, executor


def test_json_and_xml_protocol_parsers() -> None:
    assert extract_json_dict('prefix ```json\n{"ok": true}\n``` suffix') == {"ok": True}
    assert parse_action("<action>curl example</action>") == ("curl example", None)
    assert parse_action("<done>created C123</done>") == (None, "created C123")

    plan = parse_plan(
        plan_response(("Find the channel", "execute")),
        next_step_index=1,
        existing_step_count=0,
        max_steps=8,
    )
    assert plan.steps[0].goal == "Find the channel"

    with pytest.raises(ProtocolError):
        parse_plan(
            '{"thinking":"x","steps":[],"planning_complete":"yes"}',
            next_step_index=1,
            existing_step_count=0,
            max_steps=8,
        )


@pytest.mark.parametrize("current_index", [0, 1, 2])
def test_compression_continuation_is_only_the_next_sibling(current_index: int) -> None:
    steps = parse_plan(
        plan_response(("A", "decompose"), ("B", "execute"), ("C", "decompose")),
        next_step_index=1, existing_step_count=0, max_steps=8,
    ).steps
    state = SimpleNamespace(all_steps=steps, step=steps[current_index], completed={})
    pending = RExRunner._pending_siblings(state)
    assert [item.goal for item in pending] == [
        step.goal for step in steps[current_index + 1 : current_index + 2]
    ]


def test_direct_execution_records_action_observation_and_usage() -> None:
    trace, model, executor = run_script(
        [
            plan_response(("Send the requested message", "execute")),
            "<thinking>send it</thinking><action>curl https://slack.com/api/chat.postMessage</action>",
            "<thinking>confirmed</thinking><done>message ts=123 created</done>",
        ]
    )

    assert trace["completed"] is True
    assert executor.actions == ["curl https://slack.com/api/chat.postMessage"]
    assert trace["budget"] == {"model_calls": 3, "tool_calls": 1, "frames": 1}
    assert trace["usage"]["total_tokens"] == 36
    assert any(event["event"] == "tool_result" for event in trace["events"])
    assert any("<observation>" in message["content"] for message in trace["messages"])
    assert not model.responses


def test_recursive_decomposition_executes_child_frame() -> None:
    trace, _, executor = run_script(
        [
            plan_response(("Resolve and update the target", "decompose")),
            plan_response(("Resolve the target ID", "execute")),
            "<action>curl https://api.box.com/2.0/search</action>",
            "<done>resolved file id=F1</done>",
        ]
    )

    assert trace["completed"] is True
    assert trace["budget"]["frames"] == 2
    assert executor.actions == ["curl https://api.box.com/2.0/search"]
    assert any(event["frame_id"] == "root.step_1" for event in trace["events"])


@pytest.mark.parametrize("step_count", [0, 1])
def test_complete_plan_finishes_without_review(step_count: int) -> None:
    responses = [plan_response(*[("Finish", "execute")] * step_count)]
    if step_count:
        responses.append("<done>finished</done>")
    trace, model, _ = run_script(responses)

    assert trace["completed"] is True
    assert "error" not in trace
    assert len(model.requests) == len(responses)
    assert not model.responses
    assert not any(event["event"] == "frame_reviewed" for event in trace["events"])


@pytest.mark.parametrize("parent_recovers", [True, False])
def test_child_recovery_failure_bubbles_to_parent(parent_recovers: bool) -> None:
    responses = [
        plan_response(("Child goal", "decompose"), ("Stale parent step", "execute")),
        plan_response(("Child attempt", "execute"), ("Stale child step", "execute")),
        "No usable response",
        "Child attempt blocked",
        plan_response(("Child recovery", "execute"), ("Stale recovery step", "execute")),
        "Still blocked",
        "Child recovery exhausted: resource unavailable",
        plan_response(("Parent recovery", "execute")),
    ]
    responses += (
        ["<done>Parent resolved the task differently</done>"]
        if parent_recovers
        else ["Parent also blocked", "Parent recovery exhausted"]
    )
    trace, model, _ = run_script(
        responses,
        config=RExConfig(max_turns_per_step=1, compression_frame_trigger_tokens=1),
    )

    assert trace["completed"] is parent_recovers
    assert "error" not in trace
    assert not model.responses
    recoveries = [
        event
        for event in trace["events"]
        if event["event"] == "plan_created" and event["artifact"].get("recovery")
    ]
    assert [event["frame_id"] for event in recoveries] == ["root.step_1", "root"]
    started = [
        event["step_goal"]
        for event in trace["events"]
        if event["event"] == "step_started"
    ]
    assert started == ["Child goal", "Child attempt", "Child recovery", "Parent recovery"]
    aborts = [event for event in trace["events"] if event["event"] == "plan_batch_abort"]
    assert [event["artifact"]["recovery_exhausted"] for event in aborts[:3]] == [
        False, True, False,
    ]
    assert aborts[2]["artifact"]["dropped_steps"][0]["step_goal"] == "Stale parent step"
    parent_prompt = model.requests[7][-1]["content"]
    assert "Failed step: Child goal" in parent_prompt
    assert "Child recovery exhausted: resource unavailable" in parent_prompt
    assert trace["compression"]["checks"] == 0
    assert trace["compression"]["calls"] == 0
    parent_history = json.dumps(model.requests[7])
    assert "Child attempt blocked" in parent_history
    assert "Stale child step" in parent_history
    assert "[RUNNER-COMPRESSED HISTORY]" not in parent_history
    if not parent_recovers:
        assert trace["root_result"]["summary"] == "Parent recovery exhausted"


def test_successful_batch_resets_recovery_allowance() -> None:
    trace, model, _ = run_script(
        [
            plan_response(("Initial attempt", "execute")),
            "Blocked",
            "Initial attempt failed",
            plan_response(("Recovery", "execute"), planning_complete=False),
            "<done>Recovery succeeded</done>",
            plan_response(("Later step", "execute")),
            "Later step blocked",
            "Later failure can recover locally",
            plan_response(("Later recovery", "execute")),
            "<done>Later recovery succeeded</done>",
        ],
        config=RExConfig(max_turns_per_step=1),
    )

    assert trace["root_result"]["status"] == "success"
    assert "error" not in trace
    assert not model.responses
    plans = [event for event in trace["events"] if event["event"] == "plan_created"]
    assert [event["artifact"]["recovery"] for event in plans] == [False, True, False, True]
    aborts = [event for event in trace["events"] if event["event"] == "plan_batch_abort"]
    assert [event["artifact"]["recovery_exhausted"] for event in aborts] == [False, False]
    assert model.requests[5][-1]["content"].startswith("Planning phase.")
    assert "Later failure can recover locally" in model.requests[8][-1]["content"]


def test_partial_recovery_batch_success_does_not_reset_allowance() -> None:
    trace, model, _ = run_script(
        [
            plan_response(("Initial attempt", "execute")),
            "Blocked",
            "Initial failure",
            plan_response(
                ("Recovery progress", "execute"),
                ("Recovery fails", "execute"),
                ("Unused suffix", "execute"),
            ),
            "<done>First recovery step succeeded</done>",
            "Still blocked",
            "Recovery failure must propagate",
        ],
        config=RExConfig(max_turns_per_step=1),
    )

    assert trace["root_result"] == {
        "status": "failed", "summary": "Recovery failure must propagate",
    }
    assert "error" not in trace
    assert not model.responses
    aborts = [event for event in trace["events"] if event["event"] == "plan_batch_abort"]
    assert [event["artifact"]["recovery_exhausted"] for event in aborts] == [False, True]
    assert aborts[-1]["artifact"]["dropped_steps"][0]["step_goal"] == "Unused suffix"


def test_invalid_child_recovery_plan_bubbles_to_parent() -> None:
    trace, model, _ = run_script(
        [
            plan_response(("Child goal", "decompose")),
            plan_response(("Child attempt", "execute")),
            "Blocked",
            "Child failed",
            "Invalid recovery plan",
            "Still invalid after format correction",
            plan_response(("Parent recovery", "execute")),
            "<done>Parent recovered</done>",
        ],
        config=RExConfig(max_turns_per_step=1),
    )

    assert trace["completed"] is True
    assert "error" not in trace
    assert not model.responses
    assert (
        "Planning did not produce a valid recovery plan"
        in model.requests[6][-1]["content"]
    )
    assert [
        event["step_goal"]
        for event in trace["events"]
        if event["event"] == "step_started"
    ] == ["Child goal", "Child attempt", "Parent recovery"]


def test_discarded_steps_still_consume_planning_budget() -> None:
    oversized_recovery = plan_response(
        ("Recovery one", "execute"), ("Recovery two", "execute")
    )
    trace, model, _ = run_script(
        [
            plan_response(("Initial attempt", "execute"), ("Discarded step", "execute")),
            "Blocked",
            "Initial attempt failed",
            oversized_recovery,
            oversized_recovery,
        ],
        config=RExConfig(max_turns_per_step=1, max_steps_per_frame=3),
    )

    assert trace["completed"] is False
    assert "error" not in trace
    assert not model.responses
    assert "response exceeds the frame step budget" in model.requests[4][-1]["content"]
    assert [
        event["step_goal"]
        for event in trace["events"]
        if event["event"] == "step_started"
    ] == ["Initial attempt"]
    abort = next(event for event in trace["events"] if event["event"] == "plan_batch_abort")
    assert abort["artifact"]["remaining_step_budget"] == 1


def test_failed_step_uses_changed_recovery_plan() -> None:
    executor = FakeExecutor(
        [
            {
                "status": "error",
                "stdout": "",
                "stderr": "temporary failure",
                "exit_code": 1,
            },
            {
                "status": "success",
                "stdout": '{"ok":true}',
                "stderr": "",
                "exit_code": 0,
            },
        ]
    )
    trace, _, executor = run_script(
        [
            plan_response(("Create the message", "execute")),
            "<action>curl -X POST https://slack.com/api/chat.postMessage</action>",
            "<thinking>The result is still unclear.</thinking>",
            "Mutation failed; do not repeat without first changing the approach.",
            plan_response(("Verify state, then create only if absent", "execute")),
            "<action>curl https://slack.com/api/conversations.history</action>",
            "<done>verified the requested state is present</done>",
        ],
        executor=executor,
        config=RExConfig(max_turns_per_step=2),
    )

    assert trace["completed"] is True
    assert len(executor.actions) == 2
    assert "POST" in executor.actions[0]
    assert "POST" not in executor.actions[1]
    recovery_plans = [
        event
        for event in trace["events"]
        if event["event"] == "plan_created" and event["artifact"].get("recovery")
    ]
    assert len(recovery_plans) == 1


def test_invalid_initial_plan_falls_back_to_direct_execution() -> None:
    trace, _, executor = run_script(
        [
            "not json",
            "still not json",
            "<action>curl https://slack.com/api/conversations.list</action>",
            "<done>task completed directly</done>",
        ]
    )

    assert trace["completed"] is True
    assert len(executor.actions) == 1
    assert any(
        event["event"] == "plan_created" and event["artifact"].get("fallback")
        for event in trace["events"]
    )


def test_runner_can_exceed_100_model_calls_and_keeps_counting() -> None:
    responses = []
    for index in range(51):
        responses.extend([
            plan_response(
                (f"Step {index + 1}", "execute"),
                planning_complete=index == 50,
            ),
            "<done>Step completed</done>",
        ])
    trace, model, executor = run_script(
        responses, config=RExConfig(max_steps_per_frame=51),
    )

    assert trace["completed"] is True
    assert "error" not in trace
    assert trace["budget"]["model_calls"] == 102
    assert len(model.requests) == 102
    assert not model.responses
    assert not executor.actions


def test_rex_cli_has_no_global_model_call_limit() -> None:
    for parser in (benchmark.build_parser(), parallel_benchmark.build_parallel_parser()):
        args = parser.parse_args(["--model", "model"])
        assert not hasattr(args, "max_model_calls")
        assert not hasattr(parallel_benchmark.build_agent_config(args), "max_model_calls")
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "model", "--max-model-calls", "100"])


def test_tool_budget_stops_run_before_second_action() -> None:
    trace, _, executor = run_script(
        [
            plan_response(("Perform updates", "execute")),
            "<action>curl -X POST https://slack.com/api/chat.postMessage</action>",
            "<action>curl -X POST https://slack.com/api/chat.postMessage</action>",
        ],
        config=RExConfig(max_tool_calls=1),
    )

    assert trace["completed"] is False
    assert trace["error"]["type"] == "BudgetExceeded"
    assert len(executor.actions) == 1


def test_trace_callback_receives_incremental_events() -> None:
    stages: list[str] = []
    model = ScriptedModel(
        [
            plan_response(("Finish", "execute")),
            "<done>finished</done>",
        ]
    )
    trace = run_rex_runner(
        model_client=model,
        prompt="Complete the SaaS task",
        executor=FakeExecutor(),
        system_prompt="Use the current phase protocol.",
        on_trace_update=lambda _trace, stage: stages.append(stage),
    )

    assert trace["completed"] is True
    assert stages[0] == "agent_started"
    assert stages[-1] == "agent_completed"
    assert "frame_completed" in stages


def test_tree_compression_replaces_active_scope_and_uses_separate_budget(
    tmp_path: Path,
) -> None:
    middle = "MIDDLE_ONLY_IN_RAW_HISTORY"
    observation = "A" * 5000 + middle + "Z" * 5000
    trace, model, _ = run_script(
        [
            plan_response(("Resolve child task", "decompose")),
            plan_response(("Inspect the large record", "execute")),
            "<action>curl https://example.test/items/F1</action>",
            "<done>verified item id=F1</done>",
            "Item F1 was inspected and its exact identifier was verified.",
        ],
        executor=FakeExecutor(
            [
                {
                    "status": "success",
                    "stdout": observation,
                    "stderr": "",
                    "exit_code": 0,
                }
            ]
        ),
        config=RExConfig(
            compression_mode="tree",
            compression_frame_trigger_tokens=1,
        ),
    )

    # Re-run with a log directory through the public entry point because run_script
    # intentionally keeps its small convenience signature unchanged.
    logged_model = ScriptedModel(
        [
            plan_response(("Resolve child task", "decompose")),
            plan_response(("Inspect the large record", "execute")),
            "<action>curl https://example.test/items/F1</action>",
            "<done>verified item id=F1</done>",
            "Item F1 was inspected and its exact identifier was verified.",
        ]
    )
    logged_trace = run_rex_runner(
        model_client=logged_model,
        prompt="Complete the SaaS task",
        executor=FakeExecutor(
            [
                {
                    "status": "success",
                    "stdout": observation,
                    "stderr": "",
                    "exit_code": 0,
                }
            ]
        ),
        system_prompt="Use the current phase protocol.",
        config=RExConfig(
            compression_mode="tree",
            compression_frame_trigger_tokens=1,
        ),
        compression_log_dir=tmp_path,
    )

    assert trace["completed"] is True
    assert trace["budget"]["model_calls"] == 4
    assert trace["compression"]["calls"] == 1
    assert trace["compression"]["applied"] == 1
    assert trace["agent_usage"]["total_tokens"] == 48
    assert trace["compression"]["usage"]["total_tokens"] == 12
    assert trace["usage"]["total_tokens"] == 60
    assert len(model.requests) == 5
    assert model.requests[4][0]["content"].startswith(
        "Produce a compact operational handoff"
    )
    assert "[RUNNER-COMPRESSED HISTORY]" in json.dumps(trace["active_messages"])
    assert middle in json.dumps(trace["messages"])
    assert middle not in json.dumps(trace["active_messages"])
    recorded = trace["evidence"][0]
    assert recorded["call_id"] == "bash-000001"
    assert recorded["status"] == "success"
    assert recorded["error"] is None
    assert recorded["observation"] == observation
    assert recorded["output_sha256"] == hashlib.sha256(observation.encode()).hexdigest()
    assert recorded["call_id"] not in json.dumps(trace["active_messages"])
    assert recorded["output_sha256"] not in json.dumps(trace["active_messages"])

    logs = list(tmp_path.glob("*.txt"))
    assert len(logs) == 1
    log_text = logs[0].read_text(encoding="utf-8")
    assert middle in log_text
    assert '"applied": true' in log_text
    assert logged_trace["compression"]["applied"] == 1


@pytest.mark.parametrize("mode", ["execute", "decompose"])
def test_tree_compression_skips_root_and_direct_steps(mode: str) -> None:
    trace, model, _ = run_script(
        [
            plan_response(("Finish directly", mode)),
            "<done>finished</done>",
        ],
        config=RExConfig(max_depth=1, compression_frame_trigger_tokens=1),
    )

    assert trace["completed"] is True
    assert trace["compression"]["calls"] == 0
    assert len(model.requests) == 2
    assert "[RUNNER-COMPRESSED HISTORY]" not in json.dumps(trace["active_messages"])


def test_tree_compression_skips_child_below_threshold() -> None:
    trace, model, _ = run_script(
        [
            plan_response(("Resolve child", "decompose")),
            plan_response(("Finish directly", "execute")),
            "<done>finished</done>",
        ],
        config=RExConfig(compression_frame_trigger_tokens=100_000),
    )

    assert trace["completed"] is True
    assert trace["budget"]["frames"] == 2
    assert trace["compression"]["calls"] == 0
    assert len(model.requests) == 3


def test_tree_compression_returns_child_handoff_to_parent() -> None:
    trace, model, executor = run_script(
        [
            plan_response(
                ("Resolve the child investigation", "decompose"),
                ("Use child evidence", "execute"),
            ),
            plan_response(("Inspect child evidence", "execute")),
            "<action>curl https://example.test/child</action>",
            "<done>child id=C1 verified</done>",
            "The child frame verified exact identifier C1.",
            "<done>Used verified child C1</done>",
        ],
        config=RExConfig(
            compression_mode="tree",
            compression_frame_trigger_tokens=1,
        ),
    )

    assert trace["completed"] is True
    assert trace["budget"] == {"model_calls": 5, "tool_calls": 1, "frames": 2}
    assert executor.actions == ["curl https://example.test/child"]
    assert trace["compression"]["calls"] == 1
    assert trace["compression"]["applied"] == 1
    assert any(event["event"] == "frame_compress" for event in trace["events"])
    assert not any(event["event"] == "node_compress" for event in trace["events"])
    assert "[RUNNER-COMPRESSED HISTORY]" in json.dumps(trace["active_messages"])
    assert "[RUNNER-COMPRESSED HISTORY]" in json.dumps(model.requests[-1])
    assert "C1" in json.dumps(model.requests[-1])


def test_invalid_compressor_response_uses_nonfatal_fallback() -> None:
    trace, _, _ = run_script(
        [
            plan_response(("Resolve child task", "decompose")),
            plan_response(("Inspect a large record", "execute")),
            "<action>curl https://example.test/large</action>",
            "<done>verified id=A1 and performed no mutation</done>",
            " ",
        ],
        executor=FakeExecutor(
            [
                {
                    "status": "success",
                    "stdout": "verified id=A1 " + "large-result " * 3000,
                    "stderr": "",
                    "exit_code": 0,
                }
            ]
        ),
        config=RExConfig(
            compression_mode="tree",
            compression_frame_trigger_tokens=1,
        ),
    )

    assert trace["completed"] is True
    assert trace["compression"]["calls"] == 1
    assert any(event["event"] == "compression_fallback" for event in trace["events"])
    assert "verified id=A1" in json.dumps(trace["active_messages"])


def test_compression_is_not_applied_when_checkpoint_is_larger() -> None:
    trace, _, _ = run_script(
        [
            plan_response(("Resolve child task", "decompose")),
            plan_response(("Finish directly", "execute")),
            "<done>finished</done>",
            "H" * 9000,
        ],
        config=RExConfig(
            compression_mode="tree",
            compression_frame_trigger_tokens=1,
        ),
    )

    event = next(
        event for event in trace["events"] if event["event"] == "frame_compress"
    )
    assert trace["completed"] is True
    assert trace["compression"]["calls"] == 1
    assert trace["compression"]["applied"] == 0
    assert event["artifact"]["reason"] == "not_smaller"
    assert "[RUNNER-COMPRESSED HISTORY]" not in json.dumps(trace["active_messages"])


def test_compression_log_write_error_does_not_fail_task(tmp_path: Path) -> None:
    blocked_log_dir = tmp_path / "not-a-directory"
    blocked_log_dir.write_text("occupied", encoding="utf-8")
    model = ScriptedModel(
        [
            plan_response(("Resolve child task", "decompose")),
            plan_response(("Inspect a large record", "execute")),
            "<action>curl https://example.test/large</action>",
            "<done>verified A1</done>",
            "Verified exact identifier A1.",
        ]
    )

    trace = run_rex_runner(
        model_client=model,
        prompt="Complete the SaaS task",
        executor=FakeExecutor(
            [
                {
                    "status": "success",
                    "stdout": "large " * 3000,
                    "stderr": "",
                    "exit_code": 0,
                }
            ]
        ),
        system_prompt="Use the current phase protocol.",
        config=RExConfig(
            compression_mode="tree",
            compression_frame_trigger_tokens=1,
        ),
        compression_log_dir=blocked_log_dir,
    )

    assert trace["completed"] is True
    assert any(event["event"] == "compression_log_error" for event in trace["events"])


class FakeCheckpoint:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.completed: list[dict[str, Any]] = []

    def start_task(self, _key: str, record: dict[str, Any]) -> None:
        self.updates.append(dict(record))

    def update_task(self, _key: str, record: dict[str, Any]) -> None:
        self.updates.append(dict(record))

    def complete_task(self, _key: str, record: dict[str, Any]) -> None:
        self.completed.append(dict(record))


class LifecycleBash(FakeExecutor):
    instances: ClassVar[list[LifecycleBash]] = []

    def __init__(self, environment_id: str, **_kwargs: Any) -> None:
        super().__init__()
        self.environment_id = environment_id
        self.destroyed = False
        self.__class__.instances.append(self)

    def destroy_workspace(self) -> None:
        self.destroyed = True


class FakeAgentDiffClient:
    base_url = "http://agent-diff"
    api_key = "secret"

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.evaluated_expected: dict[str, Any] | None = None

    def init_env(self, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(environmentId="env-1")

    def start_run(self, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(runId="run-1")

    def evaluate_run(self, **kwargs: Any) -> dict[str, str]:
        self.evaluated_expected = kwargs["expectedOutput"]
        return {"status": "completed"}

    def get_results_for_run(self, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            status="completed",
            passed=True,
            score=1.0,
            failures=[],
            diff={},
        )

    def delete_env(self, *, envId: str) -> None:
        self.deleted.append(envId)


def test_cli_task_lifecycle_keeps_assertions_out_of_model_context(
    monkeypatch: Any,
) -> None:
    LifecycleBash.instances.clear()
    monkeypatch.setattr(benchmark, "BashExecutorProxy", LifecycleBash)
    model = ScriptedModel(
        [
            plan_response(("Send message", "execute")),
            "<action>curl https://slack.com/api/chat.postMessage</action>",
            "<done>message created</done>",
        ]
    )
    checkpoint = FakeCheckpoint()
    client = FakeAgentDiffClient()
    expected = {"assertions": [{"where": {"value": "secret_assertion_value"}}]}
    example = {
        "test_id": "slack_1",
        "test_name": "Send message",
        "question": "Send hello to general",
        "task_horizon": 2,
        "info": json.dumps(
            {
                "service": "slack",
                "seed_template": "slack_default",
                "impersonate_user_id": "U1",
            }
        ),
        "answer": json.dumps(expected),
    }

    record = benchmark.run_example(
        example=example,
        client=client,
        model_client=model,
        agent_config=RExConfig(),
        checkpoint=checkpoint,
        key="model|slack_1|0",
        model_name="model",
    )

    assert record["passed"] is True
    assert client.evaluated_expected == expected
    assert client.deleted == ["env-1"]
    assert LifecycleBash.instances[0].destroyed is True
    assert checkpoint.completed
    model_context = json.dumps(model.requests)
    assert "secret_assertion_value" not in model_context


def test_checkpoint_store_writes_one_file_per_task_and_rejects_config_change(
    tmp_path: Path,
) -> None:
    metadata = {
        "schema_version": 2,
        "agent": "rex_runner",
        "model": "model",
        "openai_base_url": "http://model",
        "agent_diff_base_url": "http://agent-diff",
        "dataset_name": "dataset",
        "dataset_split": "test",
        "dataset_fingerprint": "abc",
        "config": {"max_depth": 3},
    }
    store = benchmark.RExCheckpointStore(tmp_path, metadata)
    first_key = store.task_key("model", "slack/one")
    first = store.for_task(test_id="slack/one", task_key=first_key)
    second_key = store.task_key("model", "box_two")
    second = store.for_task(test_id="box_two", task_key=second_key)

    first.start_task(first_key, {"status": "running"})
    first.complete_task(first_key, {"status": "completed", "passed": True})

    checkpoint_files = sorted(tmp_path.glob("*.json"))
    assert len(checkpoint_files) == 2
    assert first.path.name == "slack%2Fone.json"
    assert second.path.name == "box_two.json"
    assert first.result() == {
        "status": "completed",
        "passed": True,
        "_checkpoint_key": first_key,
    }
    assert second.result() is None

    resumed = store.for_task(test_id="slack/one", task_key=first_key)
    assert resumed.is_completed() is True

    changed = {**metadata, "config": {"max_depth": 2}}
    changed_store = benchmark.RExCheckpointStore(tmp_path, changed)

    with pytest.raises(ValueError, match="Checkpoint configuration mismatch"):
        changed_store.for_task(test_id="slack/one", task_key=first_key)


def test_summary_handles_missing_scores(capsys: Any) -> None:
    results = [
        {
            "passed": False,
            "score": None,
            "time": 1.0,
            "trace": {"usage": {}, "budget": {}},
        }
    ]

    benchmark.print_summary("Incomplete score", results)

    assert "avg_score=n/a" in capsys.readouterr().out


def test_summary_normalizes_structured_scores() -> None:
    results = [
        {
            "passed": True,
            "score": {"total": 2, "passed": 1, "percent": 50},
            "time": 1.0,
            "trace": {"usage": {}, "budget": {}},
        },
        {
            "passed": True,
            "score": 1.0,
            "time": 1.0,
            "trace": {"usage": {}, "budget": {}},
        },
    ]

    summary = benchmark.metric_summary(results)

    assert summary["average_score"] == pytest.approx(0.75)
    assert summary["average_compression_calls"] == 0


@pytest.mark.parametrize("mode", ["none", "node", "hierarchical"])
def test_removed_compression_modes_are_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="compression_mode must be tree"):
        RExConfig(compression_mode=mode)
    for parser in (benchmark.build_parser(), parallel_benchmark.build_parallel_parser()):
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "model", "--compression-mode", mode])


def test_compression_defaults_to_tree_and_rejects_node_threshold() -> None:
    assert RExConfig().compression_mode == "tree"
    for parser in (benchmark.build_parser(), parallel_benchmark.build_parallel_parser()):
        args = parser.parse_args(["--model", "model"])
        assert parallel_benchmark.build_agent_config(args).compression_mode == "tree"
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--model", "model", "--compression-node-trigger-tokens", "1"]
            )


def test_compression_input_limit_option_is_rejected() -> None:
    for parser in (benchmark.build_parser(), parallel_benchmark.build_parallel_parser()):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--model", "model", "--compression-max-input-tokens", "32768"]
            )


def test_compression_cli_configuration_is_shared_by_serial_and_parallel() -> None:
    args = parallel_benchmark.build_parallel_parser().parse_args(
        [
            "--model",
            "model",
            "--compression-mode",
            "tree",
            "--compression-frame-trigger-tokens",
            "7000",
        ]
    )

    config = parallel_benchmark.build_agent_config(args)

    assert config.compression_mode == "tree"
    assert config.compression_frame_trigger_tokens == 7000


def test_parallel_runner_executes_tasks_concurrently(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dataset = [
        {"test_id": "slack_1", "test_name": "One", "task_horizon": 2},
        {"test_id": "box_2", "test_name": "Two", "task_horizon": 5},
    ]
    barrier = threading.Barrier(2)
    worker_threads: set[int] = set()
    worker_lock = threading.Lock()

    def fake_execute_task(
        *,
        example: dict[str, Any],
        checkpoint: Any,
        key: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        with worker_lock:
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

    monkeypatch.setattr(
        parallel_benchmark, "load_dataset", lambda *_args, **_kwargs: dataset
    )
    monkeypatch.setattr(parallel_benchmark, "execute_task", fake_execute_task)

    exit_code = parallel_benchmark.main(
        [
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
