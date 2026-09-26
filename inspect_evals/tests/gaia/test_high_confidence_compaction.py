import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ContentText,
    ContentToolUse,
    Model,
    ModelOutput,
)
from inspect_ai.tool import ToolCall

from inspect_evals.gaia.hierarchical_compact import (
    CompressionMode,
    CompressionScope,
    extract_compression_evidence,
    extract_inherited_evidence_ledgers,
    filter_runner_control_messages,
    prepare_compression_request,
    render_compressed_history,
)
from inspect_evals.gaia.hierarchical_compact_audit import (
    CompressionLogContext,
)
from inspect_evals.gaia.rex_runner import (
    RExConfig,
    RExPlanPatch,
    RExRunner,
    RExStatus,
    RExStep,
    RExStepResult,
)


class FakeCompressionModel:
    def __init__(
        self,
        token_counts: list[int | Exception],
        *,
        completion: str = "Concise verified handoff.",
        generate_error: Exception | None = None,
    ) -> None:
        self.token_counts = token_counts
        self.completion = completion
        self.generate_error = generate_error
        self.generate_calls: list[tuple[list[ChatMessage], list[Any]]] = []

    async def count_tokens(self, messages: list[ChatMessage]) -> int:
        value = self.token_counts.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    async def generate(
        self, messages: list[ChatMessage], *, tools: list[Any]
    ) -> ModelOutput:
        self.generate_calls.append((messages, tools))
        if self.generate_error is not None:
            raise self.generate_error
        return ModelOutput.from_content("mockllm/compressor", self.completion)


def _runner(
    model: FakeCompressionModel,
    messages: list[ChatMessage],
    *,
    log_context: CompressionLogContext | None = None,
) -> RExRunner:
    runner = RExRunner(
        model=cast(Model, model),
        messages=messages,
        tools=[],
        config=RExConfig(
            max_steps_per_frame=2,
            max_depth=2,
            max_turns_per_step=2,
            max_planning_turns=2,
            max_tool_calls=2,
            max_observation_chars=100,
            planning_tools=False,
            compression_mode="tree",
            compression_backend="standalone",
        ),
        compression_log_context=log_context,
        compression_log_context_error=(
            None if log_context is not None else "not running inside Inspect eval"
        ),
    )
    runner._event = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return runner


def _log_context(tmp_path: Path) -> CompressionLogContext:
    return CompressionLogContext(
        root=tmp_path / "eval-run",
        eval_log_parent=tmp_path,
        task_id="gaia-task-id",
        epoch=1,
        sample_uuid="sample-uuid",
        runtime_sample_id="runtime-sample",
    )


def _scope() -> CompressionScope:
    return CompressionScope(
        kind="frame",
        frame_id="root",
        step_id="step_1",
        goal="Verify the exact value",
        status="success",
        consumer_goal="Answer the question",
        depth=1,
        execution_mode="execute",
    )


@pytest.mark.parametrize("has_successor", [False, True])
def test_compressor_context_includes_parent_and_single_successor(
    has_successor: bool,
) -> None:
    scope = replace(
        _scope(),
        successor_goal="Use the verified value" if has_successor else None,
        successor_execution_mode="execute" if has_successor else None,
    )
    prepared = prepare_compression_request(scope, [])
    context = prepared.task_context

    assert context["current_task"]["goal"] == scope.goal
    assert context["parent_task"] == {"goal": "Answer the question"}
    assert context["successor_task"] == (
        {"goal": "Use the verified value", "execution_mode": "execute"}
        if has_successor
        else None
    )
    payload = prepared.messages[1].text.split("=== RUNNER TASK CONTEXT ===\n")[1]
    assert json.loads(payload.split("\n=== END TASK CONTEXT ===")[0]) == context


def test_successor_goal_truncation_is_reported() -> None:
    goal = "下一步核验数据" * 200
    context = prepare_compression_request(
        replace(_scope(), successor_goal=goal, successor_execution_mode="decompose"),
        [],
    ).task_context

    assert context["context_truncated"] is True
    assert len(context["successor_task"]["goal"]) < len(goal)
    assert context["successor_task"]["execution_mode"] == "decompose"


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
async def test_recursive_compression_context_tracks_batch_and_parent(
    monkeypatch: pytest.MonkeyPatch, recover: bool
) -> None:
    runner = _runner(FakeCompressionModel([]), [])
    runner.config = replace(runner.config, max_depth=4, max_steps_per_frame=8)
    monkeypatch.setattr(
        "inspect_evals.gaia.rex_runner.span", lambda *args, **kwargs: nullcontext()
    )

    def batch(*goals: str, complete: bool = True) -> RExPlanPatch:
        return RExPlanPatch(
            [RExStep(goal, goal, "decompose") for goal in goals], complete=complete
        )

    plans = {
        "root": [batch("a", "b", "c", complete=False), batch("d", "e")],
        "root.a": [batch("inner", "inner_next")],
    }
    contexts: dict[str, dict[str, Any]] = {}
    original_body = runner._run_frame_body

    async def plan_batch(**kwargs: Any) -> RExPlanPatch:
        pending = plans.get(kwargs["frame_id"])
        return pending.pop(0) if pending else batch()

    async def frame_body(frame_id: str, goal: str, *, depth: int) -> RExStepResult:
        if recover and goal == "a":
            return RExStepResult("failed", "Source unavailable")
        return await original_body(frame_id, goal, depth=depth)

    async def compact_scope(*, scope: CompressionScope, **kwargs: Any) -> None:
        contexts[scope.goal] = prepare_compression_request(scope, []).task_context

    monkeypatch.setattr(runner, "_plan_batch", plan_batch)
    monkeypatch.setattr(runner, "_run_frame_body", frame_body)
    monkeypatch.setattr(runner, "_compact_completed_scope", compact_scope)

    result = await runner._run_frame("root", "Root goal", depth=1)

    assert result.status == "success"
    expected = {
        "d": ("Root goal", "e"),
        "e": ("Root goal", None),
    }
    if not recover:
        expected.update(
            {
                "a": ("Root goal", "b"),
                "inner": ("a", "inner_next"),
                "inner_next": ("a", None),
                "b": ("Root goal", "c"),
                "c": ("Root goal", None),
            }
        )
    assert set(contexts) == set(expected)
    for goal, (parent, successor) in expected.items():
        assert contexts[goal]["current_task"]["goal"] == goal
        assert contexts[goal]["parent_task"] == {"goal": parent}
        assert contexts[goal]["successor_task"] == (
            {"goal": successor, "execution_mode": "decompose"}
            if successor is not None
            else None
        )


def test_filter_control_messages_preserves_tool_calls_and_evidence() -> None:
    tool_call = ToolCall(id="call-1", function="bash", arguments={"cmd": "date"})
    messages: list[ChatMessage] = [
        ChatMessageUser(content="Grounding and task decomposition phase. Ignore"),
        ChatMessageAssistant(
            content='{"status":"success","summary":"done"}',
            tool_calls=[tool_call],
        ),
        ChatMessageTool(content="2026-08-25", tool_call_id="call-1", function="bash"),
    ]

    filtered = filter_runner_control_messages(messages)
    evidence = extract_compression_evidence(messages)

    assert len(filtered) == 2
    assert isinstance(filtered[0], ChatMessageAssistant)
    assert filtered[0].text == "done"
    assert filtered[0].tool_calls == [tool_call]
    assert evidence[0].tool_name == "bash"
    assert evidence[0].tool_input == {"cmd": "date"}
    assert evidence[0].output == "2026-08-25"
    assert evidence[0].is_error is False


def test_filter_control_messages_matches_current_runner_prompts() -> None:
    messages: list[ChatMessage] = [
        ChatMessageUser(
            content="\nUse the conversation and recent tool observations to "
            "decompose goal.\n\nGoal:\nQuestion"
        ),
        ChatMessageAssistant(
            content='{"thinking":"memo","steps":[],"planning_complete":true}'
        ),
    ]

    filtered = filter_runner_control_messages(messages)

    assert filtered == []


def test_filter_control_messages_preserves_server_side_tool_evidence() -> None:
    tool_use = ContentToolUse(
        tool_type="code_execution",
        id="server-call-1",
        name="computer",
        arguments='{"action":"screenshot"}',
        result="image captured",
    )
    message = ChatMessageAssistant(
        content=[
            tool_use,
        ]
    )
    protocol_message = message.model_copy(
        update={
            "content": [
                tool_use,
                ContentText(text='{"status":"success"}'),
            ]
        }
    )

    filtered = filter_runner_control_messages([protocol_message])
    evidence = extract_compression_evidence([protocol_message])

    assert len(filtered) == 1
    assert filtered[0].content_list == [tool_use]
    assert evidence[0].tool_use_id == "server-call-1"
    assert evidence[0].output == "image captured"


def test_nested_checkpoint_evidence_is_flattened_and_deduplicated() -> None:
    child = render_compressed_history(
        _scope(),
        "The exact value was verified.",
        evidence=extract_compression_evidence(
            [
                ChatMessageAssistant(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            function="bash",
                            arguments={"cmd": "date"},
                        )
                    ],
                ),
                ChatMessageTool(
                    content="2026-08-25",
                    tool_call_id="call-1",
                    function="bash",
                ),
            ]
        ),
    )
    nested: list[ChatMessage] = [
        ChatMessageUser(content=child),
        ChatMessageUser(content=child),
    ]

    inherited = extract_inherited_evidence_ledgers(nested)
    prepared = prepare_compression_request(_scope(), nested)

    assert len(inherited) == 1
    assert "Call ID: call-1" in inherited[0]
    assert prepared.inherited_evidence_ledgers == inherited
    assert "RUNNER INHERITED EVIDENCE LEDGER" not in prepared.messages[1].text
    assert "Call ID: call-1" not in prepared.messages[1].text
    assert "Observed evidence (recorded by runner)" not in prepared.messages[1].text


@pytest.mark.asyncio
async def test_standalone_compression_uses_no_tools_and_counts_call() -> None:
    original = ChatMessageUser(content="Detailed history " * 20)
    model = FakeCompressionModel([100, 10], completion="Standalone handoff")
    runner = _runner(model, [original])
    result = RExStepResult("success", "Original summary")

    await runner._compact_completed_scope(
        scope=_scope(), start_index=0, trigger_tokens=1, result=result
    )

    assert len(model.generate_calls) == 1
    request, tools = model.generate_calls[0]
    assert [message.role for message in request] == ["system", "user"]
    assert tools == []
    assert runner.budget.model_calls == 1
    assert result.summary == "Standalone handoff"


@pytest.mark.asyncio
async def test_standalone_compression_writes_complete_text_log(
    tmp_path: Path,
) -> None:
    original = ChatMessageUser(content="Detailed history " * 20)
    model = FakeCompressionModel([100, 10], completion="Standalone handoff")
    runner = _runner(model, [original], log_context=_log_context(tmp_path))

    await runner._compact_completed_scope(
        scope=_scope(),
        start_index=0,
        trigger_tokens=1,
        result=RExStepResult("success", "Original summary"),
    )

    [path] = list((tmp_path / "eval-run").rglob("*.txt"))
    text = path.read_text(encoding="utf-8")
    assert "c000001_tree_" in path.name
    assert "=== COMPRESSOR INPUT (READABLE) ===" in text
    assert "Standalone handoff" in text
    assert '"model_output"' in text
    assert '"reason": "applied"' in text


@pytest.mark.asyncio
async def test_compression_below_threshold_does_not_call_compressor(
    tmp_path: Path,
) -> None:
    original = ChatMessageUser(content="Detailed history")
    model = FakeCompressionModel([10])
    runner = _runner(model, [original], log_context=_log_context(tmp_path))
    result = RExStepResult("success", "Original summary")

    await runner._compact_completed_scope(
        scope=_scope(), start_index=0, trigger_tokens=10, result=result
    )

    assert runner.messages == [original]
    assert model.generate_calls == []
    assert result.summary == "Original summary"
    assert runner._compression_sequence == 0
    assert not list((tmp_path / "eval-run").rglob("*.txt"))


@pytest.mark.asyncio
async def test_not_smaller_checkpoint_preserves_original_history(
    tmp_path: Path,
) -> None:
    original = ChatMessageUser(content="Detailed history")
    model = FakeCompressionModel([100, 100], completion="Standalone handoff")
    runner = _runner(model, [original], log_context=_log_context(tmp_path))
    result = RExStepResult("success", "Original summary")

    await runner._compact_completed_scope(
        scope=_scope(), start_index=0, trigger_tokens=1, result=result
    )

    assert runner.messages == [original]
    assert result.summary == "Original summary"
    assert runner.budget.model_calls == 1
    [path] = list((tmp_path / "eval-run").rglob("*.txt"))
    assert '"reason": "not_smaller"' in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_compressor_failure_preserves_original_history() -> None:
    original = ChatMessageUser(content="Detailed history")
    model = FakeCompressionModel([100], generate_error=RuntimeError("unavailable"))
    runner = _runner(model, [original])
    result = RExStepResult("success", "Original summary")

    await runner._compact_completed_scope(
        scope=_scope(), start_index=0, trigger_tokens=1, result=result
    )

    assert runner.messages == [original]
    assert result.summary == "Original summary"
    assert runner.budget.model_calls == 0


@pytest.mark.asyncio
async def test_compressor_failure_writes_error_text_log(tmp_path: Path) -> None:
    original = ChatMessageUser(content="Detailed history " * 20)
    model = FakeCompressionModel([100], generate_error=RuntimeError("unavailable"))
    runner = _runner(model, [original], log_context=_log_context(tmp_path))

    await runner._compact_completed_scope(
        scope=_scope(),
        start_index=0,
        trigger_tokens=1,
        result=RExStepResult("success", "Original summary"),
    )

    [path] = list((tmp_path / "eval-run").rglob("*.txt"))
    text = path.read_text(encoding="utf-8")
    assert "error_type: RuntimeError" in text
    assert "error_message: unavailable" in text
    assert '"reason": "compressor_error"' in text


@pytest.mark.asyncio
async def test_text_log_write_failure_does_not_change_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ChatMessageUser(content="Detailed history " * 20)
    model = FakeCompressionModel([100, 10], completion="Standalone handoff")
    runner = _runner(model, [original], log_context=_log_context(tmp_path))
    events: list[dict[str, Any]] = []

    def capture_event(event: str, **kwargs: Any) -> None:
        events.append({"event": event, **kwargs})

    def fail_write(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk unavailable")

    runner._event = capture_event  # type: ignore[method-assign]
    monkeypatch.setattr(
        "inspect_evals.gaia.rex_runner.write_compressor_text_log",
        fail_write,
    )

    result = RExStepResult("success", "Original summary")
    await runner._compact_completed_scope(
        scope=_scope(), start_index=0, trigger_tokens=1, result=result
    )

    assert len(runner.messages) == 1
    assert runner.messages[0].text.startswith("[GAIA-COMPRESSED HISTORY]")
    assert result.summary == "Standalone handoff"
    artifact = events[-1]["artifact"]
    assert artifact["applied"] is True
    assert artifact["text_log"] == {
        "status": "write_error",
        "error_type": "OSError",
        "error_message": "disk unavailable",
    }


@pytest.mark.asyncio
async def test_request_construction_failure_preserves_original_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ChatMessageUser(content="Detailed history")
    model = FakeCompressionModel([100])
    runner = _runner(model, [original], log_context=_log_context(tmp_path))
    result = RExStepResult("success", "Original summary")

    def fail_request(*args: Any, **kwargs: Any) -> None:
        raise ValueError("invalid history")

    monkeypatch.setattr(
        "inspect_evals.gaia.rex_runner.prepare_compression_request",
        fail_request,
    )

    await runner._compact_completed_scope(
        scope=_scope(), start_index=0, trigger_tokens=1, result=result
    )

    assert runner.messages == [original]
    assert result.summary == "Original summary"
    assert model.generate_calls == []
    [path] = list((tmp_path / "eval-run").rglob("*.txt"))
    text = path.read_text(encoding="utf-8")
    assert '"reason": "request_error"' in text
    assert "error_type: ValueError" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["", "x" * 13_000], ids=["empty", "oversized"])
async def test_invalid_compressor_handoff_preserves_original_history(
    tmp_path: Path,
    completion: str,
) -> None:
    original = ChatMessageUser(content="Detailed history")
    model = FakeCompressionModel([100], completion=completion)
    runner = _runner(model, [original], log_context=_log_context(tmp_path))
    result = RExStepResult("success", "Original summary")

    await runner._compact_completed_scope(
        scope=_scope(), start_index=0, trigger_tokens=1, result=result
    )

    assert runner.messages == [original]
    assert result.summary == "Original summary"
    assert runner.budget.model_calls == 1
    [path] = list((tmp_path / "eval-run").rglob("*.txt"))
    text = path.read_text(encoding="utf-8")
    assert '"reason": "invalid_compressor_handoff"' in text
    assert "error_type: InvalidHandoff" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["none", "tree"])
@pytest.mark.parametrize("status", ["success", "failed"])
async def test_compression_mode_controls_child_frame_boundary(
    mode: CompressionMode,
    status: RExStatus,
) -> None:
    runner = _runner(FakeCompressionModel([1]), [])
    runner.config = replace(runner.config, compression_mode=mode)
    captured: list[dict[str, Any]] = []

    async def run_frame_body(frame_id: str, goal: str, *, depth: int) -> RExStepResult:
        runner.messages.append(ChatMessageUser(content=f"frame {frame_id}"))
        return RExStepResult(status, goal)

    async def compact_scope(**kwargs: Any) -> None:
        captured.append(kwargs)

    runner._run_frame_body = run_frame_body  # type: ignore[method-assign]
    runner._compact_completed_scope = compact_scope  # type: ignore[method-assign]

    await runner._run_frame("root", "Root goal", depth=1)
    await runner._run_frame("root.step_1", "Child goal", depth=2)

    if mode == "none" or status != "success":
        assert captured == []
        assert [message.text for message in runner.messages] == [
            "frame root",
            "frame root.step_1",
        ]
    else:
        assert len(captured) == 1
        assert captured[0]["scope"].kind == "frame"
        assert captured[0]["scope"].execution_mode == "decomposed_frame"
        assert captured[0]["scope"].frame_id == "root.step_1"
        assert captured[0]["start_index"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,status", [("none", "success"), ("tree", "failed")])
async def test_skipped_compression_preserves_history_without_counting_or_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: CompressionMode,
    status: RExStatus,
) -> None:
    original = ChatMessageUser(content="Detailed history " * 2000)
    model = FakeCompressionModel([])
    runner = _runner(model, [original], log_context=_log_context(tmp_path))
    runner.config = replace(runner.config, compression_mode=mode)
    result = RExStepResult(status, "Original summary")
    count_tokens = AsyncMock()
    event = Mock()
    monkeypatch.setattr(model, "count_tokens", count_tokens)
    monkeypatch.setattr(runner, "_event", event)

    await runner._compact_completed_scope(
        scope=replace(_scope(), status=status),
        start_index=0,
        trigger_tokens=1,
        result=result,
    )

    count_tokens.assert_not_awaited()
    event.assert_not_called()
    assert runner.messages == [original]
    assert runner.messages[0] is original
    assert result.summary == "Original summary"
    assert model.generate_calls == []
    assert runner._compression_sequence == 0
    assert runner.budget.model_calls == 0
    assert not list((tmp_path / "eval-run").rglob("*.txt"))
