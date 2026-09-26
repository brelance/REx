import json
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai._util.registry import registry_create
from inspect_ai.agent import Agent
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model
from inspect_ai.tool import Tool, ToolCallError, tool

from inspect_evals.gaia import gaia_recap_agent
from inspect_evals.gaia.recap_agent import (
    RecapController,
    RecapState,
    extract_tool_observation,
    parse_recap_json,
)
from inspect_evals.gaia.scorer import gaia_scorer


def _output(content: str) -> ModelOutput:
    return ModelOutput.from_content("mockllm/model", content)


def _recap(think: str, *subtasks: str) -> ModelOutput:
    return _output(json.dumps({"think": think, "subtasks": list(subtasks)}))


@tool
def lookup_answer_recap() -> Tool:
    async def execute() -> str:
        """Return the answer used in ReCAP tests."""
        return "4"

    return execute


def _task(solver: Agent) -> Task:
    return Task(
        dataset=MemoryDataset([Sample(input="What is 2 + 2?", target="4")]),
        solver=solver,
        scorer=gaia_scorer(),
        message_limit=100,
    )


@pytest.mark.parametrize(
    "response",
    [
        '{"think":"plan","subtasks":["research"]}',
        '```json\n{"think":"plan","subtasks":["research"]}\n```',
        '```\n{"think":"plan","subtasks":["research"]}\n```',
        'Plan:\n```JSON\n{"think":"plan","subtasks":["research"]}\n``` trailing',
        '```json\ninvalid\n```\n```json\n{"think":"plan","subtasks":["research"]}\n```',
    ],
)
def test_parse_recap_json_accepts_complete_json(response: str) -> None:
    parsed = parse_recap_json(response)

    assert parsed is not None
    assert parsed.think == "plan"
    assert parsed.subtasks == ["research"]


@pytest.mark.parametrize(
    "response",
    [
        "",
        "not json",
        '{"think":"missing subtasks"}',
        '{"subtasks":[]}',
        '{"think":"bad type","subtasks":"research"}',
        'prefix {"think":"plan","subtasks":[]}',
        '```json\n{"think":"plan","subtasks":[]}\n',
        'Plan:\n```json\n{"think":"missing subtasks"}\n``` trailing',
        'Plan:\n```json\n{"think":"bad type","subtasks":"research"}\n```',
        '```json\n[]\n```',
    ],
)
def test_parse_recap_json_rejects_invalid_protocol(response: str) -> None:
    assert parse_recap_json(response) is None


def test_recap_controller_descends_ascends_and_finalizes() -> None:
    controller = RecapController("Root task")

    assert controller.depth == 1
    assert "Your current task:\nRoot task" in controller.initial_prompt()

    step = controller.process_assistant_text(
        '{"think":"split","subtasks":["Child task","Wrap up"]}'
    )
    assert step.state == RecapState.DOWN
    assert controller.current_task == "Child task"
    assert controller.depth == 2

    step = controller.process_assistant_text(
        '{"think":"child done","subtasks":[]}'
    )
    assert step.state == RecapState.UP
    assert step.done_task_name == "Child task"
    assert step.remaining_subtasks == ["Wrap up"]
    assert controller.current_task == "Root task"
    assert controller.depth == 1

    step = controller.process_assistant_text(
        '{"think":"root done","subtasks":[]}'
    )
    assert step.state == RecapState.FINALIZE
    assert step.continue_loop
    assert step.prompt is not None
    assert "Return only the final answer" in step.prompt

    step = controller.process_assistant_text("4")
    assert step.state == RecapState.FINALIZE
    assert step.done
    assert not step.continue_loop


def test_recap_controller_enforces_maximum_depth() -> None:
    controller = RecapController("Root task", max_depth=2)

    controller.process_assistant_text(
        '{"think":"split root","subtasks":["Level two"]}'
    )
    step = controller.process_assistant_text(
        '{"think":"split again","subtasks":["Forbidden level three"]}'
    )

    assert step.state == RecapState.DOWN
    assert step.note == "max_depth_reached"
    assert step.prompt is not None
    assert "maximum ReCAP depth (2)" in step.prompt
    assert controller.depth == 2
    assert controller.current_task == "Level two"
    assert controller.node.children == []


def test_recap_controller_caps_subtasks() -> None:
    controller = RecapController("Root task", max_subtasks=2)

    step = controller.process_assistant_text(
        '{"think":"split","subtasks":["one","two","three"]}'
    )

    assert step.remaining_subtasks == ["two"]
    assert controller.root.latest_info().subtasks == ["one", "two"]


def test_recap_controller_retries_invalid_json_once() -> None:
    controller = RecapController("Root task")

    retry = controller.process_assistant_text("invalid")
    stopped = controller.process_assistant_text("still invalid")

    assert retry.continue_loop
    assert retry.note == "invalid_recap_json_retry"
    assert retry.prompt is not None
    assert not stopped.continue_loop
    assert not stopped.done
    assert stopped.note == "non_recap_final"


def test_recap_controller_returns_to_finalize_after_tool_verification() -> None:
    controller = RecapController("Root task")
    controller.process_assistant_text(
        '{"think":"root done","subtasks":[]}'
    )

    action = controller.after_tool_action("Verification succeeded.")
    assert action.state == RecapState.ACTION_TAKEN
    assert action.prompt is not None
    assert "Verification succeeded." in action.prompt

    finalize = controller.process_assistant_text(
        '{"think":"verified","subtasks":[]}'
    )
    assert finalize.state == RecapState.FINALIZE
    assert finalize.continue_loop


def test_extract_tool_observation_includes_errors_and_truncates() -> None:
    messages = [
        ChatMessageTool(content="successful result"),
        ChatMessageTool(
            content="failed result",
            error=ToolCallError("unknown", "lookup failed"),
        ),
    ]

    observation = extract_tool_observation(messages, max_chars=35)

    assert observation.startswith("successful result\n\n[tool error]")
    assert observation.endswith("[observation truncated]")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_depth": 0}, "max_depth must be at least 1"),
        ({"max_subtasks": 0}, "max_subtasks must be at least 1"),
        ({"max_obs_chars": 0}, "max_obs_chars must be at least 1"),
        ({"max_tree_chars": 0}, "max_tree_chars must be at least 1"),
        (
            {"action_taken_prompt_variant": "invalid"},
            "action_taken_prompt_variant must be 'baseline' or 'sibling_guard'",
        ),
    ],
)
def test_recap_controller_validates_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RecapController("Root task", **kwargs)  # type: ignore[arg-type]


def test_recap_agent_factory_and_cli_registration() -> None:
    registered = registry_create(
        "agent", "inspect_evals/gaia_recap_agent", tools=[]
    )

    assert isinstance(gaia_recap_agent(tools=[]), Agent)
    assert isinstance(registered, Agent)


def test_recap_agent_factory_rejects_invalid_depth() -> None:
    with pytest.raises(ValueError, match="max_depth must be at least 1"):
        gaia_recap_agent(max_depth=0, tools=[])


def test_recap_agent_runs_gaia_end_to_end(tmp_path: Path) -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _recap("delegate lookup", "Look up the answer"),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="lookup_answer_recap",
                tool_arguments={},
            ),
            _recap("lookup complete"),
            _recap("root complete"),
            _output("4"),
        ],
    )

    [log] = eval(
        _task(gaia_recap_agent(tools=[lookup_answer_recap()])),
        model=model,
        log_dir=str(tmp_path),
        debug_errors=True,
    )

    assert log.status == "success"
    assert log.results is not None
    assert log.results.scores[0].metrics["accuracy"].value == 1.0
    assert log.samples is not None
    sample = log.samples[0]
    assert sample.output.completion == "4"
    assert any(message.role == "tool" for message in sample.messages)
    assert any(
        message.role == "user" and "return to the parent task" in message.text
        for message in sample.messages
    )


def test_recap_agent_can_verify_with_tool_during_finalize(tmp_path: Path) -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _recap("root complete"),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="lookup_answer_recap",
                tool_arguments={},
            ),
            _recap("verification complete"),
            _output("4"),
        ],
    )

    [log] = eval(
        _task(gaia_recap_agent(tools=[lookup_answer_recap()])),
        model=model,
        log_dir=str(tmp_path),
        debug_errors=True,
    )

    assert log.status == "success"
    assert log.results is not None
    assert log.results.scores[0].metrics["accuracy"].value == 1.0
    assert log.samples is not None
    sample = log.samples[0]
    assert sample.output.completion == "4"
    assert any(
        message.role == "user"
        and "Latest observation:\n4" in message.text
        for message in sample.messages
    )


def test_qwen_baseline_script_runs_all_agents_in_order() -> None:
    script = (
        Path(__file__).parents[2]
        / "src/inspect_evals/gaia/run_qwen35_9b_baseline_comparison.sh"
    ).read_text()
    solvers = [
        "gaia_plan_execute_agent",
        "gaia_reflection_agent",
        "gaia_recap_agent",
        "gaia_high_confidence_recursive_agent",
    ]

    assert [script.index(solver) for solver in solvers] == sorted(
        script.index(solver) for solver in solvers
    )
    assert 'MODEL="${GAIA_MODEL:-openai/Qwen3.5-9B}"' in script
    assert (
        'MODEL_BASE_URL="${GAIA_MODEL_BASE_URL:-http://localhost:30000/v1}"'
        in script
    )
    assert 'RUN_NAME="${GAIA_RUN_NAME:-qwen35-9b}"' in script
    assert "readonly MESSAGE_LIMIT=250" in script
    assert "CONCURRENCY=\"${GAIA_CONCURRENCY:-8}\"" in script
    assert "docker:src/inspect_evals/gaia/compose.proxy.yaml" in script
    assert '"logs/gaia-react-${RUN_NAME}-full"' in script
    assert '"logs/gaia-recap-${RUN_NAME}-full"' in script
    assert '"logs/gaia-high-confidence-${RUN_NAME}-full"' in script
