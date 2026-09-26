import json

import pytest
from inspect_ai import Task, eval
from inspect_ai._util.registry import registry_create
from inspect_ai.agent import Agent
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ModelOutput,
    get_model,
)
from inspect_ai.tool import Tool, tool

from inspect_evals.gaia import gaia_high_confidence_recursive_agent
from inspect_evals.gaia.dataset import (
    DEFAULT_INPUT_PROMPT,
    GAIA_FINAL_ANSWER_FORMAT,
)
from inspect_evals.gaia.rex_runner import (
    RExBudgetExceeded,
    RExConfig,
    RExPlanError,
    RExRunner,
    _mode_guidance,
    _split_root_prompt,
    parse_plan,
    parse_step_result,
)
from inspect_evals.gaia.scorer import gaia_scorer


def _output(content: str) -> ModelOutput:
    return ModelOutput.from_content("mockllm/model", content)


def _plan(
    steps: list[tuple[str, str]], *, complete: bool = True, thinking: str = "memo"
) -> ModelOutput:
    return _output(
        json.dumps(
            {
                "thinking": thinking,
                "steps": [
                    {"step_goal": goal, "execution_mode": mode} for goal, mode in steps
                ],
                "planning_complete": complete,
            }
        )
    )


def _step(status: str, summary: str) -> ModelOutput:
    return _output(json.dumps({"status": status, "summary": summary}))


@tool
def lookup_answer() -> Tool:
    async def execute() -> str:
        """Look up the answer used by the scripted test model."""
        return "4"

    return execute


def test_parse_plan_accepts_wrapped_json() -> None:
    patch = parse_plan(
        'Plan follows:\n```json\n{"thinking":"m","steps":'
        '[{"step_goal":"research","execution_mode":"decompose"}],'
        '"planning_complete":false}\n```',
        next_step_index=3,
        planned_step_count=2,
        max_steps=5,
    )

    assert patch.thinking == "m"
    assert patch.complete is False
    assert patch.steps[0].id == "step_3"
    assert patch.steps[0].kind == "decompose"


@pytest.mark.parametrize(
    "response",
    [
        '{"thinking":"","steps":[],"planning_complete":false}',
        '{"thinking":"","steps":['
        '{"step_goal":"a","execution_mode":"execute"},'
        '{"step_goal":"b","execution_mode":"execute"}],'
        '"planning_complete":true}',
    ],
)
def test_parse_plan_enforces_step_budget(response: str) -> None:
    with pytest.raises(RExPlanError):
        parse_plan(
            response,
            next_step_index=2,
            planned_step_count=1,
            max_steps=2,
        )


@pytest.mark.parametrize("limit", [1, 2, 3, 4])
def test_parse_plan_enforces_planning_horizon(limit: int) -> None:
    steps = ",".join(
        '{"step_goal":"work","execution_mode":"execute"}'
        for _ in range(limit + 1)
    )
    with pytest.raises(RExPlanError, match="batch limit"):
        parse_plan(
            f'{{"thinking":"","steps":[{steps}],"planning_complete":false}}',
            next_step_index=1,
            planned_step_count=0,
            max_steps=16,
            batch_limit=limit,
        )


def test_parse_plan_requires_complete_one_shot() -> None:
    with pytest.raises(RExPlanError, match="one-shot"):
        parse_plan(
            '{"thinking":"","steps":[{"step_goal":"work","execution_mode":"execute"}],"planning_complete":false}',
            next_step_index=1,
            planned_step_count=0,
            max_steps=16,
            planning_schedule="one_shot",
        )


def test_parse_step_result_treats_plain_text_as_failure() -> None:
    result = parse_step_result("Evidence collected.")

    assert result.status == "failed"
    assert result.summary == "Step returned no valid status JSON."


def test_split_root_prompt_extracts_default_question() -> None:
    root_prompt = _split_root_prompt(
        DEFAULT_INPUT_PROMPT.format(file="", question="What is 2 + 2?")
    )

    assert root_prompt.goal == "What is 2 + 2?"
    assert root_prompt.final_answer_format == GAIA_FINAL_ANSWER_FORMAT


def test_split_root_prompt_preserves_file_context() -> None:
    file_context = "The relevant file is /shared_files/example.xlsx"
    root_prompt = _split_root_prompt(
        DEFAULT_INPUT_PROMPT.format(
            file=file_context,
            question="What is the oldest entry?",
        )
    )

    assert root_prompt.goal == (
        f"{file_context}\n\nQuestion:\nWhat is the oldest entry?"
    )
    assert root_prompt.final_answer_format == GAIA_FINAL_ANSWER_FORMAT


def test_split_root_prompt_preserves_custom_prompt_as_goal() -> None:
    custom_prompt = "Solve this and explain your work: What is 2 + 2?"

    root_prompt = _split_root_prompt(custom_prompt)

    assert root_prompt.goal == custom_prompt
    assert root_prompt.final_answer_format is None


@pytest.mark.parametrize("compression_mode", ["none", "tree"])
def test_agent_factory_returns_agent(compression_mode) -> None:
    created_agent = gaia_high_confidence_recursive_agent(
        tools=[], compression_mode=compression_mode,
    )

    assert isinstance(created_agent, Agent)


def test_agent_rejects_invalid_compression_mode() -> None:
    with pytest.raises(ValueError, match="compression_mode"):
        gaia_high_confidence_recursive_agent(tools=[], compression_mode="invalid")


def test_agent_is_registered_for_cli_use() -> None:
    created_agent = registry_create(
        "agent",
        "inspect_evals/gaia_high_confidence_recursive_agent",
        max_depth=2,
        tools=[],
    )

    assert isinstance(created_agent, Agent)


def test_agent_rejects_non_positive_budgets() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        gaia_high_confidence_recursive_agent(max_depth=0, tools=[])



def test_agent_has_no_global_model_call_budget() -> None:
    runner = RExRunner(
        model=get_model("mockllm/model"),
        messages=[],
        tools=[],
        config=RExConfig(
            max_steps_per_frame=1,
            max_depth=1,
            max_turns_per_step=1,
            max_planning_turns=1,
            max_tool_calls=1,
            max_observation_chars=1,
            planning_tools=False,
        ),
    )
    runner.budget.model_calls = 10_000

    runner._check_budget()

    runner.budget.tool_calls = 2
    with pytest.raises(RExBudgetExceeded, match="tool-call budget"):
        runner._check_budget()








def test_high_confidence_agent_end_to_end(tmp_path) -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _plan([("Calculate the answer", "execute")]),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="lookup_answer",
                tool_arguments={},
                tool_call_id="lookup-answer-call",
            ),
            _step("success", "2 + 2 = 4"),
            _output("4"),
        ],
    )
    task = Task(
        dataset=MemoryDataset(
            [
                Sample(
                    input=DEFAULT_INPUT_PROMPT.format(
                        file="", question="What is 2 + 2?"
                    ),
                    target="4",
                )
            ]
        ),
        solver=gaia_high_confidence_recursive_agent(tools=[lookup_answer()]),
        scorer=gaia_scorer(),
        message_limit=100,
    )

    [log] = eval(task, model=model, log_dir=str(tmp_path), debug_errors=True)

    assert log.status == "success"
    assert log.results is not None
    assert log.results.scores[0].metrics["accuracy"].value == 1.0
    assert log.samples is not None
    assert log.samples[0].output.completion == "4"
    assert any(message.role == "tool" for message in log.samples[0].messages)
    assert all(message.source != "input" for message in log.samples[0].messages)

    system_messages = [
        message.text for message in log.samples[0].messages if message.role == "system"
    ]
    assert any(GAIA_FINAL_ANSWER_FORMAT in message for message in system_messages)

    phase_messages = [
        message.text
        for message in log.samples[0].messages
        if message.role == "user"
        and message.text.startswith(
            (
                "Grounding and task decomposition phase.",
                "Direct subtask execution.",
            )
        )
    ]
    assert phase_messages
    assert all(GAIA_FINAL_ANSWER_FORMAT not in message for message in phase_messages)

    final_messages = [
        message.text
        for message in log.samples[0].messages
        if message.role == "user"
        and message.text.startswith("Answer the original GAIA question")
    ]
    assert len(final_messages) == 1
    assert GAIA_FINAL_ANSWER_FORMAT in final_messages[0]


def test_mode_guidance_disables_decomposition_at_max_depth() -> None:
    guidance = _mode_guidance(
        3,
        3,
    )

    assert guidance == (
        '- No decomposition levels remain. Use "execute" for every step.\n'
        "- Split composite remaining work into multiple narrower execute steps."
    )


def test_failed_batch_replans_and_drops_remaining_step(tmp_path) -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _plan(
                [
                    ("Try an unavailable source", "execute"),
                    ("This step must be dropped", "execute"),
                ]
            ),
            _step("failed", "source unavailable"),
            _plan([("Use a different source", "execute")]),
            _step("success", "answer is 4"),
            _output("4"),
        ],
    )
    task = Task(
        dataset=MemoryDataset([Sample(input="What is 2 + 2?", target="4")]),
        solver=gaia_high_confidence_recursive_agent(tools=[]),
        scorer=gaia_scorer(),
        message_limit=100,
    )

    [log] = eval(task, model=model, log_dir=str(tmp_path), debug_errors=True)

    assert log.status == "success"
    assert log.samples is not None
    messages = "\n".join(message.text for message in log.samples[0].messages)
    assert "Use a different source" in messages
    assert "Current step:\nThis step must be dropped" not in messages








def test_decomposed_step_runs_in_child_frame(tmp_path) -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _plan([("Solve a narrower problem", "decompose")]),
            _plan([("Calculate", "execute")]),
            _step("success", "answer is 4"),
            _output("4"),
        ],
    )
    task = Task(
        dataset=MemoryDataset([Sample(input="What is 2 + 2?", target="4")]),
        solver=gaia_high_confidence_recursive_agent(tools=[]),
        scorer=gaia_scorer(),
        message_limit=100,
    )

    [log] = eval(task, model=model, log_dir=str(tmp_path), debug_errors=True)

    assert log.status == "success"
    assert log.samples is not None
    messages = "\n".join(message.text for message in log.samples[0].messages)
    assert "Current depth: 2" in messages
    assert log.samples[0].output.completion == "4"
