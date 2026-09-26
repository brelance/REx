import json

import pytest
from inspect_ai import Task, eval
from inspect_ai._util.registry import registry_create
from inspect_ai.agent import Agent, AgentState
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import Tool, ToolCall, ToolError, tool

from inspect_evals.gaia import gaia_plan_execute_agent, gaia_reflection_agent
from inspect_evals.gaia.baseline_agents import (
    REFLECTION_PROMPT,
    PlanExecuteProtocolError,
    _reflection_continue,
    parse_plan_execute_plan,
    parse_plan_execute_step_result,
)
from inspect_evals.gaia.scorer import gaia_scorer


def _output(content: str) -> ModelOutput:
    return ModelOutput.from_content("mockllm/model", content)


def _plan(*goals: str) -> ModelOutput:
    return _output(json.dumps({"steps": [{"goal": goal} for goal in goals]}))


def _step(complete: str, summary: str) -> ModelOutput:
    return _output(json.dumps({"complete": complete, "summary": summary}))


def _submit(answer: str) -> ModelOutput:
    return ModelOutput.for_tool_call(
        model="mockllm/model",
        tool_name="submit",
        tool_arguments={"answer": answer},
        tool_call_id="submit-baseline-call",
    )


def _parallel_tool_calls() -> ModelOutput:
    return ModelOutput(
        model="mockllm/model",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content="Look up both values.",
                    tool_calls=[
                        ToolCall(
                            id="first-baseline-call",
                            function="lookup_first_baseline",
                            arguments={},
                        ),
                        ToolCall(
                            id="second-baseline-call",
                            function="lookup_second_baseline",
                            arguments={},
                        ),
                    ],
                ),
                stop_reason="tool_calls",
            )
        ],
    )


@tool
def lookup_answer_baseline() -> Tool:
    async def execute() -> str:
        """Return the answer used in baseline tests."""
        return "4"

    return execute


@tool
def lookup_first_baseline() -> Tool:
    async def execute() -> str:
        """Return the first parallel observation."""
        return "first"

    return execute


@tool
def lookup_second_baseline() -> Tool:
    async def execute() -> str:
        """Return the second parallel observation."""
        return "second"

    return execute


@tool
def failing_lookup_baseline() -> Tool:
    async def execute() -> str:
        """Raise an error for reflection testing."""
        raise ToolError("lookup failed")

    return execute


def _task(solver: Agent) -> Task:
    return Task(
        dataset=MemoryDataset([Sample(input="What is 2 + 2?", target="4")]),
        solver=solver,
        scorer=gaia_scorer(),
        message_limit=100,
    )


def test_baseline_agent_factories_and_cli_registration() -> None:
    reflection = registry_create(
        "agent", "inspect_evals/gaia_reflection_agent", tools=[]
    )
    plan_execute = registry_create(
        "agent", "inspect_evals/gaia_plan_execute_agent", tools=[]
    )

    assert isinstance(gaia_reflection_agent(tools=[]), Agent)
    assert isinstance(gaia_plan_execute_agent(tools=[]), Agent)
    assert isinstance(reflection, Agent)
    assert isinstance(plan_execute, Agent)


@pytest.mark.asyncio
async def test_reflection_continue_only_after_tool_calls() -> None:
    state = AgentState(messages=[])
    state.output = ModelOutput.for_tool_call(
        model="mockllm/model",
        tool_name="lookup_answer_baseline",
        tool_arguments={},
    )

    assert await _reflection_continue(state) == REFLECTION_PROMPT

    state.output = _output("No tool call")
    assert await _reflection_continue(state) is True


def test_reflection_agent_adds_one_prompt_after_parallel_batch(tmp_path) -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[_parallel_tool_calls(), _submit("4")],
    )

    [log] = eval(
        _task(
            gaia_reflection_agent(
                tools=[lookup_first_baseline(), lookup_second_baseline()]
            )
        ),
        model=model,
        log_dir=str(tmp_path),
        debug_errors=True,
    )

    assert log.status == "success"
    assert log.samples is not None
    sample = log.samples[0]
    assert sample.output.completion == "4"
    assert sum(message.text == REFLECTION_PROMPT for message in sample.messages) == 1
    assert sum(message.role == "tool" for message in sample.messages) == 3


def test_reflection_agent_reflects_after_tool_error(tmp_path) -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="failing_lookup_baseline",
                tool_arguments={},
            ),
            _submit("4"),
        ],
    )

    [log] = eval(
        _task(gaia_reflection_agent(tools=[failing_lookup_baseline()])),
        model=model,
        log_dir=str(tmp_path),
        debug_errors=True,
    )

    assert log.status == "success"
    assert log.samples is not None
    assert (
        sum(message.text == REFLECTION_PROMPT for message in log.samples[0].messages)
        == 1
    )
    assert any(
        message.role == "tool" and message.error is not None
        for message in log.samples[0].messages
    )


@pytest.mark.parametrize(
    "wrapper",
    [
        "{}",
        "```json\n{}\n```",
        "```\n{}\n```",
        "Plan follows:\n```JSON\n{}\n```\nEnd of plan.",
        "```json\nnot json\n```\n```json\n{}\n```",
    ],
)
def test_parse_plan_execute_plan_accepts_json_and_fences(wrapper: str) -> None:
    response = wrapper.format(
        '{"steps":[{"goal":" research "},{"goal":"verify"}]}'
    )
    assert parse_plan_execute_plan(response, minimum_steps=2) == ["research", "verify"]


@pytest.mark.parametrize("wrapper", ["{}", "Plan:\n```json\n{}\n```\nDone."])
def test_parse_plan_execute_plan_validates_fields_and_minimum_steps(
    wrapper: str,
) -> None:
    invalid = [
        "[]",
        "not json",
        'prefix {"steps":[{"goal":"a"},{"goal":"b"}]}',
        '{"steps":[{"goal":"only one"}]}',
        '{"steps":[{"goal":"a"},{"goal":"b"}],"extra":true}',
        '{"steps":[{"goal":"a"},{"goal":""}]}',
    ]
    for response in invalid:
        with pytest.raises(PlanExecuteProtocolError):
            parse_plan_execute_plan(wrapper.format(response), minimum_steps=2)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            '{"complete":"success","summary":" verified "}',
            ("success", "verified"),
        ),
        (
            '{"complete":"false","summary":" blocked "}',
            ("false", "blocked"),
        ),
    ],
)
@pytest.mark.parametrize(
    "wrapper", ["{}", "```\n{}\n```", "Result:\n```JSON\n{}\n```\nDone."]
)
def test_parse_plan_execute_step_result(
    response: str, expected: tuple[str, str], wrapper: str
) -> None:
    result = parse_plan_execute_step_result(wrapper.format(response))

    assert (result.complete, result.summary) == expected


@pytest.mark.parametrize(
    "response",
    [
        '{"complete":true,"summary":"verified"}',
        '{"complete":"failed","summary":"blocked"}',
        '{"complete":"success","summary":""}',
        '{"complete":"success","summary":"verified","extra":1}',
        'Result: {"complete":"success","summary":"verified"}',
    ],
)
@pytest.mark.parametrize("wrapper", ["{}", "Result:\n```json\n{}\n```\nDone."])
def test_parse_plan_execute_step_result_rejects_invalid_protocol(
    response: str, wrapper: str
) -> None:
    with pytest.raises(PlanExecuteProtocolError):
        parse_plan_execute_step_result(wrapper.format(response))


def test_plan_execute_agent_runs_steps_and_synthesizes_answer(tmp_path) -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _plan("Look up the answer", "Verify the answer"),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="lookup_answer_baseline",
                tool_arguments={},
            ),
            _step("success", "The lookup returned 4."),
            _step("success", "Independent calculation confirms 4."),
            _output("4"),
        ],
    )

    [log] = eval(
        _task(gaia_plan_execute_agent(tools=[lookup_answer_baseline()])),
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
    assert (
        sum("Use more tools if needed" in message.text for message in sample.messages)
        == 1
    )


def test_plan_execute_failure_discards_tail_and_replans(tmp_path) -> None:
    dropped_goal = "THIS DROPPED STEP MUST NOT RUN"
    replacement_goal = "Use a replacement method"
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _plan("Establish a known fact", "Try a blocked source", dropped_goal),
            _step("success", "2 + 2 is the relevant expression."),
            _step("false", "The source is unavailable."),
            _plan(replacement_goal),
            _step("success", "A different method gives 4."),
            _output("4"),
        ],
    )

    [log] = eval(
        _task(gaia_plan_execute_agent(tools=[])),
        model=model,
        log_dir=str(tmp_path),
        debug_errors=True,
    )

    assert log.status == "success"
    assert log.samples is not None
    sample = log.samples[0]
    execution_prompts = [
        message.text
        for message in sample.messages
        if message.role == "user"
        and message.text.startswith("Execute only the current plan step.")
    ]
    assert len(execution_prompts) == 3
    assert all(
        f'"goal": "{dropped_goal}"' not in prompt.split("Current step:", 1)[-1]
        for prompt in execution_prompts
    )
    replan_prompt = next(
        message.text
        for message in sample.messages
        if message.role == "user"
        and message.text.startswith("The current step failed.")
    )
    assert "2 + 2 is the relevant expression." in replan_prompt
    assert "The source is unavailable." in replan_prompt
    assert dropped_goal in replan_prompt
    assert replacement_goal in "\n".join(execution_prompts)
    assert sample.output.completion == "4"


def test_plan_execute_repairs_invalid_control_responses(tmp_path) -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _output("not a plan"),
            _plan("Calculate", "Verify"),
            _output("step complete"),
            _step("success", "Calculated 4."),
            _step("success", "Verified 4."),
            _output("4"),
        ],
    )

    [log] = eval(
        _task(gaia_plan_execute_agent(tools=[])),
        model=model,
        log_dir=str(tmp_path),
        debug_errors=True,
    )

    assert log.status == "success"
    assert log.samples is not None
    messages = "\n".join(message.text for message in log.samples[0].messages)
    assert "Invalid plan response" in messages
    assert "Invalid completion response for step_1" in messages
    assert log.samples[0].output.completion == "4"
