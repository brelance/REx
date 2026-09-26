from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast

import pytest
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.model._providers.openai_compatible import (  # noautolint: private_api_imports
    OpenAICompatibleAPI,
)

from inspect_evals.model_providers.opencode import (
    OPENCODE_SESSION_HEADER,
    OpenCodeAPI,
    _sample_session_id,
)

opencode_module = import_module("inspect_evals.model_providers.opencode")


def _active_sample(
    *,
    sample_id: str = "sample-1",
    epoch: int = 1,
) -> Any:
    return SimpleNamespace(
        eval_id="eval-1",
        run_id="run-1",
        task="task-1",
        sample=SimpleNamespace(id=sample_id),
        epoch=epoch,
    )


def test_sample_session_is_stable_and_isolated() -> None:
    sample = _active_sample()

    session = _sample_session_id(sample, model_name="judge", namespace="grader")

    assert session == _sample_session_id(sample, model_name="judge", namespace="grader")
    assert session != _sample_session_id(
        _active_sample(sample_id="sample-2"),
        model_name="judge",
        namespace="grader",
    )
    assert session != _sample_session_id(
        _active_sample(epoch=2), model_name="judge", namespace="grader"
    )
    assert session != _sample_session_id(sample, model_name="judge", namespace="critic")


@pytest.mark.asyncio
async def test_generate_adds_per_sample_session_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = ModelOutput.from_content(model="opencode/judge", content="graded")
    captured_configs: list[GenerateConfig] = []

    async def generate(
        self: OpenAICompatibleAPI,
        input: Any,
        tools: Any,
        tool_choice: Any,
        config: GenerateConfig,
    ) -> ModelOutput:
        captured_configs.append(config)
        return generated

    monkeypatch.setattr(opencode_module, "sample_active", _active_sample)
    monkeypatch.setattr(OpenAICompatibleAPI, "generate", generate)

    api = OpenCodeAPI(
        "judge",
        base_url="http://localhost:3000/v1",
        api_key="test-key",
    )
    try:
        config = GenerateConfig(extra_headers={"x-existing": "value"})
        result = await api.generate([], [], "auto", config)
        await api.generate([], [], "auto", config)
    finally:
        await api.aclose()

    assert result is generated
    assert config.extra_headers == {"x-existing": "value"}
    assert len(captured_configs) == 2
    assert captured_configs[0].extra_headers is not None
    assert captured_configs[1].extra_headers is not None
    assert captured_configs[0].extra_headers["x-existing"] == "value"
    session = captured_configs[0].extra_headers[OPENCODE_SESSION_HEADER]
    assert session == captured_configs[1].extra_headers[OPENCODE_SESSION_HEADER]


@pytest.mark.asyncio
async def test_generate_preserves_explicit_session_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_config: GenerateConfig | None = None

    async def generate(
        self: OpenAICompatibleAPI,
        input: Any,
        tools: Any,
        tool_choice: Any,
        config: GenerateConfig,
    ) -> ModelOutput:
        nonlocal captured_config
        captured_config = config
        return ModelOutput.from_content(model="opencode/judge", content="graded")

    monkeypatch.setattr(OpenAICompatibleAPI, "generate", generate)
    api = OpenCodeAPI(
        "judge",
        base_url="http://localhost:3000/v1",
        api_key="test-key",
    )
    try:
        await api.generate(
            [],
            [],
            "auto",
            GenerateConfig(extra_headers={"X-OpenCode-Session": "explicit"}),
        )
    finally:
        await api.aclose()

    assert captured_config is not None
    headers = cast(dict[str, str], captured_config.extra_headers)
    assert headers == {"X-OpenCode-Session": "explicit"}
