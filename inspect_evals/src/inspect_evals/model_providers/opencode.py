from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from inspect_ai.log._samples import (  # noautolint: private_api_imports
    ActiveSample,
    sample_active,
)
from inspect_ai.model import ChatMessage, GenerateConfig, ModelCall, ModelOutput
from inspect_ai.model._providers.openai_compatible import (  # noautolint: private_api_imports
    OpenAICompatibleAPI,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from typing_extensions import override

OPENCODE_SESSION_HEADER = "x-opencode-session"


def _sample_session_id(
    sample: ActiveSample,
    *,
    model_name: str,
    namespace: str,
) -> str:
    """Return a stable routing session for one sample, epoch, and judge."""
    identity = "\0".join(
        (
            sample.eval_id,
            sample.run_id or "",
            sample.task,
            str(sample.sample.id),
            str(sample.epoch),
            model_name,
            namespace,
        )
    )
    return str(uuid5(NAMESPACE_URL, identity))


def _has_session_header(headers: dict[str, str]) -> bool:
    return any(name.lower() == OPENCODE_SESSION_HEADER for name in headers)


class OpenCodeAPI(OpenAICompatibleAPI):
    """OpenAI-compatible API with per-sample OpenCode session affinity."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        session_id: str | None = None,
        session_namespace: str = "grader",
        **model_args: Any,
    ) -> None:
        if session_id is not None and not session_id.strip():
            raise ValueError("session_id cannot be empty")
        if not session_namespace.strip():
            raise ValueError("session_namespace cannot be empty")

        self._configured_session_id = session_id
        self._session_namespace = session_namespace
        self._fallback_session_id = str(uuid4())
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="opencode",
            **model_args,
        )

    def _session_id(self) -> str:
        if self._configured_session_id is not None:
            return self._configured_session_id

        sample = sample_active()
        if sample is None:
            return self._fallback_session_id

        return _sample_session_id(
            sample,
            model_name=self.model_name,
            namespace=self._session_namespace,
        )

    @override
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        headers = dict(config.extra_headers or {})
        if not _has_session_header(headers):
            headers[OPENCODE_SESSION_HEADER] = self._session_id()

        session_config = config.model_copy(update={"extra_headers": headers})
        return await super().generate(input, tools, tool_choice, session_config)
