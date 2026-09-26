"""Google Gen AI SDK provider for Vertex AI Gemini models."""

from __future__ import annotations

import json
import base64
import random
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from ...models.content import AudioBlock, ImageBlock, TextBlock, ToolUseBlock, VideoBlock
from ...models.message import Message
from ...models.tool import ToolSpec
from ...models.trace import TokenUsage


@dataclass(frozen=True)
class GoogleVertexEndpoint:
    project: str
    location: str


_VERTEX_OPENAPI_RE = re.compile(
    r"^/v1/projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)/endpoints/openapi"
    r"(?:/chat/completions)?/?$"
)


def parse_google_vertex_endpoint(base_url: str) -> GoogleVertexEndpoint:
    """Parse a Vertex AI OpenAI-compatible endpoint URL.

    Accepts both the SDK-style base URL ending at ``/endpoints/openapi`` and
    the raw REST URL ending at ``/chat/completions``.
    """
    parsed = urlparse(base_url)
    if parsed.hostname != "aiplatform.googleapis.com":
        raise ValueError(f"Not a Vertex AI endpoint: {base_url}")
    match = _VERTEX_OPENAPI_RE.match(parsed.path.rstrip("/"))
    if not match:
        raise ValueError(
            "Unsupported Google Vertex AI base_url. Expected "
            "https://aiplatform.googleapis.com/v1/projects/{project}/"
            "locations/{location}/endpoints/openapi"
        )
    return GoogleVertexEndpoint(
        project=match.group("project"),
        location=match.group("location"),
    )


def _schema_type_to_genai(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.lower()
        return {
            "object": "OBJECT",
            "array": "ARRAY",
            "string": "STRING",
            "integer": "INTEGER",
            "number": "NUMBER",
            "boolean": "BOOLEAN",
        }.get(lowered, value.upper())
    return value


def _json_schema_to_genai_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON Schema-ish tool schemas to Gen AI Schema-compatible dicts."""
    converted: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type":
            converted[key] = _schema_type_to_genai(value)
        elif key == "properties" and isinstance(value, dict):
            converted[key] = {
                name: _json_schema_to_genai_schema(prop)
                for name, prop in value.items()
                if isinstance(prop, dict)
            }
        elif key == "items" and isinstance(value, dict):
            converted[key] = _json_schema_to_genai_schema(value)
        elif key in {"required", "description", "enum", "nullable"}:
            converted[key] = value
        elif isinstance(value, dict):
            converted[key] = _json_schema_to_genai_schema(value)
        elif isinstance(value, list):
            converted[key] = [
                _json_schema_to_genai_schema(v) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            converted[key] = value
    return converted


def _tool_spec_to_genai_declaration(spec: ToolSpec, types: Any) -> Any:
    schema = spec.input_schema or {"type": "object", "properties": {}}
    parameters = _json_schema_to_genai_schema(schema)
    return types.FunctionDeclaration(
        name=spec.name,
        description=spec.description,
        parameters=parameters,
    )


def _content_text(msg: Message) -> str:
    return "\n".join(b.text for b in msg.content if b.type == "text")


def _build_tool_use_name_map(messages: list[Message]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for msg in messages:
        for block in msg.content:
            if block.type == "tool_use":
                mapping[block.id] = block.name
    return mapping


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value.items())
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


_GOOGLE_GENAI_THOUGHT_SIGNATURE_KEY = "google_genai_thought_signature"


def _encode_thought_signature(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, str):
        return base64.b64encode(value.encode("utf-8")).decode("ascii")
    return None


def _decode_thought_signature(value: Any) -> bytes | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception:
        return None


class GoogleGenAIProvider:
    """Calls Vertex AI Gemini models via the Google Gen AI SDK."""

    def __init__(
        self,
        model_id: str,
        project: str,
        location: str,
        temperature: float | None = 0.0,
        client: Any | None = None,
        types_module: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.project = project
        self.location = location
        self.temperature = temperature

        if client is None or types_module is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise RuntimeError(
                    "Google Vertex AI endpoint requires the google-genai package. "
                    "Install dependencies with `uv sync` or `pip install google-genai`."
                ) from exc
            self.client = client or genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
            self.types = types_module or types
        else:
            self.client = client
            self.types = types_module

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> tuple[Message, TokenUsage]:
        kwargs = self._build_generate_kwargs(messages, tools)

        max_retries = 20
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(**kwargs)
                return self._parse_response(response)
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                exc_str = str(exc).lower()
                retryable = (
                    status in (429, 500, 502, 503, 529)
                    or "timeout" in exc_str
                    or "timed out" in exc_str
                    or "connection" in exc_str
                    or "server disconnected" in exc_str
                    or "unavailable" in exc_str
                    or "deadline" in exc_str
                    or "rate limit" in exc_str
                )
                if not retryable or attempt == max_retries:
                    raise
                delay = random.uniform(2, 4)
                print(
                    f"[retry] Google Gen AI error ({status or type(exc).__name__}), "
                    f"attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s ..."
                )
                time.sleep(delay)

        raise last_exc or RuntimeError("All Google Gen AI retries exhausted")

    def _build_generate_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
    ) -> dict[str, Any]:
        system_instruction = "\n\n".join(_content_text(m) for m in messages if m.role == "system")
        tool_use_names = _build_tool_use_name_map(messages)
        contents = [
            self._message_to_content(m, tool_use_names)
            for m in messages
            if m.role != "system"
        ]

        config_kwargs: dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature
        if tools:
            declarations = [
                _tool_spec_to_genai_declaration(t, self.types)
                for t in tools
            ]
            config_kwargs["tools"] = [self.types.Tool(function_declarations=declarations)]

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "contents": contents,
        }
        if config_kwargs:
            kwargs["config"] = self.types.GenerateContentConfig(**config_kwargs)
        return kwargs

    def _message_to_content(self, msg: Message, tool_use_names: dict[str, str]) -> Any:
        role = "model" if msg.role == "assistant" else "user"
        parts = []
        for block in msg.content:
            if block.type == "text":
                parts.append(self.types.Part.from_text(text=block.text))
            elif block.type == "image":
                image = block if isinstance(block, ImageBlock) else ImageBlock.model_validate(block)
                parts.append(self.types.Part.from_bytes(
                    data=base64.b64decode(image.data),
                    mime_type=image.mime_type,
                ))
            elif block.type == "audio":
                audio = block if isinstance(block, AudioBlock) else AudioBlock.model_validate(block)
                parts.append(self.types.Part.from_bytes(
                    data=base64.b64decode(audio.data),
                    mime_type=audio.mime_type,
                ))
            elif block.type == "video":
                video = block if isinstance(block, VideoBlock) else VideoBlock.model_validate(block)
                parts.append(self.types.Part.from_text(
                    text=(
                        f"[video attached: {video.source_path or 'inline'} "
                        f"({video.mime_type}, base64_bytes={len(video.data) * 3 // 4})]"
                    )
                ))
            elif block.type == "tool_use":
                args = _coerce_mapping(block.input)
                metadata = getattr(block, "provider_metadata", {}) or {}
                thought_signature = _decode_thought_signature(
                    metadata.get(_GOOGLE_GENAI_THOUGHT_SIGNATURE_KEY)
                )
                if thought_signature is None:
                    parts.append(self.types.Part.from_function_call(name=block.name, args=args))
                else:
                    try:
                        function_call = self.types.FunctionCall(name=block.name, args=args)
                        parts.append(self.types.Part(
                            function_call=function_call,
                            thought_signature=thought_signature,
                        ))
                    except Exception:
                        part = self.types.Part.from_function_call(name=block.name, args=args)
                        setattr(part, "thought_signature", thought_signature)
                        parts.append(part)
            elif block.type == "tool_result":
                content_text = "\n".join(t.text for t in block.content) if block.content else ""
                response = {
                    "content": content_text,
                    "is_error": block.is_error,
                }
                parts.append(self.types.Part.from_function_response(
                    name=tool_use_names.get(block.tool_use_id, block.tool_use_id),
                    response=response,
                ))

        if not parts:
            parts.append(self.types.Part.from_text(text=""))
        return self.types.Content(role=role, parts=parts)

    def _parse_response(self, response: Any) -> tuple[Message, TokenUsage]:
        content_blocks = []
        reasoning = None

        candidates = getattr(response, "candidates", None) or []
        parts = []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
        elif getattr(response, "text", None):
            parts = [self.types.Part.from_text(text=response.text)]

        for part in parts:
            text = getattr(part, "text", None)
            if text:
                content_blocks.append(TextBlock(text=text))
                continue

            function_call = getattr(part, "function_call", None)
            if function_call:
                name = getattr(function_call, "name", None)
                if not name:
                    continue
                args = _coerce_mapping(getattr(function_call, "args", None))
                provider_metadata = {}
                thought_signature = _encode_thought_signature(
                    getattr(part, "thought_signature", None)
                )
                if thought_signature is not None:
                    provider_metadata[_GOOGLE_GENAI_THOUGHT_SIGNATURE_KEY] = thought_signature
                content_blocks.append(ToolUseBlock(
                    id=f"google_genai_{uuid4().hex[:12]}",
                    name=name,
                    input=args,
                    provider_metadata=provider_metadata,
                ))

        usage = TokenUsage()
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None:
            usage = TokenUsage(
                input_tokens=(
                    getattr(usage_meta, "prompt_token_count", None)
                    or getattr(usage_meta, "input_token_count", None)
                    or 0
                ),
                output_tokens=(
                    getattr(usage_meta, "candidates_token_count", None)
                    or getattr(usage_meta, "output_token_count", None)
                    or 0
                ),
            )

        return Message(role="assistant", content=content_blocks, reasoning_content=reasoning), usage
