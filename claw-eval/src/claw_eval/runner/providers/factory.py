"""Provider selection helpers."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from .google_genai import GoogleVertexEndpoint, GoogleGenAIProvider, parse_google_vertex_endpoint
from .openai_compat import OpenAICompatProvider


def is_google_vertex_base_url(base_url: str | None) -> bool:
    """Return True when *base_url* points at Vertex AI's OpenAI endpoint."""
    if not base_url:
        return False
    parsed = urlparse(base_url)
    return parsed.hostname == "aiplatform.googleapis.com"


def make_model_provider(
    *,
    model_id: str,
    api_key: str | None = None,
    base_url: str | None = None,
    extra_body: dict | None = None,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = None,
) -> Any:
    """Create the model provider for the configured endpoint.

    Google Vertex AI's official OpenAI-compatible endpoint is intentionally
    routed through the native Google Gen AI SDK so ADC can refresh credentials.
    All other endpoints keep the existing OpenAI-compatible path.
    """
    if is_google_vertex_base_url(base_url):
        endpoint: GoogleVertexEndpoint = parse_google_vertex_endpoint(base_url or "")
        return GoogleGenAIProvider(
            model_id=model_id,
            project=endpoint.project,
            location=endpoint.location,
            temperature=temperature,
        )

    return OpenAICompatProvider(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        extra_body=extra_body,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
