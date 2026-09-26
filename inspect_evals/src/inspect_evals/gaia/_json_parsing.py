"""JSON response parsing for GAIA baseline agents."""

import json
import re
from typing import Any

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def json_with_fence_fallback(text: str) -> Any:
    """Parse JSON, falling back to complete Markdown code fences."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        for match in _JSON_FENCE_RE.finditer(text):
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
        raise
