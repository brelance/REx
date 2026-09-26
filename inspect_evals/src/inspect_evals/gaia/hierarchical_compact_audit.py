"""Human-readable audit logs for GAIA tree compression."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from inspect_ai.model import ChatMessage, ModelOutput

COMPRESSION_TEXT_LOG_SCHEMA_VERSION = 1
_SAFE_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_PATH_COMPONENT_CHARS = 96


@dataclass(frozen=True)
class CompressionLogContext:
    """Filesystem and sample identity for one evaluation sample."""

    root: Path
    eval_log_parent: Path
    task_id: str
    epoch: int
    sample_uuid: str
    runtime_sample_id: str


def resolve_compression_log_context(
    root_override: str | None = None,
) -> tuple[CompressionLogContext | None, str | None]:
    """Resolve Inspect's current sample context without exposing private APIs."""
    try:
        # Inspect does not currently expose sample identity and log location through
        # AgentState, so keep the compatibility dependency isolated here.
        from inspect_ai.log._samples import (  # noautolint: private_api_imports
            _sample_active,
        )
        from inspect_ai.solver._task_state import (  # noautolint: private_api_imports
            sample_state,
        )

        active = _sample_active.get()
        state = sample_state()
        if active is None or state is None:
            raise RuntimeError("Inspect sample context is unavailable")

        log_location = str(active.log_location)
        parsed = urlparse(log_location)
        if parsed.scheme and parsed.scheme != "file":
            if root_override is None:
                raise ValueError(
                    "compression text logs require a local eval log or "
                    "compression_log_root"
                )
            eval_log_path = Path.cwd() / "remote.eval"
        else:
            eval_log_path = Path(
                unquote(parsed.path) if parsed.scheme == "file" else log_location
            )

        root = (
            Path(root_override).expanduser()
            if root_override is not None
            else eval_log_path.with_suffix("")
        )
        return (
            CompressionLogContext(
                root=root,
                eval_log_parent=eval_log_path.parent,
                task_id=str(state.sample_id),
                epoch=state.epoch,
                sample_uuid=state.uuid,
                runtime_sample_id=str(active.id),
            ),
            None,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def safe_path_component(value: str) -> str:
    """Return a readable, collision-resistant filesystem component."""
    cleaned = _SAFE_PATH_COMPONENT_RE.sub("_", value).strip("._-")
    changed = cleaned != value or not cleaned
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    if not cleaned:
        cleaned = "sample"
    if len(cleaned) > _MAX_PATH_COMPONENT_CHARS:
        cleaned = cleaned[: _MAX_PATH_COMPONENT_CHARS - 13].rstrip("._-")
        changed = True
    return f"{cleaned}-{digest}" if changed else cleaned


def make_compression_id(context: CompressionLogContext, sequence: int) -> str:
    """Create a unique identifier for a triggered compression attempt."""
    seed = (f"{context.runtime_sample_id}\0{context.sample_uuid}\0{sequence}").encode(
        "utf-8"
    )
    return f"cmp_{hashlib.sha256(seed).hexdigest()[:20]}"


def compression_text_log_path(
    context: CompressionLogContext,
    *,
    sequence: int,
    scope_kind: str,
    compression_id: str,
) -> Path:
    """Return the text log path for one triggered compression attempt."""
    directory = (
        context.root
        / "samples"
        / "compression_logs"
        / safe_path_component(context.task_id)
        / f"epoch-{context.epoch}"
    )
    return directory / f"c{sequence:06d}_{scope_kind}_{compression_id}.txt"


def dump_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Serialize Inspect messages without dropping optional fields."""
    return [message.model_dump(mode="json") for message in messages]


def dump_model_output(output: ModelOutput | None) -> dict[str, Any] | None:
    """Serialize a complete model output when a compressor call completed."""
    return output.model_dump(mode="json") if output is not None else None


def _render_readable_message(message: dict[str, Any], *, index: int) -> str:
    role = str(message.get("role", "unknown")).upper()
    sections = [f"--- MESSAGE {index} ROLE={role} ---"]
    content = message.get("content")
    if isinstance(content, str):
        sections.extend(["[TEXT]", content or "(empty)"])
    elif isinstance(content, list) and content:
        for block_index, block in enumerate(content, start=1):
            if isinstance(block, dict) and block.get("type") == "text":
                sections.extend(
                    [f"[TEXT BLOCK {block_index}]", str(block.get("text", ""))]
                )
            else:
                block_type = (
                    block.get("type", "unknown")
                    if isinstance(block, dict)
                    else "unknown"
                )
                sections.extend(
                    [
                        f"[CONTENT BLOCK {block_index} TYPE={block_type}]",
                        json.dumps(block, ensure_ascii=False, sort_keys=True, indent=2),
                    ]
                )
    else:
        sections.extend(["[CONTENT]", "(empty)"])
    return "\n".join(sections)


def render_compressor_text_log(payload: dict[str, Any]) -> str:
    """Render a readable compressor call followed by its lossless JSON payload."""
    request = payload.get("request")
    request_messages = request.get("messages", []) if isinstance(request, dict) else []
    response = payload.get("response")
    response_message = response.get("message") if isinstance(response, dict) else None
    error_type = response.get("error_type") if isinstance(response, dict) else None
    error_message = (
        response.get("error_message") if isinstance(response, dict) else None
    )
    decision = payload.get("decision")

    header_keys = (
        "task_id",
        "epoch",
        "sample_uuid",
        "compression_id",
        "compression_sequence",
        "event",
        "backend",
        "model",
    )
    parts = ["COMPRESSOR CALL"]
    parts.extend(f"{key}: {payload.get(key)}" for key in header_keys)
    if isinstance(decision, dict):
        parts.extend(
            [
                f"triggered: {decision.get('triggered')}",
                f"applied: {decision.get('applied')}",
                f"reason: {decision.get('reason')}",
            ]
        )

    parts.append("\n=== COMPRESSOR INPUT (READABLE) ===")
    if request_messages:
        parts.extend(
            _render_readable_message(message, index=index)
            for index, message in enumerate(request_messages, start=1)
        )
    else:
        parts.append("(none)")

    parts.append("\n=== COMPRESSOR OUTPUT (READABLE) ===")
    if isinstance(response_message, dict):
        parts.append(_render_readable_message(response_message, index=1))
    else:
        raw_text = response.get("raw_text") if isinstance(response, dict) else None
        parts.append(str(raw_text) if raw_text else "(none)")

    parts.extend(
        [
            "\n=== COMPRESSOR ERROR ===",
            f"error_type: {error_type}",
            f"error_message: {error_message}",
            "\n=== COMPLETE CALL JSON ===",
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        ]
    )
    return "\n".join(parts) + "\n"


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def write_compressor_text_log(
    context: CompressionLogContext,
    *,
    sequence: int,
    scope_kind: str,
    compression_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Atomically write one compressor log and return transcript metadata."""
    path = compression_text_log_path(
        context,
        sequence=sequence,
        scope_kind=scope_kind,
        compression_id=compression_id,
    )
    encoded = render_compressor_text_log(payload).encode("utf-8")
    _atomic_write_bytes(path, encoded)
    try:
        display_path = path.relative_to(context.eval_log_parent)
    except ValueError:
        display_path = path
    return {
        "status": "written",
        "path": str(display_path),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


__all__ = [
    "COMPRESSION_TEXT_LOG_SCHEMA_VERSION",
    "CompressionLogContext",
    "compression_text_log_path",
    "dump_messages",
    "dump_model_output",
    "make_compression_id",
    "render_compressor_text_log",
    "resolve_compression_log_context",
    "safe_path_component",
    "write_compressor_text_log",
]
