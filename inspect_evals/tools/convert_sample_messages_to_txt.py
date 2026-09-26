#!/usr/bin/env python3
"""Convert message histories in Inspect sample JSON files to readable text."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MESSAGE_SEPARATOR = "=" * 80
SECTION_SEPARATOR = "-" * 80


def _format_json(value: Any) -> str:
    """Pretty-print structured data, including JSON encoded as a string."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _format_image(item: Mapping[str, Any]) -> str:
    image = item.get("image") or item.get("url") or item.get("source")
    if isinstance(image, str):
        if image.startswith("data:"):
            media_type = image.partition(";")[0].removeprefix("data:")
            return f"[embedded image: {media_type or 'unknown type'}]"
        return f"[image: {image}]"
    return f"[image content: {_format_json(image)}]"


def _format_content_item(item: Any) -> str:
    if not isinstance(item, Mapping):
        return str(item)

    item_type = item.get("type")
    if item_type == "text":
        return str(item.get("text", ""))
    if item_type == "reasoning":
        if item.get("redacted"):
            return "[reasoning redacted]"
        reasoning = item.get("reasoning") or item.get("summary") or ""
        return f"[reasoning]\n{reasoning}".rstrip()
    if item_type == "image":
        return _format_image(item)
    return f"[{item_type or 'content'}]\n{_format_json(item)}"


def format_content(content: Any) -> str:
    """Return human-readable text for a message content value."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        return "\n\n".join(_format_content_item(item) for item in content).strip()
    if isinstance(content, Mapping):
        return _format_content_item(content)
    return str(content)


def _tool_call_fields(call: Mapping[str, Any]) -> tuple[str, Any]:
    function = call.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name", "unknown")), function.get("arguments", {})
    return str(function or call.get("name") or "unknown"), call.get("arguments", {})


def _format_tool_calls(tool_calls: Any) -> list[str]:
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
        return ["[invalid tool_calls value]", _format_json(tool_calls)]

    lines: list[str] = []
    for index, raw_call in enumerate(tool_calls, start=1):
        if not isinstance(raw_call, Mapping):
            lines.extend((f"[TOOL CALL {index}]", _format_json(raw_call)))
            continue

        function, arguments = _tool_call_fields(raw_call)
        lines.extend((f"[TOOL CALL {index}]", f"Function: {function}"))
        if call_id := raw_call.get("id"):
            lines.append(f"Call ID: {call_id}")
        lines.extend(("Input:", _format_json(arguments)))
    return lines


def _format_tool_result(message: Mapping[str, Any]) -> list[str]:
    function = message.get("function") or message.get("name") or "unknown"
    lines = [f"Function: {function}"]
    if call_id := message.get("tool_call_id"):
        lines.append(f"Call ID: {call_id}")
    if error := message.get("error"):
        lines.extend(("Error:", _format_json(error)))
    lines.extend(("Output:", format_content(message.get("content"))))
    return lines


def render_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render a sequence of Inspect messages without sample metadata."""
    sections: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role", "unknown"))
        heading = f"[{index:03d}] {role.upper()}"
        if role == "tool":
            heading += " RESULT"
            body = _format_tool_result(message)
        else:
            body = []
            content = format_content(message.get("content"))
            if content:
                body.append(content)
            if "tool_calls" in message:
                if body:
                    body.append("")
                body.extend(_format_tool_calls(message["tool_calls"]))
            if not body:
                body.append("[empty message]")

        sections.append(
            "\n".join((MESSAGE_SEPARATOR, heading, SECTION_SEPARATOR, *body))
        )
    return "\n\n".join(sections) + "\n"


def read_messages(path: Path) -> list[Mapping[str, Any]]:
    """Read and validate the messages array from one sample JSON file."""
    try:
        sample = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(sample, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{path}: expected 'messages' to be an array")
    if not all(isinstance(message, Mapping) for message in messages):
        raise ValueError(f"{path}: every message must be a JSON object")
    return messages


def convert_file(input_path: Path, output_path: Path) -> None:
    """Convert one sample JSON file to one text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_messages(read_messages(input_path)), encoding="utf-8")


def _input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.json"))
        if not files:
            raise ValueError(f"{input_path}: no JSON files found")
        return files
    raise ValueError(f"{input_path}: input path does not exist")


def _default_output_dir(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path.with_name(f"{input_path.name}_txt")
    return input_path.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="sample JSON file or samples directory"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="output directory (default: samples_txt beside an input directory)",
    )
    args = parser.parse_args()

    try:
        input_files = _input_files(args.input)
        output_dir = args.output_dir or _default_output_dir(args.input)
        for input_file in input_files:
            convert_file(input_file, output_dir / f"{input_file.stem}.txt")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"Converted {len(input_files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()
