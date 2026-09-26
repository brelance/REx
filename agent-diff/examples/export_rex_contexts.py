"""Export REx model contexts and tool I/O as readable text files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "examples/evaluation_outputs/rex_runner/gemma-4-E4B-it_run_001"
)


def write_value(handle: TextIO, value: Any) -> None:
    if isinstance(value, str):
        handle.write(value)
        if not value.endswith("\n"):
            handle.write("\n")
        return
    handle.write(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    handle.write("\n")


def write_messages(handle: TextIO, messages: list[dict[str, Any]]) -> None:
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role", "unknown")).upper()
        handle.write(f"\n[MESSAGE {index:03d}] {role}\n")
        handle.write("-" * 80 + "\n")
        write_value(handle, message.get("content", ""))

        extra = {
            key: value
            for key, value in message.items()
            if key not in {"role", "content"}
        }
        if extra:
            handle.write("Additional message fields:\n")
            write_value(handle, extra)


def observation_content(message: dict[str, Any]) -> str | None:
    """Return the payload of an observation message, if this is one."""
    if message.get("role") != "user":
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    opening = "<observation>"
    closing = "</observation>"
    if not stripped.startswith(opening) or not stripped.endswith(closing):
        return None
    return stripped[len(opening) : -len(closing)].strip()


def extract_trace(checkpoint: dict[str, Any], source: Path) -> dict[str, Any]:
    record = checkpoint.get("record")
    if not isinstance(record, dict):
        raise TypeError(f"{source}: record must be an object")
    trace = record.get("trace")
    if not isinstance(trace, dict):
        raise TypeError(f"{source}: record.trace must be an object")
    return trace


def export_checkpoint(source: Path, destination: Path) -> tuple[int, int]:
    with source.open(encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{source}: checkpoint must be an object")

    trace = extract_trace(checkpoint, source)
    messages = trace.get("messages")
    events = trace.get("events")
    if not isinstance(messages, list) or not all(
        isinstance(message, dict) for message in messages
    ):
        raise ValueError(f"{source}: trace.messages must be a list of objects")
    if not isinstance(events, list) or not all(
        isinstance(event, dict) for event in events
    ):
        raise ValueError(f"{source}: trace.events must be a list of objects")

    assistant_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    model_events = [event for event in events if event.get("event") == "model_response"]
    tool_events = [event for event in events if event.get("event") == "tool_result"]
    if len(assistant_positions) != len(model_events):
        raise ValueError(
            f"{source}: found {len(assistant_positions)} assistant messages but "
            f"{len(model_events)} model_response events"
        )

    observation_positions = [
        index
        for index, message in enumerate(messages)
        if observation_content(message) is not None
    ]
    if len(observation_positions) != len(tool_events):
        raise ValueError(
            f"{source}: found {len(observation_positions)} observation messages but "
            f"{len(tool_events)} tool_result events"
        )
    for observation_position, event in zip(
        observation_positions, tool_events, strict=True
    ):
        artifact = event.get("artifact")
        artifact = artifact if isinstance(artifact, dict) else {}
        if observation_content(messages[observation_position]) != str(
            artifact.get("observation", "")
        ).strip():
            raise ValueError(
                f"{source}: tool observation does not match its user message"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    model_index = 0
    tool_index = 0
    message_cursor = 0
    observation_position_set = set(observation_positions)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(f"REX MODEL CONTEXT AND TOOL I/O: {source.stem}\n")
            handle.write("=" * 80 + "\n")

            for event in events:
                event_type = event.get("event")
                artifact = event.get("artifact")
                artifact = artifact if isinstance(artifact, dict) else {}

                if event_type == "model_response":
                    assistant_position = assistant_positions[model_index]
                    response = messages[assistant_position]
                    event_content = artifact.get("content")
                    if event_content != response.get("content"):
                        raise ValueError(
                            f"{source}: model response {model_index + 1} does not "
                            "match its assistant message"
                        )

                    model_index += 1
                    new_messages = [
                        messages[index]
                        for index in range(message_cursor, assistant_position)
                        if index not in observation_position_set
                    ]
                    message_cursor = assistant_position + 1
                    handle.write("\n\n" + "#" * 80 + "\n")
                    handle.write(f"MODEL CALL {model_index:03d}\n")
                    handle.write("#" * 80 + "\n")
                    handle.write(f"Phase: {artifact.get('phase', 'unknown')}\n")
                    handle.write(f"Frame: {event.get('frame_id', 'unknown')}\n")
                    if event.get("step_id") is not None:
                        handle.write(f"Step: {event['step_id']}\n")
                    handle.write(
                        "\nNEW NON-TOOL MESSAGES SINCE PREVIOUS MODEL CALL "
                        f"({len(new_messages)})\n"
                    )
                    handle.write("=" * 80 + "\n")
                    write_messages(handle, new_messages)
                    handle.write("\nMODEL RESPONSE\n")
                    handle.write("=" * 80 + "\n")
                    write_value(handle, response.get("content", ""))

                elif event_type == "tool_result":
                    tool_index += 1
                    handle.write("\n\n" + "#" * 80 + "\n")
                    handle.write(f"TOOL CALL {tool_index:03d}\n")
                    handle.write("#" * 80 + "\n")
                    handle.write(f"Frame: {event.get('frame_id', 'unknown')}\n")
                    if event.get("step_id") is not None:
                        handle.write(f"Step: {event['step_id']}\n")
                    handle.write("\nTOOL INPUT (BASH)\n")
                    handle.write("=" * 80 + "\n")
                    write_value(handle, artifact.get("action", ""))
                    handle.write("\nTOOL OUTPUT (RAW EXECUTOR RESULT)\n")
                    handle.write("=" * 80 + "\n")
                    write_value(handle, artifact.get("result"))
                    handle.write("\nOBSERVATION ADDED TO MODEL CONTEXT\n")
                    handle.write("=" * 80 + "\n")
                    write_value(handle, artifact.get("observation", ""))

        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return len(model_events), len(tool_events)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Checkpoint directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <input_dir>_txt)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_dir.with_name(f"{input_dir.name}_txt")
    )

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if output_dir == input_dir:
        print("Output directory must differ from input directory", file=sys.stderr)
        return 2

    sources = sorted(input_dir.glob("*.json"))
    if not sources:
        print(f"No JSON checkpoint files found in: {input_dir}", file=sys.stderr)
        return 2

    total_model_calls = 0
    total_tool_calls = 0
    for index, source in enumerate(sources, start=1):
        destination = output_dir / f"{source.stem}.txt"
        try:
            model_calls, tool_calls = export_checkpoint(source, destination)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"Failed to export {source}: {exc}", file=sys.stderr)
            return 1
        total_model_calls += model_calls
        total_tool_calls += tool_calls
        print(
            f"[{index}/{len(sources)}] {source.name} -> {destination.name} "
            f"({model_calls} model calls, {tool_calls} tool calls)"
        )

    print(
        f"Exported {len(sources)} files to {output_dir} "
        f"({total_model_calls} model calls, {total_tool_calls} tool calls)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
