import json
from pathlib import Path

import pytest
from tools.convert_sample_messages_to_txt import (
    convert_file,
    read_messages,
    render_messages,
)


def test_render_messages_includes_content_and_tool_io_only() -> None:
    messages = [
        {"role": "system", "content": "Follow instructions", "id": "message-1"},
        {"role": "user", "content": "查找答案", "source": "input"},
        {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "reasoning": "Need to search", "redacted": False},
                {"type": "text", "text": "I will check."},
            ],
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": "search",
                    "arguments": {"query": "示例"},
                }
            ],
            "model": "test-model",
        },
        {
            "role": "tool",
            "function": "search",
            "tool_call_id": "call-1",
            "content": "result text",
        },
    ]

    rendered = render_messages(messages)

    assert "[001] SYSTEM" in rendered
    assert "[reasoning]\nNeed to search" in rendered
    assert "Function: search" in rendered
    assert '"query": "示例"' in rendered
    assert "[004] TOOL RESULT" in rendered
    assert "Output:\nresult text" in rendered
    assert "message-1" not in rendered
    assert "test-model" not in rendered
    assert '"source"' not in rendered


def test_render_messages_supports_openai_style_tool_call() -> None:
    rendered = render_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-2",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"pwd"}',
                        },
                    }
                ],
            }
        ]
    )

    assert "Function: bash" in rendered
    assert '"command": "pwd"' in rendered


def test_convert_file_writes_utf8_text(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.json"
    output_path = tmp_path / "out" / "sample.txt"
    input_path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "你好"}]}),
        encoding="utf-8",
    )

    convert_file(input_path, output_path)

    assert "[001] USER" in output_path.read_text(encoding="utf-8")
    assert "你好" in output_path.read_text(encoding="utf-8")


def test_read_messages_rejects_missing_messages(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.json"
    input_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 'messages' to be an array"):
        read_messages(input_path)
