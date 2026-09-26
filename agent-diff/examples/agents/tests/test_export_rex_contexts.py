from __future__ import annotations

import json
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from export_rex_contexts import export_checkpoint


def test_export_prints_incremental_messages_and_tool_io_once(tmp_path: Path) -> None:
    source = tmp_path / "task.json"
    destination = tmp_path / "text" / "task.txt"
    messages = [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "Original task"},
        {
            "role": "assistant",
            "content": "<action>curl https://example.test/items</action>",
        },
        {"role": "user", "content": '<observation>{"id":"123"}</observation>'},
        {"role": "assistant", "content": "<done>Found item 123</done>"},
        {"role": "user", "content": "Review the completed step"},
        {"role": "assistant", "content": '{"status":"success"}'},
    ]
    events = [
        {
            "event": "model_response",
            "frame_id": "root",
            "step_id": "step_1",
            "artifact": {
                "phase": "direct_execution",
                "content": messages[2]["content"],
            },
        },
        {
            "event": "tool_result",
            "frame_id": "root",
            "step_id": "step_1",
            "artifact": {
                "action": "curl https://example.test/items",
                "result": {"stdout": '{"id":"123"}', "exit_code": 0},
                "observation": '{"id":"123"}',
            },
        },
        {
            "event": "model_response",
            "frame_id": "root",
            "step_id": "step_1",
            "artifact": {
                "phase": "direct_execution",
                "content": messages[4]["content"],
            },
        },
        {
            "event": "model_response",
            "frame_id": "root",
            "artifact": {
                "phase": "reduce",
                "content": messages[6]["content"],
            },
        },
    ]
    source.write_text(
        json.dumps({"record": {"trace": {"messages": messages, "events": events}}}),
        encoding="utf-8",
    )

    counts = export_checkpoint(source, destination)
    exported = destination.read_text(encoding="utf-8")

    assert counts == (3, 1)
    assert "MODEL CALL 001" in exported
    assert "NEW NON-TOOL MESSAGES SINCE PREVIOUS MODEL CALL (2)" in exported
    assert "MODEL CALL 002" in exported
    assert "NEW NON-TOOL MESSAGES SINCE PREVIOUS MODEL CALL (0)" in exported
    assert "MODEL CALL 003" in exported
    assert "NEW NON-TOOL MESSAGES SINCE PREVIOUS MODEL CALL (1)" in exported
    assert exported.count("System instructions") == 1
    assert exported.count("Original task") == 1
    assert exported.count("<action>curl https://example.test/items</action>") == 1
    assert exported.count("<done>Found item 123</done>") == 1
    assert exported.count("Review the completed step") == 1
    assert "<observation>" not in exported
    assert exported.count("TOOL CALL 001") == 1
    assert "TOOL INPUT (BASH)" in exported
    assert "curl https://example.test/items" in exported
    assert "TOOL OUTPUT (RAW EXECUTOR RESULT)" in exported
    assert '"exit_code": 0' in exported
    assert "OBSERVATION ADDED TO MODEL CONTEXT" in exported


def test_export_treats_compression_checkpoint_as_next_call_context(
    tmp_path: Path,
) -> None:
    source = tmp_path / "compressed-task.json"
    destination = tmp_path / "compressed-task.txt"
    checkpoint = "[RUNNER-COMPRESSED HISTORY]\nVerified exact item id=A1."
    messages = [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "Original task"},
        {"role": "assistant", "content": "<done>Verified A1</done>"},
        {"role": "user", "content": checkpoint},
        {"role": "user", "content": "Completion review phase."},
        {"role": "assistant", "content": '{"status":"success"}'},
    ]
    events = [
        {
            "event": "model_response",
            "frame_id": "root",
            "step_id": "step_1",
            "artifact": {
                "phase": "direct_execution",
                "content": messages[2]["content"],
            },
        },
        {
            "event": "frame_compress",
            "frame_id": "root",
            "artifact": {"applied": True},
        },
        {
            "event": "model_response",
            "frame_id": "root",
            "artifact": {
                "phase": "reduce",
                "content": messages[5]["content"],
            },
        },
    ]
    source.write_text(
        json.dumps({"record": {"trace": {"messages": messages, "events": events}}}),
        encoding="utf-8",
    )

    counts = export_checkpoint(source, destination)
    exported = destination.read_text(encoding="utf-8")

    assert counts == (2, 0)
    assert "NEW NON-TOOL MESSAGES SINCE PREVIOUS MODEL CALL (2)" in exported
    assert exported.count(checkpoint) == 1
