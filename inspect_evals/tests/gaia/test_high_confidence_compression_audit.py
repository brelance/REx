import hashlib
from pathlib import Path

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

from inspect_evals.gaia.hierarchical_compact_audit import (
    CompressionLogContext,
    compression_text_log_path,
    make_compression_id,
    safe_path_component,
    write_compressor_text_log,
)


def _context(tmp_path: Path, *, task_id: str = "task-1") -> CompressionLogContext:
    return CompressionLogContext(
        root=tmp_path / "eval-run",
        eval_log_parent=tmp_path,
        task_id=task_id,
        epoch=2,
        sample_uuid="sample-uuid",
        runtime_sample_id="runtime-sample",
    )


def _payload() -> dict[str, object]:
    request_messages = [
        ChatMessageSystem(content="压缩系统提示").model_dump(mode="json"),
        ChatMessageUser(content="完整输入").model_dump(mode="json"),
    ]
    response_message = ChatMessageAssistant(content="完整输出").model_dump(mode="json")
    return {
        "schema_version": 1,
        "task_id": "task-1",
        "epoch": 2,
        "sample_uuid": "sample-uuid",
        "compression_id": "cmp_123",
        "compression_sequence": 1,
        "event": "tree_compress",
        "backend": "standalone",
        "model": "mockllm/model",
        "request": {"messages": request_messages},
        "response": {
            "message": response_message,
            "model_output": {"model": "mockllm/model", "choices": []},
            "raw_text": "完整输出",
            "error_type": None,
            "error_message": None,
        },
        "decision": {"triggered": True, "applied": True, "reason": "applied"},
    }


def test_compression_text_log_path_is_sample_and_epoch_scoped(tmp_path: Path) -> None:
    context = _context(tmp_path, task_id="../unsafe/task")
    compression_id = make_compression_id(context, 3)

    path = compression_text_log_path(
        context,
        sequence=3,
        scope_kind="tree",
        compression_id=compression_id,
    )

    assert path.parent.parent.name == safe_path_component("../unsafe/task")
    assert path.parent.name == "epoch-2"
    assert path.name == f"c000003_tree_{compression_id}.txt"
    assert context.root in path.parents
    assert ".." not in path.relative_to(context.root).parts


def test_safe_path_component_avoids_sanitization_collisions() -> None:
    assert safe_path_component("task-1") == "task-1"
    assert safe_path_component("a/b") != safe_path_component("a?b")
    assert safe_path_component("..") not in ("", ".", "..")


def test_compressor_text_log_is_readable_complete_and_hashed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    payload = _payload()

    metadata = write_compressor_text_log(
        context,
        sequence=1,
        scope_kind="tree",
        compression_id="cmp_123",
        payload=payload,
    )

    path = tmp_path / str(metadata["path"])
    text = path.read_text(encoding="utf-8")
    assert metadata["status"] == "written"
    assert metadata["bytes"] == len(text.encode("utf-8"))
    assert metadata["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert "=== COMPRESSOR INPUT (READABLE) ===" in text
    assert "压缩系统提示" in text
    assert "完整输入" in text
    assert "完整输出" in text
    assert "=== COMPLETE CALL JSON ===" in text
    assert '"model_output"' in text




def test_compressor_text_log_removes_temporary_file_after_replace_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(
        "inspect_evals.gaia.hierarchical_compact_audit.os.replace",
        fail_replace,
    )

    with pytest.raises(OSError, match="replace failed"):
        write_compressor_text_log(
            context,
            sequence=1,
            scope_kind="tree",
            compression_id="cmp_123",
            payload=_payload(),
        )

    directory = context.root / "samples" / "compression_logs" / "task-1" / "epoch-2"
    assert not list(directory.glob("*.tmp"))
    assert not list(directory.glob("*.txt"))
