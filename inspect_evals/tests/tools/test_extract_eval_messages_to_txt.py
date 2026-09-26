import json
from pathlib import Path

import pytest
from tools.extract_eval_messages_to_txt import extract_eval_archive, process_eval
from zipfile_zstd import ZIP_STORED, ZIP_ZSTANDARD, ZipFile


def _write_eval_archive(path: Path) -> None:
    sample = {
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "Hello"},
        ]
    }
    with ZipFile(path, "w", compression=ZIP_ZSTANDARD) as archive:
        archive.writestr("header.json", "{}")
        archive.writestr(
            "samples/sample_epoch_1.json",
            json.dumps(sample, ensure_ascii=False),
        )


def test_process_eval_extracts_zstandard_and_converts_messages(
    tmp_path: Path,
) -> None:
    eval_path = tmp_path / "run.eval"
    output_dir = tmp_path / "output"
    _write_eval_archive(eval_path)

    extracted_dir, text_dir, member_count, sample_count = process_eval(
        eval_path, output_dir
    )

    assert extracted_dir == output_dir
    assert member_count == 2
    assert sample_count == 1
    assert json.loads((output_dir / "header.json").read_text(encoding="utf-8")) == {}
    rendered = (text_dir / "sample_epoch_1.txt").read_text(encoding="utf-8")
    assert "[001] USER" in rendered
    assert "你好" in rendered
    assert "[002] ASSISTANT" in rendered


def test_process_eval_rejects_nonempty_output_without_overwrite(
    tmp_path: Path,
) -> None:
    eval_path = tmp_path / "run.eval"
    output_dir = tmp_path / "output"
    _write_eval_archive(eval_path)
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="use --overwrite"):
        process_eval(eval_path, output_dir)

    assert (output_dir / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_extract_eval_archive_rejects_parent_path(tmp_path: Path) -> None:
    eval_path = tmp_path / "unsafe.eval"
    output_dir = tmp_path / "output"
    with ZipFile(eval_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("../escaped.json", "{}")

    with pytest.raises(ValueError, match="unsafe archive member path"):
        extract_eval_archive(eval_path, output_dir)

    assert not (tmp_path / "escaped.json").exists()
