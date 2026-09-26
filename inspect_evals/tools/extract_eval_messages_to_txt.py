#!/usr/bin/env python3
"""Extract an Inspect .eval archive and render its sample messages as text.

Example:
    uv run python tools/extract_eval_messages_to_txt.py logs/run.eval
"""

from __future__ import annotations

import argparse
import shutil
import stat
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

if sys.version_info >= (3, 14):
    from zipfile import BadZipFile, ZipFile
else:
    from zipfile_zstd import BadZipFile, ZipFile

if __package__:
    from tools.convert_sample_messages_to_txt import convert_file
else:
    from convert_sample_messages_to_txt import convert_file


def _member_path(name: str) -> PurePosixPath:
    """Validate an archive member and return its relative POSIX path."""
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        not name
        or "\\" in name
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return posix_path


def _is_symlink(member: Any) -> bool:
    mode = member.external_attr >> 16
    return stat.S_ISLNK(mode)


def extract_eval_archive(
    eval_path: Path, output_dir: Path, *, overwrite: bool = False
) -> int:
    """Safely extract an Inspect .eval archive, including Zstandard ZIP files."""
    if not eval_path.is_file():
        raise ValueError(f"{eval_path}: input .eval file does not exist")
    if eval_path.suffix != ".eval":
        raise ValueError(f"{eval_path}: expected a file ending in .eval")

    with ZipFile(eval_path) as archive:
        members = archive.infolist()
        if not members:
            raise ValueError(f"{eval_path}: archive is empty")

        member_paths = [_member_path(member.filename) for member in members]
        symlinks = [member.filename for member in members if _is_symlink(member)]
        if symlinks:
            raise ValueError(f"archive contains unsupported symlink: {symlinks[0]!r}")

        if output_dir.exists():
            if not output_dir.is_dir():
                raise ValueError(f"{output_dir}: output path is not a directory")
            if any(output_dir.iterdir()) and not overwrite:
                raise ValueError(
                    f"{output_dir}: output directory is not empty; use --overwrite"
                )
        output_dir.mkdir(parents=True, exist_ok=True)

        for member, relative_path in zip(members, member_paths, strict=True):
            destination = output_dir.joinpath(*relative_path.parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    return len(members)


def convert_extracted_samples(output_dir: Path) -> tuple[Path, int]:
    """Convert all JSON files in the extracted samples directory to text."""
    samples_dir = output_dir / "samples"
    if not samples_dir.is_dir():
        raise ValueError(f"{samples_dir}: extracted archive has no samples directory")

    sample_files = sorted(samples_dir.glob("*.json"))
    if not sample_files:
        raise ValueError(f"{samples_dir}: no sample JSON files found")

    text_dir = output_dir / "samples_txt"
    for sample_file in sample_files:
        convert_file(sample_file, text_dir / f"{sample_file.stem}.txt")
    return text_dir, len(sample_files)


def process_eval(
    eval_path: Path, output_dir: Path | None = None, *, overwrite: bool = False
) -> tuple[Path, Path, int, int]:
    """Extract one .eval file and convert every sample message history."""
    resolved_output_dir = output_dir or eval_path.with_suffix("")
    member_count = extract_eval_archive(
        eval_path, resolved_output_dir, overwrite=overwrite
    )
    text_dir, sample_count = convert_extracted_samples(resolved_output_dir)
    return resolved_output_dir, text_dir, member_count, sample_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_file", type=Path, help="Inspect .eval archive")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="extraction directory (default: beside the archive without .eval)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite archive members and text files in a non-empty output directory",
    )
    args = parser.parse_args()

    try:
        output_dir, text_dir, member_count, sample_count = process_eval(
            args.eval_file, args.output_dir, overwrite=args.overwrite
        )
    except (BadZipFile, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"Extracted {member_count} file(s) to {output_dir}")
    print(f"Converted {sample_count} sample(s) to {text_dir}")


if __name__ == "__main__":
    main()
