"""Environment helpers."""

from __future__ import annotations

from pathlib import Path


def load_dotenv_files() -> None:
    """Load .env files without overriding variables already exported by the shell."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    project_root = Path(__file__).resolve().parents[2]
    candidates = [project_root / ".env", Path.cwd() / ".env"]

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            load_dotenv(dotenv_path=path, override=False)
