"""Data root injected by entry files, keeping src kernel-free."""

from __future__ import annotations

from pathlib import Path

_ROOT: Path | None = None


def apply_data_root(root) -> None:
    """Entry files call this once before any registry lookup."""
    global _ROOT
    _ROOT = Path(root)


def get_data_root() -> Path:
    """Injected root, or a refusal naming the missing call."""
    if _ROOT is None:
        raise Exception("data root unset; the entry file must call apply_data_root first")
    return _ROOT
