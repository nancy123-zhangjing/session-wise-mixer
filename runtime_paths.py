"""Resolve recorded project paths without rewriting historical evidence.

Only the explicitly known project roots are relocated. Unrelated absolute
paths remain the caller's choice; this module is not a filesystem hook.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_PROJECT_ROOTS = (
    "/home/jingzhang/experiment1",
    "/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1",
    "C:/Users/nancy/Documents/experiment1",
)


def resolve_project_path(value, *, project_root=None, relative_to=None) -> Path:
    """Prefer the new root for known old paths, even while the old file exists.

    Relative paths use ``relative_to`` (the project root by default). When
    that base is inside the project, traversal and symlinks may not escape it.
    Unknown absolute paths are returned unchanged, including foreign-platform
    paths, rather than silently interpreting them relative to this checkout.
    """
    root = Path(project_root or PROJECT_ROOT).resolve()
    text = str(value)
    normalized = text.replace("\\", "/")
    for previous in LEGACY_PROJECT_ROOTS:
        windows = PureWindowsPath(previous).is_absolute()
        candidate_text = normalized.casefold() if windows else normalized
        prefix = previous.casefold() if windows else previous
        if candidate_text != prefix and not candidate_text.startswith(prefix + "/"):
            continue
        suffix = normalized[len(previous):].lstrip("/")
        parts = PurePosixPath(suffix).parts
        if any(part == ".." or ":" in part for part in parts):
            raise ValueError(f"Unsafe relocated project path: {value}")
        result = (root / PurePosixPath(suffix)).resolve()
        if not result.is_relative_to(root):
            raise ValueError(f"Relocated path escapes project root: {value}")
        return result

    path = Path(value)
    if PurePosixPath(normalized).is_absolute() or PureWindowsPath(text).is_absolute():
        if path.is_absolute() and path.is_relative_to(root):
            result = path.resolve()
            if not result.is_relative_to(root):
                raise ValueError(f"Current project path escapes project root: {value}")
            return result
        return path
    base = Path(relative_to) if relative_to is not None else root
    base_started_inside = base.absolute().is_relative_to(root)
    base = base.resolve()
    if base_started_inside and not base.is_relative_to(root):
        raise ValueError(f"Relative path base escapes project root: {relative_to}")
    result = (base / path).resolve()
    if base.is_relative_to(root) and not result.is_relative_to(root):
        raise ValueError(f"Relative path escapes project root: {value}")
    return result
