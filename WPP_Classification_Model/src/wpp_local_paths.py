"""Configuration-boundary path adaptation for local WSL/Windows defaults.

Stored provenance and cache metadata deliberately retain their original strings.
Only caller-supplied configuration paths pass through this module, before a
platform ``Path.resolve()`` can interpret a WSL mount path incorrectly.
"""
from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath

# On Windows, Path("/mnt/d/x") is rooted on the current drive, while a path
# passed through PureWindowsPath can render as "\\mnt\\d\\x".  Both mean the
# WSL mount convention only when their first two components are mnt/<letter>.
_WSL_DRIVE_MOUNT = re.compile(r"^[\\/]+mnt[\\/]([A-Za-z])(?:[\\/](.*))?$")


def normalize_local_path_text(value: str | os.PathLike[str], *, windows: bool | None = None) -> str:
    """Return a native-Windows spelling for a WSL drive mount when applicable.

    ``windows`` exists for focused platform-independent tests.  In normal use it
    follows the running platform.  Native drive paths, UNC paths, relative paths,
    and every non-WSL path are returned unchanged.
    """
    text = os.fspath(value)
    if not (os.name == "nt" if windows is None else windows):
        return text
    # A double-leading separator is an explicit UNC path, including a host
    # coincidentally named ``mnt``; it must not be mistaken for a rooted
    # PureWindowsPath rendering of /mnt/<drive>.
    if text.startswith(("\\\\", "//")):
        return text
    match = _WSL_DRIVE_MOUNT.fullmatch(text)
    if match is None:
        return text
    drive, remainder = match.groups()
    # PureWindowsPath supplies the canonical native separator without touching
    # the filesystem (and therefore without changing user-supplied semantics).
    root = PureWindowsPath(f"{drive.upper()}:\\")
    if not remainder:
        return str(root)
    return str(root / PureWindowsPath(remainder.replace("/", "\\")))


def local_config_path(value: str | os.PathLike[str], *, windows: bool | None = None) -> Path:
    """Adapt a configuration path before any caller resolves it on this host."""
    return Path(normalize_local_path_text(value, windows=windows)).expanduser()


__all__ = ["local_config_path", "normalize_local_path_text"]
