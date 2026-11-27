"""Where installed extensions live, and how a running process finds them.

The directory is chosen, in order, by :func:`configure_extensions_dir` (a host
that owns a data directory), the ``LLM_ROSETTA_EXTENSIONS_DIR`` environment
variable, then a per-user default.  The default never falls back to a path
under ``/``: a container uid with no passwd entry gets ``HOME=/``, and an
install there would write into the image root or fail late and obscurely.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from ._discovery import NAMESPACE

#: The environment variable that names the extensions directory.
ENV_VAR = "LLM_ROSETTA_EXTENSIONS_DIR"

_CONFIGURED: list[Path] = []


class ExtensionInstallError(Exception):
    """An extension could not be installed, removed or located."""


class ExtensionsDirError(ExtensionInstallError):
    """No usable extensions directory can be derived from the environment."""


def configure_extensions_dir(path: str | os.PathLike[str] | None) -> None:
    """Set the extensions directory explicitly.

    An explicit directory wins over the environment variable and the default.

    Args:
        path: The directory, or ``None`` to clear a previous setting.
    """
    _CONFIGURED.clear()
    if path is not None:
        _CONFIGURED.append(Path(path).expanduser())


def _unusable_home(reason: str) -> ExtensionsDirError:
    return ExtensionsDirError(
        f"cannot choose a default extensions directory: {reason}. "
        f"Set {ENV_VAR} to a writable directory."
    )


def _home() -> Path:
    home = os.path.expanduser("~")
    if home == "~" or not os.path.isabs(home):
        raise _unusable_home("the home directory is unknown")
    path = Path(home)
    if path == Path(path.anchor):
        raise _unusable_home(f"the home directory is the filesystem root ({home!r})")
    return path


def _default_dir() -> Path:
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA")
        if base and os.path.isabs(base):
            return Path(base) / "llm-rosetta" / "extensions"
        return _home() / "AppData" / "Local" / "llm-rosetta" / "extensions"
    if sys.platform == "darwin":
        return (
            _home() / "Library" / "Application Support" / "llm-rosetta" / "extensions"
        )
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg and os.path.isabs(xdg):
        return Path(xdg) / "llm-rosetta" / "extensions"
    return _home() / ".local" / "share" / "llm-rosetta" / "extensions"


def extensions_dir() -> Path:
    """Return the directory extensions are installed into.

    Returns:
        The configured directory, else ``$LLM_ROSETTA_EXTENSIONS_DIR``, else the
        per-user default.  The directory need not exist yet.

    Raises:
        ExtensionsDirError: Nothing was configured and the home directory is
            unknown or is ``/``.
    """
    if _CONFIGURED:
        return _CONFIGURED[0]
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    return _default_dir()


#: The environment variable that names the directory external CLIs install into.
TOOLS_ENV_VAR = "LLM_ROSETTA_TOOLS_DIR"


def tools_dir() -> Path:
    """Return the directory ``ext install-cli`` installs external programs into.

    Returns:
        ``$LLM_ROSETTA_TOOLS_DIR`` when set, else ``tools`` beside the
        extensions directory (``/config/tools`` when that is
        ``/config/extensions``).  The directory need not exist yet.

    Raises:
        ExtensionsDirError: Nothing names it and no extensions directory can be
            derived.
    """
    env = os.environ.get(TOOLS_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return extensions_dir().parent / "tools"


def tool_bin_dir(ext_id: str) -> Path:
    """Return the directory holding the entry links of an extension's programs."""
    return tools_dir() / ext_id / "bin"


def activate_extensions_dir(path: str | os.PathLike[str] | None = None) -> Path:
    """Make the extensions in a directory importable in this process.

    Puts the directory first on ``sys.path`` (once), invalidates the import
    caches and drops a cached ``llm_rosetta_ext`` namespace module, so an
    extension installed after the first import is found without a restart.

    Args:
        path: The directory; :func:`extensions_dir` by default.

    Returns:
        The activated directory.
    """
    target = Path(path) if path is not None else extensions_dir()
    entry = str(target)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    importlib.invalidate_caches()
    cached = sys.modules.get(NAMESPACE)
    if cached is not None and getattr(cached, "__file__", None) is None:
        del sys.modules[NAMESPACE]
    return target


def activate_if_present() -> Path | None:
    """Activate the extensions directory when it exists; otherwise do nothing."""
    try:
        target = extensions_dir()
    except ExtensionInstallError:
        return None
    if not target.is_dir():
        return None
    return activate_extensions_dir(target)
