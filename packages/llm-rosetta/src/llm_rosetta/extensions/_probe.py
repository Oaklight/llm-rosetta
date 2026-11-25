"""Import an installed extension where a failure cannot take the caller down.

``llm-rosetta ext doctor`` wants to know whether an extension's code imports,
whether its ``commands`` module exposes what its manifest lists and whether its
``transport`` module does the same.  A broken extension can raise anything at
import, or exit the interpreter, so the check runs in a child process; a host
that has no Python interpreter to spawn (the compiled binary) checks in process
instead and says so.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from llm_rosetta.contracts.extension import ExtensionManifest

from ._dirs import extensions_dir

#: Seconds a child may take to import one extension.
IMPORT_TIMEOUT = 30.0

_SCRIPT = r"""
import importlib, json, sys

directory, ext_id, verbs, transports = sys.argv[1:5]
sys.path.insert(0, directory)
problems = []
try:
    importlib.import_module("llm_rosetta_ext." + ext_id)
    for module, attribute, wanted in (
        ("commands", "COMMANDS", json.loads(verbs)),
        ("transport", "TRANSPORTS", json.loads(transports)),
    ):
        if not wanted:
            continue
        table = getattr(
            importlib.import_module(f"llm_rosetta_ext.{ext_id}.{module}"), attribute, None
        )
        if not isinstance(table, dict):
            problems.append(f"{module} has no {attribute} mapping")
        elif set(table) != set(wanted):
            problems.append(
                f"{attribute} has {sorted(table)} but the manifest lists {sorted(wanted)}"
            )
except BaseException as exc:
    problems.append(f"{type(exc).__name__}: {exc}")
print(json.dumps(problems))
"""


@dataclass(frozen=True)
class ImportCheck:
    """The result of importing one extension.

    Attributes:
        ok: Whether everything imported and matched the manifest.
        detail: ``"imported"``, or the problems found.
        isolated: ``True`` when a child process ran the check; ``False`` when
            the host checked in its own process.
    """

    ok: bool
    detail: str
    isolated: bool


def _can_spawn() -> bool:
    exe = Path(sys.executable or "")
    return exe.is_file() and exe.name.lower().startswith("python")


def _run_in_process(
    directory: str, ext_id: str, verbs: list[str], transports: list[str]
) -> list[str]:
    namespace: dict[str, object] = {}
    saved_argv = sys.argv
    saved_out = sys.stdout
    import io

    sys.argv = ["", directory, ext_id, json.dumps(verbs), json.dumps(transports)]
    sys.stdout = captured = io.StringIO()
    try:
        exec(compile(_SCRIPT, "<ext-doctor>", "exec"), namespace)  # noqa: S102
    finally:
        sys.argv = saved_argv
        sys.stdout = saved_out
    return json.loads(captured.getvalue() or "[]")


def check_import(
    ext_id: str,
    manifest: ExtensionManifest | None,
    *,
    directory: str | os.PathLike[str] | None = None,
    timeout: float = IMPORT_TIMEOUT,
) -> ImportCheck:
    """Import an extension and check its modules against its manifest.

    Args:
        ext_id: The extension id.
        manifest: Its manifest, when it could be read; the commands and
            transports it lists are checked.
        directory: The extensions directory; :func:`extensions_dir` by default.
        timeout: Seconds the child may run.

    Returns:
        The result; a timeout, a crash or a bad answer is a failed check, not
        an exception.
    """
    root = str(Path(directory) if directory is not None else extensions_dir())
    verbs = sorted(manifest.commands) if manifest else []
    transports = sorted(manifest.transports) if manifest else []
    if not _can_spawn():
        try:
            problems = _run_in_process(root, ext_id, verbs, transports)
        except Exception as exc:  # noqa: BLE001 - the check reports, never raises
            problems = [f"{type(exc).__name__}: {exc}"]
        return ImportCheck(not problems, "; ".join(problems) or "imported", False)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [item for item in sys.path if item and os.path.isdir(item)]
    )
    try:
        run = subprocess.run(
            [
                sys.executable,
                "-c",
                _SCRIPT,
                root,
                ext_id,
                json.dumps(verbs),
                json.dumps(transports),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ImportCheck(False, f"the import did not finish in {timeout:g}s", True)
    except OSError as exc:
        return ImportCheck(False, f"cannot start a child process: {exc}", True)
    try:
        problems = json.loads(run.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        tail = [line for line in run.stderr.splitlines() if line.strip()]
        detail = tail[-1] if tail else f"the child exited {run.returncode}"
        return ImportCheck(False, f"the import crashed: {detail}", True)
    return ImportCheck(not problems, "; ".join(problems) or "imported", True)
