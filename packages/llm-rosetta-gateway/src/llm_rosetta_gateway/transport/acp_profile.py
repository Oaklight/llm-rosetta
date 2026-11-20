"""The gateway-owned ``agy`` profile: how phase 1 keeps the agent's tools off.

The agent CLI behind an ACP lane is an upstream, not the gateway's client: it
must never run a tool on the gateway host for an API request.  ``agy`` has no
single switch for that, so the lane layers four, each measured on ``agy``
1.5.13 (see the shim README):

1. ``--gemini-dir <dir>``: ``agy`` reads its settings, hooks, agents and MCP
   servers from a directory the gateway owns, not from the host user's
   ``~/.gemini``.  The host's MCP servers (and so ``call_mcp_tool``,
   ``list_resources``, ``read_resource``) are not loaded.
2. ``--agent gateway-text-only`` (``config/agents/.../agent.md``): a custom
   agent with ``tools: []`` and ``excludeDefaultComponents: true``.  The model
   is told it has one tool at most (``manage_task``), which is the only one the
   agent definition cannot remove.
3. ``config/hooks.json``: a ``PreToolUse`` hook for every tool that answers
   ``deny``.  A hook that exits non-zero, prints invalid JSON or times out
   also blocks the tool (measured), so it fails closed.
4. ``antigravity-cli/settings.json`` with ``toolPermission: strict``: every
   file, shell and write tool asks before it runs, and the gateway's ACP
   client refuses every ask.  Without it ``read_file`` of the system temp dir
   and ``search_web`` run without asking.

The transport re-asserts all of it before every session, so a change the agent
makes to its own profile is undone.  The ACP client also stops a turn at the
first tool frame it sees (the guard in ``acp.py``).

Only an ``agy`` command is hardened this way: the flags are ``agy``'s.  An
operator-supplied ``acp_argv`` is used as written and is not hardened.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("llm-rosetta-gateway")

#: The custom agent the lane runs: no tools, no default components.
AGENT_NAME = "gateway-text-only"

#: The flag that makes ``agy`` ask before a tool runs, instead of approving
#: every request itself (its default for ``agy acp`` is to approve).
PERMISSION_FLAG = "--dangerously-skip-permissions=false"

#: The sentence a blocked tool call carries back to the model.
DENY_REASON = "agent tools are disabled by the gateway"

_SETTINGS_PATH = ("antigravity-cli", "settings.json")
_HOOKS_PATH = ("config", "hooks.json")
_AGENT_PATH = ("config", "agents", AGENT_NAME, "agent.md")

_HOOK_COMMAND = f"cat >/dev/null; printf '%s' '{json.dumps({'decision': 'deny', 'reason': DENY_REASON}, separators=(',', ':'))}'"

#: ``hooks.json`` as the gateway writes it.  ``matcher: "*"`` matches every tool.
HOOKS: dict[str, Any] = {
    "gateway-deny": {
        "PreToolUse": [
            {
                "matcher": "*",
                "hooks": [{"type": "command", "command": _HOOK_COMMAND, "timeout": 10}],
            }
        ]
    }
}

AGENT_MD = f"""---
name: {AGENT_NAME}
description: Text-only assistant. It has no tools.
tools: []
excludeDefaultComponents: true
mainAgent: true
subagent: false
---
# System Prompt
You are a helpful text-only assistant. You have no tools: answer directly in
text, and say so when a request needs one.
"""

_GEMINI_FLAGS = ("--gemini-dir", "--gemini_dir")
_AGENT_FLAGS = ("--agent",)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.:-]+")


def is_agy(argv: list[str] | tuple[str, ...]) -> bool:
    """Whether *argv* starts the ``agy`` CLI (the only agent this hardens)."""
    return bool(argv) and Path(argv[0]).name in ("agy", "agy.real", "agy.exe")


def fallback_root() -> Path:
    """Where profiles live when the gateway has no data directory."""
    return Path.home() / ".config" / "llm-rosetta" / "acp"


def slug_of(provider: str) -> str:
    """A directory name for a provider: readable, never hidden, no separators."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", provider).lstrip(".") or "provider"


def state_dir_for(data_dir: str | None, provider: str) -> str:
    """The directory the gateway keeps one ACP provider's agent profile in."""
    root = Path(data_dir) / "acp" if data_dir else fallback_root()
    return str(root / slug_of(provider))


def flag_value(argv: list[str], names: tuple[str, ...]) -> str | None:
    """The value of the last ``--flag value`` or ``--flag=value`` in *argv*."""
    found: str | None = None
    for index, arg in enumerate(argv):
        for name in names:
            if arg == name and index + 1 < len(argv):
                found = argv[index + 1]
            elif arg.startswith(name + "="):
                found = arg.split("=", 1)[1]
    return found


def _without_flag(argv: list[str], names: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in names:
            skip = True
            continue
        if any(arg.startswith(name + "=") for name in names):
            continue
        out.append(arg)
    return out


def gemini_dir_of(argv: list[str]) -> str | None:
    """The ``--gemini-dir`` an argv names, if any."""
    return flag_value(argv, _GEMINI_FLAGS)


def harden_argv(argv: list[str], state_dir: str) -> list[str]:
    """Add the flags that put ``agy`` in the gateway-owned profile.

    A ``--gemini-dir`` already in *argv* is kept: it is another profile the
    gateway owns (an entry's own login profile), and the hardening is written
    into that one.  The agent flag is always ours.
    """
    out = list(argv)
    if not any(a.startswith("--dangerously-skip-permissions") for a in out):
        out.append(PERMISSION_FLAG)
    if gemini_dir_of(out) is None:
        out += ["--gemini-dir", state_dir]
    out = _without_flag(out, _AGENT_FLAGS)
    out += ["--agent", AGENT_NAME]
    return out


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, 0o600)
        # ``replace`` swaps the name itself, so a symlink planted at *path* is
        # replaced rather than followed.
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _settings_text(existing: str | None) -> tuple[str, bool]:
    """The settings document with ``toolPermission: strict``; whether it drifted.

    Other keys are kept: a sign-in made under this profile may store its auth
    method there.
    """
    document: dict[str, Any] = {}
    if existing is not None:
        try:
            loaded = json.loads(existing)
            if isinstance(loaded, dict):
                document = loaded
        except ValueError:
            document = {}
    drifted = existing is not None and document.get("toolPermission") != "strict"
    document["toolPermission"] = "strict"
    return json.dumps(document, indent=2, sort_keys=True) + "\n", drifted


def ensure_profile(gemini_dir: str) -> list[str]:
    """Write (or re-assert) the hardened profile; name what had been changed.

    Idempotent and cheap: a file is rewritten only when it differs.  Called at
    start and before every session.

    Returns:
        The relative names of files that existed and were altered since the
        gateway wrote them (empty on first creation and on an untouched profile).
    """
    root = Path(gemini_dir)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(root, 0o700)
    except OSError:  # pragma: no cover - a filesystem that refuses chmod
        pass
    drift: list[str] = []

    settings = root.joinpath(*_SETTINGS_PATH)
    existing = _read(settings)
    text, drifted = _settings_text(existing)
    if existing is None or _normalised(existing) != _normalised(text):
        _write_atomic(settings, text)
    if drifted:
        drift.append("/".join(_SETTINGS_PATH))

    for parts, want in (
        (_HOOKS_PATH, json.dumps(HOOKS, indent=2) + "\n"),
        (_AGENT_PATH, AGENT_MD),
    ):
        path = root.joinpath(*parts)
        have = _read(path)
        if have != want:
            _write_atomic(path, want)
            if have is not None:
                drift.append("/".join(parts))
    return drift


def _normalised(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def tool_name_of(update: dict[str, Any]) -> str | None:
    """The tool a ``session/update`` payload names, or ``None`` if it is no tool frame.

    ``agy`` puts the tool's own name in ``name``; an agent that does not gets
    its ``kind`` back.  Only the name is returned (and clipped): a title or
    ``rawInput`` can carry a command line or file contents.
    """
    body = update.get("update")
    if isinstance(body, dict):
        update = body
    kind = update.get("sessionUpdate") or update.get("session_update") or ""
    if kind not in ("tool_call", "tool_call_update"):
        return None
    name = update.get("name")
    if not isinstance(name, str) or not name:
        name = f"kind:{update.get('kind') or 'unknown'}"
    return _SAFE_NAME.sub("_", name)[:64]
