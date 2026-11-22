"""The gateway-owned ``agy`` profile: agent tools off, client tools bridged.

The agent CLI behind an ACP lane is an upstream, not the gateway's client: it
must never run a tool on the gateway host for an API request.  Client tools
travel the other way — the request declares them, the transport serves them
from a loopback MCP endpoint named on ``session/new``, and the agent may call
those.  ``agy`` has no single switch for the first part, so the lane layers
four, each measured on ``agy`` 1.5.13:

1. ``--gemini-dir <dir>``: ``agy`` reads its settings, hooks, agents and MCP
   servers from a directory the gateway owns, not from the host user's
   ``~/.gemini``.  The host's MCP servers are not loaded.
2. ``--agent gateway-text-only`` (``config/agents/.../agent.md``): a custom
   agent with ``tools: []`` and ``excludeDefaultComponents: true``, so the
   built-in file, shell, web and task tools are not offered.  Its prompt names
   the client's MCP server, because excluding the default prompt sections also
   removes the server-name hint the model would otherwise get.
3. ``config/hooks.json``: a ``PreToolUse`` hook for every tool.  It allows the
   gateway's MCP server in the two shapes agents use — ``agy`` 1.5.14 and later
   name each MCP tool directly as ``mcp_<server>_<tool>``, older versions use
   the ``call_mcp_tool`` meta-tool (and the read-only resource readers) with
   the server in the arguments — and denies everything else.  A hook that
   exits non-zero, prints invalid JSON or times out also blocks the tool
   (measured), so it fails closed.
4. ``antigravity-cli/settings.json`` with ``toolPermission: strict`` and
   ``permissions.allow: ["mcp(*)"]``: the permission layer lets the MCP call
   reach the hook, which is what narrows it to the gateway's server.  Without
   the allow rule a headless turn auto-denies the MCP call; without ``strict``
   file reads and web search run without asking.

The transport re-asserts all of it before every session, so a change the agent
makes to its own profile is undone.  The ACP client also stops a turn at the
first tool frame that is not a bridged call (the guard in ``acp.py``).

Only an ``agy`` command is hardened this way: the flags are ``agy``'s.  An
operator-supplied ``acp_argv`` is used as written and is not hardened.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

from .acp_mcp import MCP_SERVER_NAME

logger = logging.getLogger("llm-rosetta-gateway")

#: The custom agent the lane runs: no built-in tools, no default components.
AGENT_NAME = "gateway-text-only"

#: The flag that makes ``agy`` ask before a tool runs, instead of approving
#: every request itself (its default for ``agy acp`` is to approve).
PERMISSION_FLAG = "--dangerously-skip-permissions=false"

#: The sentence a blocked tool call carries back to the model.
DENY_REASON = "agent tools are disabled by the gateway"

_SETTINGS_PATH = ("antigravity-cli", "settings.json")
_HOOKS_PATH = ("config", "hooks.json")
_HOOK_SCRIPT_PATH = ("config", "hooks", "allow_client_tools.py")
_AGENT_PATH = ("config", "agents", AGENT_NAME, "agent.md")

#: The read-only MCP meta-tools a model may use to inspect the client's
#: server.  They are allowed only for the gateway's own server; ``acp_mcp``
#: answers them from the bridge.
_MCP_READERS = ("list_resources", "read_resource", "list_prompts", "get_prompt")

_HOOK_SCRIPT_TEMPLATE = '''"""Allow the gateway's bridged MCP tools; deny every host tool.

The gateway writes this file.  ``agy`` runs it for every ``PreToolUse`` event
through ``hooks.json``.  It allows the client's MCP server and nothing else:
newer ``agy`` names each MCP tool ``mcp_<server>_<tool>``, older versions use
``call_mcp_tool`` with the server in its arguments, and the read-only resource
and prompt readers carry only the server.
"""

import json
import sys

SERVER = __SERVER__
PREFIX = "mcp_" + SERVER + "_"
READERS = __READERS__
DENY = __DENY__


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print(json.dumps({"decision": "deny", "reason": "unreadable hook input"}))
        return
    tool = payload.get("toolCall") or {}
    name = tool.get("name") or ""
    args = tool.get("args") or {}
    if name.startswith(PREFIX) and len(name) > len(PREFIX):
        print(json.dumps({"decision": "allow"}))
        return
    server = args.get("ServerName") or args.get("serverName") or ""
    declared = args.get("ToolName") or args.get("toolName") or ""
    if name == "call_mcp_tool" and server == SERVER and declared:
        print(json.dumps({"decision": "allow"}))
        return
    if name in READERS and server == SERVER:
        print(json.dumps({"decision": "allow"}))
        return
    print(json.dumps({"decision": "deny", "reason": DENY}))


if __name__ == "__main__":
    main()
'''

AGENT_MD = f"""---
name: {AGENT_NAME}
description: Text assistant with client-provided MCP tools only.
tools: []
excludeDefaultComponents: true
mainAgent: true
subagent: false
---
# System Prompt
You are a helpful assistant. The client provides tools through an MCP server
named `{MCP_SERVER_NAME}`; use those tools when the request calls for one.
Never use a file, shell, web, or task-management tool.
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
    into that one.  The agent flag is always ours.  ``--process-per-session``
    is what lets ``agy`` load an HTTP MCP server named on ``session/new``
    (measured on 1.5.13); without it the client tools are not listed.
    """
    out = list(argv)
    if not any(a.startswith("--dangerously-skip-permissions") for a in out):
        out.append(PERMISSION_FLAG)
    if "--process-per-session" not in out:
        out.append("--process-per-session")
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


def _hook_script_text() -> str:
    """The ``PreToolUse`` handler the gateway writes into the profile."""
    return (
        _HOOK_SCRIPT_TEMPLATE.replace("__SERVER__", repr(MCP_SERVER_NAME))
        .replace("__READERS__", repr(_MCP_READERS))
        .replace("__DENY__", repr(DENY_REASON))
    )


def _hooks_document(command: str) -> dict[str, Any]:
    """``hooks.json`` with one ``PreToolUse`` group for every tool."""
    return {
        "gateway-client-tools": {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": command, "timeout": 10}],
                }
            ]
        }
    }


def _settings_text(existing: str | None) -> tuple[str, bool]:
    """The settings document with the gateway's permission policy.

    ``toolPermission: strict`` makes the agent ask before a tool runs.
    ``permissions.allow: ["mcp(*)"]`` lets the MCP meta-tool reach the hook;
    the hook is what narrows it to the gateway's own server.  Other keys are
    kept: a sign-in made under this profile may store its auth method there.

    Returns:
        The document text and whether an existing document had to change.
    """
    document: dict[str, Any] = {}
    if existing is not None:
        try:
            loaded = json.loads(existing)
            if isinstance(loaded, dict):
                document = loaded
        except ValueError:
            document = {}
    before = json.dumps(document, sort_keys=True)
    document["toolPermission"] = "strict"
    permissions = document.get("permissions")
    if not isinstance(permissions, dict):
        permissions = {}
    allow = permissions.get("allow")
    if not isinstance(allow, list):
        allow = []
    if "mcp(*)" not in allow:
        allow.append("mcp(*)")
    permissions["allow"] = allow
    document["permissions"] = permissions
    drifted = existing is not None and json.dumps(document, sort_keys=True) != before
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

    hook_script = root.joinpath(*_HOOK_SCRIPT_PATH)
    hook_command = f"{shlex.quote(sys.executable)} {shlex.quote(str(hook_script))}"
    for parts, want in (
        (_HOOK_SCRIPT_PATH, _hook_script_text()),
        (_HOOKS_PATH, json.dumps(_hooks_document(hook_command), indent=2) + "\n"),
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
