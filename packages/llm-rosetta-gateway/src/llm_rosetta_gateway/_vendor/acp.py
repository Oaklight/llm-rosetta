# /// zerodep
# version = "0.3.7"
# deps = ["jsonrpc"]
# tier = "subsystem"
# category = "protocol"
# note = "Install/update via `zerodep add acp`"
# ///

"""ACP (Agent Client Protocol) -- Zero-dependency Python implementation.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

The Agent Client Protocol standardizes communication between code editors
(Clients) and AI coding agents (Agents).  It uses JSON-RPC 2.0 over stdio
(newline-delimited JSON), similar to how the Language Server Protocol (LSP)
standardized language-server integration.

This single-file module provides:

* **JSONRPCTransport** -- async read/write of newline-delimited JSON-RPC 2.0
  messages over arbitrary ``asyncio.StreamReader`` / ``asyncio.StreamWriter``
  pairs (typically stdin/stdout of a subprocess).

* **Protocol data types** -- pure-dataclass representations of every message
  and structure defined by the ACP specification (protocol versions 1 and 2).

* **ACPClient** -- high-level async helper that spawns an agent subprocess,
  performs the ``initialize`` handshake, creates sessions, sends prompts, and
  yields ``session/update`` notifications as an async iterator.

* **ACPAgent** -- abstract base class for implementing an ACP-compatible agent.
  Subclass it and override the ``on_*`` handler methods; call ``agent.run()``
  to start the stdio event loop.

Requires Python >= 3.10.  No third-party packages are needed -- only the
standard library (``asyncio``, ``json``, ``dataclasses``, ``enum``, ``typing``,
``sys``, ``abc``, ``uuid``, ``logging``).

Protocol versions: this module speaks ACP v1 and v2.  ``ACPClient`` opens
with its highest supported version (2) and retries once with the v1 field
names when the agent rejects it, so existing v1 agents keep working.  An
agent selects the highest version it supports at or below the client's
request; when none qualifies, initialization fails with ``-32022`` and
``{requested, supported}`` data.

Quickstart -- Client side::

    async def main():
        client = ACPClient(["python", "-m", "my_agent"])
        await client.start()
        init = await client.initialize()
        session = await client.new_session("/home/user/project")
        async for update in client.prompt(session.session_id, "Hello!"):
            print(update)
        await client.stop()

Quickstart -- Agent side::

    class EchoAgent(ACPAgent):
        async def on_initialize(self, params):
            return InitializeResult(protocol_version=1)

        async def on_new_session(self, params):
            return NewSessionResult(session_id="sess_1")

        async def on_prompt(self, params):
            text = ""
            for block in params.prompt:
                if isinstance(block, TextContent):
                    text = block.text
            await self.send_update(params.session_id,
                AgentMessageChunkUpdate(
                    content=TextContent(text=f"Echo: {text}")))
            return PromptResult(stop_reason=StopReason.END_TURN)

    if __name__ == "__main__":
        asyncio.run(EchoAgent().run())
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Union,
)

# ── Protocol version cap ───────────────────────────────────────────────────

#: Highest ACP protocol version implemented by this module.
#:
#: ACP wire compatibility is negotiated during ``initialize``. Bump this
#: constant only when a new major protocol version is implemented; additive
#: features within a version do not change it. See
#: https://agentclientprotocol.com/protocol/ for the current specification.
ACP_PROTOCOL_VERSION = 2

#: Protocol versions this module can negotiate, newest first.
SUPPORTED_PROTOCOL_VERSIONS: tuple[int, ...] = (2, 1)

#: ACP v2 error code for failed protocol-version negotiation.
UNSUPPORTED_PROTOCOL_VERSION = -32022

#: Errors that may mean a v1-only peer saw our v2 initialize request.
_VERSION_RETRY_CODES = frozenset({UNSUPPORTED_PROTOCOL_VERSION, -32600, -32602})

#: Date of the last audit against the published ACP v1 and v2 specifications.
#: Update this together with the feature lists below.
ACP_SPEC_REVISION = "2026-10-31"

#: Stable ACP methods implemented by this module (client or agent side).
ACP_IMPLEMENTED_METHODS = frozenset(
    {
        "initialize",
        "session/new",
        "session/load",
        "session/prompt",
        "session/cancel",
        "session/set_mode",
        "session/set_config_option",
        "session/list",
        "session/resume",
        "session/close",
        "session/delete",
        "authenticate",
        "logout",
        "auth/login",
        "auth/logout",
    }
)

#: ``session/update`` variants modeled by this module.
ACP_IMPLEMENTED_UPDATES = frozenset(
    {
        "agent_message_chunk",
        "user_message_chunk",
        "agent_thought_chunk",
        "tool_call",
        "tool_call_update",
        "plan",
        "available_commands_update",
        "current_mode_update",
        "config_option_update",
        "session_info_update",
        "usage_update",
        "notice",
        "compaction_update",
        "compaction_summary_chunk",
        # ACP v2 variants
        "user_message",
        "agent_message",
        "agent_thought",
        "state_update",
        "tool_call_content_chunk",
        "terminal_update",
        "terminal_output_chunk",
        "plan_update",
    }
)

#: Methods removed in ACP v2.  A v2 connection must reject them.
ACP_V2_REMOVED_METHODS = frozenset(
    {
        "authenticate",
        "logout",
        "session/load",
        "session/set_mode",
        "fs/read_text_file",
        "fs/write_text_file",
        "terminal/create",
        "terminal/output",
        "terminal/wait_for_exit",
        "terminal/kill",
        "terminal/release",
    }
)


def _ensure_sibling_path(name: str) -> str:
    """Add sibling module paths to ``sys.path`` for flat and nested layouts."""
    base = os.path.dirname(os.path.abspath(__file__))
    for candidate in [base, os.path.normpath(os.path.join(base, "..", name))]:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    return base


try:  # flat package layout: pkg.acp + pkg.jsonrpc
    from .jsonrpc import (  # ty: ignore[unresolved-import]  # noqa: E402
        INTERNAL_ERROR,
        METHOD_NOT_FOUND,
        JSONRPCError,
        JSONRPCException,
        JSONRPCTransport,
    )
except ImportError:
    try:  # nested package layout: pkg.acp.acp + pkg.jsonrpc
        from ..jsonrpc import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]  # noqa: E402
            INTERNAL_ERROR,
            METHOD_NOT_FOUND,
            JSONRPCError,
            JSONRPCException,
            JSONRPCTransport,
        )
    except ImportError:  # flat/nested sibling files on sys.path
        _ensure_sibling_path("jsonrpc")
        from jsonrpc import (  # noqa: E402
            INTERNAL_ERROR,
            METHOD_NOT_FOUND,
            JSONRPCError,
            JSONRPCException,
            JSONRPCTransport,
        )


class UnsupportedProtocolVersion(JSONRPCException):
    """Raised when a peer negotiates an ACP version this module does not support.

    ACP v2 defines error code ``-32022`` with ``{requested, supported}``
    data for failed version negotiation.
    """

    def __init__(self, version: Any) -> None:
        super().__init__(
            JSONRPCError(
                code=UNSUPPORTED_PROTOCOL_VERSION,
                message=(
                    f"Unsupported ACP protocol version {version!r}; "
                    f"supported versions: {SUPPORTED_PROTOCOL_VERSIONS}"
                ),
                data={
                    "requested": version,
                    "supported": list(SUPPORTED_PROTOCOL_VERSIONS),
                },
            )
        )
        self.version = version


__all__ = [
    # Protocol version
    "ACP_PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "ACP_SPEC_REVISION",
    "ACP_IMPLEMENTED_METHODS",
    "ACP_IMPLEMENTED_UPDATES",
    "ACP_V2_REMOVED_METHODS",
    "UNSUPPORTED_PROTOCOL_VERSION",
    "UnsupportedProtocolVersion",
    "negotiate_protocol_version",
    # JSON-RPC (re-exported from jsonrpc module)
    "JSONRPCTransport",
    "JSONRPCException",
    "JSONRPCError",
    # Enums
    "StopReason",
    "ToolKind",
    "ToolCallStatus",
    "PermissionOptionKind",
    "PlanEntryPriority",
    "PlanEntryStatus",
    # Content blocks
    "TextContent",
    "ImageContent",
    "AudioContent",
    "ResourceContent",
    "ResourceLinkContent",
    # Protocol types
    "ImplementationInfo",
    "FsCapabilities",
    "ClientCapabilities",
    "SessionClientCapabilities",
    "ElicitationCapabilities",
    "supports_update",
    "PromptCapabilities",
    "McpCapabilities",
    "SessionCapabilities",
    "AgentCapabilities",
    "AuthMethod",
    "InitializeParams",
    "InitializeResult",
    "EnvVariable",
    "McpServerStdio",
    "McpServerHttp",
    "HttpHeader",
    "NewSessionParams",
    "NewSessionResult",
    "LoadSessionParams",
    "ResumeSessionParams",
    "CloseSessionParams",
    "DeleteSessionParams",
    "AuthenticateParams",
    "LogoutParams",
    "PromptParams",
    "PromptResult",
    "CancelParams",
    "SessionMode",
    "SessionModeState",
    "SetModeParams",
    "ConfigOptionValue",
    "ConfigOption",
    "SetConfigOptionParams",
    "SetConfigOptionResult",
    "ListSessionsParams",
    "SessionInfo",
    "ListSessionsResult",
    "ToolCallLocation",
    "ToolCallContentItem",
    "DiffContent",
    "TerminalContent",
    "PermissionOption",
    "RequestPermissionParams",
    "RequestPermissionResult",
    "PermissionOutcome",
    "PlanEntry",
    "AvailableCommandInput",
    "AvailableCommand",
    # Session updates
    "AgentMessageChunkUpdate",
    "UserMessageChunkUpdate",
    "AgentThoughtChunkUpdate",
    "ThoughtMessageChunkUpdate",
    "ToolCallUpdate",
    "ToolCallStatusUpdate",
    "PlanUpdate",
    "AvailableCommandsUpdate",
    "CurrentModeUpdate",
    "ConfigOptionUpdate",
    "SessionInfoUpdate",
    "UsageUpdate",
    "NoticeUpdate",
    "CompactionUpdate",
    "CompactionSummaryChunkUpdate",
    "UserMessageUpdate",
    "AgentMessageUpdate",
    "AgentThoughtUpdate",
    "StateUpdate",
    "ToolCallContentChunkUpdate",
    "TerminalUpdate",
    "TerminalOutputChunkUpdate",
    "PlanItemsUpdate",
    # File system
    "ReadTextFileParams",
    "ReadTextFileResult",
    "WriteTextFileParams",
    # Terminal
    "CreateTerminalParams",
    "CreateTerminalResult",
    "TerminalOutputParams",
    "TerminalOutputResult",
    "TerminalExitStatus",
    "WaitForExitParams",
    "KillTerminalParams",
    "ReleaseTerminalParams",
    # Client / Agent
    "ACPClient",
    "ACPAgent",
]

__version__ = "0.3.7"

logger = logging.getLogger("acp")

# ── Helper: dataclass <-> dict conversion ──────────────────────────────────

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])([A-Z])")


def _to_camel(name: str) -> str:
    """Convert a snake_case name to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(w.capitalize() for w in parts[1:])


def _to_snake(name: str) -> str:
    """Convert a camelCase name to snake_case."""
    return _CAMEL_RE.sub(r"_\1", name).lower()


def to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass instance to a JSON-friendly dict.

    * ``None`` values and empty collections are omitted.
    * Snake-case field names are converted to camelCase per the ACP spec.
    * Enum values are serialized as their ``.value``.

    Args:
        obj: A dataclass instance, dict, list, or primitive.

    Returns:
        A JSON-serializable value.
    """
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    to_wire = getattr(obj, "__acp_to_dict__", None)
    if callable(to_wire):
        return to_wire()
    if hasattr(obj, "__dataclass_fields__"):
        result: dict[str, Any] = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            if val is None:
                continue
            if isinstance(val, (list, tuple)) and len(val) == 0:
                continue
            if isinstance(val, dict) and len(val) == 0:
                continue
            result[_to_camel(f.name)] = to_dict(val)
        return result
    return obj


def from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert camelCase keys in *raw* to snake_case.

    Args:
        raw: A dictionary with camelCase keys.

    Returns:
        A new dictionary with snake_case keys.
    """
    return {_to_snake(k): v for k, v in raw.items()}


def _enum_from_value(enum_cls: type[Enum], value: Any) -> Any:
    """Look up an enum member by its *value* string."""
    if isinstance(value, enum_cls):
        return value
    for member in enum_cls:
        if member.value == value:
            return member
    raise ValueError(f"Unknown {enum_cls.__name__} value: {value!r}")


# ── Enums ──────────────────────────────────────────────────────────────────


class StopReason(str, Enum):
    """Reason an agent stopped a prompt turn."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    MAX_TURN_REQUESTS = "max_turn_requests"
    REFUSAL = "refusal"
    CANCELLED = "cancelled"
    # ACP v2 adds an explicit error stop reason.
    ERROR = "error"
    # Local sentinel for a reason introduced by a later ACP revision.  Never
    # sent on the wire; ``PromptResult.raw_stop_reason`` keeps the original.
    UNKNOWN = "unknown"


class ToolKind(str, Enum):
    """Category of a tool being invoked."""

    READ = "read"
    EDIT = "edit"
    DELETE = "delete"
    MOVE = "move"
    SEARCH = "search"
    EXECUTE = "execute"
    THINK = "think"
    FETCH = "fetch"
    OTHER = "other"


class ToolCallStatus(str, Enum):
    """Execution status of a tool call."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    # ACP v2 adds an explicit cancelled status.
    CANCELLED = "cancelled"


class PermissionOptionKind(str, Enum):
    """Kind of permission option presented to the user."""

    ALLOW_ONCE = "allow_once"
    ALLOW_ALWAYS = "allow_always"
    REJECT_ONCE = "reject_once"
    REJECT_ALWAYS = "reject_always"


class PlanEntryPriority(str, Enum):
    """Priority of a plan entry."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PlanEntryStatus(str, Enum):
    """Execution status of a plan entry."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


# ── Content Blocks (shared with MCP) ───────────────────────────────────────


@dataclass
class TextContent:
    """Plain text content block.

    Attributes:
        text: The text payload.
        type: Discriminator (always ``"text"``).
    """

    text: str
    type: str = "text"


@dataclass
class ImageContent:
    """Base64-encoded image content block.

    Attributes:
        data: Base64-encoded image data.
        mime_type: MIME type such as ``"image/png"``.
        type: Discriminator (always ``"image"``).
    """

    data: str
    mime_type: str
    type: str = "image"


@dataclass
class AudioContent:
    """Base64-encoded audio content block.

    Attributes:
        data: Base64-encoded audio data.
        mime_type: MIME type such as ``"audio/wav"``.
        type: Discriminator (always ``"audio"``).
    """

    data: str
    mime_type: str
    type: str = "audio"


@dataclass
class _TextResource:
    """Embedded text resource."""

    uri: str
    text: str
    mime_type: str | None = None


@dataclass
class ResourceContent:
    """Embedded resource content block.

    Attributes:
        resource: The embedded resource (text or blob).
        type: Discriminator (always ``"resource"``).
    """

    resource: _TextResource
    type: str = "resource"


@dataclass
class ResourceLinkContent:
    """Reference to an external resource.

    Attributes:
        uri: URI of the resource.
        name: Human-readable resource name.
        mime_type: Optional MIME type.
        title: Optional display title.
        description: Optional description.
        size: Optional size in bytes.
        type: Discriminator (always ``"resource_link"``).
    """

    uri: str
    name: str
    mime_type: str | None = None
    title: str | None = None
    description: str | None = None
    size: int | None = None
    type: str = "resource_link"


ContentBlock = Union[
    TextContent,
    ImageContent,
    AudioContent,
    ResourceContent,
    ResourceLinkContent,
]


def _content_from_dict(raw: dict[str, Any]) -> ContentBlock:
    """Deserialize a content block from a raw dictionary."""
    t = raw.get("type", "text")
    if t == "text":
        return TextContent(text=raw["text"])
    if t == "image":
        return ImageContent(data=raw["data"], mime_type=raw["mimeType"])
    if t == "audio":
        return AudioContent(data=raw["data"], mime_type=raw["mimeType"])
    if t == "resource":
        res = raw["resource"]
        return ResourceContent(
            resource=_TextResource(
                uri=res["uri"],
                text=res.get("text", ""),
                mime_type=res.get("mimeType"),
            )
        )
    if t == "resource_link":
        return ResourceLinkContent(
            uri=raw["uri"],
            name=raw["name"],
            mime_type=raw.get("mimeType"),
            title=raw.get("title"),
            description=raw.get("description"),
            size=raw.get("size"),
        )
    # Fallback -- treat unknown as text
    return TextContent(text=str(raw))


# ── Protocol Types -- Initialization ───────────────────────────────────────


@dataclass
class ImplementationInfo:
    """Information about a client or agent implementation.

    Attributes:
        name: Programmatic identifier.
        version: Version string.
        title: Human-readable display name.
    """

    name: str
    version: str = ""
    title: str | None = None


@dataclass
class FsCapabilities:
    """Client file-system capabilities.

    Attributes:
        read_text_file: Whether ``fs/read_text_file`` is available.
        write_text_file: Whether ``fs/write_text_file`` is available.
    """

    read_text_file: bool = False
    write_text_file: bool = False


@dataclass
class SessionClientCapabilities:
    """Client support for session-level notifications.

    Attributes:
        notices: If present, ``notice`` updates are supported.
        compaction: If present, compaction updates are supported.
        config_options: If present, boolean config options are supported.
    """

    notices: dict[str, Any] | None = None
    compaction: dict[str, Any] | None = None
    config_options: dict[str, Any] | None = None


@dataclass
class ElicitationCapabilities:
    """Client elicitation capabilities.

    Attributes:
        form: If present, form elicitation is supported.
        url: If present, URL elicitation is supported.
    """

    form: dict[str, Any] | None = None
    url: dict[str, Any] | None = None


@dataclass
class ClientCapabilities:
    """Capabilities supported by the client.

    Attributes:
        fs: File-system method availability.
        terminal: Whether all ``terminal/*`` methods are available.
        session: Session-level notification capabilities.
        elicitation: Structured user-input capabilities.
        auth: Authentication-related client capabilities.
    """

    fs: FsCapabilities | None = None
    terminal: bool = False
    session: SessionClientCapabilities | None = None
    elicitation: ElicitationCapabilities | None = None
    auth: dict[str, Any] | None = None


_CAPABILITY_GATED_UPDATES: dict[str, str] = {
    "notice": "notices",
    "compaction_update": "compaction",
    "compaction_summary_chunk": "compaction",
}


def supports_update(caps: ClientCapabilities | None, session_update: str) -> bool:
    """Return whether a client advertising *caps* accepts *session_update*.

    Some session updates are gated on a client capability and must not be
    sent otherwise.  Ungated updates always return ``True``.

    Args:
        caps: Client capabilities from ``initialize`` (or ``None``).
        session_update: Wire value of the ``sessionUpdate`` discriminator.

    Returns:
        ``True`` if the update may be sent to this client.
    """
    gate = _CAPABILITY_GATED_UPDATES.get(session_update)
    if gate is None:
        return True
    if caps is None or caps.session is None:
        return False
    return getattr(caps.session, gate) is not None


@dataclass
class PromptCapabilities:
    """Content types the agent supports in prompts.

    Attributes:
        image: Whether image content is supported.
        audio: Whether audio content is supported.
        embedded_context: Whether embedded resource content is supported.
    """

    image: bool = False
    audio: bool = False
    embedded_context: bool = False


@dataclass
class McpCapabilities:
    """MCP transport capabilities.

    Attributes:
        stdio: Whether stdio MCP transport is supported.
        http: Whether HTTP MCP transport is supported.
        sse: Whether SSE MCP transport is supported (ACP v1 only).
    """

    stdio: bool = False
    http: bool = False
    sse: bool = False


@dataclass
class SessionListCapability:
    """Marker for session/list support."""

    pass


@dataclass
class SessionCapabilities:
    """Session-level capabilities.

    Attributes:
        list: If present, ``session/list`` is supported.
        resume: If present, ``session/resume`` is supported.
        close: If present, ``session/close`` is supported.
        delete: If present, ``session/delete`` is supported.
        additional_directories: If present, additional workspace roots are
            supported on session lifecycle requests.
    """

    list: SessionListCapability | None = None
    resume: dict[str, Any] | None = None
    close: dict[str, Any] | None = None
    delete: dict[str, Any] | None = None
    additional_directories: dict[str, Any] | None = None


@dataclass
class AgentCapabilities:
    """Capabilities supported by the agent.

    Attributes:
        load_session: Whether ``session/load`` is available.
        prompt_capabilities: Supported content types in prompts.
        mcp_capabilities: Supported MCP transports.
        session_capabilities: Session-level capabilities.
        auth: Authentication capabilities (for example ``logout``).
    """

    load_session: bool = False
    prompt_capabilities: PromptCapabilities | None = None
    mcp_capabilities: McpCapabilities | None = None
    session_capabilities: SessionCapabilities | None = None
    auth: dict[str, Any] | None = None


@dataclass
class AuthMethod:
    """An authentication method advertised by the agent.

    Attributes:
        id: Unique identifier for this auth method.
        name: Human-readable name.
        description: Optional description.
        type: Optional method kind (for example ``"agent"`` or ``"terminal"``).
    """

    id: str
    name: str
    description: str | None = None
    type: str | None = None


@dataclass
class InitializeParams:
    """Parameters for the ``initialize`` request.

    Attributes:
        protocol_version: Latest protocol version the client supports.
        client_capabilities: Capabilities the client supports.
        client_info: Information about the client implementation.
    """

    protocol_version: int = ACP_PROTOCOL_VERSION
    client_capabilities: ClientCapabilities | None = None
    client_info: ImplementationInfo | None = None


@dataclass
class InitializeResult:
    """Result of the ``initialize`` request.

    Attributes:
        protocol_version: Negotiated protocol version.
        agent_capabilities: Capabilities the agent supports.
        agent_info: Information about the agent implementation.
        auth_methods: Available authentication methods.
    """

    protocol_version: int = 1
    agent_capabilities: AgentCapabilities | None = None
    agent_info: ImplementationInfo | None = None
    auth_methods: list[AuthMethod] | None = None


# ── Protocol Types -- Session Setup ────────────────────────────────────────


@dataclass
class EnvVariable:
    """An environment variable.

    Attributes:
        name: Variable name.
        value: Variable value.
    """

    name: str
    value: str


@dataclass
class McpServerStdio:
    """Stdio MCP server specification.

    Attributes:
        name: Human-readable server name.
        command: Path to the MCP server executable.
        args: Command-line arguments.
        env: Environment variables.
    """

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: list[EnvVariable] | None = None


@dataclass
class HttpHeader:
    """An HTTP header.

    Attributes:
        name: Header name.
        value: Header value.
    """

    name: str
    value: str


@dataclass
class McpServerHttp:
    """HTTP MCP server specification.

    Attributes:
        name: Human-readable server name.
        url: URL of the MCP server.
        headers: HTTP headers.
        type: Transport type (``"http"`` or ``"sse"``).
    """

    name: str
    url: str
    headers: list[HttpHeader] = field(default_factory=list)
    type: str = "http"


McpServer = Union[McpServerStdio, McpServerHttp]


@dataclass
class NewSessionParams:
    """Parameters for ``session/new``.

    Attributes:
        cwd: Absolute path to the working directory.
        mcp_servers: MCP servers to connect to.
        additional_directories: Additional workspace roots.
    """

    cwd: str
    mcp_servers: list[dict[str, Any]] | None = None
    additional_directories: list[str] | None = None


@dataclass
class SessionMode:
    """An operating mode for the agent.

    Attributes:
        id: Unique mode identifier.
        name: Human-readable mode name.
        description: Optional description.
    """

    id: str
    name: str
    description: str | None = None


@dataclass
class SessionModeState:
    """Current mode state for a session.

    Attributes:
        current_mode_id: The currently active mode.
        available_modes: All available modes.
    """

    current_mode_id: str
    available_modes: list[SessionMode] = field(default_factory=list)


@dataclass
class ConfigOptionValue:
    """A possible value for a configuration option.

    Attributes:
        value: Value identifier.
        name: Human-readable name.
        description: Optional description.
    """

    value: str
    name: str
    description: str | None = None


@dataclass
class ConfigOption:
    """A session configuration option.

    Attributes:
        id: Unique option identifier.
        name: Human-readable label.
        type: Input control type (currently only ``"select"``).
        current_value: Currently selected value.
        options: Available values.
        description: Optional description.
        category: Optional semantic category.
    """

    id: str
    name: str
    type: str
    current_value: str
    options: list[ConfigOptionValue] = field(default_factory=list)
    description: str | None = None
    category: str | None = None


@dataclass
class NewSessionResult:
    """Result of ``session/new``.

    Attributes:
        session_id: Unique identifier for the created session.
        modes: Optional mode state.
        config_options: Optional configuration options.
    """

    session_id: str
    modes: SessionModeState | None = None
    config_options: list[ConfigOption] | None = None
    available_commands: list[AvailableCommand] | None = None


@dataclass
class LoadSessionParams:
    """Parameters for ``session/load``.

    Attributes:
        session_id: Session to resume.
        cwd: Working directory.
        mcp_servers: MCP servers to connect to.
        additional_directories: Additional workspace roots.
    """

    session_id: str
    cwd: str
    mcp_servers: list[dict[str, Any]] | None = None
    additional_directories: list[str] | None = None


@dataclass
class ResumeSessionParams:
    """Parameters for ``session/resume``.

    Attributes:
        session_id: Session to resume.
        cwd: Working directory.
        mcp_servers: MCP servers to connect to.
        additional_directories: Additional workspace roots.
    """

    session_id: str
    cwd: str
    mcp_servers: list[dict[str, Any]] | None = None
    additional_directories: list[str] | None = None
    replay_from: dict[str, Any] | None = None


@dataclass
class CloseSessionParams:
    """Parameters for ``session/close``.

    Attributes:
        session_id: Session to close.
    """

    session_id: str


@dataclass
class DeleteSessionParams:
    """Parameters for ``session/delete``.

    Attributes:
        session_id: Session to delete.
    """

    session_id: str


@dataclass
class AuthenticateParams:
    """Parameters for ``authenticate``.

    Attributes:
        method_id: Authentication method id advertised by the agent.
    """

    method_id: str


@dataclass
class LogoutParams:
    """Parameters for ``logout`` (currently empty)."""


# ── Protocol Types -- Prompt Turn ──────────────────────────────────────────


@dataclass
class PromptParams:
    """Parameters for ``session/prompt``.

    Attributes:
        session_id: Target session.
        prompt: Content blocks forming the user message.
    """

    session_id: str
    prompt: list[ContentBlock] = field(default_factory=list)


@dataclass
class PromptResult:
    """Result of ``session/prompt``.

    Attributes:
        stop_reason: Why the agent stopped.  ACP v2 reports this in an
            ``idle`` ``state_update`` instead of the prompt response.
        raw_stop_reason: Original wire value when ``stop_reason`` is
            :attr:`StopReason.UNKNOWN`.
        message_id: ACP v2 user-message id acknowledged by the response.
    """

    stop_reason: StopReason
    raw_stop_reason: str | None = None
    message_id: str | None = None


@dataclass
class CancelParams:
    """Parameters for ``session/cancel`` notification.

    Attributes:
        session_id: Session to cancel.
    """

    session_id: str


# ── Protocol Types -- Session Modes & Config ───────────────────────────────


@dataclass
class SetModeParams:
    """Parameters for ``session/set_mode``.

    Attributes:
        session_id: Target session.
        mode_id: Mode to switch to.
    """

    session_id: str
    mode_id: str


@dataclass
class SetConfigOptionParams:
    """Parameters for ``session/set_config_option``.

    Attributes:
        session_id: Target session.
        config_id: Configuration option id.
        value: New value.
    """

    session_id: str
    config_id: str
    value: Any


@dataclass
class SetConfigOptionResult:
    """Result of ``session/set_config_option``.

    Attributes:
        config_options: Complete list of config options with current values.
    """

    config_options: list[ConfigOption] = field(default_factory=list)


# ── Protocol Types -- Session List ─────────────────────────────────────────


@dataclass
class ListSessionsParams:
    """Parameters for ``session/list``.

    Attributes:
        cwd: Optional directory filter.
        cursor: Optional pagination cursor.
    """

    cwd: str | None = None
    cursor: str | None = None


@dataclass
class SessionInfo:
    """Metadata about an existing session.

    Attributes:
        session_id: Unique session identifier.
        cwd: Working directory.
        title: Optional human-readable title.
        updated_at: Optional ISO 8601 timestamp.
    """

    session_id: str
    cwd: str
    title: str | None = None
    updated_at: str | None = None


@dataclass
class ListSessionsResult:
    """Result of ``session/list``.

    Attributes:
        sessions: List of session metadata.
        next_cursor: Pagination cursor for the next page.
    """

    sessions: list[SessionInfo] = field(default_factory=list)
    next_cursor: str | None = None


# ── Protocol Types -- Tool Calls ───────────────────────────────────────────


@dataclass
class ToolCallLocation:
    """File location affected by a tool call.

    Attributes:
        path: Absolute file path.
        line: Optional line number (1-based).
    """

    path: str
    line: int | None = None


@dataclass
class DiffContent:
    """A file diff produced by a tool call.

    Attributes:
        path: Absolute path of the file being modified.
        new_text: New content after modification.
        old_text: Original content (``None`` for new files).
        type: Discriminator (always ``"diff"``).
    """

    path: str
    new_text: str
    old_text: str | None = None
    type: str = "diff"


@dataclass
class TerminalContent:
    """Reference to terminal output embedded in a tool call.

    Attributes:
        terminal_id: Id of the terminal created with ``terminal/create``.
        type: Discriminator (always ``"terminal"``).
    """

    terminal_id: str
    type: str = "terminal"


@dataclass
class ToolCallContentItem:
    """A content item within a tool call (wraps a ContentBlock).

    Attributes:
        content: The wrapped content block.
        type: Discriminator (always ``"content"``).
    """

    content: ContentBlock
    type: str = "content"


ToolCallContent = Union[ToolCallContentItem, DiffContent, TerminalContent]


@dataclass
class PermissionOption:
    """A permission option presented to the user.

    Attributes:
        option_id: Unique option identifier.
        name: Human-readable label.
        kind: Permission kind hint.
    """

    option_id: str
    name: str
    kind: PermissionOptionKind


@dataclass
class PermissionOutcome:
    """Outcome of a permission request.

    Attributes:
        outcome: ``"selected"`` or ``"cancelled"``.
        option_id: The selected option id (if ``outcome == "selected"``).
    """

    outcome: str
    option_id: str | None = None


@dataclass
class RequestPermissionParams:
    """Parameters for ``session/request_permission`` (agent -> client).

    Attributes:
        session_id: Target session.
        tool_call: Tool call update with details about the operation.
        options: Available permission options.
    """

    session_id: str
    tool_call: dict[str, Any] = field(default_factory=dict)
    options: list[PermissionOption] = field(default_factory=list)


@dataclass
class RequestPermissionResult:
    """Result of ``session/request_permission``.

    Attributes:
        outcome: The user's decision.
    """

    outcome: PermissionOutcome


# ── Protocol Types -- Plans ────────────────────────────────────────────────


@dataclass
class PlanEntry:
    """A single entry in an agent's execution plan.

    Attributes:
        content: Human-readable task description.
        priority: Relative importance.
        status: Current execution status.
    """

    content: str
    priority: PlanEntryPriority
    status: PlanEntryStatus


# ── Protocol Types -- Slash Commands ───────────────────────────────────────


@dataclass
class AvailableCommandInput:
    """Input specification for a slash command.

    Attributes:
        hint: Placeholder text shown when no input has been provided.
    """

    hint: str


@dataclass
class AvailableCommand:
    """A slash command advertised by the agent.

    Attributes:
        name: Command name (e.g. ``"web"``).
        description: Human-readable description.
        input: Optional input specification.
    """

    name: str
    description: str
    input: AvailableCommandInput | None = None


# ── Protocol Types -- File System (client methods) ─────────────────────────


@dataclass
class ReadTextFileParams:
    """Parameters for ``fs/read_text_file`` (agent -> client).

    Attributes:
        session_id: Target session.
        path: Absolute file path.
        line: Optional start line (1-based).
        limit: Optional max number of lines.
    """

    session_id: str
    path: str
    line: int | None = None
    limit: int | None = None


@dataclass
class ReadTextFileResult:
    """Result of ``fs/read_text_file``.

    Attributes:
        content: File text content.
    """

    content: str


@dataclass
class WriteTextFileParams:
    """Parameters for ``fs/write_text_file`` (agent -> client).

    Attributes:
        session_id: Target session.
        path: Absolute file path.
        content: Text content to write.
    """

    session_id: str
    path: str
    content: str


# ── Protocol Types -- Terminals (client methods) ───────────────────────────


@dataclass
class CreateTerminalParams:
    """Parameters for ``terminal/create`` (agent -> client).

    Attributes:
        session_id: Target session.
        command: Command to execute.
        args: Command arguments.
        env: Environment variables.
        cwd: Working directory (absolute path).
        output_byte_limit: Max bytes of output to retain.
    """

    session_id: str
    command: str
    args: list[str] | None = None
    env: list[EnvVariable] | None = None
    cwd: str | None = None
    output_byte_limit: int | None = None


@dataclass
class CreateTerminalResult:
    """Result of ``terminal/create``.

    Attributes:
        terminal_id: Unique terminal identifier.
    """

    terminal_id: str


@dataclass
class TerminalOutputParams:
    """Parameters for ``terminal/output``.

    Attributes:
        session_id: Target session.
        terminal_id: Terminal to query.
    """

    session_id: str
    terminal_id: str


@dataclass
class TerminalExitStatus:
    """Terminal process exit status.

    Attributes:
        exit_code: Process exit code (may be ``None``).
        signal: Termination signal (may be ``None``).
    """

    exit_code: int | None = None
    signal: str | None = None


@dataclass
class TerminalOutputResult:
    """Result of ``terminal/output``.

    Attributes:
        output: Captured terminal output.
        truncated: Whether output was truncated.
        exit_status: Present only if the command has exited.
    """

    output: str = ""
    truncated: bool = False
    exit_status: TerminalExitStatus | None = None


@dataclass
class WaitForExitParams:
    """Parameters for ``terminal/wait_for_exit``.

    Attributes:
        session_id: Target session.
        terminal_id: Terminal to wait on.
    """

    session_id: str
    terminal_id: str


@dataclass
class KillTerminalParams:
    """Parameters for ``terminal/kill``.

    Attributes:
        session_id: Target session.
        terminal_id: Terminal to kill.
    """

    session_id: str
    terminal_id: str


@dataclass
class ReleaseTerminalParams:
    """Parameters for ``terminal/release``.

    Attributes:
        session_id: Target session.
        terminal_id: Terminal to release.
    """

    session_id: str
    terminal_id: str


# ── Session Update Types (agent -> client notifications) ───────────────────


@dataclass
class AgentMessageChunkUpdate:
    """Agent text output chunk.

    Attributes:
        content: The content block.
        session_update: Discriminator.
    """

    content: ContentBlock
    session_update: str = "agent_message_chunk"


@dataclass
class UserMessageChunkUpdate:
    """Replayed user message chunk (used in ``session/load``).

    Attributes:
        content: The content block.
        session_update: Discriminator.
    """

    content: ContentBlock
    session_update: str = "user_message_chunk"


@dataclass
class AgentThoughtChunkUpdate:
    """Agent internal reasoning chunk.

    Attributes:
        content: The content block.
        session_update: Discriminator (``agent_thought_chunk``).
    """

    content: ContentBlock
    session_update: str = "agent_thought_chunk"


# Backward-compatible alias for the pre-0.4 class name.
ThoughtMessageChunkUpdate = AgentThoughtChunkUpdate


@dataclass
class ToolCallUpdate:
    """Initial tool call notification.

    Attributes:
        tool_call_id: Unique tool call identifier.
        title: Human-readable description.
        kind: Tool category.
        status: Current execution status.
        content: Tool call content items.
        locations: File locations affected.
        raw_input: Raw input parameters.
        raw_output: Raw output.
        session_update: Discriminator.
    """

    tool_call_id: str
    title: str
    kind: ToolKind = ToolKind.OTHER
    status: ToolCallStatus = ToolCallStatus.PENDING
    content: list[ToolCallContent] | None = None
    locations: list[ToolCallLocation] | None = None
    raw_input: dict[str, Any] | None = None
    raw_output: dict[str, Any] | None = None
    session_update: str = "tool_call"


@dataclass
class ToolCallStatusUpdate:
    """Tool call progress/result update.

    Attributes:
        tool_call_id: The tool call being updated.
        status: New status.
        content: Optional new content.
        title: Optional updated title.
        locations: Optional updated locations.
        session_update: Discriminator.
    """

    tool_call_id: str
    status: ToolCallStatus | None = None
    content: list[ToolCallContent] | None = None
    title: str | None = None
    locations: list[ToolCallLocation] | None = None
    session_update: str = "tool_call_update"


@dataclass
class PlanUpdate:
    """Agent execution plan.

    Attributes:
        entries: Plan entries.
        session_update: Discriminator.
    """

    entries: list[PlanEntry] = field(default_factory=list)
    session_update: str = "plan"


@dataclass
class AvailableCommandsUpdate:
    """Update to available slash commands.

    Attributes:
        available_commands: Current list of commands.
        session_update: Discriminator.
    """

    available_commands: list[AvailableCommand] = field(default_factory=list)
    session_update: str = "available_commands_update"


@dataclass
class CurrentModeUpdate:
    """Notification that the agent changed its mode.

    Attributes:
        mode_id: New mode identifier.
        session_update: Discriminator.
    """

    mode_id: str
    session_update: str = "current_mode_update"


@dataclass
class ConfigOptionUpdate:
    """Notification that config options changed.

    Attributes:
        config_options: Complete configuration state.
        session_update: Discriminator.
    """

    config_options: list[ConfigOption] = field(default_factory=list)
    session_update: str = "config_option_update"


@dataclass
class SessionInfoUpdate:
    """Update to session metadata.

    Attributes:
        title: Updated session title.
        updated_at: Updated timestamp.
        session_update: Discriminator.
    """

    title: str | None = None
    updated_at: str | None = None
    session_update: str = "session_info_update"


@dataclass
class UsageUpdate:
    """Session context usage and cumulative cost.

    Attributes:
        used: Tokens currently used in the session context.
        size: Context window size in tokens.
        cost: Optional ``{amount, currency}`` cost object.
        session_update: Discriminator.
    """

    used: int
    size: int
    cost: dict[str, Any] | None = None
    session_update: str = "usage_update"


@dataclass
class NoticeUpdate:
    """Live user-facing notice.

    Attributes:
        severity: ``info``, ``warning``, ``error``, or an open value.
        title: Short plain-text title.
        description: Optional supporting detail.
        session_update: Discriminator.
    """

    severity: str
    title: str
    description: str | None = None
    session_update: str = "notice"


@dataclass
class CompactionUpdate:
    """Context compaction status update.

    Attributes:
        compaction_id: Opaque compaction identifier.
        status: ``in_progress``, ``completed``, ``failed``, or ``cancelled``.
        summary: Optional retained summary content.
        error: Optional human-readable error.
        session_update: Discriminator.
    """

    compaction_id: str
    status: str
    summary: list[ContentBlock] | None = None
    error: str | None = None
    session_update: str = "compaction_update"


@dataclass
class CompactionSummaryChunkUpdate:
    """Incremental chunk of a compaction summary.

    Attributes:
        compaction_id: Opaque compaction identifier.
        content: Appended content block.
        session_update: Discriminator.
    """

    compaction_id: str
    content: ContentBlock
    session_update: str = "compaction_summary_chunk"


# ── ACP v2 session updates ─────────────────────────────────────────────────


@dataclass
class UserMessageUpdate:
    """ACP v2 upsert of a user message.

    Attributes:
        message_id: Message id, unique among user messages in the session.
        content: Complete replacement content, or ``None`` to leave unchanged.
        session_update: Discriminator.
    """

    message_id: str
    content: list[ContentBlock] | None = None
    session_update: str = "user_message"


@dataclass
class AgentMessageUpdate:
    """ACP v2 upsert of an agent message.

    Attributes:
        message_id: Message id, unique among agent messages in the session.
        content: Complete replacement content, or ``None`` to leave unchanged.
        session_update: Discriminator.
    """

    message_id: str
    content: list[ContentBlock] | None = None
    session_update: str = "agent_message"


@dataclass
class AgentThoughtUpdate:
    """ACP v2 upsert of an agent thought message.

    Attributes:
        message_id: Message id, unique among agent thoughts in the session.
        content: Complete replacement content, or ``None`` to leave unchanged.
        session_update: Discriminator.
    """

    message_id: str
    content: list[ContentBlock] | None = None
    session_update: str = "agent_thought"


@dataclass
class StateUpdate:
    """ACP v2 foreground-activity snapshot.

    Attributes:
        state: ``running``, ``idle``, or ``requires_action``.
        stop_reason: Why foreground work stopped (on ``idle``).
        error: Optional error object for ``stop_reason == "error"``.
        session_update: Discriminator.
    """

    state: str
    stop_reason: str | None = None
    error: dict[str, Any] | None = None
    session_update: str = "state_update"


@dataclass
class ToolCallContentChunkUpdate:
    """ACP v2 streamed item of tool-call content.

    Attributes:
        tool_call_id: Tool call this content belongs to.
        content: One content item produced by the tool call.
        session_update: Discriminator.
    """

    tool_call_id: str
    content: ToolCallContent
    session_update: str = "tool_call_content_chunk"


@dataclass
class TerminalUpdate:
    """ACP v2 upsert of an agent-owned terminal.

    Attributes:
        terminal_id: Terminal identifier.
        command: Command being run.
        cwd: Absolute working directory.
        output: Authoritative replacement snapshot of terminal output.
        exit_status: Exit information; a concrete object marks exit.
        size: Terminal size the command runs with.
        session_update: Discriminator.
    """

    terminal_id: str
    command: str | None = None
    cwd: str | None = None
    output: dict[str, Any] | None = None
    exit_status: dict[str, Any] | None = None
    size: dict[str, Any] | None = None
    session_update: str = "terminal_update"


@dataclass
class TerminalOutputChunkUpdate:
    """ACP v2 chunk of bytes appended to a terminal's output.

    Attributes:
        terminal_id: Terminal receiving these bytes.
        data: Base64-encoded output bytes.
        session_update: Discriminator.
    """

    terminal_id: str
    data: str
    session_update: str = "terminal_output_chunk"


@dataclass
class PlanItemsUpdate:
    """ACP v2 plan update carrying structured entries.

    The wire shape nests the entries under ``plan``:
    ``{"plan": {"type": "items", "planId": ..., "entries": [...]}}``.

    Attributes:
        plan_id: Plan identifier.
        entries: Complete list of plan entries.
        session_update: Discriminator.
    """

    plan_id: str
    entries: list[PlanEntry] = field(default_factory=list)
    session_update: str = "plan_update"

    def __acp_to_dict__(self) -> dict[str, Any]:
        """Serialize to the ACP v2 nested ``plan`` shape."""
        return {
            "sessionUpdate": self.session_update,
            "plan": {
                "type": "items",
                "planId": self.plan_id,
                "entries": [to_dict(e) for e in self.entries],
            },
        }


SessionUpdate = Union[
    AgentMessageChunkUpdate,
    UserMessageChunkUpdate,
    AgentThoughtChunkUpdate,
    ToolCallUpdate,
    ToolCallStatusUpdate,
    PlanUpdate,
    AvailableCommandsUpdate,
    CurrentModeUpdate,
    ConfigOptionUpdate,
    SessionInfoUpdate,
    UsageUpdate,
    NoticeUpdate,
    CompactionUpdate,
    CompactionSummaryChunkUpdate,
    UserMessageUpdate,
    AgentMessageUpdate,
    AgentThoughtUpdate,
    StateUpdate,
    ToolCallContentChunkUpdate,
    TerminalUpdate,
    TerminalOutputChunkUpdate,
    PlanItemsUpdate,
]


# ── Parsing helpers ────────────────────────────────────────────────────────


def negotiate_protocol_version(requested: int) -> int:
    """Return the protocol version to use for an ``initialize`` response.

    ACP v2 requires the agent to select the highest version it supports
    that is at or below the version requested by the client.  When no
    supported version qualifies, negotiation fails with
    :class:`UnsupportedProtocolVersion` (error code ``-32022``).

    Args:
        requested: Protocol version requested by the client.

    Returns:
        The negotiated protocol version.

    Raises:
        UnsupportedProtocolVersion: If *requested* is malformed or no
            supported version is at or below it.
    """
    if isinstance(requested, bool) or not isinstance(requested, (int, float, str)):
        raise UnsupportedProtocolVersion(requested)
    try:
        version = int(requested)
    except (TypeError, ValueError):
        raise UnsupportedProtocolVersion(requested) from None
    if isinstance(requested, float) and requested != version:
        raise UnsupportedProtocolVersion(requested)
    candidates = [v for v in SUPPORTED_PROTOCOL_VERSIONS if v <= version]
    if not candidates:
        raise UnsupportedProtocolVersion(requested)
    return max(candidates)


def _require_supported_version(value: Any) -> int:
    """Return *value* as a supported int version, or raise.

    Raises:
        UnsupportedProtocolVersion: If *value* is not an integer version in
            :data:`SUPPORTED_PROTOCOL_VERSIONS`.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise UnsupportedProtocolVersion(value)
    try:
        version = int(value)
    except (TypeError, ValueError):
        raise UnsupportedProtocolVersion(value) from None
    if version != value or version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise UnsupportedProtocolVersion(value)
    return version


def _parse_implementation_info(raw: Any) -> ImplementationInfo | None:
    """Parse an implementation info object from wire format."""
    if not isinstance(raw, dict) or not raw:
        return None
    return ImplementationInfo(
        name=raw.get("name", ""),
        version=raw.get("version", ""),
        title=raw.get("title"),
    )


def _parse_agent_capabilities(raw: Any) -> AgentCapabilities | None:
    """Parse agent capabilities from an ``initialize`` response.

    Accepts ACP v1 (``loadSession``, ``promptCapabilities``,
    ``mcpCapabilities``, ``sessionCapabilities``) and ACP v2 (``session``,
    ``auth``) shapes.
    """
    if not isinstance(raw, dict) or not raw:
        return None
    r = from_raw(raw)

    # ACP v2: session capabilities are nested under ``session`` and use
    # presence-based objects instead of v1's boolean/marker mix.
    v2_session = r.get("session")
    if isinstance(v2_session, dict):
        prompt_raw = v2_session.get("prompt") or {}
        mcp_raw = v2_session.get("mcp") or {}
        prompt_caps = (
            PromptCapabilities(
                image=bool(prompt_raw.get("image", False)),
                audio=bool(prompt_raw.get("audio", False)),
                embedded_context=bool(prompt_raw.get("embeddedContext", False)),
            )
            if prompt_raw
            else None
        )
        mcp_caps = (
            McpCapabilities(
                stdio=bool(mcp_raw.get("stdio", False)),
                http=bool(mcp_raw.get("http", False)),
            )
            if mcp_raw
            else None
        )
        auth_raw = r.get("auth")
        return AgentCapabilities(
            prompt_capabilities=prompt_caps,
            mcp_capabilities=mcp_caps,
            session_capabilities=SessionCapabilities(
                list=SessionListCapability(),
                resume={},
                close={},
                delete=v2_session.get("delete"),
                additional_directories=v2_session.get("additionalDirectories"),
            ),
            auth=from_raw(auth_raw) if isinstance(auth_raw, dict) else None,
        )

    prompt_raw = r.get("prompt_capabilities") or {}
    mcp_raw = r.get("mcp_capabilities") or {}
    session_raw = r.get("session_capabilities") or {}

    prompt_caps = None
    if prompt_raw:
        prompt_caps = PromptCapabilities(
            image=bool(prompt_raw.get("image", False)),
            audio=bool(prompt_raw.get("audio", False)),
            embedded_context=bool(prompt_raw.get("embeddedContext", False)),
        )

    mcp_caps = None
    if mcp_raw:
        mcp_caps = McpCapabilities(
            stdio=bool(mcp_raw.get("stdio", False)),
            http=bool(mcp_raw.get("http", False)),
            sse=bool(mcp_raw.get("sse", False)),
        )

    session_caps = None
    if session_raw:
        session_caps = SessionCapabilities(
            list=SessionListCapability() if "list" in session_raw else None,
            resume=session_raw.get("resume"),
            close=session_raw.get("close"),
            delete=session_raw.get("delete"),
            additional_directories=session_raw.get("additionalDirectories"),
        )

    auth_raw = r.get("auth")
    return AgentCapabilities(
        load_session=bool(r.get("load_session", False)),
        prompt_capabilities=prompt_caps,
        mcp_capabilities=mcp_caps,
        session_capabilities=session_caps,
        auth=from_raw(auth_raw) if isinstance(auth_raw, dict) else None,
    )


def _parse_session_modes(raw: Any) -> SessionModeState | None:
    """Parse ``modes`` from a ``session/new`` response."""
    if not isinstance(raw, dict) or not raw:
        return None
    modes = raw.get("availableModes", raw.get("available_modes", []))
    return SessionModeState(
        current_mode_id=raw.get("currentModeId", raw.get("current_mode_id", "")),
        available_modes=[
            SessionMode(
                id=m.get("id", ""),
                name=m.get("name", ""),
                description=m.get("description"),
            )
            for m in modes
            if isinstance(m, dict)
        ],
    )


def _parse_config_options(raw: Any) -> list[ConfigOption] | None:
    """Parse ``configOptions`` from a session response."""
    if not isinstance(raw, list):
        return None
    result: list[ConfigOption] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        result.append(
            ConfigOption(
                id=item.get("configId", item.get("id", "")),
                name=item.get("name", ""),
                type=item.get("type", ""),
                current_value=item.get("currentValue", item.get("current_value", "")),
                options=[
                    ConfigOptionValue(
                        value=v.get("value", ""),
                        name=v.get("name", ""),
                        description=v.get("description"),
                    )
                    for v in item.get("options", [])
                    if isinstance(v, dict)
                ],
                description=item.get("description"),
                category=item.get("category"),
            )
        )
    return result or None


def _parse_available_commands(raw: Any) -> list[AvailableCommand] | None:
    """Parse ``availableCommands`` from a session response."""
    if not isinstance(raw, list):
        return None
    result: list[AvailableCommand] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        input_raw = item.get("input")
        result.append(
            AvailableCommand(
                name=item.get("name", ""),
                description=item.get("description", ""),
                input=(
                    AvailableCommandInput(hint=input_raw.get("hint", ""))
                    if isinstance(input_raw, dict)
                    else None
                ),
            )
        )
    return result or None


def _parse_client_capabilities(raw: Any) -> ClientCapabilities | None:
    """Parse ``clientCapabilities`` from an ``initialize`` request."""
    if not isinstance(raw, dict):
        return None
    cc = raw or {}
    fs = None
    if "fs" in cc:
        f = cc["fs"] or {}
        fs = FsCapabilities(
            read_text_file=f.get("readTextFile", False),
            write_text_file=f.get("writeTextFile", False),
        )
    session_caps = None
    if "session" in cc:
        s = cc["session"] or {}
        session_caps = SessionClientCapabilities(
            notices=s.get("notices"),
            compaction=s.get("compaction"),
            config_options=s.get("configOptions"),
        )
    elicitation = None
    if "elicitation" in cc:
        e = cc["elicitation"] or {}
        elicitation = ElicitationCapabilities(
            form=e.get("form"),
            url=e.get("url"),
        )
    auth = cc.get("auth")
    return ClientCapabilities(
        fs=fs,
        terminal=cc.get("terminal", False),
        session=session_caps,
        elicitation=elicitation,
        auth=from_raw(auth) if isinstance(auth, dict) else None,
    )


def _parse_initialize_params(params: dict[str, Any]) -> InitializeParams:
    """Parse an ``initialize`` request into :class:`InitializeParams`.

    Accepts both ACP v1 field names (``clientInfo``/``clientCapabilities``)
    and ACP v2 names (``info``/``capabilities``).
    """
    rp = from_raw(params)
    client_info = None
    info_raw = rp.get("info", rp.get("client_info"))
    if isinstance(info_raw, dict):
        client_info = ImplementationInfo(
            name=info_raw.get("name", ""),
            version=info_raw.get("version", ""),
            title=info_raw.get("title"),
        )
    client_caps = None
    caps_raw = rp.get("capabilities", rp.get("client_capabilities"))
    if isinstance(caps_raw, dict):
        client_caps = _parse_client_capabilities(caps_raw)
    return InitializeParams(
        protocol_version=rp.get("protocol_version", ACP_PROTOCOL_VERSION),
        client_capabilities=client_caps,
        client_info=client_info,
    )


def _parse_prompt_result(raw: Any) -> PromptResult:
    """Parse a ``session/prompt`` response into a :class:`PromptResult`."""
    if not isinstance(raw, dict):
        return PromptResult(stop_reason=StopReason.END_TURN)
    message_id = raw.get("messageId", raw.get("message_id"))
    if message_id is not None and "stopReason" not in raw and "stop_reason" not in raw:
        # ACP v2 acknowledges insertion with a message id; the stop reason
        # arrives in an idle ``state_update``.
        return PromptResult(
            stop_reason=StopReason.UNKNOWN,
            message_id=str(message_id),
        )
    value = raw.get("stopReason", raw.get("stop_reason", "end_turn"))
    try:
        return PromptResult(
            stop_reason=StopReason(value),
            message_id=str(message_id) if message_id is not None else None,
        )
    except ValueError:
        # A future ACP revision may add stop reasons.  Surface them as UNKNOWN
        # instead of claiming a normal end of turn, and keep the raw value.
        return PromptResult(
            stop_reason=StopReason.UNKNOWN,
            raw_stop_reason=str(value),
            message_id=str(message_id) if message_id is not None else None,
        )


# ── Wire serialization (v1 / v2) ───────────────────────────────────────────


def _default_implementation() -> ImplementationInfo:
    """Return the default implementation info advertised on v2 ``initialize``."""
    return ImplementationInfo(name="zerodep-acp", version=__version__)


def _client_capabilities_to_wire(
    caps: ClientCapabilities | None, version: int
) -> dict[str, Any]:
    """Serialize client capabilities for the negotiated *version*."""
    if version < 2:
        return to_dict(caps) or {}
    payload: dict[str, Any] = {}
    if caps is not None:
        if caps.auth is not None:
            payload["auth"] = caps.auth
        if caps.elicitation is not None:
            payload["elicitation"] = to_dict(caps.elicitation)
    return payload


def _agent_capabilities_to_wire(
    caps: AgentCapabilities | None, version: int
) -> dict[str, Any]:
    """Serialize agent capabilities for the negotiated *version*."""
    if version < 2:
        return to_dict(caps) or {}
    session: dict[str, Any] = {}
    if caps is not None:
        if caps.prompt_capabilities is not None:
            prompt: dict[str, Any] = {}
            if caps.prompt_capabilities.image:
                prompt["image"] = {}
            if caps.prompt_capabilities.audio:
                prompt["audio"] = {}
            if caps.prompt_capabilities.embedded_context:
                prompt["embeddedContext"] = {}
            if prompt:
                session["prompt"] = prompt
        if caps.mcp_capabilities is not None:
            mcp: dict[str, Any] = {}
            if caps.mcp_capabilities.stdio:
                mcp["stdio"] = {}
            if caps.mcp_capabilities.http:
                mcp["http"] = {}
            if mcp:
                session["mcp"] = mcp
        sc = caps.session_capabilities
        if sc is not None:
            if sc.delete is not None:
                session["delete"] = sc.delete or {}
            if sc.additional_directories is not None:
                session["additionalDirectories"] = sc.additional_directories or {}
    payload: dict[str, Any] = {"session": session}
    if caps is not None and caps.auth is not None:
        payload["auth"] = caps.auth
    return payload


def _initialize_params_to_wire(
    params: InitializeParams, version: int
) -> dict[str, Any]:
    """Serialize ``initialize`` request parameters for *version*."""
    if version < 2:
        return to_dict(params)
    payload: dict[str, Any] = {
        "protocolVersion": params.protocol_version,
        "info": to_dict(params.client_info or _default_implementation()),
        "capabilities": _client_capabilities_to_wire(
            params.client_capabilities, version
        ),
    }
    return payload


def _auth_method_to_wire(method: AuthMethod, version: int) -> dict[str, Any]:
    """Serialize an auth method.

    ACP v2 renames the descriptor's ``id`` to ``methodId`` and requires a
    ``type`` discriminator.
    """
    if version < 2:
        return to_dict(method)
    payload: dict[str, Any] = {
        "methodId": method.id,
        "name": method.name,
        "type": method.type or "agent",
    }
    if method.description is not None:
        payload["description"] = method.description
    return payload


def _initialize_result_to_wire(
    result: InitializeResult, version: int
) -> dict[str, Any]:
    """Serialize an ``initialize`` result for *version*."""
    if version < 2:
        return to_dict(result)
    payload: dict[str, Any] = {
        "protocolVersion": result.protocol_version,
        "info": to_dict(result.agent_info or _default_implementation()),
        "capabilities": _agent_capabilities_to_wire(result.agent_capabilities, version),
    }
    if result.auth_methods:
        payload["authMethods"] = [
            _auth_method_to_wire(method, version) for method in result.auth_methods
        ]
    return payload


def _config_option_to_wire(option: ConfigOption, version: int) -> dict[str, Any]:
    """Serialize a session config option for *version*."""
    if version < 2:
        return to_dict(option)
    payload: dict[str, Any] = {
        "configId": option.id,
        "name": option.name,
        "type": option.type or "select",
        "currentValue": option.current_value,
        "options": [
            {
                "value": value.value,
                "name": value.name,
                **(
                    {"description": value.description}
                    if value.description is not None
                    else {}
                ),
            }
            for value in option.options
        ],
    }
    if option.description is not None:
        payload["description"] = option.description
    if option.category is not None:
        payload["category"] = option.category
    return payload


def _config_options_to_wire(
    options: list[ConfigOption] | None, version: int
) -> list[dict[str, Any]]:
    """Serialize a config-option list for *version*."""
    return [_config_option_to_wire(option, version) for option in options or []]


# ── ACP Client ─────────────────────────────────────────────────────────────


class ACPClient:
    """High-level async client for communicating with an ACP agent subprocess.

    Usage::

        client = ACPClient(["python", "-m", "my_agent"])
        await client.start()
        init = await client.initialize()
        session = await client.new_session("/project")
        async for update in client.prompt(session.session_id, "Hello"):
            print(update)
        await client.stop()

    Args:
        command: Command and arguments to spawn the agent process.
        env: Optional environment variables for the subprocess.
    """

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._env = env
        self._process: asyncio.subprocess.Process | None = None
        self._transport: JSONRPCTransport | None = None
        self._pending: dict[Union[int, str], asyncio.Future[dict[str, Any]]] = {}
        self._notification_handlers: dict[str, Callable[..., Any]] = {}
        # One queue per ACP session: updates are routed by sessionId so
        # concurrent prompts cannot consume each other's notifications.
        self._update_queues: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._request_handler: Callable[[str, dict[str, Any]], Any] | None = None
        # Last prompt result per session; the shared attribute is kept for
        # backwards compatibility and always reflects the latest turn.
        self._prompt_results: dict[str, PromptResult] = {}
        self._last_prompt_result: PromptResult | None = None
        # Negotiated in initialize(); None until the handshake completes.
        self._negotiated_version: int | None = None

    async def start(self) -> None:
        """Launch the agent subprocess and begin reading messages."""
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        reader = asyncio.StreamReader()
        reader.set_transport(self._process.stdout._transport)  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
        self._transport = JSONRPCTransport(
            reader=self._process.stdout,  # type: ignore[arg-type]
            writer=self._process.stdin,  # type: ignore[arg-type]
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        """Terminate the agent subprocess and clean up."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._transport:
            await self._transport.close()
        self._update_queues.clear()
        self._prompt_results.clear()
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()

    def set_request_handler(
        self, handler: Callable[[str, dict[str, Any]], Any]
    ) -> None:
        """Register a handler for incoming requests from the agent.

        This is used for requests like ``session/request_permission``,
        ``fs/read_text_file``, etc.

        Args:
            handler: Async callable ``(method, params) -> result``.
        """
        self._request_handler = handler

    @property
    def negotiated_version(self) -> int | None:
        """Protocol version negotiated by :meth:`initialize`, if any."""
        return self._negotiated_version

    def prompt_result(self, session_id: str) -> PromptResult | None:
        """Return the last prompt result for *session_id*, if any."""
        return self._prompt_results.get(session_id)

    @property
    def last_prompt_result(self) -> PromptResult | None:
        """Result of the most recent ``session/prompt`` turn, if any.

        Kept for compatibility; use :meth:`prompt_result` for concurrent
        sessions.
        """
        return self._last_prompt_result

    def _is_response(self, msg: dict[str, Any]) -> bool:
        """Check if *msg* is a JSON-RPC response (has ``id`` and result/error)."""
        return "id" in msg and ("result" in msg or "error" in msg)

    def _resolve_pending_future(self, msg: dict[str, Any]) -> None:
        """Resolve or reject the future associated with a response message."""
        req_id = msg["id"]
        fut = self._pending.pop(req_id, None)
        if not fut or fut.done():
            return
        if "error" in msg:
            fut.set_exception(JSONRPCException(JSONRPCError.from_dict(msg["error"])))
        else:
            fut.set_result(msg.get("result") or {})

    async def _handle_notification(self, msg: dict[str, Any]) -> None:
        """Process an incoming notification from the agent."""
        method = msg["method"]
        params = msg.get("params", {})
        if method == "session/update":
            session_id = params.get("sessionId", "")
            if not session_id:
                logger.debug("session/update without sessionId: %r", params)
            await self._queue_for(session_id).put(params)
        handler = self._notification_handlers.get(method)
        if handler:
            asyncio.create_task(handler(params))

    async def _read_loop(self) -> None:
        """Background task that reads and dispatches incoming messages."""
        assert self._transport is not None
        while True:
            msg = await self._transport.read_message()
            if msg is None:
                # EOF -- signal every waiting prompt iterator
                for queue in list(self._update_queues.values()):
                    queue.put_nowait(None)
                break

            if self._is_response(msg):
                self._resolve_pending_future(msg)
            elif "method" in msg and "id" not in msg:
                await self._handle_notification(msg)
            elif "method" in msg and "id" in msg:
                asyncio.create_task(self._handle_agent_request(msg))

    async def _handle_agent_request(self, msg: dict[str, Any]) -> None:
        """Handle an incoming request from the agent side."""
        assert self._transport is not None
        req_id = msg["id"]
        method = msg["method"]
        params = msg.get("params", {})
        if self._request_handler:
            try:
                result = self._request_handler(method, params)
                if asyncio.iscoroutine(result):
                    result = await result
                await self._transport.send_result(req_id, result)
            except Exception as exc:
                await self._transport.send_error(
                    req_id,
                    JSONRPCError(code=INTERNAL_ERROR, message=str(exc)),
                )
        else:
            await self._transport.send_error(
                req_id,
                JSONRPCError(
                    code=METHOD_NOT_FOUND,
                    message=f"No handler for {method}",
                ),
            )

    async def _call(self, method: str, params: Any = None) -> Any:
        """Send a request and wait for its response.

        Args:
            method: JSON-RPC method name.
            params: Method parameters.

        Returns:
            The result payload from the agent.

        Raises:
            JSONRPCException: If the agent returns an error.
        """
        assert self._transport is not None
        sent = await self._transport.send_request(method, params)
        req_id = sent["id"]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = fut
        return await fut

    # -- Public API ---------------------------------------------------------

    async def initialize(
        self,
        params: InitializeParams | None = None,
    ) -> InitializeResult:
        """Perform the ``initialize`` handshake with the agent.

        Args:
            params: Initialization parameters (defaults provided if ``None``).

        Returns:
            The agent's initialization result.

        Raises:
            UnsupportedProtocolVersion: If the agent negotiated a protocol
                version this client does not implement.  The subprocess is
                left running; call :meth:`stop` before retrying.
        """
        if params is None:
            params = InitializeParams()
        # A client sends the highest version it supports.  Use v2 field names
        # when requesting v2 or newer; v1 peers read the legacy names.
        requested = params.protocol_version
        wire_version = 2 if isinstance(requested, int) and requested >= 2 else 1
        try:
            raw = await self._call(
                "initialize", _initialize_params_to_wire(params, wire_version)
            )
        except JSONRPCException as exc:
            # A v2-only-first client can meet a v1-only agent.  Retry once
            # with the v1 shape when the agent rejects the v2 request.
            if wire_version != 2 or exc.error.code not in _VERSION_RETRY_CODES:
                raise
            params.protocol_version = 1
            raw = await self._call("initialize", _initialize_params_to_wire(params, 1))
        raw = raw or {}
        r = from_raw(raw)
        version = _require_supported_version(
            r.get("protocol_version", ACP_PROTOCOL_VERSION)
        )
        self._negotiated_version = version
        auth_methods = [
            AuthMethod(
                id=m.get("methodId", m.get("id", "")),
                name=m.get("name", ""),
                description=m.get("description"),
                type=m.get("type"),
            )
            for m in (r.get("auth_methods") or [])
            if isinstance(m, dict)
        ]
        return InitializeResult(
            protocol_version=version,
            agent_capabilities=_parse_agent_capabilities(
                r.get("capabilities", r.get("agent_capabilities"))
            ),
            agent_info=_parse_implementation_info(r.get("info", r.get("agent_info"))),
            auth_methods=auth_methods or None,
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        additional_directories: list[str] | None = None,
    ) -> NewSessionResult:
        """Create a new conversation session.

        Args:
            cwd: Absolute path to the working directory.
            mcp_servers: Optional MCP server configurations.
            additional_directories: Additional workspace roots.

        Returns:
            The session creation result including the session id, modes,
            and configuration options.
        """
        p: dict[str, Any] = {"cwd": cwd, "mcpServers": mcp_servers or []}
        if additional_directories:
            p["additionalDirectories"] = additional_directories
        raw = await self._call("session/new", p)
        raw = raw or {}
        r = from_raw(raw)
        return NewSessionResult(
            session_id=r.get("session_id", ""),
            modes=_parse_session_modes(r.get("modes")),
            config_options=_parse_config_options(r.get("config_options")),
            available_commands=_parse_available_commands(r.get("available_commands")),
        )

    async def load_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        additional_directories: list[str] | None = None,
    ) -> None:
        """Load (resume) an existing session.

        The agent will replay conversation history as ``session/update``
        notifications before responding.

        Args:
            session_id: Session to resume.
            cwd: Working directory.
            mcp_servers: Optional MCP server configurations.
            additional_directories: Additional workspace roots.
        """
        if self._negotiated_version is not None and self._negotiated_version >= 2:
            # ACP v2 removed session/load; replay is requested on resume.
            await self.resume_session(
                session_id,
                cwd,
                mcp_servers,
                additional_directories,
                replay_from={"type": "start"},
            )
            return
        p: dict[str, Any] = {
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": mcp_servers or [],
        }
        if additional_directories:
            p["additionalDirectories"] = additional_directories
        await self._call("session/load", p)

    @staticmethod
    def _build_prompt_blocks(
        text: str,
        extra_content: list[ContentBlock] | None,
    ) -> list[dict[str, Any]]:
        """Build the prompt content block list from text and extras."""
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if extra_content:
            for b in extra_content:
                blocks.append(to_dict(b))
        return blocks

    def _queue_for(self, session_id: str) -> asyncio.Queue[dict[str, Any] | None]:
        """Return the update queue for *session_id*, creating it if needed."""
        queue = self._update_queues.get(session_id)
        if queue is None:
            queue = asyncio.Queue()
            self._update_queues[session_id] = queue
        return queue

    def _existing_queue(
        self, session_id: str
    ) -> asyncio.Queue[dict[str, Any] | None] | None:
        """Return the queue for *session_id* without creating one."""
        return self._update_queues.get(session_id)

    def _drain_stale_updates(self, session_id: str) -> None:
        """Discard stale updates queued for *session_id* before a prompt."""
        queue = self._existing_queue(session_id)
        if queue is None:
            return
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def prompt(
        self,
        session_id: str,
        text: str,
        extra_content: list[ContentBlock] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a prompt and yield ``session/update`` notifications.

        This is an async generator.  For ACP v1 it yields each raw update
        until the ``session/prompt`` response is received.  For ACP v2 the
        response only acknowledges insertion, so iteration continues until
        an ``idle`` ``state_update`` reports that the turn ended.

        Args:
            session_id: Target session.
            text: User message text.
            extra_content: Additional content blocks to include.

        Yields:
            Raw ``session/update`` parameter dictionaries.
        """
        blocks = self._build_prompt_blocks(text, extra_content)
        params = {"sessionId": session_id, "prompt": blocks}
        queue = self._queue_for(session_id)
        self._drain_stale_updates(session_id)
        self._last_prompt_result = None
        self._prompt_results.pop(session_id, None)
        is_v2 = self._negotiated_version is not None and self._negotiated_version >= 2
        ack_result: PromptResult | None = None
        observed_stop_reason: str | None = None
        idle_seen = False

        assert self._transport is not None
        sent = await self._transport.send_request("session/prompt", params)
        req_id = sent["id"]
        loop = asyncio.get_running_loop()
        prompt_fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = prompt_fut

        def finish() -> PromptResult:
            result = ack_result or _parse_prompt_result({})
            if observed_stop_reason is not None:
                try:
                    result.stop_reason = StopReason(observed_stop_reason)
                    result.raw_stop_reason = None
                except ValueError:
                    result.stop_reason = StopReason.UNKNOWN
                    result.raw_stop_reason = observed_stop_reason
            return result

        def store(result: PromptResult) -> None:
            self._prompt_results[session_id] = result
            self._last_prompt_result = result

        while True:
            # Wait for an update, and for the prompt response until it lands.
            wait_on = [prompt_fut] if ack_result is None else []
            update_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                [update_task, *wait_on],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if update_task in done:
                update = update_task.result()
                if update is None:
                    # EOF: keep a completed turn's result, if any.
                    if ack_result is not None:
                        store(finish())
                    return
                stop = self._stop_reason_from_update(update)
                if stop is not None:
                    observed_stop_reason = stop
                    idle_seen = True
                yield update
                if idle_seen and ack_result is not None:
                    store(finish())
                    for u in self._drain_remaining_updates(session_id):
                        yield u
                    return
            else:
                update_task.cancel()

            if ack_result is None and prompt_fut.done():
                ack_result = _parse_prompt_result(prompt_fut.result())
                # Some agents still return a stop reason with the v2 ack;
                # treat that as turn completion instead of waiting for idle.
                explicit_stop = (
                    ack_result.raw_stop_reason is not None
                    or ack_result.stop_reason is not StopReason.UNKNOWN
                )
                if not is_v2 or idle_seen or explicit_stop:
                    store(finish() if is_v2 else ack_result)
                    for u in self._drain_remaining_updates(session_id):
                        yield u
                    return

    @staticmethod
    def _stop_reason_from_update(update: dict[str, Any]) -> str | None:
        """Return the stop reason carried by an idle ``state_update``."""
        payload = update.get("update")
        if not isinstance(payload, dict):
            return None
        if payload.get("sessionUpdate") != "state_update":
            return None
        if payload.get("state") != "idle":
            return None
        raw = payload.get("stopReason")
        return raw if isinstance(raw, str) else None

    def _drain_remaining_updates(self, session_id: str) -> list[dict[str, Any]]:
        """Collect remaining non-None updates queued for *session_id*."""
        remaining: list[dict[str, Any]] = []
        queue = self._existing_queue(session_id)
        if queue is None:
            return remaining
        while not queue.empty():
            update = queue.get_nowait()
            if update is not None:
                remaining.append(update)
        return remaining

    async def prompt_simple(
        self,
        session_id: str,
        text: str,
    ) -> tuple[list[dict[str, Any]], PromptResult]:
        """Send a prompt and collect all updates, returning them with the result.

        A simpler alternative to the async-generator ``prompt()`` method.

        Args:
            session_id: Target session.
            text: User message text.

        Returns:
            Tuple of (list of update dicts, PromptResult).
        """
        updates: list[dict[str, Any]] = []
        async for u in self.prompt(session_id, text):
            updates.append(u)
        return updates, self._prompt_results.get(session_id) or PromptResult(
            stop_reason=StopReason.END_TURN
        )

    async def cancel(self, session_id: str) -> None:
        """Cancel the current prompt turn.

        Args:
            session_id: Session to cancel.
        """
        assert self._transport is not None
        await self._transport.send_notification(
            "session/cancel", {"sessionId": session_id}
        )

    async def set_mode(self, session_id: str, mode_id: str) -> None:
        """Switch the agent operating mode.

        Args:
            session_id: Target session.
            mode_id: Mode to switch to.
        """
        await self._call(
            "session/set_mode",
            {"sessionId": session_id, "modeId": mode_id},
        )

    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
    ) -> ListSessionsResult:
        """List sessions known to the agent.

        Args:
            cwd: Optional working-directory filter.
            cursor: Optional pagination cursor.

        Returns:
            List of session metadata.
        """
        p: dict[str, Any] = {}
        if cwd:
            p["cwd"] = cwd
        if cursor:
            p["cursor"] = cursor
        raw = await self._call("session/list", p)
        raw = raw or {}
        sessions: list[SessionInfo] = []
        for s in raw.get("sessions", []):
            sr = from_raw(s)
            sessions.append(
                SessionInfo(
                    session_id=sr.get("session_id", ""),
                    cwd=sr.get("cwd", ""),
                    title=sr.get("title"),
                    updated_at=sr.get("updated_at"),
                )
            )
        return ListSessionsResult(
            sessions=sessions,
            next_cursor=raw.get("nextCursor"),
        )

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        additional_directories: list[str] | None = None,
        replay_from: dict[str, Any] | None = None,
    ) -> None:
        """Resume an existing session without replaying history.

        Args:
            session_id: Session to resume.
            cwd: Working directory.
            mcp_servers: Optional MCP server configurations.
            additional_directories: Additional workspace roots.
            replay_from: ACP v2 replay cursor, for example
                ``{"type": "start"}`` to replay all retained history.
        """
        p: dict[str, Any] = {
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": mcp_servers or [],
        }
        if additional_directories:
            p["additionalDirectories"] = additional_directories
        if replay_from is not None:
            p["replayFrom"] = replay_from
        await self._call("session/resume", p)

    async def close_session(self, session_id: str) -> None:
        """Close an active session and free its resources.

        Args:
            session_id: Session to close.
        """
        await self._call("session/close", {"sessionId": session_id})
        self._update_queues.pop(session_id, None)
        self._prompt_results.pop(session_id, None)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session from the agent's history.

        Args:
            session_id: Session to delete.
        """
        await self._call("session/delete", {"sessionId": session_id})

    async def set_config_option(
        self, session_id: str, config_id: str, value: Any
    ) -> SetConfigOptionResult:
        """Set a session configuration option.

        Args:
            session_id: Target session.
            config_id: Configuration option id.
            value: New value.  ACP v2 expects a tagged value on the wire;
                strings are sent as ``{"type": "id"}`` and booleans as
                ``{"type": "boolean"}``.

        Returns:
            The complete configuration state after the change.
        """
        payload: dict[str, Any]
        if self._negotiated_version is not None and self._negotiated_version >= 2:
            if isinstance(value, bool):
                payload = {
                    "sessionId": session_id,
                    "configId": config_id,
                    "type": "boolean",
                    "value": value,
                }
            else:
                payload = {
                    "sessionId": session_id,
                    "configId": config_id,
                    "type": "id",
                    "value": value,
                }
        else:
            payload = {
                "sessionId": session_id,
                "configId": config_id,
                "value": value,
            }
        raw = await self._call("session/set_config_option", payload)
        r = from_raw(raw or {})
        return SetConfigOptionResult(
            config_options=_parse_config_options(r.get("config_options")) or []
        )

    async def authenticate(self, method_id: str) -> None:
        """Authenticate with an agent-advertised auth method.

        Args:
            method_id: Auth method id from ``initialize``.
        """
        if self._negotiated_version is not None and self._negotiated_version >= 2:
            await self._call("auth/login", {"methodId": method_id})
            return
        await self._call("authenticate", {"methodId": method_id})

    async def logout(self) -> None:
        """End the current authenticated state."""
        if self._negotiated_version is not None and self._negotiated_version >= 2:
            await self._call("auth/logout", {})
            return
        await self._call("logout", {})


# ── ACP Agent (server-side base class) ─────────────────────────────────────


class ACPAgent(ABC):
    """Abstract base class for implementing an ACP-compatible agent.

    Subclass this and override the ``on_*`` methods.  Call ``await agent.run()``
    to start the stdio event loop.

    Example::

        class MyAgent(ACPAgent):
            async def on_initialize(self, params):
                return InitializeResult(protocol_version=1)
            async def on_new_session(self, params):
                return NewSessionResult(session_id=str(uuid.uuid4()))
            async def on_prompt(self, params):
                await self.send_update(params.session_id,
                    AgentMessageChunkUpdate(content=TextContent(text="Hi!")))
                return PromptResult(stop_reason=StopReason.END_TURN)
    """

    def __init__(self) -> None:
        self._transport: JSONRPCTransport | None = None
        self._running = False
        self._client_capabilities: ClientCapabilities | None = None
        self._negotiated_version: int | None = None

    @property
    def negotiated_version(self) -> int | None:
        """Protocol version negotiated by ``initialize``, if any."""
        return self._negotiated_version

    # -- Handler methods (override these) -----------------------------------

    async def on_initialize(self, params: InitializeParams) -> InitializeResult:
        """Handle the ``initialize`` request.

        The default implementation negotiates the protocol version and
        advertises no optional capabilities.  Override to advertise agent
        capabilities or authentication methods; the base class rejects any
        returned version outside :data:`SUPPORTED_PROTOCOL_VERSIONS`.

        Args:
            params: Initialization parameters from the client.

        Returns:
            Initialization result with agent capabilities.
        """
        return InitializeResult(
            protocol_version=negotiate_protocol_version(params.protocol_version)
        )

    @abstractmethod
    async def on_new_session(self, params: NewSessionParams) -> NewSessionResult:
        """Handle ``session/new``.

        Args:
            params: Session creation parameters.

        Returns:
            Result containing the new session id.
        """
        ...

    @abstractmethod
    async def on_prompt(self, params: PromptParams) -> PromptResult:
        """Handle ``session/prompt``.

        Use ``self.send_update()`` to stream updates back to the client
        before returning the final result.

        Args:
            params: Prompt parameters including user message.

        Returns:
            Result with the stop reason.
        """
        ...

    async def on_load_session(self, params: LoadSessionParams) -> None:
        """Handle ``session/load``.  Override to support session resumption.

        Args:
            params: Session load parameters.
        """
        raise JSONRPCException(
            JSONRPCError(
                code=METHOD_NOT_FOUND,
                message="session/load not supported",
            )
        )

    async def on_cancel(self, params: CancelParams) -> None:
        """Handle ``session/cancel`` notification.

        Args:
            params: Cancel parameters.
        """

    async def on_set_mode(self, params: SetModeParams) -> None:
        """Handle ``session/set_mode``.

        Args:
            params: Mode change parameters.
        """

    async def on_set_config_option(
        self, params: SetConfigOptionParams
    ) -> SetConfigOptionResult:
        """Handle ``session/set_config_option``.

        Args:
            params: Config option change parameters.

        Returns:
            Complete configuration state.
        """
        return SetConfigOptionResult()

    async def on_list_sessions(self, params: ListSessionsParams) -> ListSessionsResult:
        """Handle ``session/list``.

        Args:
            params: List sessions parameters.

        Returns:
            List of session metadata.
        """
        return ListSessionsResult()

    async def on_resume_session(self, params: ResumeSessionParams) -> None:
        """Handle ``session/resume``. Override to support resumption.

        Args:
            params: Session resume parameters.
        """
        raise JSONRPCException(
            JSONRPCError(
                code=METHOD_NOT_FOUND,
                message="session/resume not supported",
            )
        )

    async def on_close_session(self, params: CloseSessionParams) -> None:
        """Handle ``session/close``. Default: no-op.

        Args:
            params: Session close parameters.
        """

    async def on_delete_session(self, params: DeleteSessionParams) -> None:
        """Handle ``session/delete``. Override to support deletion.

        Args:
            params: Session delete parameters.
        """
        raise JSONRPCException(
            JSONRPCError(
                code=METHOD_NOT_FOUND,
                message="session/delete not supported",
            )
        )

    async def on_authenticate(self, params: AuthenticateParams) -> None:
        """Handle ``authenticate``. Default: no-op.

        Args:
            params: Authentication parameters.
        """

    async def on_logout(self, params: LogoutParams) -> None:
        """Handle ``logout``. Default: no-op.

        Args:
            params: Logout parameters.
        """

    # -- Outgoing helpers ---------------------------------------------------

    def _require_v1_method(self, method: str) -> None:
        """Raise ``METHOD_NOT_FOUND`` when *method* was removed in ACP v2."""
        if self._negotiated_version is not None and self._negotiated_version >= 2:
            raise JSONRPCException(
                JSONRPCError(
                    code=METHOD_NOT_FOUND,
                    message=f"{method} was removed in ACP v2",
                )
            )

    def supports_update(self, session_update: str) -> bool:
        """Return whether the connected client accepts *session_update*.

        Use this before sending capability-gated updates such as ``notice``
        or ``compaction_update``; the client advertises support in
        ``initialize``.

        Args:
            session_update: Wire value of the ``sessionUpdate`` discriminator.
        """
        return supports_update(self._client_capabilities, session_update)

    async def send_update(self, session_id: str, update: SessionUpdate) -> None:
        """Send a ``session/update`` notification to the client.

        Args:
            session_id: Target session.
            update: The update payload.
        """
        assert self._transport is not None
        await self._transport.send_notification(
            "session/update",
            {"sessionId": session_id, "update": to_dict(update)},
        )

    async def request_permission(
        self,
        session_id: str,
        tool_call: dict[str, Any],
        options: list[PermissionOption],
        *,
        title: str | None = None,
        description: str | None = None,
        subject: dict[str, Any] | None = None,
    ) -> PermissionOutcome:
        """Request permission from the client for a tool call.

        Args:
            session_id: Target session.
            tool_call: Tool call details (ACP v1 ``toolCall``; ACP v2 uses it
                as the ``tool_call`` subject when *subject* is omitted).
            options: Available permission options.
            title: ACP v2 permission-prompt title.  Defaults to the tool
                call's title.
            description: ACP v2 optional explanation.
            subject: ACP v2 structured subject.  Defaults to
                ``{"type": "tool_call", "toolCall": tool_call}``.

        Returns:
            The user's decision.

        Raises:
            JSONRPCException: If the client returns an error.
        """
        assert self._transport is not None
        is_v2 = self._negotiated_version is not None and self._negotiated_version >= 2
        payload: dict[str, Any]
        if is_v2:
            payload = {
                "sessionId": session_id,
                "title": title or str(tool_call.get("title", "Permission required")),
                "options": [to_dict(o) for o in options],
            }
            if description is not None:
                payload["description"] = description
            payload["subject"] = (
                subject
                if subject is not None
                else {"type": "tool_call", "toolCall": tool_call}
            )
        else:
            payload = {
                "sessionId": session_id,
                "toolCall": tool_call,
                "options": [to_dict(o) for o in options],
            }
        sent = await self._transport.send_request("session/request_permission", payload)
        # We need to wait for the response in-line.  Since run() owns the
        # read loop, we store a future that run() will resolve.
        req_id = sent["id"]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[req_id] = fut
        raw = await fut
        outcome = raw.get("outcome", {})
        return PermissionOutcome(
            outcome=outcome.get("outcome", "cancelled"),
            option_id=outcome.get("optionId"),
        )

    async def read_text_file(
        self,
        session_id: str,
        path: str,
        *,
        line: int | None = None,
        limit: int | None = None,
    ) -> str:
        """Read a text file via the client's ``fs/read_text_file`` method.

        Args:
            session_id: Target session.
            path: Absolute file path.
            line: Optional start line (1-based).
            limit: Optional max lines.

        Returns:
            File text content.
        """
        assert self._transport is not None
        self._require_v1_method("fs/read_text_file")
        p: dict[str, Any] = {"sessionId": session_id, "path": path}
        if line is not None:
            p["line"] = line
        if limit is not None:
            p["limit"] = limit
        sent = await self._transport.send_request("fs/read_text_file", p)
        req_id = sent["id"]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[req_id] = fut
        raw = await fut
        return raw.get("content", "")

    async def write_text_file(self, session_id: str, path: str, content: str) -> None:
        """Write a text file via the client's ``fs/write_text_file`` method.

        Args:
            session_id: Target session.
            path: Absolute file path.
            content: Text content to write.
        """
        assert self._transport is not None
        self._require_v1_method("fs/write_text_file")
        sent = await self._transport.send_request(
            "fs/write_text_file",
            {"sessionId": session_id, "path": path, "content": content},
        )
        req_id = sent["id"]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[req_id] = fut
        await fut

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        *,
        args: list[str] | None = None,
        cwd: str | None = None,
    ) -> str:
        """Create a terminal via the client's ``terminal/create`` method.

        Args:
            session_id: Target session.
            command: Command to execute.
            args: Command arguments.
            cwd: Working directory.

        Returns:
            Terminal id.
        """
        assert self._transport is not None
        self._require_v1_method("terminal/create")
        p: dict[str, Any] = {"sessionId": session_id, "command": command}
        if args:
            p["args"] = args
        if cwd:
            p["cwd"] = cwd
        sent = await self._transport.send_request("terminal/create", p)
        req_id = sent["id"]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[req_id] = fut
        raw = await fut
        return raw.get("terminalId", "")

    # -- Main event loop ----------------------------------------------------

    async def run(self) -> None:
        """Run the agent, reading from stdin and writing to stdout.

        This blocks until the client closes the connection (EOF on stdin).
        """
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        read_transport, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
        )

        # Build a proper StreamWriter backed by a StreamReaderProtocol so
        # that drain() works correctly on all Python versions.
        write_reader = asyncio.StreamReader()
        write_protocol = asyncio.StreamReaderProtocol(write_reader)
        write_transport, _ = await loop.connect_write_pipe(
            lambda: write_protocol, sys.stdout
        )
        writer = asyncio.StreamWriter(
            write_transport,
            write_protocol,
            write_reader,
            loop,
        )

        self._transport = JSONRPCTransport(reader, writer)
        self._running = True
        self._pending_requests: dict[
            Union[int, str], asyncio.Future[dict[str, Any]]
        ] = {}

        logger.info("Agent started, waiting for messages on stdin...")
        try:
            while self._running:
                msg = await self._transport.read_message()
                if msg is None:
                    break
                asyncio.create_task(self._dispatch(msg))
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            await self._transport.close()
            read_transport.close()

    def _resolve_agent_response(self, msg: dict[str, Any]) -> bool:
        """Resolve a response to one of our outgoing requests.

        Returns:
            ``True`` if the message was a response and was handled.
        """
        if not ("id" in msg and ("result" in msg or "error" in msg)):
            return False
        req_id = msg["id"]
        fut = self._pending_requests.pop(req_id, None)
        if not fut or fut.done():
            return True
        if "error" in msg:
            fut.set_exception(JSONRPCException(JSONRPCError.from_dict(msg["error"])))
        else:
            fut.set_result(msg.get("result") or {})
        return True

    async def _dispatch_and_respond(
        self,
        method: str | None,
        params: dict[str, Any],
        req_id: int | str | None,
    ) -> None:
        """Invoke the method handler and send back the JSON-RPC result/error."""
        assert self._transport is not None
        is_notification = req_id is None
        try:
            result = await self._handle_method(method, params)
            if not is_notification:
                assert req_id is not None
                result_dict = to_dict(result) if result is not None else None
                await self._transport.send_result(req_id, result_dict)
        except JSONRPCException as exc:
            if not is_notification:
                await self._transport.send_error(req_id, exc.error)
        except Exception as exc:
            logger.exception("Unhandled error in %s", method)
            if not is_notification:
                await self._transport.send_error(
                    req_id,
                    JSONRPCError(code=INTERNAL_ERROR, message=str(exc)),
                )

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        """Route an incoming JSON-RPC message to the appropriate handler."""
        if self._resolve_agent_response(msg):
            return
        await self._dispatch_and_respond(
            msg.get("method"), msg.get("params", {}), msg.get("id")
        )

    async def _handle_initialize(self, params: dict[str, Any]) -> Any:
        """Handle ``initialize`` and serialize the negotiated result."""
        init_params = _parse_initialize_params(params)
        self._client_capabilities = init_params.client_capabilities
        result = await self.on_initialize(init_params)
        negotiated = _require_supported_version(result.protocol_version)
        requested = init_params.protocol_version
        if (
            isinstance(requested, int)
            and not isinstance(requested, bool)
            and negotiated > requested
        ):
            # The agent must not pick a version above the client's.
            raise UnsupportedProtocolVersion(requested)
        result.protocol_version = negotiated
        self._negotiated_version = negotiated
        return _initialize_result_to_wire(result, negotiated)

    def _reject_v2_removed(self, method: str | None) -> None:
        """Reject methods that ACP v2 removed from the wire surface."""
        if (
            self._negotiated_version is not None
            and self._negotiated_version >= 2
            and method in ACP_V2_REMOVED_METHODS
        ):
            raise JSONRPCException(
                JSONRPCError(
                    code=METHOD_NOT_FOUND,
                    message=f"{method} was removed in ACP v2",
                )
            )

    async def _handle_prompt(self, params: dict[str, Any]) -> Any:
        """Handle ``session/prompt`` for the negotiated protocol version."""
        rp = from_raw(params)
        prompt_blocks: list[ContentBlock] = []
        for block in rp.get("prompt", []):
            prompt_blocks.append(_content_from_dict(block))
        session_id = rp.get("session_id", "")
        is_v2 = self._negotiated_version is not None and self._negotiated_version >= 2
        message_id: str | None = None
        if is_v2:
            message_id = f"msg_{uuid.uuid4().hex}"
            await self.send_update(session_id, StateUpdate(state="running"))
            await self.send_update(
                session_id,
                UserMessageUpdate(message_id=message_id, content=prompt_blocks),
            )
        result = await self.on_prompt(
            PromptParams(
                session_id=session_id,
                prompt=prompt_blocks,
            )
        )
        if not is_v2:
            return result
        stop_reason = result.raw_stop_reason
        if stop_reason is None and result.stop_reason is not StopReason.UNKNOWN:
            stop_reason = result.stop_reason.value
        await self.send_update(
            session_id,
            StateUpdate(state="idle", stop_reason=stop_reason),
        )
        # ACP v2 acknowledges insertion; completion is the idle state_update.
        return {"messageId": message_id}

    async def _handle_new_session(self, params: dict[str, Any]) -> Any:
        """Handle ``session/new`` and serialize the negotiated result."""
        rp = from_raw(params)
        result = await self.on_new_session(
            NewSessionParams(
                cwd=rp.get("cwd", "."),
                mcp_servers=rp.get("mcp_servers"),
                additional_directories=rp.get("additional_directories"),
            )
        )
        if self._negotiated_version is None or self._negotiated_version < 2:
            return result
        payload: dict[str, Any] = {"sessionId": result.session_id}
        if result.config_options:
            payload["configOptions"] = _config_options_to_wire(result.config_options, 2)
        if result.available_commands:
            payload["availableCommands"] = [
                to_dict(command) for command in result.available_commands
            ]
        return payload

    async def _handle_set_config_option(self, params: dict[str, Any]) -> Any:
        """Handle ``session/set_config_option`` and serialize the result."""
        rp = from_raw(params)
        value: Any = rp.get("value", "")
        if rp.get("type") == "boolean":
            value = bool(value)
        result = await self.on_set_config_option(
            SetConfigOptionParams(
                session_id=rp.get("session_id", ""),
                config_id=rp.get("config_id", ""),
                value=value,
            )
        )
        if self._negotiated_version is None or self._negotiated_version < 2:
            return result
        return {"configOptions": _config_options_to_wire(result.config_options, 2)}

    async def _handle_method(self, method: str | None, params: dict[str, Any]) -> Any:
        """Dispatch a method call to the correct handler."""
        if method == "initialize":
            return await self._handle_initialize(params)

        self._reject_v2_removed(method)

        if method == "session/new":
            return await self._handle_new_session(params)

        if method == "session/load":
            rp = from_raw(params)
            return await self.on_load_session(
                LoadSessionParams(
                    session_id=rp.get("session_id", ""),
                    cwd=rp.get("cwd", "."),
                    mcp_servers=rp.get("mcp_servers"),
                    additional_directories=rp.get("additional_directories"),
                )
            )

        if method == "session/resume":
            rp = from_raw(params)
            session_id = rp.get("session_id", "")
            await self.on_resume_session(
                ResumeSessionParams(
                    session_id=session_id,
                    cwd=rp.get("cwd", "."),
                    mcp_servers=rp.get("mcp_servers"),
                    additional_directories=rp.get("additional_directories"),
                    replay_from=rp.get("replay_from"),
                )
            )
            if self._negotiated_version is not None and self._negotiated_version >= 2:
                # ACP v2 requires a current state update before the response.
                await self.send_update(session_id, StateUpdate(state="idle"))
            return None

        if method == "session/close":
            rp = from_raw(params)
            await self.on_close_session(
                CloseSessionParams(session_id=rp.get("session_id", ""))
            )
            return None

        if method == "session/delete":
            rp = from_raw(params)
            await self.on_delete_session(
                DeleteSessionParams(session_id=rp.get("session_id", ""))
            )
            return None

        if method == "authenticate":
            rp = from_raw(params)
            await self.on_authenticate(
                AuthenticateParams(method_id=rp.get("method_id", ""))
            )
            return None

        if method == "logout":
            await self.on_logout(LogoutParams())
            return None

        if method == "auth/login":
            rp = from_raw(params)
            await self.on_authenticate(
                AuthenticateParams(method_id=rp.get("method_id", ""))
            )
            return None

        if method == "auth/logout":
            await self.on_logout(LogoutParams())
            return None

        if method == "session/prompt":
            return await self._handle_prompt(params)

        if method == "session/cancel":
            rp = from_raw(params)
            await self.on_cancel(CancelParams(session_id=rp.get("session_id", "")))
            return None

        if method == "session/set_mode":
            rp = from_raw(params)
            await self.on_set_mode(
                SetModeParams(
                    session_id=rp.get("session_id", ""),
                    mode_id=rp.get("mode_id", ""),
                )
            )
            return None

        if method == "session/set_config_option":
            return await self._handle_set_config_option(params)

        if method == "session/list":
            rp = from_raw(params)
            return await self.on_list_sessions(
                ListSessionsParams(
                    cwd=rp.get("cwd"),
                    cursor=rp.get("cursor"),
                )
            )

        raise JSONRPCException(
            JSONRPCError(
                code=METHOD_NOT_FOUND,
                message=f"Unknown method: {method}",
            )
        )


# ── Example Usage ──────────────────────────────────────────────────────────


class _EchoAgent(ACPAgent):
    """Minimal echo agent for demonstration purposes."""

    async def on_initialize(self, params: InitializeParams) -> InitializeResult:
        return InitializeResult(
            protocol_version=negotiate_protocol_version(params.protocol_version),
            agent_info=ImplementationInfo(
                name="echo-agent", version="0.1.0", title="Echo Agent"
            ),
        )

    async def on_new_session(self, params: NewSessionParams) -> NewSessionResult:
        return NewSessionResult(session_id=f"sess_{uuid.uuid4().hex[:12]}")

    async def on_prompt(self, params: PromptParams) -> PromptResult:
        # Collect text from the prompt
        text_parts: list[str] = []
        for block in params.prompt:
            if isinstance(block, TextContent):
                text_parts.append(block.text)

        user_text = " ".join(text_parts) if text_parts else "(empty)"

        # Echo back as an agent message chunk
        await self.send_update(
            params.session_id,
            AgentMessageChunkUpdate(content=TextContent(text=f"Echo: {user_text}")),
        )
        return PromptResult(stop_reason=StopReason.END_TURN)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    print(
        "Starting echo agent. Send JSON-RPC messages on stdin.",
        file=sys.stderr,
    )
    asyncio.run(_EchoAgent().run())
