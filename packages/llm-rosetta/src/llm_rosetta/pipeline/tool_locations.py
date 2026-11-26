"""Where a request body declares its tools, per wire format.

An upstream that rejects a tool schema names the tool the way *it* sees the
request — ``tools[16].function.name``, ``function 'send_message'``.  The client
never wrote that list: it declared its tools in its own dialect, and a
conversion may have flattened containers, dropped a definition or rewritten a
name since.  Attributing the rejection to the client's own request therefore
needs one thing this module supplies: for a given format and body, the ordered
declarations, each with the path it sits at, the field that carries its schema,
and the name the wire gives it.

A declaration's path is spelled the way the body spells it, so a Google body
reports ``tools[0].function_declarations[3]`` and a Chat Completions body
reports ``tools[16]``.  ``wrapper`` names a nesting step between the path and
the name/schema (Chat's ``function``, ``custom``); ``schema_field`` names the
field the schema lives under (``parameters``, ``input_schema``).

The module reads a plain dict and imports nothing from the
converters, so the gateway can use it on a body it holds without converting it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class ToolDeclaration:
    """One tool declaration in a request body, and the path that reaches it.

    Attributes:
        name: The name the body gives the tool, or ``None`` for an entry that
            declares no name (a built-in server tool, a malformed entry).
        path: The JSON path of the declaration in this body, e.g.
            ``tools[16]`` or ``tools[0].functionDeclarations[3]``.
        wrapper: The key between *path* and the name/schema fields — Chat
            Completions nests both under ``function`` (or ``custom``).  Empty
            when the declaration carries them directly.
        schema_field: The key the tool's JSON schema lives under in this
            format, or ``None`` when the declaration carries no schema.
        number: 1-based position among the body's *named* declarations, which
            is the position a client can count in its own request.  ``0`` for
            an entry without a name.
    """

    name: str | None
    path: str
    wrapper: str = ""
    schema_field: str | None = None
    number: int = 0

    def field_path(self, key: str) -> str:
        """The path of *key* inside this declaration, as this body spells it.

        Args:
            key: A field of the declaration, e.g. ``name`` or a schema field.

        Returns:
            ``<path>.<wrapper>.<key>`` with the wrapper omitted when empty.
        """
        base = f"{self.path}.{self.wrapper}" if self.wrapper else self.path
        return f"{base}.{key}"


def read_tool_declarations(provider: str, body: Any) -> list[ToolDeclaration]:
    """List the tool declarations a request *body* carries, in wire order.

    Args:
        provider: A format or shim name; legacy spellings are accepted.  A
            name this module does not know yields an empty list.
        body: A provider request body.  Anything that is not a mapping, or
            that carries no ``tools`` list, yields an empty list.

    Returns:
        One :class:`ToolDeclaration` per declared tool, numbered among the
        named ones.  Empty for a body with no tools.
    """
    if not isinstance(body, dict):
        return []
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    reader = _READERS.get(_format_name(provider))
    if reader is None:
        return []
    numbered: list[ToolDeclaration] = []
    number = 0
    for declaration in reader(tools):
        if declaration.name:
            number += 1
            declaration = replace(declaration, number=number)
        numbered.append(declaration)
    return numbered


def _format_name(provider: str) -> str:
    """The format a provider/shim name denotes.

    Resolving shim names keeps a caller from having to know whether the route
    named a format or a vendor profile of one: ``openai--open_responses`` and
    ``open_responses`` describe the same body shape.
    """
    from llm_rosetta.contracts.resolver import resolve_format

    return resolve_format(provider)


# ---------------------------------------------------------------------------
# Per-format readers
# ---------------------------------------------------------------------------


def _read_chat_completions(tools: list[Any]) -> list[ToolDeclaration]:
    """Chat Completions: ``{"type": "function", "function": {...}}``."""
    declarations: list[ToolDeclaration] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        path = f"tools[{index}]"
        nested = tool.get("function")
        if isinstance(nested, dict):
            declarations.append(
                ToolDeclaration(
                    name=_str_or_none(nested.get("name")),
                    path=path,
                    wrapper="function",
                    schema_field="parameters",
                )
            )
            continue
        custom = tool.get("custom")
        if isinstance(custom, dict):
            declarations.append(
                ToolDeclaration(
                    name=_str_or_none(custom.get("name")),
                    path=path,
                    wrapper="custom",
                )
            )
            continue
        # A server tool (``{"type": "web_search"}``) or an unrecognised entry:
        # it can be pointed at by position, but it names no tool.
        declarations.append(
            ToolDeclaration(name=_str_or_none(tool.get("name")), path=path)
        )
    return declarations


def _read_messages(tools: list[Any]) -> list[ToolDeclaration]:
    """Anthropic Messages: flat entries with ``input_schema``."""
    declarations: list[ToolDeclaration] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        declarations.append(
            ToolDeclaration(
                name=_str_or_none(tool.get("name")),
                path=f"tools[{index}]",
                schema_field="input_schema",
            )
        )
    return declarations


def _read_open_responses(tools: list[Any]) -> list[ToolDeclaration]:
    """Open Responses: flat entries, plus ``type: namespace`` containers."""
    declarations: list[ToolDeclaration] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        path = f"tools[{index}]"
        children = tool.get("tools")
        if tool.get("type") == "namespace" and isinstance(children, list):
            for child_index, child in enumerate(children):
                if not isinstance(child, dict):
                    continue
                declarations.append(
                    ToolDeclaration(
                        name=_str_or_none(child.get("name")),
                        path=f"{path}.tools[{child_index}]",
                        schema_field=_schema_field(child),
                    )
                )
            continue
        declarations.append(
            ToolDeclaration(
                name=_str_or_none(tool.get("name")),
                path=path,
                schema_field=_schema_field(tool),
            )
        )
    return declarations


def _read_google_generate(tools: list[Any]) -> list[ToolDeclaration]:
    """Google generateContent: ``function_declarations`` inside each tool."""
    declarations: list[ToolDeclaration] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        path = f"tools[{index}]"
        key = _declaration_key(tool)
        if key is None:
            declarations.append(
                ToolDeclaration(
                    name=_str_or_none(tool.get("name")),
                    path=path,
                    schema_field="parameters",
                )
            )
            continue
        for child_index, child in enumerate(tool[key]):
            if not isinstance(child, dict):
                continue
            declarations.append(
                ToolDeclaration(
                    name=_str_or_none(child.get("name")),
                    path=f"{path}.{key}[{child_index}]",
                    schema_field="parameters",
                )
            )
    return declarations


def _read_google_interactions(tools: list[Any]) -> list[ToolDeclaration]:
    """Google Interactions: flat entries with ``parameters``."""
    declarations: list[ToolDeclaration] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        declarations.append(
            ToolDeclaration(
                name=_str_or_none(tool.get("name")),
                path=f"tools[{index}]",
                schema_field="parameters",
            )
        )
    return declarations


def _schema_field(tool: Mapping[Any, Any]) -> str | None:
    """The schema field a Responses-family declaration uses, if any.

    A custom tool carries its argument shape under ``schema``; a function
    carries it under ``parameters``.  Every other type (a server tool) carries
    none.
    """
    if tool.get("type") == "custom":
        return "schema" if isinstance(tool.get("schema"), dict) else None
    return "parameters"


def _declaration_key(tool: dict[str, Any]) -> str | None:
    """The key holding a Google tool's declarations, spelled as the body spells it."""
    for key in ("functionDeclarations", "function_declarations"):
        value = tool.get(key)
        if isinstance(value, list):
            return key
    return None


def _str_or_none(value: Any) -> str | None:
    """Return *value* when it is a non-empty string, else ``None``."""
    return value if isinstance(value, str) and value else None


_READERS = {
    "chat_completions": _read_chat_completions,
    "messages": _read_messages,
    "open_responses": _read_open_responses,
    "google_generate": _read_google_generate,
    "google_interactions": _read_google_interactions,
}

__all__ = ["ToolDeclaration", "read_tool_declarations"]
