"""ACP tool operations: what this dialect can and cannot say.

A client tool reaches an ACP agent through MCP, not through the prompt.  The
converter writes the declaration in MCP ``tools/list`` shape; the transport
serves it from a loopback MCP endpoint named on ``session/new`` and turns the
agent's ``tools/call`` into an IR tool call.  The client's result comes back on
a later request and is handed to the same pending MCP call.

The agent's own tools are a different thing: the gateway refuses them.  The
reader methods below understand ACP's ``toolCallId``/``title``/``rawInput``
shape so an agent-side call can be inspected when it is the thing being
refused.
"""

from __future__ import annotations

import json
from typing import Any

from llm_rosetta.ir.model.parts import ToolCallPart, ToolResultPart
from llm_rosetta.ir.model.tools import ToolCallConfig, ToolChoice, ToolDefinition
from ...base.tools.ops import BaseToolOps
from llm_rosetta.conversion_warnings import DROPPED, report_loss


def _drop(warnings: Any, subject: str, reason: str) -> None:
    """Report one declaration this lane cannot carry."""
    report_loss(reason, code=DROPPED, subject=subject, warnings=warnings)


class AcpToolOps(BaseToolOps):
    """Tool vocabulary of the ACP lane: MCP declarations, and no choice field."""

    @staticmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> Any:
        """Write one IR tool as an MCP ``tools/list`` item."""
        name = ir_tool.get("name")
        if not isinstance(name, str) or not name:
            return None
        parameters = ir_tool.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            parameters = {"type": "object", "properties": {}}
        return {
            "name": name,
            "description": str(ir_tool.get("description") or ""),
            "parameters": parameters,
        }

    @staticmethod
    def p_tool_definition_to_ir(
        provider_tool: Any, **kwargs: Any
    ) -> ToolDefinition | None:
        if not isinstance(provider_tool, dict):
            return None
        name = provider_tool.get("name") or provider_tool.get("title")
        if not isinstance(name, str) or not name:
            return None
        return ToolDefinition(
            type="function",
            name=name,
            description=str(provider_tool.get("description") or ""),
            parameters=provider_tool.get("inputSchema") or {"type": "object"},
        )

    @staticmethod
    def ir_tool_choice_to_p(ir_tool_choice: ToolChoice, **kwargs: Any) -> Any:
        _drop(
            kwargs.get("warnings"),
            "tool_choice",
            "ACP lanes expose tools over MCP and cannot force a choice; "
            "the setting was dropped.",
        )
        return None

    @staticmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        # The lane has no spelling for a choice, so whatever arrives reads as
        # the neutral one; the reader exists only to satisfy the contract.
        return ToolChoice(mode="auto")

    @staticmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        call = provider_tool_call if isinstance(provider_tool_call, dict) else {}
        raw = (
            call.get("rawInput")
            or call.get("raw_input")
            or call.get("tool_input")
            or {}
        )
        # A bridged call names the tool in ``name``; a built-in one may only
        # have a title.  The transport supplies the authoritative name for a
        # call it captured over MCP, so this reader is for the agent-side shape.
        name = call.get("name") or call.get("tool_name") or call.get("title") or ""
        if isinstance(name, str) and name.startswith("mcp"):
            # ACP clients commonly prefix MCP tools with ``mcp_<server>_``;
            # the suffix is the declared name, not a second tool.
            for separator in ("__", "_"):
                if separator in name:
                    name = name.rsplit(separator, 1)[-1]
                    break
        return ToolCallPart(
            type="tool_call",
            tool_call_id=str(call.get("toolCallId") or call.get("tool_call_id") or ""),
            tool_name=str(name),
            tool_input=raw if isinstance(raw, dict) else {},
            tool_type="function",
        )

    @staticmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        result = provider_tool_result if isinstance(provider_tool_result, dict) else {}
        content = result.get("content")
        if isinstance(content, (dict, list)):
            text = json.dumps(content, ensure_ascii=False)
        else:
            text = str(content or "")
        return ToolResultPart(
            type="tool_result",
            tool_call_id=str(result.get("toolCallId") or ""),
            result=text,
            is_error=bool(result.get("isError") or result.get("is_error")),
        )

    @staticmethod
    def ir_tool_config_to_p(ir_tool_config: ToolCallConfig, **kwargs: Any) -> Any:
        _drop(
            kwargs.get("warnings"),
            "tool_config",
            "ACP lanes carry no tool configuration; the setting was dropped.",
        )
        return None

    @staticmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        # No configuration travels on this lane; an empty one is the truth.
        return ToolCallConfig()


__all__ = ["AcpToolOps"]
