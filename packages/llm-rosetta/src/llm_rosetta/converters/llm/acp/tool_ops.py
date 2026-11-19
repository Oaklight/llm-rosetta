"""ACP tool operations: what this dialect can and cannot say.

Phase 1 of the Antigravity channel carries text, thinking and images.  The
agent behind ACP owns its own tools, and the client's tools reach it over an
MCP bridge this dialect does not build yet, so a tool declared on the request
cannot be honoured.  Refusing it here keeps the loss visible: the pipeline
reports the warning, and a client is never told its tools are available when
they are not.

The provider→IR readers exist because the base class requires them.  They are
written for the shape ACP uses (a call with ``toolCallId``/``title`` and a
JSON ``rawInput``), so a later phase that surfaces agent-side calls can read
them without rewriting this module.
"""

from __future__ import annotations

import json
from typing import Any

from ....types.ir.parts import ToolCallPart, ToolResultPart
from ....types.ir.tools import ToolCallConfig, ToolChoice, ToolDefinition
from ...base.tools.ops import BaseToolOps
from llm_rosetta.conversion_warnings import DROPPED, report_loss


def _drop(warnings: Any, subject: str) -> None:
    """Report one declaration this lane cannot carry."""
    report_loss(
        "ACP lanes cannot declare client tools yet; the declaration was dropped.",
        code=DROPPED,
        subject=subject,
        warnings=warnings,
    )


class AcpToolOps(BaseToolOps):
    """Tool vocabulary of the ACP lane: none, and says so."""

    @staticmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> Any:
        _drop(kwargs.get("warnings"), f"tools.{ir_tool.get('name', '')}")
        return None

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
        _drop(kwargs.get("warnings"), "tool_choice")
        return None

    @staticmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        # The lane has no spelling for a choice, so whatever arrives reads as
        # the neutral one; the reader exists only to satisfy the contract.
        return ToolChoice(mode="auto")

    @staticmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        call = provider_tool_call if isinstance(provider_tool_call, dict) else {}
        raw = call.get("rawInput") or call.get("raw_input") or {}
        return ToolCallPart(
            type="tool_call",
            tool_call_id=str(call.get("toolCallId") or call.get("tool_call_id") or ""),
            tool_name=str(call.get("title") or call.get("name") or ""),
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
        _drop(kwargs.get("warnings"), "tool_config")
        return None

    @staticmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        # No configuration travels on this lane; an empty one is the truth.
        return ToolCallConfig()


__all__ = ["AcpToolOps"]
