"""
LLM-Rosetta - Google Interactions Tool Operations

Bidirectional conversion between Interactions API Tool/FunctionCallStep
and IR ToolDefinition/ToolCallPart/ToolResultPart.
"""

import json
from typing import Any

from ...types.ir import ToolCallPart, ToolResultPart
from ...types.ir.tools import ToolChoice, ToolDefinition
from ..base import BaseToolOps


_SERVER_CALL_TYPES: dict[str, str] = {
    "google_search_call": "google_search",
    "code_execution_call": "code_execution",
    "google_maps_call": "google_maps",
    "file_search_call": "file_search",
    "retrieval_call": "retrieval",
    "url_context_call": "url_context",
}

_SERVER_RESULT_TYPES: dict[str, str] = {
    "google_search_result": "google_search",
    "code_execution_result": "code_execution",
    "google_maps_result": "google_maps",
    "file_search_result": "file_search",
    "retrieval_result": "retrieval",
    "url_context_result": "url_context",
}

_INTRINSIC_KIND_TO_CALL_TYPE = {v: k for k, v in _SERVER_CALL_TYPES.items()}
_INTRINSIC_KIND_TO_RESULT_TYPE = {v: k for k, v in _SERVER_RESULT_TYPES.items()}


class GoogleInteractionsToolOps(BaseToolOps):
    """Google Interactions API tool conversion operations."""

    # ── Tool definitions ───────────────────────────────────────────

    @staticmethod
    def ir_tool_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> dict:
        """IR ToolDefinition → Interactions Function tool."""
        result: dict[str, Any] = {
            "type": "function",
            "name": ir_tool["name"],
        }
        if ir_tool.get("description"):
            result["description"] = ir_tool["description"]
        if ir_tool.get("parameters"):
            result["parameters"] = ir_tool["parameters"]
        return result

    @staticmethod
    def p_tool_to_ir(provider_tool: Any, **kwargs: Any) -> ToolDefinition:
        """Interactions Function tool → IR ToolDefinition."""
        tool_type = provider_tool.get("type", "function")
        ir_type = "mcp" if tool_type == "mcp_server" else "function"
        result: ToolDefinition = {
            "type": ir_type,
            "name": provider_tool.get("name", ""),
            "description": provider_tool.get("description", ""),
            "parameters": provider_tool.get("parameters", {}),
        }
        if tool_type == "mcp_server":
            result["metadata"] = {
                "url": provider_tool.get("url"),
                "provider_tool_type": "mcp_server",
            }
        return result

    # ── Tool calls (FunctionCallStep) ──────────────────────────────

    @staticmethod
    def p_function_call_to_ir(step: dict) -> ToolCallPart:
        """Interactions FunctionCallStep → IR ToolCallPart."""
        return {
            "type": "tool_call",
            "tool_call_id": step["id"],
            "tool_name": step["name"],
            "tool_input": step.get("arguments", {}),
        }

    @staticmethod
    def ir_function_call_to_p(ir_part: ToolCallPart) -> dict:
        """IR ToolCallPart → Interactions FunctionCallStep."""
        if ir_part.get("tool_type") == "intrinsic":
            pm = ir_part.get("provider_metadata") or {}
            intrinsic_kind = pm.get("intrinsic_kind", ir_part["tool_name"])
            step_type = _INTRINSIC_KIND_TO_CALL_TYPE.get(
                intrinsic_kind, f"{intrinsic_kind}_call"
            )
            return {
                "type": step_type,
                "id": ir_part["tool_call_id"],
                "name": ir_part["tool_name"],
                "arguments": ir_part["tool_input"],
            }
        return {
            "type": "function_call",
            "id": ir_part["tool_call_id"],
            "name": ir_part["tool_name"],
            "arguments": ir_part["tool_input"],
        }

    # ── Server step calls ─────────────────────────────────────────

    @staticmethod
    def p_server_call_to_ir(step: dict) -> ToolCallPart:
        """Interactions server call step (google_search_call, etc.) → IR ToolCallPart."""
        step_type = step.get("type", "")
        intrinsic_kind = _SERVER_CALL_TYPES.get(step_type, step_type)
        return {
            "type": "tool_call",
            "tool_call_id": step.get("id", ""),
            "tool_name": step.get("name", step_type),
            "tool_input": step.get("arguments", {}),
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": intrinsic_kind},
        }

    # ── Tool results (FunctionResultStep) ──────────────────────────

    @staticmethod
    def p_function_result_to_ir(step: dict) -> ToolResultPart:
        """Interactions FunctionResultStep → IR ToolResultPart."""
        result_val = step.get("result", "")
        if isinstance(result_val, list):
            texts = [
                item.get("text", "") for item in result_val if isinstance(item, dict)
            ]
            result_val = "\n".join(texts) if texts else ""
        elif isinstance(result_val, dict):
            result_val = json.dumps(result_val)
        part: ToolResultPart = {
            "type": "tool_result",
            "tool_call_id": step.get("call_id", step.get("id", "")),
            "result": result_val,
        }
        if step.get("is_error"):
            part["is_error"] = True
        return part

    @staticmethod
    def ir_function_result_to_p(ir_part: ToolResultPart) -> dict:
        """IR ToolResultPart → Interactions FunctionResultStep."""
        if ir_part.get("tool_type") == "intrinsic":
            pm = ir_part.get("provider_metadata") or {}
            intrinsic_kind = pm.get("intrinsic_kind", "")
            step_type = _INTRINSIC_KIND_TO_RESULT_TYPE.get(
                intrinsic_kind, f"{intrinsic_kind}_result"
            )
            result: dict[str, Any] = {
                "type": step_type,
                "call_id": ir_part["tool_call_id"],
                "result": ir_part["result"],
            }
            if ir_part.get("is_error"):
                result["is_error"] = True
            return result
        result = {
            "type": "function_result",
            "call_id": ir_part["tool_call_id"],
            "result": ir_part["result"],
        }
        if ir_part.get("is_error"):
            result["is_error"] = True
        return result

    # ── Server step results ───────────────────────────────────────

    @staticmethod
    def p_server_result_to_ir(step: dict) -> ToolResultPart:
        """Interactions server result step (google_search_result, etc.) → IR ToolResultPart."""
        step_type = step.get("type", "")
        intrinsic_kind = _SERVER_RESULT_TYPES.get(step_type, step_type)
        result_val = step.get("result", "")
        if isinstance(result_val, list):
            texts = [
                item.get("text", "") for item in result_val if isinstance(item, dict)
            ]
            result_val = "\n".join(texts) if texts else ""
        elif isinstance(result_val, dict):
            result_val = json.dumps(result_val)
        part: ToolResultPart = {
            "type": "tool_result",
            "tool_call_id": step.get("call_id", step.get("id", "")),
            "result": result_val,
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": intrinsic_kind},
        }
        if step.get("is_error"):
            part["is_error"] = True
        return part

    # ── Tool choice ────────────────────────────────────────────────

    @staticmethod
    def ir_tool_choice_to_p(ir_tool_choice: ToolChoice, **kwargs: Any) -> Any:
        """IR ToolChoice → Interactions tool_choice."""
        mode = ir_tool_choice["mode"]
        if mode in ("auto", "any", "none"):
            return mode
        if mode == "tool":
            tool_name = ir_tool_choice.get("tool_name")
            if tool_name:
                return {
                    "allowed_tools": {
                        "mode": "any",
                        "tools": [tool_name],
                    }
                }
            return "any"
        return "auto"

    @staticmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        """Interactions tool_choice → IR ToolChoice."""
        if isinstance(provider_tool_choice, str):
            if provider_tool_choice in ("auto", "any", "none"):
                return {"mode": provider_tool_choice}
            return {"mode": "auto"}
        if isinstance(provider_tool_choice, dict):
            allowed = provider_tool_choice.get("allowed_tools", {})
            tools = allowed.get("tools", [])
            if tools:
                # IR ToolChoice only supports a single tool name; extra tools are dropped
                return {"mode": "tool", "tool_name": tools[0]}
            mode = allowed.get("mode", "auto")
            return {"mode": mode if mode in ("auto", "any", "none") else "auto"}
        return {"mode": "auto"}

    # ── Base class abstract method implementations ──────────────────

    @staticmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> Any:
        return GoogleInteractionsToolOps.ir_tool_to_p(ir_tool, **kwargs)

    @staticmethod
    def p_tool_definition_to_ir(
        provider_tool: Any, **kwargs: Any
    ) -> ToolDefinition | list[ToolDefinition] | None:
        return GoogleInteractionsToolOps.p_tool_to_ir(provider_tool, **kwargs)

    @staticmethod
    def ir_tool_call_to_p(ir_tool_call: ToolCallPart, **kwargs: Any) -> Any:
        return GoogleInteractionsToolOps.ir_function_call_to_p(ir_tool_call)

    @staticmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        return GoogleInteractionsToolOps.p_function_call_to_ir(provider_tool_call)

    @staticmethod
    def ir_tool_result_to_p(ir_tool_result: ToolResultPart, **kwargs: Any) -> Any:
        return GoogleInteractionsToolOps.ir_function_result_to_p(ir_tool_result)

    @staticmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        return GoogleInteractionsToolOps.p_function_result_to_ir(provider_tool_result)

    @staticmethod
    def ir_tool_config_to_p(ir_tool_config: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> Any:
        return {}
