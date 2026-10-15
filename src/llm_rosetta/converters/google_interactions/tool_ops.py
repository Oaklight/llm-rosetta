"""
LLM-Rosetta - Google Interactions Tool Operations

Bidirectional conversion between Interactions API Tool/FunctionCallStep
and IR ToolDefinition/ToolCallPart/ToolResultPart.
"""

import json
import logging
from typing import Any, cast

from ...types.ir import ToolCallPart, ToolResultPart
from ...types.ir.tools import ToolChoice, ToolDefinition
from ..base import BaseToolOps
from ..base.tools.intrinsic import (
    get_native_definition,
    intrinsic_kind_of,
    make_intrinsic_tool_definition,
    make_intrinsic_tool_call,
    make_intrinsic_tool_result,
)


logger = logging.getLogger(__name__)


SERVER_CALL_TYPES: dict[str, str] = {
    "google_search_call": "google_search",
    "code_execution_call": "code_execution",
    "google_maps_call": "google_maps",
    "file_search_call": "file_search",
    "retrieval_call": "retrieval",
    "url_context_call": "url_context",
}

SERVER_RESULT_TYPES: dict[str, str] = {
    "google_search_result": "google_search",
    "code_execution_result": "code_execution",
    "google_maps_result": "google_maps",
    "file_search_result": "file_search",
    "retrieval_result": "retrieval",
    "url_context_result": "url_context",
}

INTRINSIC_KIND_TO_CALL_TYPE = {v: k for k, v in SERVER_CALL_TYPES.items()}
INTRINSIC_KIND_TO_RESULT_TYPE = {v: k for k, v in SERVER_RESULT_TYPES.items()}

# Native Interactions server tools for client-declared intrinsic kinds.
_GI_INTRINSIC_TOOLS: dict[str, dict[str, Any]] = {
    "google_search": {"type": "google_search"},
    "code_execution": {"type": "code_execution"},
    "url_context": {"type": "url_context"},
    "google_maps": {"type": "google_maps"},
    "file_search": {"type": "file_search"},
    "retrieval": {"type": "retrieval"},
}

# Native Interactions server-tool type → canonical intrinsic kind.
_GI_NATIVE_TYPE_TO_KIND: dict[str, str] = {
    v["type"]: k for k, v in _GI_INTRINSIC_TOOLS.items()
}


class GoogleInteractionsToolOps(BaseToolOps):
    """Google Interactions API tool conversion operations."""

    # ── Tool definitions ───────────────────────────────────────────

    @staticmethod
    def ir_tool_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> dict:
        """IR ToolDefinition → Interactions Function tool."""
        if ir_tool.get("type") == "intrinsic":
            stored = get_native_definition(ir_tool, "google_interactions")
            if stored is not None:
                return dict(stored)
            kind = intrinsic_kind_of(ir_tool)
            native = _GI_INTRINSIC_TOOLS.get(kind)
            if native is None:
                logger.warning(
                    "Google Interactions has no server tool for intrinsic kind"
                    " %r; dropping",
                    kind,
                )
                return {}
            return dict(native)

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
        kind = _GI_NATIVE_TYPE_TO_KIND.get(tool_type)
        if kind is not None:
            return cast(
                ToolDefinition,
                make_intrinsic_tool_definition(
                    kind,
                    description=provider_tool.get("description", ""),
                    native=provider_tool,
                    native_base="google_interactions",
                ),
            )
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
    def p_function_call_to_ir(step: dict, **kwargs: Any) -> ToolCallPart:
        """Interactions FunctionCallStep → IR ToolCallPart."""
        result: dict[str, Any] = {
            "type": "tool_call",
            "tool_call_id": step["id"],
            "tool_name": step["name"],
            "tool_input": step.get("arguments", {}),
        }
        preserve_metadata = kwargs.get("preserve_metadata", True)
        signature = step.get("signature")
        if preserve_metadata and signature:
            result.setdefault("provider_metadata", {}).setdefault("google", {})[
                "thought_signature"
            ] = signature
        return cast(ToolCallPart, result)

    @staticmethod
    def ir_function_call_to_p(ir_part: ToolCallPart, **kwargs: Any) -> dict:
        """IR ToolCallPart → Interactions FunctionCallStep."""
        preserve_metadata = kwargs.get("preserve_metadata", True)
        if ir_part.get("tool_type") == "intrinsic":
            # Native server-call steps (``google_search_call``, ...) take no
            # ``name``: the API rejects a body carrying one with ``400 Unknown
            # parameter 'name' at 'input[0]'``.  They do carry a ``signature``
            # on the wire, so restore the one the read path kept.
            intrinsic_kind = intrinsic_kind_of(ir_part)
            step_type = INTRINSIC_KIND_TO_CALL_TYPE.get(intrinsic_kind)
            if step_type is None:
                # A kind this dialect cannot render (``web_search`` arriving
                # from another lane, …) has no native step type; drop the call
                # rather than fabricate ``{kind}_call`` — a step type the API
                # does not define — or downgrade it to a client
                # ``function_call``.  Matches the streaming writer and the
                # sibling converters.
                return {}
            step: dict[str, Any] = {
                "type": step_type,
                "id": ir_part["tool_call_id"],
                "arguments": ir_part["tool_input"],
            }
            pm = ir_part.get("provider_metadata") or {}
            sig = pm.get("google", {}).get("thought_signature")
            if preserve_metadata and sig:
                step["signature"] = sig
            return step
        step: dict[str, Any] = {
            "type": "function_call",
            "id": ir_part["tool_call_id"],
            "name": ir_part["tool_name"],
            "arguments": ir_part["tool_input"],
        }
        # ``provider_metadata`` may be present-but-None (optional IR field),
        # so guard with ``or {}`` rather than a default.
        pm = ir_part.get("provider_metadata") or {}
        sig = pm.get("google", {}).get("thought_signature")
        if preserve_metadata and sig:
            step["signature"] = sig
        return step

    # ── Server step calls ─────────────────────────────────────────

    @staticmethod
    def p_server_call_to_ir(step: dict, **kwargs: Any) -> ToolCallPart:
        """Interactions server call step (google_search_call, etc.) → IR ToolCallPart.

        The native step carries no ``name``; its identity is the canonical
        intrinsic kind derived from the step type.  It does carry a
        ``signature``, which a replay needs, so it is kept on the part's
        ``provider_metadata`` and restored by ``ir_function_call_to_p``.
        """
        step_type = step.get("type", "")
        intrinsic_kind = SERVER_CALL_TYPES.get(step_type, step_type)
        part = cast(
            ToolCallPart,
            make_intrinsic_tool_call(
                tool_call_id=step.get("id", ""),
                tool_name=intrinsic_kind,
                tool_input=step.get("arguments", {}),
                intrinsic_kind=intrinsic_kind,
            ),
        )
        preserve_metadata = kwargs.get("preserve_metadata", True)
        signature = step.get("signature")
        if preserve_metadata and signature:
            part.setdefault("provider_metadata", {}).setdefault("google", {})[
                "thought_signature"
            ] = signature
        return part

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
        # Keep the function name the step carries so it survives a passthrough
        # and a result-only replay (no function_call in the batch to resolve it
        # from) — see ``ir_function_result_to_p``.
        name = step.get("name")
        if name:
            part.setdefault("provider_metadata", {}).setdefault("google", {})[
                "tool_name"
            ] = name
        if step.get("is_error"):
            part["is_error"] = True
        return part

    @staticmethod
    def ir_function_result_to_p(ir_part: ToolResultPart, **kwargs: Any) -> dict:
        """IR ToolResultPart → Interactions FunctionResultStep.

        A ``function_result`` step requires the function ``name``; the IR part
        only carries ``tool_call_id``, so the caller passes a
        ``tool_call_index`` (``tool_call_id → tool_name``) resolved from the
        message history. Without it the API rejects the step with
        ``function_response.name: Name cannot be empty``.
        """
        if ir_part.get("tool_type") == "intrinsic":
            intrinsic_kind = intrinsic_kind_of(ir_part)
            step_type = INTRINSIC_KIND_TO_RESULT_TYPE.get(intrinsic_kind)
            if step_type is None:
                # Unrenderable kind: drop the result rather than fabricate
                # ``{kind}_result`` (see ``ir_function_call_to_p``).
                return {}
            result: dict[str, Any] = {
                "type": step_type,
                "call_id": ir_part["tool_call_id"],
                "result": ir_part["result"],
            }
            if ir_part.get("is_error"):
                result["is_error"] = True
            return result

        call_id = ir_part["tool_call_id"]
        result = {
            "type": "function_result",
            "call_id": call_id,
            "result": ir_part["result"],
        }
        tool_call_index = kwargs.get("tool_call_index") or {}
        tool_name = tool_call_index.get(call_id)
        if not tool_name:
            # A name captured on the way in (passthrough / result-only replay).
            tool_name = (
                (ir_part.get("provider_metadata") or {})
                .get("google", {})
                .get("tool_name")
            )
        if not tool_name:
            # No function_call in the history and no stored name — fall back to
            # the call_id (mirrors google_generate's behaviour). A result with
            # an empty call_id and no name at all is unrecoverable: GI requires
            # a non-empty ``name``, so fail loudly rather than emit a body the
            # API will reject with ``function_response.name: Name cannot be
            # empty``.
            if not call_id:
                raise ValueError(
                    "function_result has no resolvable function name: the"
                    " tool_call_id is empty and no matching function_call was"
                    " found in the history"
                )
            tool_name = call_id
            logger.warning(
                "Could not resolve tool name for function_result %r; using the"
                " call_id as the function name",
                call_id,
            )
        result["name"] = tool_name
        if ir_part.get("is_error"):
            result["is_error"] = True
        return result

    # ── Server step results ───────────────────────────────────────

    @staticmethod
    def p_server_result_to_ir(step: dict) -> ToolResultPart:
        """Interactions server result step (google_search_result, etc.) → IR ToolResultPart.

        The result payload is kept exactly as the wire carries it.  A
        ``google_search_result`` is an array of result blocks and a
        ``code_execution_result`` is a string; the API rejects a replayed step
        whose ``result`` does not match that shape (``Expected an array, got
        string``), so the two must not be flattened to a common string.
        """
        step_type = step.get("type", "")
        intrinsic_kind = SERVER_RESULT_TYPES.get(step_type, step_type)
        return cast(
            ToolResultPart,
            make_intrinsic_tool_result(
                tool_call_id=step.get("call_id", step.get("id", "")),
                result=step.get("result", ""),
                intrinsic_kind=intrinsic_kind,
                is_error=bool(step.get("is_error")),
            ),
        )

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

    @classmethod
    def ir_tool_call_to_p(cls, ir_tool_call: ToolCallPart, **kwargs: Any) -> Any:
        return GoogleInteractionsToolOps.ir_function_call_to_p(ir_tool_call, **kwargs)

    @staticmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        return GoogleInteractionsToolOps.p_function_call_to_ir(
            provider_tool_call, **kwargs
        )

    @classmethod
    def ir_tool_result_to_p(cls, ir_tool_result: ToolResultPart, **kwargs: Any) -> Any:
        return GoogleInteractionsToolOps.ir_function_result_to_p(
            ir_tool_result, **kwargs
        )

    @staticmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        return GoogleInteractionsToolOps.p_function_result_to_ir(provider_tool_result)

    @staticmethod
    def ir_tool_config_to_p(ir_tool_config: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> Any:
        return {}
