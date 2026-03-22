"""
LLM-Rosetta - OpenAI Chat Tool Operations

OpenAI Chat Completions API tool conversion operations.
Handles bidirectional conversion of tool definitions, calls, results,
choice strategies, and call configurations.

Self-contained: does not depend on utils/ToolCallConverter or utils/ToolConverter.
"""

import json
from typing import Any, cast

from ...types.ir import (
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)
from ...types.ir.tools import ToolCallConfig
from ..base import BaseToolOps

# JSON Schema keywords not supported by OpenAI / Vertex AI compatible
# endpoints.  These are valid per the JSON Schema spec but upstream servers
# (e.g. Vertex AI's OpenAI-compatible layer) reject them with Pydantic
# ``extra='forbid'`` validation errors.
_UNSUPPORTED_SCHEMA_KEYS: set[str] = {
    "propertyNames",
    "const",
    "$comment",
    "$id",
    "$anchor",
    "$dynamicAnchor",
    "$dynamicRef",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
    "deprecated",
    "readOnly",
    "writeOnly",
    "examples",
}

# Keys that hold definition maps (consumed for $ref resolution, then removed).
_DEFS_KEYS: set[str] = {"$defs", "definitions"}


def _flatten_combination(schema: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``anyOf``/``oneOf`` nullable patterns into a simple typed schema.

    Vertex AI's OpenAI-compatible layer does not support ``anyOf``/``oneOf``
    at all.  The most common pattern is a nullable union like
    ``{"anyOf": [{"type": "string"}, {"type": "null"}]}``, which we convert to
    ``{"type": "string", "nullable": true}``.

    For single-variant unions we unwrap directly.  For multi-type (non-null)
    unions we keep only the first non-null variant (lossy but safe).

    ``allOf`` with a single element is simply unwrapped.

    Args:
        schema: A schema dict that may contain ``anyOf``/``oneOf``/``allOf``.

    Returns:
        A new dict with combination keywords resolved.
    """
    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue

        non_null = [v for v in variants if v.get("type") != "null"]
        has_null = len(non_null) < len(variants)

        # Preserve sibling metadata (description, title, etc.)
        base: dict[str, Any] = {
            k: v for k, v in schema.items() if k not in ("anyOf", "oneOf", "allOf")
        }

        if len(non_null) == 1:
            # Common nullable pattern: merge the single real type
            base.update(non_null[0])
        elif len(non_null) > 1:
            # Multiple non-null types: pick the first (lossy but avoids rejection)
            base.update(non_null[0])
        # else: all variants are null → just mark nullable

        if has_null:
            base["nullable"] = True

        return base

    # allOf with a single element: unwrap
    all_of = schema.get("allOf")
    if isinstance(all_of, list) and len(all_of) == 1 and isinstance(all_of[0], dict):
        base = {k: v for k, v in schema.items() if k != "allOf"}
        base.update(all_of[0])
        return base

    return schema


def _resolve_ref(ref: str, defs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Resolve a JSON Schema ``$ref`` pointer against collected definitions.

    Only local definition references (``#/$defs/Name`` or
    ``#/definitions/Name``) are supported.  Unresolvable refs return an
    empty dict so the caller can proceed without crashing.

    Args:
        ref: The ``$ref`` string value.
        defs: Merged definitions from ``$defs`` and ``definitions``.

    Returns:
        The referenced schema dict, or ``{}`` if unresolvable.
    """
    for prefix in ("#/$defs/", "#/definitions/"):
        if ref.startswith(prefix):
            name = ref[len(prefix) :]
            return defs.get(name, {})
    return {}


def _sanitize_schema(
    schema: dict[str, Any],
    defs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recursively remove unsupported JSON Schema keywords.

    Also resolves ``$ref`` references by inlining the referenced definition,
    and flattens ``anyOf``/``oneOf``/``allOf`` combination keywords into
    simple typed schemas, as required by Vertex AI's OpenAI-compatible layer
    which does not support these constructs at all.

    Args:
        schema: A JSON Schema dict (or sub-schema).
        defs: Collected ``$defs``/``definitions`` from the top-level schema.
            Populated automatically on the first call if the schema contains
            definition maps.

    Returns:
        A new dict with unsupported keys removed at every level.
    """
    # On first call, collect $defs/definitions for $ref resolution.
    if defs is None:
        defs = {}
        for key in _DEFS_KEYS:
            d = schema.get(key)
            if isinstance(d, dict):
                defs.update(d)

    # Resolve $ref: inline the referenced definition (merge siblings).
    ref = schema.get("$ref")
    if isinstance(ref, str) and defs:
        resolved = _resolve_ref(ref, defs)
        if resolved:
            # Siblings of $ref (e.g. description) are kept; $ref itself is
            # replaced by the resolved definition's content.
            merged = {k: v for k, v in schema.items() if k != "$ref"}
            merged.update(resolved)
            return _sanitize_schema(merged, defs)

    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS or key in _DEFS_KEYS:
            continue
        if key == "$ref":
            # Unresolvable $ref — drop it to avoid upstream rejection.
            continue
        if isinstance(value, dict):
            result[key] = _sanitize_schema(value, defs)
        elif isinstance(value, list):
            result[key] = [
                _sanitize_schema(item, defs) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    # Flatten combination keywords (anyOf/oneOf/allOf) into simple types.
    if result.keys() & {"anyOf", "oneOf", "allOf"}:
        result = _flatten_combination(result)

    return result


class OpenAIChatToolOps(BaseToolOps):
    """OpenAI Chat Completions tool conversion operations.

    All methods are static and stateless. Handles tool definitions,
    calls, results, choice strategies, and call configurations.
    """

    # ==================== Tool Definition ====================

    @staticmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> dict:
        """IR ToolDefinition → OpenAI Chat tool definition.

        Converts flat IR format to nested OpenAI ``{"type":"function","function":{...}}``
        format.  Unsupported JSON Schema keywords (e.g. ``propertyNames``) are
        recursively stripped from ``parameters``.

        Args:
            ir_tool: IR tool definition.

        Returns:
            OpenAI Chat tool definition dict.
        """
        if ir_tool.get("type", "function") == "function":
            func_def: dict[str, Any] = {
                "name": ir_tool["name"],
                "description": ir_tool.get("description", ""),
            }
            parameters = ir_tool.get("parameters")
            if parameters and isinstance(parameters, dict):
                func_def["parameters"] = _sanitize_schema(parameters)
            elif parameters:
                func_def["parameters"] = parameters
            return {"type": "function", "function": func_def}

        # Non-function tool types: wrap as custom
        raw_params = ir_tool.get("parameters", {})
        params = (
            _sanitize_schema(raw_params) if isinstance(raw_params, dict) else raw_params
        )
        return {
            "type": "function",
            "function": {
                "name": f"{ir_tool['type']}_{ir_tool['name']}",
                "description": ir_tool.get("description", ""),
                "parameters": params,
            },
        }

    @staticmethod
    def p_tool_definition_to_ir(provider_tool: Any, **kwargs: Any) -> ToolDefinition:
        """OpenAI Chat tool definition → IR ToolDefinition.

        Converts nested OpenAI format to flat IR format.

        Args:
            provider_tool: OpenAI Chat tool definition dict.

        Returns:
            IR ToolDefinition.
        """
        func = provider_tool.get("function", {})
        result: dict[str, Any] = {
            "type": "function",
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        }

        # Extract required_parameters from JSON Schema if available
        parameters = func.get("parameters", {})
        if isinstance(parameters, dict) and "required" in parameters:
            result["required_parameters"] = parameters["required"]
        else:
            result["required_parameters"] = []

        result["metadata"] = {}
        return cast(ToolDefinition, result)

    # ==================== Tool Choice ====================

    @staticmethod
    def ir_tool_choice_to_p(ir_tool_choice: ToolChoice, **kwargs: Any) -> str | dict:
        """IR ToolChoice → OpenAI Chat tool_choice parameter.

        Mapping:
        - ``mode:"none"`` → ``"none"``
        - ``mode:"auto"`` → ``"auto"``
        - ``mode:"any"`` → ``"required"``
        - ``mode:"tool"`` → ``{"type":"function","function":{"name":"..."}}``

        Args:
            ir_tool_choice: IR tool choice.

        Returns:
            OpenAI tool_choice value (string or dict).
        """
        mode = ir_tool_choice.get("mode", "auto")

        if mode == "none":
            return "none"
        elif mode == "auto":
            return "auto"
        elif mode == "any":
            return "required"
        elif mode == "tool":
            tool_name = ir_tool_choice.get("tool_name")
            if tool_name:
                return {"type": "function", "function": {"name": tool_name}}
            return "required"

        return "auto"

    @staticmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        """OpenAI Chat tool_choice → IR ToolChoice.

        Mapping:
        - ``"none"`` → ``mode:"none"``
        - ``"auto"`` → ``mode:"auto"``
        - ``"required"`` → ``mode:"any"``
        - ``{"type":"function","function":{"name":"..."}}`` → ``mode:"tool"``

        Args:
            provider_tool_choice: OpenAI tool_choice value.

        Returns:
            IR ToolChoice.
        """
        if isinstance(provider_tool_choice, str):
            if provider_tool_choice == "none":
                return cast(ToolChoice, {"mode": "none", "tool_name": ""})
            elif provider_tool_choice == "auto":
                return cast(ToolChoice, {"mode": "auto", "tool_name": ""})
            elif provider_tool_choice == "required":
                return cast(ToolChoice, {"mode": "any", "tool_name": ""})
            return cast(ToolChoice, {"mode": "auto", "tool_name": ""})

        if isinstance(provider_tool_choice, dict):
            if provider_tool_choice.get("type") == "function":
                func = provider_tool_choice.get("function", {})
                return cast(
                    ToolChoice, {"mode": "tool", "tool_name": func.get("name", "")}
                )

        return cast(ToolChoice, {"mode": "auto", "tool_name": ""})

    # ==================== Tool Call ====================

    @staticmethod
    def ir_tool_call_to_p(ir_tool_call: ToolCallPart, **kwargs: Any) -> dict:
        """IR ToolCallPart → OpenAI Chat tool call.

        Converts ``tool_input`` dict to JSON string ``arguments``.

        Args:
            ir_tool_call: IR tool call part.

        Returns:
            OpenAI Chat tool call dict.
        """
        tool_input = ir_tool_call.get("tool_input", {})
        arguments = (
            json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
        )

        return {
            "id": ir_tool_call["tool_call_id"],
            "type": "function",
            "function": {
                "name": ir_tool_call["tool_name"],
                "arguments": arguments,
            },
        }

    @staticmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        """OpenAI Chat tool call → IR ToolCallPart.

        Parses JSON string ``arguments`` back to dict.

        Args:
            provider_tool_call: OpenAI Chat tool call dict.

        Returns:
            IR ToolCallPart.
        """
        func = provider_tool_call.get("function", {})
        arguments_str = func.get("arguments", "{}")

        try:
            tool_input = json.loads(arguments_str) if arguments_str else {}
        except (json.JSONDecodeError, TypeError):
            tool_input = {"raw_arguments": arguments_str}

        return ToolCallPart(
            type="tool_call",
            tool_call_id=provider_tool_call.get("id", ""),
            tool_name=func.get("name", ""),
            tool_input=tool_input,
            tool_type="function",
        )

    # ==================== Tool Result ====================

    @staticmethod
    def ir_tool_result_to_p(ir_tool_result: ToolResultPart, **kwargs: Any) -> dict:
        """IR ToolResultPart → OpenAI Chat tool role message.

        Args:
            ir_tool_result: IR tool result part.

        Returns:
            OpenAI tool role message dict.
        """
        result = ir_tool_result.get("result", "")
        content = str(result) if result is not None else ""

        return {
            "role": "tool",
            "tool_call_id": ir_tool_result["tool_call_id"],
            "content": content,
        }

    @staticmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        """OpenAI Chat tool role message → IR ToolResultPart.

        Args:
            provider_tool_result: OpenAI tool role message dict.

        Returns:
            IR ToolResultPart.
        """
        return ToolResultPart(
            type="tool_result",
            tool_call_id=provider_tool_result.get("tool_call_id", ""),
            result=provider_tool_result.get("content", ""),
        )

    # ==================== Tool Config ====================

    @staticmethod
    def ir_tool_config_to_p(ir_tool_config: ToolCallConfig, **kwargs: Any) -> dict:
        """IR ToolCallConfig → OpenAI Chat tool call config fields.

        Mapping:
        - ``disable_parallel`` → ``parallel_tool_calls`` (inverted)

        Args:
            ir_tool_config: IR tool call config.

        Returns:
            Dict of OpenAI request fields to merge.
        """
        result: dict[str, Any] = {}

        if "disable_parallel" in ir_tool_config:
            result["parallel_tool_calls"] = not ir_tool_config["disable_parallel"]

        # max_calls is not supported by OpenAI Chat
        return result

    @staticmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        """OpenAI Chat tool call config → IR ToolCallConfig.

        Mapping:
        - ``parallel_tool_calls`` → ``disable_parallel`` (inverted)

        Args:
            provider_tool_config: Dict with OpenAI tool config fields.

        Returns:
            IR ToolCallConfig.
        """
        result: dict[str, Any] = {}

        if isinstance(provider_tool_config, dict):
            parallel = provider_tool_config.get("parallel_tool_calls")
            if parallel is not None:
                result["disable_parallel"] = not parallel

        return cast(ToolCallConfig, result)
