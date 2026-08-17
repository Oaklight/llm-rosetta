"""
LLM-Rosetta - Google GenAI Tool Operations

Google GenAI API tool conversion operations.
Handles bidirectional conversion of tool definitions, calls, results,
choice strategies, and call configurations.

Self-contained: does not depend on utils/ToolCallConverter or utils/ToolConverter.

Google-specific:
- Tool definitions use FunctionDeclaration wrapped in a Tool object
- Tool calls use function_call Part (name + args dict)
- Tool results use function_response Part (name + response dict)
- Tool choice uses ToolConfig → FunctionCallingConfig (mode: NONE/AUTO/ANY)
"""

import warnings
from typing import Any, cast

from ...types.ir import (
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)
from ...types.ir.tools import ToolCallConfig
from ..base import BaseToolOps
from ..base.helpers import sanitize_schema
from ._constants import generate_tool_call_id


def _normalize_schema_types(schema: Any, to_upper: bool = False) -> Any:
    """Recursively normalize ``type`` values in a JSON Schema dict.

    Google's API uses uppercase type names (``STRING``, ``OBJECT``, …)
    while JSON Schema / other providers use lowercase.

    Args:
        schema: A JSON Schema dict (or sub-schema value).
        to_upper: If True, convert to Google uppercase; otherwise lowercase.
    """
    if not isinstance(schema, dict):
        return schema
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            result[key] = value.upper() if to_upper else value.lower()
        elif key == "properties" and isinstance(value, dict):
            result[key] = {
                k: _normalize_schema_types(v, to_upper) for k, v in value.items()
            }
        elif key == "items":
            result[key] = _normalize_schema_types(value, to_upper)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            result[key] = [_normalize_schema_types(v, to_upper) for v in value]
        else:
            result[key] = value
    return result


def _get_result_content(ir_tool_result: ToolResultPart) -> Any:
    """Extract and normalize result content from an IR ToolResultPart.

    Structured content (list of content blocks, dicts) is preserved as-is
    for Google's Struct-based ``functionResponse.response``.  Scalar values
    are returned unchanged.
    """
    result_content = (
        ir_tool_result.get("result")
        or ir_tool_result.get("content")
        or ir_tool_result.get("output")
        or ""
    )
    if isinstance(result_content, dict):
        import json

        return json.dumps(result_content)
    return result_content


class GoogleGenAIToolOps(BaseToolOps):
    """Google GenAI tool conversion operations.

    All methods are static and stateless. Handles tool definitions,
    calls, results, choice strategies, and call configurations.
    """

    # ==================== Tool Definition ====================

    @staticmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> dict:
        """IR ToolDefinition → Google GenAI FunctionDeclaration (wrapped in Tool).

        Google wraps function declarations in a Tool object with
        ``function_declarations`` list.

        Args:
            ir_tool: IR tool definition.

        Returns:
            Google Tool dict with function_declarations.
        """
        func_decl: dict[str, Any] = {
            "name": ir_tool["name"],
            "description": ir_tool.get("description", ""),
        }
        parameters = ir_tool.get("parameters")
        if parameters:
            sanitized = (
                sanitize_schema(
                    parameters,
                    extra_strip_keys={"additionalProperties", "title"},
                )
                if isinstance(parameters, dict)
                else parameters
            )
            func_decl["parameters"] = _normalize_schema_types(sanitized, to_upper=True)

        return {"function_declarations": [func_decl]}

    @staticmethod
    def p_tool_definition_to_ir(
        provider_tool: Any, **kwargs: Any
    ) -> ToolDefinition | list[ToolDefinition] | None:
        """Google GenAI FunctionDeclaration → IR ToolDefinition(s).

        A single Google Tool dict may contain multiple function declarations.
        Returns a list when multiple declarations are present, or a single
        ToolDefinition for backward compatibility when there is exactly one.

        Returns ``None`` for tool entries that contain neither
        ``function_declarations`` / ``functionDeclarations`` nor a bare
        ``name`` field — typically Google built-in tool types such as
        ``googleSearch`` or ``codeExecution`` whose capabilities cannot be
        mapped to a generic function-call IR.

        Supports both snake_case (``function_declarations``) and camelCase
        (``functionDeclarations``) keys for REST API compatibility.

        Args:
            provider_tool: Google Tool dict with function_declarations.

        Returns:
            IR ToolDefinition, list of ToolDefinitions, or None for
            entries that cannot be converted.
        """
        # Handle both snake_case and camelCase, wrapped and unwrapped formats
        func_decls = provider_tool.get("function_declarations") or provider_tool.get(
            "functionDeclarations"
        )

        if func_decls:
            results: list[ToolDefinition] = []
            for func in func_decls:
                parameters = _normalize_schema_types(func.get("parameters", {}))
                td: dict[str, Any] = {
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": parameters,
                }
                if isinstance(parameters, dict) and "required" in parameters:
                    td["required_parameters"] = parameters["required"]
                else:
                    td["required_parameters"] = []
                td["metadata"] = {}
                results.append(cast(ToolDefinition, td))
            if len(results) == 1:
                return results[0]
            return results

        # Bare function declaration (no wrapper)
        func = provider_tool
        if not func.get("name"):
            return None
        parameters = _normalize_schema_types(func.get("parameters", {}))
        result: dict[str, Any] = {
            "type": "function",
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": parameters,
        }

        # Extract required_parameters from JSON Schema if available
        if isinstance(parameters, dict) and "required" in parameters:
            result["required_parameters"] = parameters["required"]
        else:
            result["required_parameters"] = []

        result["metadata"] = {}
        return cast(ToolDefinition, result)

    # ==================== Tool Choice ====================

    @staticmethod
    def ir_tool_choice_to_p(ir_tool_choice: ToolChoice, **kwargs: Any) -> dict | None:
        """IR ToolChoice → Google GenAI ToolConfig/FunctionCallingConfig.

        Mapping:
        - ``mode:"none"`` → ``{"function_calling_config": {"mode": "NONE"}}``
        - ``mode:"auto"`` → ``{"function_calling_config": {"mode": "AUTO"}}``
        - ``mode:"any"`` → ``{"function_calling_config": {"mode": "ANY"}}``
        - ``mode:"tool"`` → ``{"function_calling_config": {"mode": "ANY", "allowed_function_names": [...]}}``

        Args:
            ir_tool_choice: IR tool choice.

        Returns:
            Google ToolConfig dict, or None if mode is unrecognized.
        """
        mode = ir_tool_choice.get("mode")

        if mode == "none":
            return {"function_calling_config": {"mode": "NONE"}}
        elif mode == "auto":
            return {"function_calling_config": {"mode": "AUTO"}}
        elif mode == "any":
            return {"function_calling_config": {"mode": "ANY"}}
        elif mode == "tool":
            config: dict[str, Any] = {"function_calling_config": {"mode": "ANY"}}
            tool_name = ir_tool_choice.get("tool_name")
            if tool_name:
                cast(dict, config["function_calling_config"])[
                    "allowed_function_names"
                ] = [tool_name]
            return config

        return None

    @staticmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        """Google GenAI ToolConfig → IR ToolChoice.

        Args:
            provider_tool_choice: Google ToolConfig dict.

        Returns:
            IR ToolChoice.
        """
        if not isinstance(provider_tool_choice, dict):
            return cast(ToolChoice, {"mode": "auto", "tool_name": ""})

        fcc = provider_tool_choice.get(
            "function_calling_config"
        ) or provider_tool_choice.get("functionCallingConfig", {})
        mode = fcc.get("mode", "AUTO")

        mode_map = {
            "NONE": "none",
            "AUTO": "auto",
            "ANY": "any",
        }

        ir_mode = mode_map.get(mode, "auto")

        # Check for specific tool names
        allowed_names = fcc.get("allowed_function_names") or fcc.get(
            "allowedFunctionNames", []
        )
        if allowed_names and ir_mode == "any":
            return cast(ToolChoice, {"mode": "tool", "tool_name": allowed_names[0]})

        return cast(ToolChoice, {"mode": ir_mode, "tool_name": ""})

    # ==================== Tool Call ====================

    @staticmethod
    def ir_tool_call_to_p(ir_tool_call: ToolCallPart, **kwargs: Any) -> dict:
        """IR ToolCallPart → Google GenAI function_call Part.

        Google uses ``function_call`` with ``name`` and ``args`` (dict, not JSON string).

        Args:
            ir_tool_call: IR tool call part.

        Returns:
            Google function_call Part dict.
        """
        tool_name = ir_tool_call.get("tool_name", ir_tool_call.get("name", ""))
        tool_input = ir_tool_call.get("tool_input", ir_tool_call.get("arguments", {}))

        func_call: dict[str, Any] = {
            "name": tool_name,
            "args": tool_input,
        }
        tool_call_id = ir_tool_call.get("tool_call_id")
        if tool_call_id:
            func_call["id"] = tool_call_id

        part: dict[str, Any] = {"functionCall": func_call}

        preserve_metadata = kwargs.get("preserve_metadata", True)
        if preserve_metadata and "provider_metadata" in ir_tool_call:
            metadata = ir_tool_call["provider_metadata"]
            if "google" in metadata and "thought_signature" in metadata["google"]:
                part["thoughtSignature"] = metadata["google"]["thought_signature"]
            part["_provider_metadata"] = metadata

        return part

    @staticmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        """Google GenAI function_call Part → IR ToolCallPart.

        Supports both SDK naming (``function_call``) and REST API naming
        (``functionCall``).

        Args:
            provider_tool_call: Google Part dict with function_call.

        Returns:
            IR ToolCallPart.
        """
        func_call = provider_tool_call.get("function_call") or provider_tool_call.get(
            "functionCall"
        )
        if not func_call:
            raise ValueError("Part does not contain function_call")

        # Google function_call may not have id field, generate a unique ID
        tool_call_id = func_call.get("id")
        if not tool_call_id:
            tool_call_id = generate_tool_call_id()

        tool_call_kwargs: dict[str, Any] = {
            "type": "tool_call",
            "tool_call_id": tool_call_id,
            "tool_name": func_call["name"],
            "tool_input": func_call.get("args", {}),
            "tool_type": "function",
        }

        preserve_metadata = kwargs.get("preserve_metadata", True)
        if preserve_metadata:
            pm = provider_tool_call.get("_provider_metadata")
            if pm:
                tool_call_kwargs["provider_metadata"] = pm
            thought_sig = provider_tool_call.get(
                "thoughtSignature"
            ) or provider_tool_call.get("thought_signature")
            if thought_sig:
                pm = tool_call_kwargs.setdefault("provider_metadata", {})
                pm.setdefault("google", {})["thought_signature"] = thought_sig

        return cast(ToolCallPart, tool_call_kwargs)

    # ==================== Tool Result ====================

    @staticmethod
    def ir_tool_result_to_p(ir_tool_result: ToolResultPart, **kwargs: Any) -> dict:
        """IR ToolResultPart → Google GenAI function_response Part.

        Note: Google's function_response.name should be the function name,
        not the tool_call_id. When context (ir_input) is available, use
        ``ir_tool_result_to_p_with_context`` instead.

        Args:
            ir_tool_result: IR tool result part.

        Returns:
            Google function_response Part dict.
        """
        tool_name = ir_tool_result.get("tool_call_id", "")

        result_content = _get_result_content(ir_tool_result)

        response_data: dict[str, Any] = {"output": result_content}
        if ir_tool_result.get("is_error"):
            response_data = {"error": result_content}

        func_response: dict[str, Any] = {
            "name": tool_name,
            "response": response_data,
        }
        tool_call_id = ir_tool_result.get("tool_call_id")
        if tool_call_id:
            func_response["id"] = tool_call_id

        return {"functionResponse": func_response}

    @staticmethod
    def ir_tool_result_to_p_with_context(
        ir_tool_result: ToolResultPart, ir_input: Any
    ) -> dict:
        """IR ToolResultPart → Google GenAI function_response Part with context.

        Looks up the corresponding tool_call in the message history to find
        the actual function name (since Google requires function name, not
        tool_call_id).

        Args:
            ir_tool_result: IR tool result part.
            ir_input: Full IR input (message list) for context lookup.

        Returns:
            Google function_response Part dict.
        """
        from ...types.ir import is_message, is_tool_call_part

        tool_name = None
        tool_call_id = ir_tool_result.get("tool_call_id")

        for msg in ir_input:
            if not is_message(msg):
                continue
            for part in msg.get("content", []):
                if is_tool_call_part(part) and part.get("tool_call_id") == tool_call_id:
                    tool_name = part.get("tool_name")
                    break
            if tool_name:
                break

        if not tool_name:
            warnings.warn(
                f"Could not find corresponding tool call for tool_call_id "
                f"'{tool_call_id}'. Using tool_call_id as function name, "
                f"which may cause issues with Google GenAI."
            )
            tool_name = tool_call_id

        result_content = _get_result_content(ir_tool_result)

        response_data: dict[str, Any] = {"output": result_content}
        if ir_tool_result.get("is_error"):
            response_data = {"error": result_content}

        func_response: dict[str, Any] = {
            "name": tool_name,
            "response": response_data,
        }
        tool_call_id = ir_tool_result.get("tool_call_id")
        if tool_call_id:
            func_response["id"] = tool_call_id

        return {"functionResponse": func_response}

    @staticmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        """Google GenAI function_response Part → IR ToolResultPart.

        Supports both SDK naming (``function_response``) and REST API naming
        (``functionResponse``).  Structured content (lists) is preserved
        as-is for multimodal tool result round-tripping.

        Args:
            provider_tool_result: Google Part dict with function_response.

        Returns:
            IR ToolResultPart.
        """
        func_response = provider_tool_result.get(
            "function_response"
        ) or provider_tool_result.get("functionResponse")
        response_data = func_response.get("response", {})

        is_error = "error" in response_data
        content = response_data.get("error" if is_error else "output", "")

        if isinstance(content, list):
            result: Any = content
        elif isinstance(content, dict):
            import json

            result = json.dumps(content)
        else:
            result = str(content)

        return ToolResultPart(
            type="tool_result",
            tool_call_id=func_response.get("id", func_response.get("name", "")),
            result=result,
            is_error=is_error,
        )

    # ==================== Tool Config ====================

    @staticmethod
    def ir_tool_config_to_p(ir_tool_config: ToolCallConfig, **kwargs: Any) -> dict:
        """IR ToolCallConfig → Google GenAI tool config fields.

        Google does not have a direct parallel_tool_calls equivalent.

        Args:
            ir_tool_config: IR tool call config.

        Returns:
            Dict of Google request fields to merge (may be empty).
        """
        # Google doesn't have a direct mapping for disable_parallel
        # or max_calls. Return empty dict.
        return {}

    @staticmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        """Google GenAI tool config → IR ToolCallConfig.

        Args:
            provider_tool_config: Dict with Google tool config fields.

        Returns:
            IR ToolCallConfig.
        """
        return cast(ToolCallConfig, {})
