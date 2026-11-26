"""
LLM-Rosetta - OpenAI Responses Tool Operations

OpenAI Responses API tool conversion operations.
Handles bidirectional conversion of tool definitions, calls, results,
choice strategies, and call configurations.

Self-contained: does not depend on utils/ToolCallConverter or utils/ToolConverter.

Note: Responses API uses flat items (function_call, function_call_output)
instead of nested tool_calls within messages. Tool definitions use a flat
format with type/name/description/parameters at the top level.
"""

import json
import logging
from typing import Any, cast

from ...base.tools.content import (
    convert_content_blocks_to_ir,
    convert_ir_content_blocks_to_p,
)
from llm_rosetta.ir.model import (
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)
from llm_rosetta.ir.model.tools import ToolCallConfig
from ...base import BaseToolOps
from llm_rosetta.conversion_warnings import DOWNGRADED, report_loss
from ...base.helpers.opaque import CARRIER_KEY
from llm_rosetta.ir.primitives.tool_group import (
    tool_call_group,
)
from ...base.tools import (
    allowed_tool_names,
    get_native_definition,
    intrinsic_kind_of,
    ir_strict,
    make_intrinsic_tool_definition,
    mark_tool_error,
    provider_allowed_tools_to_ir,
    sanitize_tool_call_id,
    sanitize_tool_parameters,
    warn_tool_error,
    write_ir_strict,
)

from .tool_intrinsic_ops import IntrinsicToolOpsMixin
from .tool_items import (  # noqa: F401
    ADDITIONAL_TOOLS_ITEM_TYPE,
    _DEFAULT_WEB_SEARCH_ACTION,
    _INTRINSIC_KIND_TO_ITEM,
    _INTRINSIC_KIND_TO_RESULT_ITEM,
    _INTRINSIC_RESULT_TYPES,
    _ITEM_TO_INTRINSIC_KIND,
    _RESPONSES_INTRINSIC_TOOLS,
    _RESPONSES_NATIVE_TYPE_TO_KIND,
    _RESULT_ITEM_TOOL_TYPES,
    _SERVER_ITEM_ID_PREFIX,
    _SERVER_ITEM_INPUT_KEYS,
    _SERVER_ITEM_STRUCTURAL_KEYS,
    _SERVER_ITEM_TYPES,
    _build_function_call_item,
    _build_intrinsic_result_item,
    _build_server_item,
    _collect_additional_tools,
    _custom_tool_call_to_ir,
    _declared_child_name,
    _declared_child_names,
    _flatten_namespace_tool,
    _flattened_child,
    _p_server_item_to_ir,
    _server_item_renderable,
    _server_item_tool_input,
    _synthesize_passthrough_tool,
    _tag_child_group,
    _warn_dropped_intrinsic_result,
    _wire_tool_call_group,
    custom_call_input_text,
    function_call_item_id,
    harvest_additional_tools,
    strip_additional_tools_items,
)

logger = logging.getLogger(__name__)


class OpenResponsesToolOps(IntrinsicToolOpsMixin, BaseToolOps):
    """OpenAI Responses API tool conversion operations.

    All methods are static and stateless. Handles tool definitions,
    calls, results, choice strategies, and call configurations.
    """

    # ==================== Tool Definition ====================

    @staticmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> dict:
        """IR ToolDefinition → OpenAI Responses tool definition.

        Responses API uses a flat format:
        ``{"type": "function", "name": "...", "description": "...", "parameters": {...}}``

        Non-function passthrough tools (e.g. ``web_search``) stored in
        ``_passthrough`` are returned as-is.

        Args:
            ir_tool: IR tool definition.

        Returns:
            OpenAI Responses tool definition dict.
        """
        if ir_tool.get("type") == "intrinsic":
            stored = get_native_definition(ir_tool, "openai_responses")
            if stored is not None:
                return dict(stored)
            kind = intrinsic_kind_of(ir_tool)
            native = _RESPONSES_INTRINSIC_TOOLS.get(kind)
            if native is None:
                report_loss(
                    f"OpenAI Responses has no server tool for intrinsic kind "
                    f"{kind!r}; dropping",
                    subject=kind,
                )
                return {}
            return dict(native)

        # Passthrough tools (web_search, etc.) go back as-is except for the
        # name, since a rename must follow the tool upstream or nothing else
        # in the request agrees on it.
        #
        # The `provider_tool.get("name")` half is required, not defensive:
        # `_synthesize_passthrough_tool` falls back to the type string as the
        # IR name, which is truthy, so testing `renamed` alone would give
        # every bare {"type": "web_search"} a `name` it never had.
        passthrough = ir_tool.get("_passthrough")
        if passthrough is not None:
            provider_tool = dict(passthrough)
            renamed = ir_tool.get("name")
            if renamed and provider_tool.get("name"):
                provider_tool["name"] = renamed
            return provider_tool

        tool_type = ir_tool.get("type", "function")

        if tool_type == "custom":
            result: dict[str, Any] = {
                "type": "custom",
                "name": ir_tool["name"],
            }
            desc = ir_tool.get("description", "")
            if desc:
                result["description"] = desc
            metadata = ir_tool.get("metadata") or {}
            fmt = metadata.get("format")
            if fmt:
                result["format"] = fmt
            return result

        parameters = ir_tool.get("parameters", {})
        if isinstance(parameters, dict):
            parameters = sanitize_tool_parameters(
                parameters,
                tool=ir_tool,
                warnings=kwargs.get("warnings"),
            )
        metadata = ir_tool.get("metadata") or {}
        strict = ir_strict(ir_tool)
        result: dict[str, Any] = {
            "type": "function",
            "name": ir_tool["name"],
            "description": ir_tool.get("description", ""),
            "parameters": parameters,
            # Responses defaults the flag, so a source that said nothing gets
            # the wire default rather than an absent field.
            "strict": False if strict is None else strict,
        }
        output_schema = metadata.get("output_schema")
        if output_schema is not None:
            result["output_schema"] = output_schema
        return result

    @staticmethod
    def p_tool_definition_to_ir(
        provider_tool: Any, **kwargs: Any
    ) -> ToolDefinition | list[ToolDefinition]:
        """OpenAI Responses tool definition → IR ToolDefinition.

        Handles both flat format (Responses API native) and nested format
        (with ``function`` key).  ``type: "namespace"`` containers are
        expanded into individual child tools via ``_flatten_namespace_tool``.
        A tool whose type is outside the IR set (e.g. ``web_search``) is
        stored as passthrough, keeping its original type in
        ``metadata["provider_type"]`` so ``ir_tool_definition_to_p`` can
        restore it unmodified.  Everything else is an IR type already —
        ``function``, ``mcp`` or ``custom`` — and keeps it.  Notably a Codex
        ``custom`` tool such as ``apply_patch`` is *not* downgraded here; a
        provider that cannot accept one is served later and separately, by
        :func:`~llm_rosetta.ir.passes.downgrade_custom_tools`, which
        records itself under ``metadata["_downgraded_from"]``.

        Args:
            provider_tool: OpenAI Responses tool definition dict.

        Returns:
            IR ToolDefinition, or list of ToolDefinitions for namespace
            containers.
        """
        _IR_ALLOWED_TYPES = {"function", "mcp", "custom", "intrinsic"}

        # Handle nested format ({"type": "function", "function": {...}})
        if "function" in provider_tool and isinstance(provider_tool["function"], dict):
            func = provider_tool["function"]
            result: dict[str, Any] = {
                "type": "function",
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            }
        else:
            tool_type = provider_tool.get("type", "function")
            if tool_type == "namespace":
                return _flatten_namespace_tool(provider_tool)
            kind = _RESPONSES_NATIVE_TYPE_TO_KIND.get(tool_type)
            if kind is not None:
                return cast(
                    ToolDefinition,
                    make_intrinsic_tool_definition(
                        kind,
                        description=provider_tool.get("description", ""),
                        native=provider_tool,
                        native_base="openai_responses",
                    ),
                )
            # Non-function tools outside the IR type set (e.g. web_search or
            # Codex custom apply_patch) are stored as passthrough to avoid
            # lossy conversion. IR ``type`` is forced to "function" to
            # satisfy validation; ``ir_tool_definition_to_p`` restores the
            # original payload on the outbound leg.
            if tool_type != "function" and tool_type not in _IR_ALLOWED_TYPES:
                return _synthesize_passthrough_tool(provider_tool, tool_type)

            # Flat format (Responses API native).
            # Custom tools use "schema" instead of "parameters".
            params = provider_tool.get("parameters", {})
            if tool_type != "function" and not params:
                params = provider_tool.get("schema", {})
            # No downgrade to make: every type outside the IR set has already
            # returned above as a passthrough, so what reaches here is an IR
            # type and stays one.
            result = {
                "type": tool_type,
                "name": provider_tool.get("name", ""),
                "description": provider_tool.get("description", ""),
                "parameters": params,
            }

        # Extract required_parameters from JSON Schema if available
        parameters = result.get("parameters", {})
        if isinstance(parameters, dict) and "required" in parameters:
            result["required_parameters"] = parameters["required"]
        else:
            result["required_parameters"] = []

        meta: dict[str, Any] = {}
        fmt = provider_tool.get("format")
        if fmt:
            meta["format"] = fmt
        output_schema = provider_tool.get("output_schema")
        if output_schema is not None:
            meta["output_schema"] = output_schema
        result["metadata"] = meta
        # The strict flag is a first-class IR field now; metadata["strict"] was
        # where it used to live and is still read by ``ir_strict``.
        write_ir_strict(result, provider_tool.get("strict"))
        return cast(ToolDefinition, result)

    # ==================== Tool Choice ====================

    @staticmethod
    def ir_tool_choice_to_p(
        ir_tool_choice: ToolChoice,
        *,
        allowed_tools: dict[str, Any] | None = None,
        tool_types: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str | dict:
        """IR ToolChoice → OpenAI Responses tool_choice parameter.

        Mapping:
        - ``mode:"none"`` → ``"none"``
        - ``mode:"auto"`` → ``"auto"``
        - ``mode:"any"`` → ``"required"``
        - ``mode:"required"`` → ``"required"``
        - ``mode:"tool"`` → ``{"type":"function","function":{"name":"..."}}``

        Also supports legacy ``type`` field for backward compatibility.

        Args:
            ir_tool_choice: IR tool choice.
            allowed_tools: Legacy copy of the wire object, retrieved from
                ``provider_extensions``.  Only used when the IR carries no
                ``allowed_tools`` of its own, so IR built before that field
                existed still round-trips.
            **kwargs: ``warnings`` (list[str] | None) — optional warning sink
                for a restriction naming a tool the request does not declare.
            tool_types: Declared tool name → IR type, used to spell each
                allow-list entry.  An entry's ``type`` must match how the tool
                was declared — the API answers "Tool choice 'custom' not found
                in 'tools' parameter" for a function-shaped entry naming a
                custom tool, and the reverse — so it is derived from the
                declaration rather than carried in the IR, where a downgraded
                custom tool would leave it stale.  A name the map does not
                know is written as ``function``.

        Returns:
            OpenAI tool_choice value (string or dict).
        """
        mode = ir_tool_choice.get("mode") or ir_tool_choice.get("type")
        wire_mode = {
            "any": "required",
            "required": "required",
            "none": "none",
            "auto": "auto",
        }.get(str(mode), "auto")

        # The IR states the restriction as tool identities, so rebuild the
        # wire object around the IR mode.
        names = allowed_tool_names(ir_tool_choice)
        if names:
            declared_types = tool_types or {}
            warnings = kwargs.get("warnings")
            if tool_types is not None:
                # The upstream refuses a restriction naming a tool it was not
                # sent ("Tool choice 'x' not found in 'tools' parameter"), so
                # the client is told here as well as by the naming pass — a
                # library caller driving the converter directly has no naming
                # pass to hear it from.
                undeclared = [n for n in names if n not in declared_types]
                if undeclared:
                    report_loss(
                        "tool_choice.allowed_tools names "
                        f"{', '.join(repr(n) for n in undeclared)}, which the "
                        "request does not declare; the provider rejects a "
                        "restriction naming a tool it was not sent",
                        code=DOWNGRADED,
                        subject="allowed_tools",
                        warnings=warnings,
                    )
            return {
                "type": "allowed_tools",
                "tools": [
                    {"type": declared_types.get(name, "function"), "name": name}
                    for name in names
                ],
                "mode": wire_mode,
            }

        # Legacy ``provider_extensions`` passthrough, re-emitted verbatim with
        # its ``mode`` refreshed from the IR mode.
        if (
            isinstance(allowed_tools, dict)
            and allowed_tools.get("type") == "allowed_tools"
        ):
            out = dict(allowed_tools)
            out["mode"] = wire_mode
            return out

        # Support both "mode" and legacy "type" field
        if mode == "none":
            return "none"
        elif mode == "auto":
            return "auto"
        elif mode in ("any", "required"):
            return "required"
        elif mode in ("tool", "function"):
            tool_name = ir_tool_choice.get("tool_name")
            if not tool_name and "function" in ir_tool_choice:
                tool_name = cast(dict, ir_tool_choice)["function"].get("name")
            if tool_name:
                if ir_tool_choice.get("tool_type") == "custom":
                    return {"type": "custom", "name": tool_name}
                return {"type": "function", "name": tool_name}
            return "required"

        return "auto"

    @staticmethod
    def p_tool_choice_to_ir(
        provider_tool_choice: Any,
        *,
        extensions: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolChoice:
        """OpenAI Responses tool_choice → IR ToolChoice.

        Mapping:
        - ``"none"`` → ``mode:"none"``
        - ``"auto"`` → ``mode:"auto"``
        - ``"required"`` → ``mode:"any"``
        - ``{"type":"function","function":{"name":"..."}}`` → ``mode:"tool"``
        - ``{"type":"allowed_tools","tools":[...],"mode":...}`` → the same
          mode, plus ``allowed_tools`` identities.  Entries carry a name, and
          the ``namespace`` beside it when the source echoed a container.
        - ``{"type":"namespace","name":...,"tools":[...]}`` → ``mode:"auto"``,
          reported on *warnings*.  The IR ``ToolChoice`` has no group slot, so
          a forced container child cannot be carried and the call is no longer
          forced (parent issue #953).

        Args:
            provider_tool_choice: OpenAI tool_choice value.
            extensions: Unused; kept so older call sites still pass it.
            warnings: Optional warning sink.  Receives one entry when a
                namespace container form is dropped.

        Returns:
            IR ToolChoice.
        """
        if (
            isinstance(provider_tool_choice, dict)
            and provider_tool_choice.get("type") == "allowed_tools"
        ):
            return provider_allowed_tools_to_ir(
                provider_tool_choice,
                warnings=warnings,
                selector="tool_choice.allowed_tools",
            )

        if isinstance(provider_tool_choice, str):
            if provider_tool_choice == "none":
                return cast(ToolChoice, {"mode": "none", "tool_name": ""})
            elif provider_tool_choice == "auto":
                return cast(ToolChoice, {"mode": "auto", "tool_name": ""})
            elif provider_tool_choice == "required":
                return cast(ToolChoice, {"mode": "any", "tool_name": ""})
            return cast(ToolChoice, {"mode": "auto", "tool_name": ""})

        if isinstance(provider_tool_choice, dict):
            if provider_tool_choice.get("type") == "custom":
                return cast(
                    ToolChoice,
                    {
                        "mode": "tool",
                        "tool_name": provider_tool_choice.get("name", ""),
                        "tool_type": "custom",
                    },
                )
            if provider_tool_choice.get("type") == "function":
                tool_name = provider_tool_choice.get("name", "")
                if not tool_name:
                    func = provider_tool_choice.get("function", {})
                    tool_name = func.get("name", "")
                return cast(ToolChoice, {"mode": "tool", "tool_name": tool_name})
            if provider_tool_choice.get("type") == "namespace":
                report_loss(
                    "Dropped a namespace tool_choice the IR cannot "
                    "represent; fell back to 'auto', so the call is no "
                    "longer forced.",
                    code=DOWNGRADED,
                    subject="tool_choice",
                    warnings=warnings,
                )
                return cast(ToolChoice, {"mode": "auto", "tool_name": ""})

        return cast(ToolChoice, {"mode": "auto", "tool_name": ""})

    # ==================== Tool Call ====================

    @classmethod
    def ir_client_call_to_p(cls, ir_tool_call: ToolCallPart, **kwargs: Any) -> dict:
        """IR ToolCallPart → OpenAI Responses client tool call item.

        Converts to function_call or mcp_call depending on tool_type/tool_name.

        Args:
            ir_tool_call: IR tool call part.

        Returns:
            OpenAI Responses tool call item dict.
        """
        tool_type = ir_tool_call.get("tool_type", "function")
        tool_call_id = sanitize_tool_call_id(
            ir_tool_call.get("tool_call_id", ir_tool_call.get("id", ""))
        )
        tool_name = ir_tool_call.get("tool_name", ir_tool_call.get("name", ""))
        tool_input = ir_tool_call.get("tool_input", ir_tool_call.get("arguments", {}))

        # Serialize tool_input
        arguments = (
            json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
        )

        is_mcp = tool_type == "mcp" or (tool_name and tool_name.startswith("mcp://"))
        if is_mcp:
            return {
                "type": "mcp_call",
                "id": tool_call_id,
                "name": tool_name,
                "arguments": arguments,
                "server_label": ir_tool_call.get("server_name", "default"),
                "status": "calling",
            }
        elif tool_type == "function":
            return _build_function_call_item(
                ir_tool_call, tool_call_id, tool_name, arguments
            )
        elif tool_type == "custom":
            # Custom tool calls use plain text 'input' instead of JSON
            # 'arguments'; one unwrap rule, shared with the body fold.
            input_str = custom_call_input_text(tool_input)
            item: dict[str, Any] = {
                "type": "custom_tool_call",
                "call_id": tool_call_id,
                "name": tool_name,
                "input": input_str,
            }
            # Same round-trip as a function_call above.
            metadata = ir_tool_call.get("provider_metadata") or {}
            namespace = tool_call_group(ir_tool_call)
            if namespace:
                item["namespace"] = namespace
            if metadata:
                item[CARRIER_KEY] = metadata
            return item
        else:
            # Default to function_call
            return {
                "type": "function_call",
                "call_id": tool_call_id,
                "name": f"{tool_type}_{tool_name}",
                "arguments": arguments,
            }

    @staticmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        """OpenAI Responses tool call item → IR ToolCallPart.

        Handles function_call, mcp_call, custom_tool_call, and server
        tool item types (shell_call, computer_call, code_interpreter_call,
        web_search_call, file_search_call) which map to intrinsic.

        Args:
            provider_tool_call: OpenAI Responses tool call item dict.

        Returns:
            IR ToolCallPart.
        """
        item_type = provider_tool_call.get("type")

        # Parse arguments
        arguments = provider_tool_call.get("arguments", {})
        if isinstance(arguments, dict):
            tool_input = arguments
        elif isinstance(arguments, str):
            try:
                tool_input = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                tool_input = {"input": arguments}
        else:
            tool_input = {}

        if item_type == "function_call":
            # Responses API has both 'id' (item ID, fc_ prefix) and
            # 'call_id' (correlation ID, call_ prefix). Store call_id as
            # tool_call_id for correlation, preserve 'id' in provider_metadata
            # for lossless round-trip.
            call_id = provider_tool_call.get("call_id")
            item_id = provider_tool_call.get("id", "")
            if not call_id:
                # Fallback: derive call_ prefix from fc_ prefix
                if item_id.startswith("fc_"):
                    call_id = "call_" + item_id[3:]
                else:
                    call_id = item_id
            part = ToolCallPart(
                type="tool_call",
                tool_call_id=call_id,
                tool_name=provider_tool_call.get("name", ""),
                tool_input=tool_input,
                tool_type="function",
            )
            # Keep it: (name, group) is what the request leg maps back.  The
            # group is first-class on the call now; a legacy metadata spelling
            # (a carrier from a foreign hop) folds into it rather than staying
            # a second copy.
            restored = provider_tool_call.get(CARRIER_KEY)
            pm: dict[str, Any] = dict(restored) if isinstance(restored, dict) else {}
            if item_id and item_id != call_id:
                pm["responses_item_id"] = item_id
            group = _wire_tool_call_group(provider_tool_call, pm)
            pm.pop("namespace", None)
            if group:
                part["tool_group"] = group
            if pm:
                part["provider_metadata"] = pm
            return part
        elif item_type == "mcp_call":
            # MCP call may use server/tool fields or name field
            server = provider_tool_call.get("server", "")
            tool = provider_tool_call.get("tool", provider_tool_call.get("name", ""))
            tool_name = f"mcp://{server}/{tool}" if server and tool else tool

            return ToolCallPart(
                type="tool_call",
                tool_call_id=provider_tool_call.get("id", ""),
                tool_name=tool_name,
                tool_input=tool_input,
                tool_type="mcp",
            )
        elif item_type in _SERVER_ITEM_TYPES:
            return _p_server_item_to_ir(provider_tool_call, item_type)
        elif item_type == "custom_tool_call":
            return _custom_tool_call_to_ir(provider_tool_call)
        else:
            raise ValueError(f"Unsupported OpenAI Responses item type: {item_type}")

    # ==================== Tool Result ====================

    @classmethod
    def ir_client_result_to_p(
        cls, ir_tool_result: ToolResultPart, **kwargs: Any
    ) -> dict:
        """IR ToolResultPart → OpenAI Responses tool call output item.

        Emits ``custom_tool_call_output`` when the context indicates the
        tool call was custom, otherwise ``function_call_output``.

        The Responses API has no tool error field, so an ``is_error`` flag is
        re-encoded as a trailing ``input_text`` block (see
        ``ir.tools.error_marker``).

        An intrinsic result reaches this path only when its call was written as
        a client ``function_call``; the hook pair below decides that.

        Args:
            ir_tool_result: IR tool result part.
            **kwargs: ``warnings`` (list[str]) receives a note when the error
                flag had to be re-encoded or dropped.
        """
        result_content = ir_tool_result.get("result") or ir_tool_result.get(
            "content", ""
        )

        if isinstance(result_content, list):
            from .content_ops import OpenResponsesContentOps

            output = convert_ir_content_blocks_to_p(
                result_content, OpenResponsesContentOps
            )
        elif isinstance(result_content, dict):
            output = json.dumps(result_content)
        elif isinstance(result_content, str):
            output = result_content
        else:
            output = str(result_content)

        # Sanitize here to match the ID registered by ir_tool_call_to_p / streaming start.
        call_id = sanitize_tool_call_id(ir_tool_result["tool_call_id"])

        # A status read from a Responses item is replayed as the item's own
        # status rather than re-encoded as an error marker: ``incomplete`` is
        # the native spelling of the same fact, and the marker would mutate
        # model-visible output on a same-format round-trip.  Only an error
        # flag with no such status takes the marker path.
        provider_metadata = ir_tool_result.get("provider_metadata") or {}
        responses_status = provider_metadata.get("responses_status")
        if not isinstance(responses_status, str) or not responses_status:
            responses_status = None
        if ir_tool_result.get("is_error") and responses_status != "incomplete":
            # Both function_call_output and custom_tool_call_output accept a
            # list of input_* content blocks, so the marker rides along as one.
            output = mark_tool_error(output, text_block_type="input_text")
            warn_tool_error(
                kwargs.get("warnings"),
                format_name="OpenAI Responses",
                tool_call_id=call_id,
            )

        # The IR part wins: it survives request history, where the context's
        # tool-type map is empty because only streaming populates it.
        tool_type = ir_tool_result.get("tool_type")
        if tool_type is None:
            ctx = kwargs.get("context")
            tool_type = ctx.get_tool_type(call_id) if ctx is not None else "function"

        if tool_type == "custom":
            return {
                "type": "custom_tool_call_output",
                "call_id": call_id,
                "output": output,
                "status": responses_status or "completed",
            }
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
            "status": responses_status or "completed",
        }

    @staticmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        """OpenAI Responses tool call output → IR ToolResultPart.

        Handles function_call_output, custom_tool_call_output, and
        mcp_call_output.

        Args:
            provider_tool_result: OpenAI Responses tool result item dict.

        Returns:
            IR ToolResultPart.
        """
        output = provider_tool_result.get("output", "")
        # String outputs are opaque tool data, even when they contain JSON.
        # Only an actual list represents multimodal content blocks.
        if isinstance(output, list):
            from .content_ops import OpenResponsesContentOps

            output = convert_content_blocks_to_ir(output, OpenResponsesContentOps)

        part = ToolResultPart(
            type="tool_result",
            tool_call_id=provider_tool_result.get("call_id", ""),
            result=output,
        )
        # Only a boolean is copied: the IR field is typed ``bool``, so a
        # malformed value would fail IR validation and take the request down.
        is_error = provider_tool_result.get("is_error")
        item_type = provider_tool_result.get("type", "")
        # An incomplete function output is authoritative over a conflicting
        # flag.  The status rides along so a Responses target can replay it
        # instead of re-encoding the same fact as an error marker.
        if (
            item_type == "function_call_output"
            and provider_tool_result.get("status") == "incomplete"
        ):
            part["is_error"] = True
            part.setdefault("provider_metadata", {})["responses_status"] = "incomplete"
        elif isinstance(is_error, bool):
            part["is_error"] = is_error
        # Record the non-default types so the outbound leg can emit the
        # same item kind.  The context it would otherwise consult is only
        # populated while streaming a response, never by parsing history.
        tool_type = _RESULT_ITEM_TOOL_TYPES.get(item_type)
        if tool_type is not None:
            part["tool_type"] = tool_type
        intrinsic_kind = _INTRINSIC_RESULT_TYPES.get(item_type)
        if intrinsic_kind is not None:
            part["tool_type"] = "intrinsic"
            part["intrinsic_kind"] = intrinsic_kind
        return part

    # ==================== Tool Config ====================

    @staticmethod
    def ir_tool_config_to_p(ir_tool_config: ToolCallConfig, **kwargs: Any) -> dict:
        """IR ToolCallConfig → OpenAI Responses tool call config fields.

        Mapping:
        - ``disable_parallel`` → ``parallel_tool_calls`` (inverted)
        - ``max_calls`` → ``max_tool_calls``

        Args:
            ir_tool_config: IR tool call config.

        Returns:
            Dict of OpenAI request fields to merge.
        """
        result: dict[str, Any] = {}

        if "disable_parallel" in ir_tool_config:
            result["parallel_tool_calls"] = not ir_tool_config["disable_parallel"]

        if "max_calls" in ir_tool_config:
            result["max_tool_calls"] = ir_tool_config["max_calls"]

        return result

    @staticmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        """OpenAI Responses tool call config → IR ToolCallConfig.

        Mapping:
        - ``parallel_tool_calls`` → ``disable_parallel`` (inverted)
        - ``max_tool_calls`` → ``max_calls``

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

            max_calls = provider_tool_config.get("max_tool_calls")
            if max_calls is not None:
                result["max_calls"] = max_calls

        return cast(ToolCallConfig, result)
