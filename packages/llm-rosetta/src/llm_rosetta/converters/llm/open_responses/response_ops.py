"""Response/body assembly helpers for the Open Responses converter."""

from typing import Any, cast

from llm_rosetta.ir.model import (
    is_text_part,
    is_tool_call_part,
    is_tool_result_part,
    is_reasoning_part,
    is_citation_part,
)
from llm_rosetta.ir.model.response import IRResponse, UsageInfo
from ...base.context import ConversionContext
from ...base.helpers.citations import report_dropped_citation
from llm_rosetta.conversion_warnings import DOWNGRADED, report_loss
from llm_rosetta.ir.primitives.tool_group import (
    GENERATED_GROUP_MARKER,
)
from ._mixin_host import OpenResponsesMixinHost
from ._constants import (
    generate_message_id,
)
from .message_ops import OpenResponsesMessageOps
from .tool_ops import (
    custom_call_input_text,
    function_call_item_id,
)
from ._wire_helpers import (
    _insert_after_call,
    _capture_item_metadata,
)


class OpenResponsesResponseMixin(OpenResponsesMixinHost):
    """Body shaping, citation, usage and preserved-metadata helpers."""

    @staticmethod
    def _renumber_message_items(body: dict[str, Any], response_id: str) -> None:
        """Number message items by their place in the output, as a body does.

        Each choice's fold starts its own count, so two choices would name
        their first item the same; the body numbers them after the items
        already emitted.
        """
        count = 0
        for item in body.get("output") or []:
            if item.get("type") == "message":
                item["id"] = generate_message_id(response_id, count)
                count += 1

    def _apply_body_shape(
        self,
        body: dict[str, Any],
        ir_response: IRResponse,
        *,
        context: ConversionContext,
        kwargs: dict[str, Any],
    ) -> None:
        """Add what a whole body carries and no stream event carries.

        Args:
            body: The response payload the fold produced; mutated in place.
            ir_response: The IR response it was folded from.
            context: The conversion context, for the warning sink.
            kwargs: The writer's keyword arguments, for the same.
        """
        if "created" in ir_response:
            body["created_at"] = ir_response["created"]
        if "service_tier" in ir_response:
            body["service_tier"] = ir_response["service_tier"]

        for item in body.get("output") or []:
            if item.get("type") == "message":
                item["status"] = "completed"
                for part in item.get("content") or []:
                    if part.get("type") == "output_text":
                        part.setdefault("annotations", [])
                        part.setdefault("logprobs", [])
            elif item.get("type") == "function_call" and item.get("id") == item.get(
                "call_id"
            ):
                # The events name the item with the bare call id when the
                # source carried no item id — that is the stream's own
                # contract — while a body has always used the API's ``fc_``
                # spelling.
                item["id"] = function_call_item_id(str(item.get("call_id", "")))

        self._restore_tool_results(body, ir_response, context=context, kwargs=kwargs)
        self._restore_custom_inputs(body, ir_response)
        self._attach_reasoning_content(body, ir_response)
        self._attach_citations(body, ir_response, context=context, kwargs=kwargs)

    def _restore_tool_results(
        self,
        body: dict[str, Any],
        ir_response: IRResponse,
        *,
        context: ConversionContext,
        kwargs: dict[str, Any],
    ) -> None:
        """Write the client tool results the events never carry into the body.

        A client result is not a stream item at all: the stream leg writes it
        on the *request* leg, and a body has always written it after the call
        it answers.  A server result is the stream's to place — including the
        report when it has no form — so only client results are restored here.

        Args:
            body: The folded body, mutated in place.
            ir_response: The IR response it was folded from.
            context: The conversion context, for the warning sink.
            kwargs: The writer's keyword arguments, for the same.
        """
        output = body.get("output") or []
        # Only a result item means the result was written; the call it answers
        # carries the same call id.
        present = {
            item.get("call_id")
            for item in output
            if str(item.get("type", "")).endswith("_output")
        }
        warnings = self._citation_warnings(context, kwargs)
        for choice in ir_response.get("choices") or []:
            for part in (choice.get("message") or {}).get("content") or []:
                if not is_tool_result_part(part):
                    continue
                if part.get("tool_type") == "intrinsic":
                    continue
                call_id = part.get("tool_call_id")
                if call_id in present:
                    continue
                item = self.tool_ops.ir_client_result_to_p(
                    part, leg="response", warnings=warnings
                )
                if not item:
                    continue
                _insert_after_call(output, str(call_id), item)
                present.add(call_id)

    @staticmethod
    def _restore_custom_inputs(body: dict[str, Any], ir_response: IRResponse) -> None:
        """Give each custom tool call back its plain-text ``input``.

        The events carry a tool call's arguments as JSON; a custom tool takes
        text, and the IR wraps it as ``{"input": ...}`` when it is not itself
        a JSON object.  One unwrap rule, the same one the body builder uses.
        """
        parts = {
            part["tool_call_id"]: part
            for choice in ir_response.get("choices") or []
            for part in (choice.get("message") or {}).get("content") or []
            if is_tool_call_part(part) and "tool_call_id" in part
        }
        for item in body.get("output") or []:
            if item.get("type") != "custom_tool_call":
                continue
            part = parts.get(item.get("call_id"))
            if part is not None:
                item["input"] = custom_call_input_text(part.get("tool_input"))

    @staticmethod
    def _attach_reasoning_content(
        body: dict[str, Any], ir_response: IRResponse
    ) -> None:
        """Give each reasoning item back the raw ``content`` a body carries.

        The spec's structured ``summary`` is what a stream carries; the raw
        ``reasoning_text`` content lives on the part in a body, beside it.
        """
        reasoning_items = [
            item for item in body.get("output") or [] if item.get("type") == "reasoning"
        ]
        index = -1
        for choice in ir_response.get("choices") or []:
            for part in (choice.get("message") or {}).get("content") or []:
                if not is_reasoning_part(part):
                    continue
                index += 1
                metadata = part.get("provider_metadata") or {}
                raw = metadata.get("responses_reasoning_content")
                if raw and 0 <= index < len(reasoning_items):
                    reasoning_items[index]["content"] = raw

    def _attach_citations(
        self,
        body: dict[str, Any],
        ir_response: IRResponse,
        *,
        context: ConversionContext,
        kwargs: dict[str, Any],
    ) -> None:
        """Hang each citation on the text part it annotates.

        The events carry a citation as its own ``citation_delta``; a whole body
        carries it on the ``output_text`` part of the nearest preceding text,
        which is the adjacency ``expand_response`` gives the event.
        """
        text_parts = [
            part
            for item in body.get("output") or []
            if item.get("type") == "message"
            for part in item.get("content") or []
            if part.get("type") == "output_text"
        ]
        warnings = self._citation_warnings(context, kwargs)
        index = -1
        for choice in ir_response.get("choices") or []:
            for part in (choice.get("message") or {}).get("content") or []:
                if is_text_part(part):
                    if index + 1 < len(text_parts):
                        index += 1
                elif is_citation_part(part):
                    annotation = self.content_ops.ir_citation_to_p(part)
                    if annotation is None:
                        report_dropped_citation(part, warnings=warnings)
                    elif index >= 0:
                        text_parts[index]["annotations"].append(annotation)
                    else:
                        report_dropped_citation(
                            part, warnings=warnings, why="no text part to annotate"
                        )

    # ------------------------------------------------------------------
    # Cross-provider consistency helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _citation_warnings(
        context: ConversionContext | None, kwargs: dict[str, Any]
    ) -> list[str] | None:
        """The sink a citation drop should be reported to, when there is one."""
        sink = kwargs.get("warnings")
        if isinstance(sink, list):
            return sink
        return getattr(context, "warnings", None)

    @staticmethod
    def _report_generated_groups(ir_tools: list[Any]) -> None:
        """Report each group whose name had to be invented.

        A container the client left unnamed is malformed input at every
        upstream that implements containers — OpenAI, DeepSeek and Volcengine
        all reject a missing or empty name — so the client should hear that
        their tools are now addressed under a name we chose.

        Emitted here, where the marker and the grouping are on hand, so
        every request carrying such a container reports it.  Once per group,
        not once per child: the children of one container share the name and
        the cause.
        """
        reported: set[str] = set()
        for tool in ir_tools:
            meta = tool.get("metadata")
            if not isinstance(meta, dict) or not meta.get(GENERATED_GROUP_MARKER):
                continue
            group = tool.get("tool_group")
            if not isinstance(group, str) or group in reported:
                continue
            reported.add(group)
            report_loss(
                "A tool container arrived with no name of its own, so its "
                f"tools are addressed under the generated group {group!r}.  "
                "Calls route correctly, but the model reads a less meaningful "
                "group than a named container would have given it; name the "
                "container to fix that",
                code=DOWNGRADED,
                subject=group,
            )

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> UsageInfo:
        """Build IR usage dict from Responses API usage."""
        usage_info: dict[str, Any] = {
            "prompt_tokens": p_usage.get("input_tokens") or 0,
            "completion_tokens": p_usage.get("output_tokens") or 0,
            "total_tokens": p_usage.get("total_tokens") or 0,
        }
        p_input_details = p_usage.get("input_tokens_details")
        if p_input_details:
            if "cached_tokens" in p_input_details:
                usage_info["cache_read_tokens"] = p_input_details["cached_tokens"]
        p_output_details = p_usage.get("output_tokens_details")
        if p_output_details:
            if "reasoning_tokens" in p_output_details:
                usage_info["reasoning_tokens"] = p_output_details["reasoning_tokens"]
        # Nullable on the wire (still null in practice).  Copy it when the key
        # is present, so "absent" and "present but null" stay distinguishable.
        if "compute_units" in p_usage:
            usage_info["compute_units"] = p_usage["compute_units"]
        return cast(UsageInfo, usage_info)

    @staticmethod
    def _stamp_text_part_phase(part: Any, phase: str) -> None:
        """Record the message item's phase on one of its text parts.

        Takes ``Any`` because not every member of the content-part union
        declares ``provider_metadata``.  Text is the part a phased item
        carries in practice; a refusal part is left to the message-level
        phase the write leg falls back on.

        Args:
            part: IR content part to stamp; non-text parts are left alone.
            phase: Phase of the message item the part came from.
        """
        if not isinstance(part, dict) or not is_text_part(part):
            return
        metadata = part.get("provider_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            part["provider_metadata"] = metadata
        metadata["responses_phase"] = phase

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: Any) -> dict[str, Any]:
        """Build Responses API usage dict from IR usage.

        The body writer folds its usage out of the events, but the base
        contract still asks every dialect for this mapping, and a caller may
        reach it directly.
        """
        usage: dict[str, Any] = {
            "input_tokens": ir_usage.get("prompt_tokens") or 0,
            "output_tokens": ir_usage.get("completion_tokens") or 0,
            "total_tokens": ir_usage.get("total_tokens") or 0,
            "input_tokens_details": {
                "cached_tokens": ir_usage.get("cache_read_tokens", 0),
            },
            "output_tokens_details": {
                "reasoning_tokens": ir_usage.get("reasoning_tokens", 0),
            },
        }
        # The same shape ``_build_finish_usage`` writes, so it carries the same
        # field: a caller that reaches this mapping directly must not get a
        # usage object two other builders would have filled (#1446 review).
        if "compute_units" in ir_usage:
            usage["compute_units"] = ir_usage["compute_units"]
        return usage

    def _capture_preserve_metadata(
        self,
        provider_response: dict[str, Any],
        ir_response: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Capture extra fields from provider response for lossless round-trip."""
        output_items = provider_response.get("output", [])
        preserve = self._get_preserve_fields(ctx)
        extras = {
            k: v
            for k, v in provider_response.items()
            if k in preserve and v is not None
        }
        if extras:
            ctx.store_response_extras(extras)

        items_meta = [
            _capture_item_metadata(item)
            for item in output_items
            if OpenResponsesMessageOps.is_portable_response_output_item(item)
        ]
        if items_meta:
            ctx.store_output_items_meta(items_meta)

    def _apply_preserve_metadata(
        self,
        provider_response: dict[str, Any],
        ir_response: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Re-inject captured metadata fields in *preserve* mode."""
        echo = ctx.get_echo_fields()
        core_keys = {
            "id",
            "object",
            "created_at",
            "model",
            "output",
            "status",
            "usage",
        }
        # Apply required defaults first, then override with actual echo
        for k, v in self._get_required_defaults(ctx).items():
            if k not in core_keys and k not in provider_response:
                provider_response[k] = v
        for k, v in echo.items():
            if k not in core_keys:
                provider_response[k] = v

        # Ensure echoed tools have required 'strict' field
        for tool in provider_response.get("tools", []):
            if isinstance(tool, dict) and tool.get("type") == "function":
                tool.setdefault("strict", None)

        # Restore per-output-item metadata
        items_meta = ctx.get_output_items_meta()
        output = provider_response.get("output", [])
        for i, meta in enumerate(items_meta):
            if i >= len(output):
                break
            item = output[i]
            if "id" in meta:
                item["id"] = meta["id"]
            if "status" in meta:
                item["status"] = meta["status"]
            # Restore per-content-part metadata
            content_meta = meta.get("content_meta", [])
            content = item.get("content", [])
            for j, pm in enumerate(content_meta):
                if j >= len(content):
                    break
                if "annotations" in pm:
                    content[j]["annotations"] = pm["annotations"]
                if "logprobs" in pm:
                    content[j]["logprobs"] = pm["logprobs"]
