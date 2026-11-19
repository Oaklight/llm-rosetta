"""
LLM-Rosetta - OpenAI Responses API Converter

Top-level converter implementing the 6 explicit interfaces.
Composes ContentOps, ToolOps, MessageOps, and ConfigOps for full bidirectional
conversion between IR and OpenAI Responses API format.

Note: Responses API uses a flat list of items (input/output) instead of
nested messages. The converter handles this structural difference.
"""

import time
from typing import Any, ClassVar, cast

from ....types.ir import (
    TextPart,
    is_text_part,
)
from ....types.ir.passthrough import ProviderPassthroughItem
from ....types.ir.request import IRRequest
from ....types.ir.response import IRResponse
from ...base import BaseConverter
from ...base.context import ConversionContext, StreamContext
from ....ir.primitives import expand_response
from ....ir.passes.tool_names import STRICT_TOOL_NAME_PATTERN
from ...base.helpers.extensions import extension_loss_warning, wire_extensions
from llm_rosetta.conversion_warnings import DOWNGRADED, report_loss
from ...base.helpers.tool_group import (
    ToolGroup,
    group_tools,
)
from .stream_context import OpenResponsesStreamContext
from ._constants import (
    OPEN_RESPONSES_PRESERVE_FIELDS,
    OPEN_RESPONSES_REQUIRED_DEFAULTS,
    RESPONSES_PRESERVE_FIELDS,
    RESPONSES_REQUIRED_DEFAULTS,
    TERMINAL_EVENTS,
)
from .config_ops import OpenResponsesConfigOps
from .content_ops import OpenResponsesContentOps
from .message_ops import OpenResponsesMessageOps
from .tool_ops import (
    _INTRINSIC_RESULT_TYPES,
    _ITEM_TO_INTRINSIC_KIND,
    OpenResponsesToolOps,
    harvest_additional_tools,
)
from .response_ops import OpenResponsesResponseMixin
from .stream_finish_ops import OpenResponsesStreamFinishMixin
from .stream_from_ops import OpenResponsesStreamFromMixin
from .stream_to_ops import OpenResponsesStreamToMixin
from ._wire_helpers import (  # noqa: F401
    _capture_item_metadata,
    _declared_tool_types,
    _forward_passthrough_request_fields,
    _PROFILE_REQUEST_FIELDS,
    _insert_after_call,
    _response_failure_metadata,
    _response_status_reason,
    _stream_finish_reason,
)


#: Server-tool output item types this format reads (``web_search_call``,
#: ``code_interpreter_call``, …).  A streamed item of one of these types is a
#: provider-executed call; the value is its canonical intrinsic kind.
_SERVER_ITEM_INTRINSIC_KINDS: frozenset[str] = frozenset(_ITEM_TO_INTRINSIC_KIND)

#: Server *result* item types (``code_interpreter_call_output``, …).  A
#: streamed item of one of these types is a provider-executed result, carried
#: as a one-shot ``ToolResultEvent``.
_SERVER_RESULT_ITEM_TYPES: frozenset[str] = frozenset(_INTRINSIC_RESULT_TYPES)


class OpenResponsesConverter(
    OpenResponsesResponseMixin,
    OpenResponsesStreamFromMixin,
    OpenResponsesStreamToMixin,
    OpenResponsesStreamFinishMixin,
    BaseConverter,
):
    """Open Responses spec converter (vendor-neutral base).

    Implements the 6 explicit conversion interfaces defined by BaseConverter.

    Uses composition of Ops classes for modular, testable conversion logic.

    Note: the spec uses ``input`` for request items and ``output`` for response
    items, with a flat item list structure.  :class:`OpenAIResponsesConverter`
    derives from this class and layers the OpenAI-specific profile on top.
    """

    content_ops_class = OpenResponsesContentOps
    tool_ops_class = OpenResponsesToolOps
    message_ops_class = OpenResponsesMessageOps
    config_ops_class = OpenResponsesConfigOps
    _CONVERTER_TAG = "open_responses"
    _PASSTHROUGH_RESTORE_KEY = "output"
    # The spec publishes ``^[a-zA-Z0-9_-]{1,64}$`` for a function name.
    _TOOL_NAME_PATTERN = STRICT_TOOL_NAME_PATTERN

    # Preserve-mode echo fields and required response defaults.  These are the
    # spec-level defaults (no OpenAI-only lifecycle fields, no forced
    # ``store: true``); the conversion pipeline layers a provider shim's
    # ``preserve_fields`` / ``store_default`` settings on top via
    # ``context.options``, and ``OpenAIResponsesConverter`` overrides the class
    # attributes directly for direct, context-free use.
    _PRESERVE_FIELDS: ClassVar[set[str]] = OPEN_RESPONSES_PRESERVE_FIELDS
    _REQUIRED_DEFAULTS: ClassVar[dict[str, Any]] = OPEN_RESPONSES_REQUIRED_DEFAULTS

    # Default response ID prefix.  The vendor-neutral spec has none; the OpenAI
    # profile sets ``"resp_"``.  Used as a fallback when no shim-driven prefix
    # (``context.options["response_id_prefix"]``) is available.
    _RESPONSE_ID_PREFIX = ""

    def _get_response_id_prefix(
        self, context: ConversionContext | StreamContext | None = None
    ) -> str:
        """Return the response ID prefix from context or class default.

        Reads ``response_id_prefix`` from ``context.options`` if set by
        the conversion pipeline (populated from the provider shim's
        ``response_id_prefix`` field).  Falls back to the class constant
        ``_RESPONSE_ID_PREFIX`` for direct converter usage without a
        pipeline.
        """
        if context is not None:
            prefix = context.options.get("response_id_prefix")
            if prefix is not None:
                return prefix
        return self._RESPONSE_ID_PREFIX

    def _get_preserve_fields(
        self, context: ConversionContext | StreamContext | None = None
    ) -> set[str]:
        """Return the effective preserve/echo field set.

        Starts from the class-level spec set and adds any shim-declared
        ``preserve_fields`` carried in ``context.options`` (populated by the
        pipeline from the provider shim).  Direct converter use without a
        pipeline falls back to the class attribute.
        """
        fields = set(self._PRESERVE_FIELDS)
        if context is not None:
            extra = context.options.get("preserve_fields")
            if extra:
                fields.update(extra)
        return fields

    def _get_required_defaults(
        self, context: ConversionContext | StreamContext | None = None
    ) -> dict[str, Any]:
        """Return the effective response required-defaults map.

        Starts from the class-level spec defaults and, when the provider shim
        declares a ``store_default`` in ``context.options``, sets the ``store``
        field accordingly (the OpenAI profile stores responses server-side).
        """
        defaults = dict(self._REQUIRED_DEFAULTS)
        if context is not None:
            store_default = context.options.get("store_default")
            if store_default is not None:
                defaults["store"] = store_default
        return defaults

    @staticmethod
    def _completion_timestamp() -> int:
        """Current Unix timestamp for completed_at fields."""
        return int(time.time())

    def __init__(self):
        self.content_ops = self.content_ops_class()
        self.tool_ops = self.tool_ops_class()
        self.message_ops = self.message_ops_class(self.content_ops, self.tool_ops)
        self.config_ops = self.config_ops_class()

    @classmethod
    def create_stream_context(cls) -> OpenResponsesStreamContext:
        """Create a stream context with Responses API specific state."""
        return OpenResponsesStreamContext()

    # ==================== Top-level Interfaces ====================

    def _do_request_to_provider(
        self,
        ir_request: IRRequest,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ctx = context
        result: dict[str, Any] = {"model": ir_request["model"]}

        # 1. System instruction → instructions field (plain string)
        system_parts = ir_request.get("system_instruction")
        if system_parts:
            result["instructions"] = " ".join(
                p["text"] for p in system_parts if is_text_part(p)
            )

        # 2. Messages → input items — the pairing repair already ran at the
        # converter boundary, so every target sees the same repaired history.
        ir_messages = ir_request.get("messages", [])
        # Batch ids and the stripped tool config were applied at the
        # converter boundary, over the same repaired list.
        items, msg_warnings = self.message_ops.ir_messages_to_p(
            ir_messages, target_provider=self._CONVERTER_TAG
        )
        ctx.warnings.extend(msg_warnings)
        result["input"] = items

        # 3-5. Tools + tool_choice + tool_config
        self._apply_tool_config(ir_request, result, ctx)

        # 6. Generation config
        gen_config = ir_request.get("generation")
        if gen_config:
            gen_fields = self.config_ops.ir_generation_config_to_p(gen_config)
            result.update(gen_fields)

        # 7. Response format
        resp_format = ir_request.get("response_format")
        if resp_format:
            rf_fields = self.config_ops.ir_response_format_to_p(resp_format)
            result.update(rf_fields)

        # 8. Stream config
        stream = ir_request.get("stream")
        if stream:
            stream_fields = self.config_ops.ir_stream_config_to_p(stream)
            result.update(stream_fields)
            if stream.get("include_usage"):
                report_loss(
                    "OpenAI Responses API reports usage in every stream; "
                    "include_usage is not a request field",
                    subject="stream.include_usage",
                )

        # 9. Reasoning config
        reasoning = ir_request.get("reasoning")
        if reasoning:
            rc_kwargs: dict[str, Any] = {}
            if ctx and "reasoning_cap" in ctx.options:
                rc_kwargs["reasoning_cap"] = ctx.options["reasoning_cap"]
            reasoning_fields = self.config_ops.ir_reasoning_config_to_p(
                reasoning, **rc_kwargs
            )
            result.update(reasoning_fields)

        # 10. Cache config
        cache = ir_request.get("cache")
        if cache:
            cache_fields = self.config_ops.ir_cache_config_to_p(cache)
            result.update(cache_fields)

        # 11. Service tier — the two OpenAI-family formats share a value space.
        service_tier = ir_request.get("service_tier")
        if service_tier:
            result["service_tier"] = service_tier

        # 12. Provider extensions (pass-through)
        extensions = ir_request.get("provider_extensions")
        if extensions:
            # Merge _text_verbosity back into the text object
            verbosity = extensions.get("_text_verbosity")
            if verbosity is not None:
                result.setdefault("text", {})["verbosity"] = verbosity
            # ``wire_extensions`` drops internal ``_``-prefixed keys, so the
            # spec-form ``_open_responses_allowed_tools`` (re-emitted inside
            # ``tool_choice`` by ``_apply_tool_config``) never leaks as a
            # top-level field.  A legacy top-level ``allowed_tools`` is a plain
            # key and still passes through.
            merged = wire_extensions(ir_request, self._CONVERTER_TAG)
            # A profile-only field reaches this body only when the target
            # declares it; otherwise it is reported, because a neutral
            # upstream answering 400 to a field it never defined is a failure
            # this gateway caused.
            declared = frozenset(ctx.options.get("request_fields") or ())
            withheld = {
                key
                for key in merged
                if key in _PROFILE_REQUEST_FIELDS and key not in declared
            }
            for key in sorted(withheld):
                merged.pop(key)
                # One warning per field, with its IR path as the subject:
                # ``explained()`` matches a subject against the path it
                # explains, and a comma-joined list matches none of them.
                report_loss(
                    extension_loss_warning([key], self._CONVERTER_TAG),
                    subject=f"provider_extensions.{key}",
                )
            result.update(merged)

        return result

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ir_request: dict[str, Any] = {
            "model": provider_request.get("model", ""),
            "messages": [],
        }

        # 1. Instructions → system_instruction (list[TextPart])
        instructions = provider_request.get("instructions")
        if instructions:
            ir_request["system_instruction"] = [
                TextPart(type="text", text=instructions)
            ]

        # 2. Input items → messages
        input_items = provider_request.get("input", [])
        if isinstance(input_items, str):
            input_items = [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_items}],
                }
            ]
        # Codex transports its tool definitions inside `input` as
        # `additional_tools` items instead of the top-level `tools` field.
        # Harvest them before message conversion so the spent items can be
        # dropped: routed through the passthrough path they would strand a
        # content-less assistant message, which the Chat converter emits as
        # `{"role": "assistant", "content": ""}`.
        top_level_tools = list(provider_request.get("tools") or [])
        nested_tools, input_items = harvest_additional_tools(input_items)

        if isinstance(input_items, list):
            ir_messages = self.message_ops.p_messages_to_ir(
                input_items, provider=self._CONVERTER_TAG
            )
            ir_request["messages"] = ir_messages

        # 3. Tools (provider→IR, uncached after #1410)
        # Filter out disabled tools before converting (external_web_access=False)
        tools = top_level_tools + nested_tools
        if tools:
            active_tools = [
                t
                for t in tools
                if not (isinstance(t, dict) and t.get("external_web_access") is False)
            ]
            if active_tools:
                ir_tools = self._convert_p_tools_to_ir(active_tools)
                if ir_tools:
                    # Naming is the target's call and happens there; this
                    # leg only reports the containers that arrived unnamed.
                    self._report_generated_groups(ir_tools)
                    ir_request["tools"] = ir_tools

        # 4-5. Tool choice + tool config
        self._convert_p_tool_config_to_ir(provider_request, ir_request)

        # 6. Generation config
        gen_config = self.config_ops.p_generation_config_to_ir(provider_request)
        if gen_config:
            ir_request["generation"] = gen_config

        # 7. Response format (text field in Responses API)
        text_format = provider_request.get("text")
        if text_format:
            ir_request["response_format"] = self.config_ops.p_response_format_to_ir(
                text_format
            )

        # 8. Reasoning config
        reasoning = self.config_ops.p_reasoning_config_to_ir(provider_request)
        if reasoning:
            ir_request["reasoning"] = reasoning

        # 9. Stream config
        stream = provider_request.get("stream")
        stream_options = provider_request.get("stream_options")
        if stream is not None or stream_options:
            ir_request["stream"] = self.config_ops.p_stream_config_to_ir(
                {"stream": stream, "stream_options": stream_options}
            )

        # 10. Cache config
        self._convert_p_cache_to_ir(provider_request, ir_request)

        # 11. Service tier — Chat Completions and Responses share one value
        # space, so the tier travels as the source wrote it.
        service_tier = provider_request.get("service_tier")
        if service_tier:
            ir_request["service_tier"] = service_tier

        # 12. Provider extensions (passthrough fields like allowed_tools)
        allowed_tools = provider_request.get("allowed_tools")
        if allowed_tools is not None:
            ir_request.setdefault("provider_extensions", {})["allowed_tools"] = (
                allowed_tools
            )

        # 13. Responses-specific request fields not modelled in IR.
        ctx = context if context is not None else ConversionContext()
        _forward_passthrough_request_fields(provider_request, ir_request)

        # Preserve mode: capture request fields for echo-back in response
        if ctx.metadata_mode == "preserve":
            preserve = self._get_preserve_fields(ctx)
            echo = {k: v for k, v in provider_request.items() if k in preserve}
            if echo:
                ctx.store_request_echo(echo)

        return ir_request

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:

        choices = []
        output_items = provider_response.get("output", [])

        finish_reason_val = _response_status_reason(provider_response)

        # Convert output items to IR message content.  Unlike the request leg,
        # no ``provider=`` is passed: non-portable response items (e.g.
        # ``compaction``) are captured as ``provider_passthrough_items`` by the
        # caller, so this leg must not also emit ``ProviderPassthroughItem``.
        ir_items = self.message_ops.p_messages_to_ir(output_items)

        # Collect all content parts into a single assistant message
        message_content, message_metadata = self._flatten_message_items(ir_items)

        if finish_reason_val in {"error", "cancelled"}:
            message_metadata = {
                **message_metadata,
                **_response_failure_metadata(provider_response, context),
            }

        # Empty output still needs a choice carrying its terminal condition.
        if message_content or finish_reason_val in {
            "error",
            "cancelled",
            "length",
            "content_filter",
        }:
            ir_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message_content,
            }
            if message_metadata:
                ir_msg["provider_metadata"] = message_metadata
            choices.append(
                {
                    "index": 0,
                    "message": ir_msg,
                    "finish_reason": {"reason": finish_reason_val},
                }
            )

        ir_response: dict[str, Any] = {
            "id": self.strip_response_id_prefix(
                provider_response.get("id", ""),
                self._get_response_id_prefix(context),
            ),
            "object": "response",
            "created": int(provider_response.get("created_at", 0)),
            "model": provider_response.get("model", ""),
            "choices": choices,
        }
        passthrough_items = [
            ProviderPassthroughItem(
                type="provider_passthrough_item",
                provider=self._CONVERTER_TAG,
                payload=dict(item),
                position=position,
            )
            for position, item in enumerate(output_items)
            if isinstance(item, dict)
            and not self.message_ops.is_portable_response_output_item(item)
        ]
        if passthrough_items:
            ir_response["provider_passthrough_items"] = passthrough_items

        # Handle usage
        p_usage = provider_response.get("usage")
        if p_usage:
            ir_response["usage"] = self._build_p_usage_to_ir(p_usage)

        if provider_response.get("service_tier") is not None:
            ir_response["service_tier"] = provider_response["service_tier"]

        return ir_response

    def _flatten_message_items(
        self, ir_items: list[Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Collapse a response's IR items into the one message a choice holds.

        A choice holds a single IR message, so a response's several message
        items — the commentary and the final answer an agent run reports
        apart — collapse into it here.  Each item's phase is recorded on its
        text parts so the boundary stays recoverable: without it the write
        leg rebuilds one item whose body holds both replies (#997).

        Args:
            ir_items: IR items read from the response's ``output``.

        Returns:
            The message's content parts, and the provider metadata of the
            first item that carried any.
        """
        message_content: list[dict[str, Any]] = []
        message_metadata: dict[str, Any] = {}
        for ir_item in ir_items:
            if not (isinstance(ir_item, dict) and "role" in ir_item):
                continue
            content = cast(list, ir_item.get("content", []))
            item_phase = (ir_item.get("provider_metadata") or {}).get("responses_phase")
            if item_phase:
                for part in content:
                    self._stamp_text_part_phase(part, item_phase)
            message_content.extend(content)
            pm = ir_item.get("provider_metadata")
            if pm and not message_metadata:
                message_metadata = pm
        return message_content, message_metadata

    def _do_response_to_provider(
        self,
        ir_response: IRResponse,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Write the turn as a whole body, by folding the stream it would be.

        A body is not a second content model: it is the event sequence
        :func:`~llm_rosetta.ir.primitives.expand_response` renders, written by
        the same per-event code the streaming leg uses, and read back off the
        terminal event.  What a whole body carries and an event does not — the
        message's ``status``, a text part's ``annotations`` / ``logprobs``,
        ``service_tier`` — is applied afterwards by :meth:`_apply_body_shape`.
        """
        stream_context = self.create_stream_context()
        stream_context.options.update(getattr(context, "options", None) or {})

        choices = ir_response.get("choices") or []
        body: dict[str, Any] = {}
        output: list[dict[str, Any]] = []
        for choice in choices or [None]:
            # One fold per choice keeps every choice's items in that choice's
            # own output entry; the body numbers message items across the whole
            # response afterwards (`_renumber_message_items`), because each
            # fold starts its own count.
            one = cast(IRResponse, dict(ir_response))
            if choice is not None:
                one["choices"] = [choice]
            folded = self._fold_events(one, stream_context)
            output.extend(folded.get("output") or [])
            body = folded
        if choices:
            body["output"] = output
            self._renumber_message_items(body, str(ir_response.get("id", "")))

        self._apply_body_shape(body, ir_response, context=context, kwargs=kwargs)
        return body

    def _fold_events(
        self,
        ir_response: IRResponse,
        stream_context: OpenResponsesStreamContext,
    ) -> dict[str, Any]:
        """Drive the stream writer over *ir_response*'s expansion.

        Args:
            ir_response: The turn to fold.
            stream_context: The writer context, shared across choices so the
                response envelope (id, model, created) is built once.

        Returns:
            The response payload of the terminal event, or ``{}``.
        """
        body: dict[str, Any] = {}
        for event in expand_response(ir_response):
            result = self.stream_response_to_provider(event, context=stream_context)
            for chunk in result if isinstance(result, list) else [result]:
                if isinstance(chunk, dict) and chunk.get("type") in TERMINAL_EVENTS:
                    body = dict(chunk.get("response") or {})
        return body

    def _apply_tool_config(
        self,
        ir_request: IRRequest,
        result: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Apply tools, tool_choice, and tool_config to provider request."""
        tools = ir_request.get("tools")
        if tools:
            result["tools"] = self._tools_to_p(tools, ctx)
        tool_choice = ir_request.get("tool_choice")
        if tool_choice:
            allowed_tools = (ir_request.get("provider_extensions") or {}).get(
                "_open_responses_allowed_tools"
            )
            result["tool_choice"] = self.tool_ops.ir_tool_choice_to_p(
                tool_choice,
                allowed_tools=allowed_tools,
                tool_types=_declared_tool_types(tools),
            )
        tool_config = ir_request.get("tool_config")
        if tool_config:
            tc_fields = self.tool_ops.ir_tool_config_to_p(tool_config)
            result.update(tc_fields)

    def _tools_to_p(self, tools: list[Any], ctx: ConversionContext) -> list[Any]:
        """Render the tool list, groups and all.

        A target that carries tool groups gets the client's grouping back, as a
        container per group; every other target gets the flat list it has
        always had, with the group folded into the tool names.  Whether this
        target is one of the former is the shim's ``tools.tool_groups``,
        passed down on the context, because the converter cannot see a shim.

        The grouping is decided here rather than in the IR: the IR keeps every
        tool flat, with ``tool_group`` recording where it came from, and only a
        writer knows whether its wire format can say that.  Which is also why
        ``group_tools`` is a helper and not a pipeline pass — the ordering and
        the entry shape are the writer's business.

        Children go out under the names the naming pass settled on.  For a
        target that carries groups those are the client's own child names:
        that pass is told this target carries groups, so it qualifies a child
        only against the members of its own container — measured, an upstream
        accepts a child named like a top-level tool and like a child of another
        container.  Nothing is shortened here, because the name map is built
        from the IR, and rewriting a child at this layer would leave a replayed
        history call transliterated to a spelling this function had just
        discarded.
        """
        if not ctx.options.get("tool_groups"):
            return self._get_cached_ir_tools_to_p(tools)

        rendered: list[Any] = []
        reported_containers: set[str] = set()
        for entry in group_tools(tools, carries_groups=True):
            if not isinstance(entry, ToolGroup):
                rendered.extend(self._get_cached_ir_tools_to_p([entry]))
                continue
            children = self._get_cached_ir_tools_to_p(list(entry.tools))
            # A group whose every member this target declined is not a group: a
            # childless container is what every upstream that accepts one
            # rejects.  The list conversion drops each declined entry on its
            # own, so this covers only a writer that declines all of them.
            # Nothing the pipeline produces reaches it — a tool the target
            # cannot express is dropped from the IR and its group never forms —
            # but the IR is a public TypedDict and the converter is public API,
            # so a caller can hand one to ``request_to_provider`` directly, and
            # that is the path the test covers.
            if not children:
                continue
            # The container name goes out as the client wrote it.  It is the
            # client's own identity for these tools, and a rename would need a
            # wire-group → client-group translation the response leg could undo
            # — which does not exist, so a repaired container would come back
            # under a namespace the client never wrote.  The target's name rule
            # still applies to that name, though, and when it fails the whole
            # request is refused upstream.  Report it here, where the failure
            # can be attributed to the container before the upstream answers
            # (#1610 review).
            pattern = self._TOOL_NAME_PATTERN
            if (
                pattern is not None
                and not pattern.fullmatch(entry.name)
                and entry.name not in reported_containers
            ):
                reported_containers.add(entry.name)
                report_loss(
                    f"Tool container {entry.name!r} is not a legal group name "
                    f"for this target (it must match {pattern.pattern!r}), so "
                    "it goes upstream unchanged and the upstream is expected "
                    "to reject the request; rename the container, or send "
                    "these tools to a target that flattens the group instead",
                    code=DOWNGRADED,
                    subject=entry.name,
                )
            # ``description`` is required on a container by OpenAI and
            # OpenRouter: probed, an absent one is `400 Missing required
            # parameter: 'tools[0].description'` and `400 Invalid Responses API
            # request` respectively, while an empty string is accepted by both.
            # A container the client gave no description to therefore goes out
            # with "", rather than with text the client never wrote — a longer
            # placeholder is accepted everywhere but costs input tokens on
            # DeepSeek and says nothing the children do not already say.
            container: dict[str, Any] = {
                "type": "namespace",
                "name": entry.name,
                "description": entry.description,
                "tools": children,
            }
            rendered.append(container)
        return rendered

    def _convert_p_tool_config_to_ir(
        self,
        provider_request: dict[str, Any],
        ir_request: dict[str, Any],
    ) -> None:
        """Extract tool_choice and tool_config from provider request into IR.

        Args:
            provider_request: Responses request body.
            ir_request: IR request under construction.
        """
        tool_choice = provider_request.get("tool_choice")
        if tool_choice is not None:
            # Collect into a scratch dict so an empty ``provider_extensions`` is
            # not created when the tool choice carries no provider-bound object.
            extensions: dict[str, Any] = {}
            ir_request["tool_choice"] = self.tool_ops.p_tool_choice_to_ir(
                tool_choice, extensions=extensions
            )
            if extensions:
                ir_request.setdefault("provider_extensions", {}).update(extensions)
        tool_config_fields: dict[str, Any] = {}
        if "parallel_tool_calls" in provider_request:
            tool_config_fields["parallel_tool_calls"] = provider_request[
                "parallel_tool_calls"
            ]
        if "max_tool_calls" in provider_request:
            tool_config_fields["max_tool_calls"] = provider_request["max_tool_calls"]
        if tool_config_fields:
            ir_request["tool_config"] = self.tool_ops.p_tool_config_to_ir(
                tool_config_fields
            )

    def _convert_p_cache_to_ir(
        self,
        provider_request: dict[str, Any],
        ir_request: dict[str, Any],
    ) -> None:
        """Extract cache config from provider request into IR."""
        cache_fields: dict[str, Any] = {}
        if "prompt_cache_key" in provider_request:
            cache_fields["prompt_cache_key"] = provider_request["prompt_cache_key"]
        if "prompt_cache_retention" in provider_request:
            cache_fields["prompt_cache_retention"] = provider_request[
                "prompt_cache_retention"
            ]
        if cache_fields:
            ir_request["cache"] = self.config_ops.p_cache_config_to_ir(cache_fields)


class OpenAIResponsesConverter(OpenResponsesConverter):
    """OpenAI Responses API converter (OpenAI profile preset).

    A thin preset over the vendor-neutral :class:`OpenResponsesConverter` that
    turns the OpenAI profile on for **direct, context-free use**:

    - the ``resp_`` response-id prefix (the spec defines no prefix);
    - OpenAI-only lifecycle echo fields (``billing``, ``moderation``, ...);
    - OpenAI's default of server-side storage (``store: true``).

    The conversion pipeline no longer selects this class: ``openai_responses``
    is a *shim* over the ``open_responses`` base, and the shim's
    ``response_id_prefix`` / ``preserve_fields`` / ``store_default`` settings
    carry the same traits through ``context.options``.  This class is retained
    as an importable preset so existing direct users keep their behavior.
    """

    _CONVERTER_TAG = "openai_responses"
    _RESPONSE_ID_PREFIX = "resp_"
    _PRESERVE_FIELDS = RESPONSES_PRESERVE_FIELDS
    _REQUIRED_DEFAULTS = RESPONSES_REQUIRED_DEFAULTS
