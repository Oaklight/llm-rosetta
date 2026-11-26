"""Type-checking host for the ``OpenResponsesConverter`` mixins.

The converter's stream and response logic lives in mixins that call members
defined on the final class (``BaseConverter``, the converter's own helpers) or
on a sibling mixin.  A type checker cannot see those members from inside a
mixin, so each mixin derives from :data:`OpenResponsesMixinHost`.

At runtime the host is plain ``object``; the declarations below exist only
for the type checker, so the MRO of the converter is unchanged.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...base import BaseConverter
    from ...base.context import ConversionContext, StreamContext
    from llm_rosetta.ir.model.stream import ContentBlockEndEvent, ContentBlockStartEvent
    from llm_rosetta.ir.model.stream import RefusalDeltaEvent
    from llm_rosetta.ir.model.response import UsageInfo
    from .content_ops import OpenResponsesContentOps
    from .stream_context import MessageItemState, OpenResponsesStreamContext

    class OpenResponsesMixinHost(BaseConverter):
        """Members the mixins read from the final converter or a sibling."""

        _CONVERTER_TAG: str
        content_ops: OpenResponsesContentOps
        # Kept ``Any`` like ``BaseConverter.tool_ops``: this declaration is
        # inherited by the converter, and a narrower class type would leak
        # ``OpenResponsesToolOps``'s ``ToolDefinition | list`` return union
        # into every caller of ``converter.tool_ops``.
        tool_ops: Any

        def _get_response_id_prefix(
            self, context: ConversionContext | StreamContext | None = None
        ) -> str: ...

        def _get_preserve_fields(
            self, context: ConversionContext | StreamContext | None = None
        ) -> set[str]: ...

        def _get_required_defaults(
            self, context: ConversionContext | StreamContext | None = None
        ) -> dict[str, Any]: ...

        @staticmethod
        def _completion_timestamp() -> int: ...

        @staticmethod
        def _citation_warnings(
            context: ConversionContext | None, kwargs: dict[str, Any]
        ) -> list[str] | None: ...

        @staticmethod
        def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> UsageInfo: ...

        @staticmethod
        def _build_finish_usage(pending_usage: dict[str, Any]) -> dict[str, Any]: ...

        @staticmethod
        def _terminal_events(response: dict[str, Any]) -> list[dict[str, Any]]: ...

        @staticmethod
        def _content_part_done_events(
            item: MessageItemState,
        ) -> list[dict[str, Any]]: ...

        def _close_message_item_events(
            self, item: MessageItemState
        ) -> list[dict[str, Any]]: ...

        @staticmethod
        def _build_stream_function_call_item(
            context: OpenResponsesStreamContext | None,
            *,
            item_id: str,
            call_id: str,
            tool_name: str,
            arguments: str,
            status: str,
        ) -> dict[str, Any]: ...

        @staticmethod
        def _build_stream_custom_tool_call_item(
            context: OpenResponsesStreamContext | None,
            *,
            item_id: str,
            call_id: str,
            tool_name: str,
            input_str: str,
            status: str,
        ) -> dict[str, Any]: ...

        def _handle_ir_content_block_start_to_p(
            self,
            event: ContentBlockStartEvent,
            context: OpenResponsesStreamContext | None,
        ) -> dict[str, Any] | list[dict[str, Any]]: ...

        def _handle_ir_content_block_end_to_p(
            self,
            event: ContentBlockEndEvent,
            context: OpenResponsesStreamContext | None,
        ) -> dict[str, Any] | list[dict[str, Any]]: ...

        def _handle_ir_refusal_delta_to_p(
            self,
            event: RefusalDeltaEvent,
            context: StreamContext | None,
        ) -> dict[str, Any] | list[dict[str, Any]]: ...
else:
    OpenResponsesMixinHost = object
