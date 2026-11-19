"""Intrinsic (server-tool) leg of the Responses tool ops."""

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from ...base.tools.content import (
    convert_ir_content_blocks_to_p,
)
from ....types.ir import (
    ToolCallPart,
    ToolResultPart,
)
from ...base import IntrinsicLeg
from llm_rosetta.conversion_warnings import report_loss
from ...base.tools import (
    intrinsic_kind_of,
    sanitize_tool_call_id,
)

from .tool_items import (
    _INTRINSIC_KIND_TO_ITEM,
    _INTRINSIC_KIND_TO_RESULT_ITEM,
    _build_intrinsic_result_item,
    _build_server_item,
    _server_item_renderable,
    _warn_dropped_intrinsic_result,
)

logger = logging.getLogger(__name__)


class IntrinsicToolOpsMixin:
    """Intrinsic-kind rendering and streaming items for ``OpenResponsesToolOps``."""

    if TYPE_CHECKING:
        # Defined on ``OpenResponsesToolOps``, the final class of this mixin.
        @classmethod
        def ir_client_result_to_p(
            cls, ir_tool_result: ToolResultPart, **kwargs: Any
        ) -> dict: ...

    @staticmethod
    def stream_intrinsic_item(
        kind: str,
        item_id: str,
        tool_input: dict[str, Any] | None = None,
        *,
        status: str = "in_progress",
    ) -> dict[str, Any] | None:
        """Build a native server item for a streamed intrinsic call.

        The streaming counterpart of ``_build_server_item``: it lets the
        writer re-emit the provider's own server item type for an intrinsic
        ``ToolCallStartEvent`` instead of downgrading it to a client
        ``function_call``.  Returns ``None`` for a kind this dialect cannot
        render as a server item, so the caller drops the event rather than
        emitting a client call.
        """
        item_type = _INTRINSIC_KIND_TO_ITEM.get(kind)
        if item_type is None:
            return None
        return _build_server_item(
            item_type,
            item_id=item_id,
            intrinsic_kind=kind,
            fields=tool_input or {},
            arguments="",
            status=status,
        )

    @staticmethod
    def stream_result_item(
        kind: str, call_id: str, result: Any
    ) -> dict[str, Any] | None:
        """Build a server result item for a streamed ``ToolResultEvent``.

        The streaming counterpart of the result half of ``ir_tool_result_to_p``:
        it produces the provider's own ``*_call_output`` item.  Returns ``None``
        for a kind with no separate output item (``web_search`` — its result
        rides on the call item) or one this dialect cannot render, so the caller
        drops the event rather than re-encoding it as a client
        ``function_call_output``.
        """
        item_type = _INTRINSIC_KIND_TO_RESULT_ITEM.get(kind)
        if item_type is None:
            return None
        output: Any = result
        if isinstance(result, list):
            from .content_ops import OpenResponsesContentOps

            output = convert_ir_content_blocks_to_p(result, OpenResponsesContentOps)
        elif isinstance(result, dict):
            output = json.dumps(result)
        elif result is None:
            output = ""
        return {"type": item_type, "call_id": call_id, "output": output}

    @classmethod
    def intrinsic_leg_from_flags(
        cls, kwargs: Any, *, default: IntrinsicLeg
    ) -> IntrinsicLeg:
        """Map this writer's ``for_history`` flag onto :data:`IntrinsicLeg`.

        The Responses write path says which leg it is on — ``message_ops``
        passes ``for_history=True``, the response writer says nothing — so the
        same hook pair answers differently per leg without a second predicate.
        A caller that says nothing takes the template's default, which is what
        this writer answered before the hooks existed.

        Args:
            kwargs: The keyword arguments the write template was called with.
            default: The leg when ``for_history`` is absent.

        Returns:
            ``"history"`` for a history write, else ``"response"``.
        """
        if "for_history" in kwargs:
            return "history" if kwargs["for_history"] else "response"
        return default

    @classmethod
    def intrinsic_renderable(
        cls, kind: str, fields: Mapping[str, Any], *, leg: IntrinsicLeg
    ) -> bool:
        """Whether a server item can be written for *kind* on this leg.

        A kind with no native item type (``google_search`` arriving from
        another lane, …) has one: it is written as a client ``function_call``
        on both legs.  A native item the IR cannot complete — the API requires
        a ``container_id`` on a code interpreter and queries with complete
        results on a file search, verified live 2026-10-13 — is still written
        natively on the response leg (the turn the provider just returned) and
        degraded on the history leg, where replaying it would be rejected.

        Args:
            kind: Canonical intrinsic kind (``intrinsic_kind_of``).
            fields: The IR ``tool_input`` for the call.
            leg: Which leg the part is written on.

        Returns:
            True when the call is written natively (see
            :meth:`ir_intrinsic_call_to_p`).
        """
        item_type = _INTRINSIC_KIND_TO_ITEM.get(kind)
        if item_type is None:
            return False
        # A `web_search_call` has a valid replay shape for history (an
        # `action`, no `call_id`/`arguments`), so it is the one kind whose
        # payload the API does not require on a replay.
        if item_type == "web_search_call" or leg == "response":
            return True
        return _server_item_renderable(item_type, fields if fields else {})

    @classmethod
    def intrinsic_call_degrades_on_history(cls, ir_tool_call: ToolCallPart) -> bool:
        """Whether the history-leg writer degrades this call to a client call.

        The result writer pairs with the call that is actually written, so it
        must consult the same decision.  See :meth:`intrinsic_renderable`.

        Args:
            ir_tool_call: IR tool call part with ``tool_type == "intrinsic"``.

        Returns:
            True when the call is written as a client ``function_call``.
        """
        fields = ir_tool_call.get("tool_input")
        return not cls.intrinsic_renderable(
            intrinsic_kind_of(ir_tool_call),
            fields if isinstance(fields, dict) else {},
            leg="history",
        )

    @classmethod
    def ir_intrinsic_call_to_p(
        cls,
        ir_tool_call: ToolCallPart,
        *,
        leg: IntrinsicLeg = "history",
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """IR intrinsic ToolCallPart → Responses native server item.

        The shape is a client ``function_call`` on neither leg for an
        unrenderable kind: :meth:`intrinsic_renderable` decides, and the
        template's ``non_native_intrinsic_call_to_p`` writes the degradation,
        so the call and its result cannot disagree about which shape was
        written.

        Args:
            ir_tool_call: IR tool call part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on.
            **kwargs: Extra arguments.

        Returns:
            Native server item, or ``None`` when the call is degraded.
        """
        kind = intrinsic_kind_of(ir_tool_call)
        tool_input = ir_tool_call.get("tool_input")
        fields = tool_input if isinstance(tool_input, dict) else {}
        if not cls.intrinsic_renderable(kind, fields, leg=leg):
            return None
        # A history part keys off the kind only.  It is deliberately not gated
        # by the target shim (which gates *definitions*): a history part is
        # context, and its shape here is independent of whether the same kind
        # is currently declared.  See the note in pipeline.py / #839.
        item_type = _INTRINSIC_KIND_TO_ITEM[kind]
        arguments = (
            json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
        )
        tool_call_id = sanitize_tool_call_id(
            ir_tool_call.get("tool_call_id", ir_tool_call.get("id", ""))
        )
        # A web search restores the status the read side lifted; every other
        # server call the IR knows is complete.
        status = (ir_tool_call.get("provider_metadata") or {}).get(
            "responses_status"
        ) or "completed"
        return _build_server_item(
            item_type,
            item_id=tool_call_id,
            intrinsic_kind=kind,
            fields=fields,
            arguments=arguments,
            status=status,
        )

    @classmethod
    def non_native_intrinsic_call_to_p(
        cls,
        ir_tool_call: ToolCallPart,
        *,
        leg: IntrinsicLeg,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Write an intrinsic call as the client ``function_call`` it degrades to.

        A Responses ``function_call`` is the only shape left for a server call
        this dialect cannot replay (see :meth:`intrinsic_renderable`), and it
        is the shape the matching result pairs with, so it carries the
        canonical kind as its name and the IR input as its arguments.

        Args:
            ir_tool_call: IR tool call part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on.
            **kwargs: Extra arguments.

        Returns:
            A client ``function_call`` item.
        """
        tool_input = ir_tool_call.get("tool_input")
        return {
            "type": "function_call",
            "call_id": sanitize_tool_call_id(
                ir_tool_call.get("tool_call_id", ir_tool_call.get("id", ""))
            ),
            "name": intrinsic_kind_of(ir_tool_call),
            "arguments": (
                json.dumps(tool_input)
                if isinstance(tool_input, dict)
                else str(tool_input)
            ),
        }

    @classmethod
    def ir_intrinsic_result_to_p(
        cls,
        ir_tool_result: ToolResultPart,
        *,
        leg: IntrinsicLeg = "history",
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """IR intrinsic ToolResultPart → Responses native server result item.

        Only the response leg has a native result item: on the history leg a
        native server call accepts no output item (its ``*_call_output`` type
        is rejected for a replayed call, and a ``function_call_output`` with a
        server call's id pairs with no client call), so the hook declines and
        the template's ``non_native_intrinsic_result_to_p`` decides between
        dropping the result and pairing it with a degraded call.

        Args:
            ir_tool_result: IR tool result part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on.
            **kwargs: Extra arguments.

        Returns:
            Native ``*_call_output`` item on the response leg, ``None``
            otherwise.
        """
        if leg != "response":
            return None
        result_item_type = _INTRINSIC_KIND_TO_RESULT_ITEM.get(
            intrinsic_kind_of(ir_tool_result)
        )
        if result_item_type is None:
            return None
        return _build_intrinsic_result_item(ir_tool_result, result_item_type, kwargs)

    @classmethod
    def non_native_intrinsic_result_to_p(
        cls,
        ir_tool_result: ToolResultPart,
        *,
        leg: IntrinsicLeg,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Write (or drop) an intrinsic result with no native item.

        A result is written as a client ``function_call_output`` only on the
        **history** leg, and only for a call that was degraded to a client
        ``function_call``: every other case answers ``{}`` — a ``web_search``
        result rides on its call item, and a native server call pairs with no
        output item on either leg.  On the **response** leg a server result is
        never handed to the client as a client output — the client did not run
        the call — so it is dropped with a warning.

        Args:
            ir_tool_result: IR tool result part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on.
            **kwargs: ``degraded_intrinsic_calls`` (dict[str, bool]) — intrinsic
                call id → whether the call writer degrades it, so a result
                pairs with the call that is actually written.

        Returns:
            A client ``function_call_output`` for a degraded call, else ``{}``.
        """
        kind = intrinsic_kind_of(ir_tool_result)
        if _INTRINSIC_KIND_TO_ITEM.get(kind) == "web_search_call":
            # Its payload rides on the call item; no separate output item on
            # either leg.
            _warn_dropped_intrinsic_result(ir_tool_result, kwargs)
            return {}
        if leg != "history":
            _warn_dropped_intrinsic_result(ir_tool_result, kwargs)
            report_loss(
                f"OpenAI Responses has no server result item for kind "
                f"{kind!r}; dropped the result of tool_call_id "
                f"{ir_tool_result.get('tool_call_id')!r}",
                subject=kind,
                leg="response",
                warnings=kwargs.get("warnings"),
            )
            return {}
        degraded = kwargs.get("degraded_intrinsic_calls")
        call_id = sanitize_tool_call_id(ir_tool_result.get("tool_call_id", ""))
        if degraded is not None and not degraded.get(call_id, True):
            _warn_dropped_intrinsic_result(ir_tool_result, kwargs)
            return {}
        return cast(dict[str, Any], cls.ir_client_result_to_p(ir_tool_result, **kwargs))
