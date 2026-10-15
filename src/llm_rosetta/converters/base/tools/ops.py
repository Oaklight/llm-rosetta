"""Base Tool Operations — abstract interface for the tool lifecycle.

Defines :class:`BaseToolOps`, the abstract base class that every
provider-specific ``ToolOps`` must implement.  Covers:
definition → choice → call → result → config.

An intrinsic (provider-executed) tool call — ``tool_type == "intrinsic"``, a
web search, a code execution — is a *derived* case of an ordinary tool call:
it differs in how it is written natively, not in being a call at all.  The
call/result templates below own that dispatch, so a converter supplies its
native rendering in the ``ir_intrinsic_*_to_p`` hooks and its kind↔native
tables, and nothing else.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal, cast

from ....types.ir import (
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)
from ....types.ir.tools import ToolCallConfig
from .intrinsic import intrinsic_kind_of, is_intrinsic_part

#: Which leg of a conversion an intrinsic part is being written on.
#:
#: ``"history"`` is a request body replayed from IR, where the server tool ran
#: on the provider side in an earlier turn; ``"response"`` is the turn the
#: provider has just returned.  A dialect may render the same kind differently
#: on each (open_responses writes a native server item only for a *response*,
#: degrading a history call the API would reject).
IntrinsicLeg = Literal["history", "response"]

#: The values :data:`IntrinsicLeg` accepts, for validating a caller's own word.
_INTRINSIC_LEGS: tuple[str, ...] = ("history", "response")


def _report_dropped_intrinsic(part: Any, kwargs: dict[str, Any]) -> None:
    """Report a response-leg intrinsic drop on the caller's warnings sink."""
    warnings = kwargs.get("warnings")
    if not isinstance(warnings, list):
        return
    kind = intrinsic_kind_of(part)
    warnings.append(
        f"Server tool {'result' if part.get('type') == 'tool_result' else 'call'} "
        f"of kind {kind!r} has no {kwargs.get('target', 'target')} form; dropped "
        f"from the response"
    )


class BaseToolOps(ABC):
    """Abstract base class for tool conversion operations.

    Uniformly handles all stages of the tool lifecycle:
    definition → choice → call → result → config.
    """

    # ==================== Intrinsic hooks ====================
    #
    # The default of every hook is "this dialect has no native form for an
    # intrinsic part", so a converter that carries no intrinsic vocabulary
    # (chat_completions) overrides none of them and every intrinsic part takes
    # the ordinary client path.

    @classmethod
    def intrinsic_leg(
        cls, kwargs: Mapping[str, Any], default: IntrinsicLeg
    ) -> IntrinsicLeg:
        """Which leg an intrinsic part is being written on.

        A caller may state it outright with ``leg=``; otherwise the converter
        maps its own write-path flag through :meth:`intrinsic_leg_from_flags`.
        The templates pass the writer's own historical default, because a
        dialect may have answered differently for a call and for a result;
        phase 4 is where one default can take over.

        Args:
            kwargs: The keyword arguments the write template was called with.
            default: The leg when neither the caller nor a converter flag says.

        Returns:
            ``"history"`` or ``"response"``.
        """
        leg = kwargs.get("leg")
        if leg in _INTRINSIC_LEGS:
            return cast(IntrinsicLeg, leg)
        return cls.intrinsic_leg_from_flags(kwargs, default=default)

    @classmethod
    def intrinsic_leg_from_flags(
        cls, kwargs: Mapping[str, Any], *, default: IntrinsicLeg
    ) -> IntrinsicLeg:
        """Map this converter's own leg flag onto :data:`IntrinsicLeg`.

        A converter whose write path already distinguishes the two legs maps
        its flag here (open_responses: ``for_history``); a dialect that does
        not answer per leg takes the default its template passed.

        Args:
            kwargs: The keyword arguments the write template was called with.
            default: The leg when no flag is present.

        Returns:
            ``"history"`` or ``"response"``.
        """
        return default

    @classmethod
    def intrinsic_renderable(
        cls, kind: str, fields: Mapping[str, Any], *, leg: IntrinsicLeg
    ) -> bool:
        """Whether this dialect can write *kind* natively on this leg.

        The one predicate behind both the native and the non-native outcome,
        so a call writer and a result writer cannot disagree about which shape
        the call takes.  Answers "could this be written natively at all", not
        "does this call have a payload worth writing": a dialect that degrades
        a history call which lacks a required field returns ``False`` here and
        still writes it natively on the response leg.

        Args:
            kind: Canonical intrinsic kind (``intrinsic_kind_of``).
            fields: The IR ``tool_input`` for the call.
            leg: Which leg the part is written on.

        Returns:
            True when :meth:`ir_intrinsic_call_to_p` can render it.
        """
        return False

    @classmethod
    def intrinsic_requires_result(cls, kind: str) -> bool:
        """Whether a replayed server call of *kind* must be followed by its result.

        Anthropic rejects a ``server_tool_use`` with no matching result block,
        while other dialects carry the result on the call item or omit it.  A
        call the target cannot replay alone is degraded by the history pass.

        Args:
            kind: Canonical intrinsic kind.

        Returns:
            True when the dialect needs the paired result in history.
        """
        return False

    @classmethod
    def ir_intrinsic_call_to_p(
        cls,
        ir_tool_call: ToolCallPart,
        *,
        leg: IntrinsicLeg = "history",
        **kwargs: Any,
    ) -> Any | None:
        """IR intrinsic ToolCallPart → this dialect's native server call.

        Args:
            ir_tool_call: IR tool call part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on.
            **kwargs: The caller's keyword arguments.

        Returns:
            The native server call, or ``None`` when this dialect has no
            native form for the part — the template then applies
            :meth:`non_native_intrinsic_call_to_p`.
        """
        return None

    @classmethod
    def ir_intrinsic_result_to_p(
        cls,
        ir_tool_result: ToolResultPart,
        *,
        leg: IntrinsicLeg = "history",
        **kwargs: Any,
    ) -> Any | None:
        """IR intrinsic ToolResultPart → this dialect's native server result.

        Args:
            ir_tool_result: IR tool result part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on.
            **kwargs: The caller's keyword arguments.

        Returns:
            The native server result, or ``None`` when this dialect has no
            native form for it — the template then applies
            :meth:`non_native_intrinsic_result_to_p`.
        """
        return None

    @classmethod
    def p_intrinsic_call_to_ir(
        cls, provider_part: Any, **kwargs: Any
    ) -> ToolCallPart | None:
        """Provider native server call → IR intrinsic ToolCallPart.

        The read counterpart of :meth:`ir_intrinsic_call_to_p`: a reader asks
        this *first*, before the client shapes, so a provider-executed call is
        never read as a client call that merely shares its name.  A dialect
        whose wire has no server call parts of its own leaves it at ``None``.

        Args:
            provider_part: The provider's content part or item.
            **kwargs: Extra arguments.

        Returns:
            IR ToolCallPart, or ``None`` when the part is not a native server
            call — the reader then takes its ordinary path.
        """
        return None

    @classmethod
    def p_intrinsic_result_to_ir(
        cls, provider_part: Any, **kwargs: Any
    ) -> ToolResultPart | None:
        """Provider native server result → IR intrinsic ToolResultPart.

        The read counterpart of :meth:`ir_intrinsic_result_to_p`, asked the way
        :meth:`p_intrinsic_call_to_ir` is.

        Args:
            provider_part: The provider's content part or item.
            **kwargs: Extra arguments.

        Returns:
            IR ToolResultPart, or ``None`` when the part is not a native server
            result.
        """
        return None

    @classmethod
    def dropped_intrinsic_call(
        cls, ir_tool_call: ToolCallPart, *, leg: IntrinsicLeg, **kwargs: Any
    ) -> Any:
        """Drop an intrinsic call the target cannot render on the response leg.

        A response is a call the *client* did not run.  Handing it to the client
        as an ordinary client call would tell the client to execute something it
        never asked for, so the response and stream legs never degrade — what
        they cannot render is dropped and reported.  (History is the opposite:
        it is context the client already saw, so there the call degrades to a
        plain function pair.  See :meth:`non_native_intrinsic_call_to_p`.)

        Args:
            ir_tool_call: IR tool call part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on (always ``"response"`` here).
            **kwargs: The caller's keyword arguments; a ``warnings`` list is
                appended a message naming the drop.

        Returns:
            ``{}`` — nothing is written.
        """
        _report_dropped_intrinsic(ir_tool_call, kwargs)
        return {}

    @classmethod
    def dropped_intrinsic_result(
        cls, ir_tool_result: ToolResultPart, *, leg: IntrinsicLeg, **kwargs: Any
    ) -> Any:
        """Drop an intrinsic result the target cannot render on the response leg.

        The result half of :meth:`dropped_intrinsic_call`, with the same reason.

        Args:
            ir_tool_result: IR tool result part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on (always ``"response"`` here).
            **kwargs: The caller's keyword arguments.

        Returns:
            ``{}`` — nothing is written.
        """
        _report_dropped_intrinsic(ir_tool_result, kwargs)
        return {}

    @classmethod
    def non_native_intrinsic_call_to_p(
        cls, ir_tool_call: ToolCallPart, *, leg: IntrinsicLeg, **kwargs: Any
    ) -> Any:
        """Write an intrinsic call this dialect does not render natively.

        Reached on the **history** leg only: the template routes a
        response-leg part to :meth:`dropped_intrinsic_call`.  The default is the
        parent path — treat it as an ordinary client call — which is what a
        dialect with no intrinsic vocabulary does.  A dialect that drops such a
        call instead (messages, google_generate, google_interactions)
        overrides this to return its own empty outcome.

        Args:
            ir_tool_call: IR tool call part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on.
            **kwargs: The caller's keyword arguments.

        Returns:
            The dialect's non-native write: a client call, or an empty drop.
        """
        return cls.ir_client_call_to_p(ir_tool_call, **kwargs)

    @classmethod
    def non_native_intrinsic_result_to_p(
        cls, ir_tool_result: ToolResultPart, *, leg: IntrinsicLeg, **kwargs: Any
    ) -> Any:
        """Write an intrinsic result this dialect does not render natively.

        The default is the parent path, mirroring
        :meth:`non_native_intrinsic_call_to_p`.

        Args:
            ir_tool_result: IR tool result part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on.
            **kwargs: The caller's keyword arguments.

        Returns:
            The dialect's non-native write: a client result, or an empty drop.
        """
        return cls.ir_client_result_to_p(ir_tool_result, **kwargs)

    # ==================== Intrinsic-aware write templates ====================

    @classmethod
    def ir_tool_call_to_p(cls, ir_tool_call: ToolCallPart, **kwargs: Any) -> Any:
        """IR ToolCallPart → provider tool call.

        The template every caller uses: an intrinsic call asks the native hook
        first and otherwise takes the converter's non-native outcome, so the
        intrinsic branch lives here rather than in each dialect's writer.

        Args:
            ir_tool_call: IR tool call part.
            **kwargs: Extra arguments.

        Returns:
            Provider tool call.
        """
        if not is_intrinsic_part(ir_tool_call):
            return cls.ir_client_call_to_p(ir_tool_call, **kwargs)

        # A dialect's two writers have historically defaulted to different
        # legs (open_responses: a call without ``for_history`` answered for the
        # response, a result for the replay), so each template passes its own.
        leg = cls.intrinsic_leg(kwargs, "response")
        forwarded = {key: value for key, value in kwargs.items() if key != "leg"}
        native = cls.ir_intrinsic_call_to_p(ir_tool_call, leg=leg, **forwarded)
        if native is not None:
            return native
        if leg == "response":
            return cls.dropped_intrinsic_call(ir_tool_call, leg=leg, **forwarded)
        return cls.non_native_intrinsic_call_to_p(ir_tool_call, leg=leg, **forwarded)

    @classmethod
    def ir_tool_result_to_p(cls, ir_tool_result: ToolResultPart, **kwargs: Any) -> Any:
        """IR ToolResultPart → provider tool result.

        The result half of :meth:`ir_tool_call_to_p`, with the same dispatch.

        Args:
            ir_tool_result: IR tool result part.
            **kwargs: Extra arguments.

        Returns:
            Provider tool result.
        """
        if not is_intrinsic_part(ir_tool_result):
            return cls.ir_client_result_to_p(ir_tool_result, **kwargs)

        leg = cls.intrinsic_leg(kwargs, "history")
        forwarded = {key: value for key, value in kwargs.items() if key != "leg"}
        native = cls.ir_intrinsic_result_to_p(ir_tool_result, leg=leg, **forwarded)
        if native is not None:
            return native
        if leg == "response":
            return cls.dropped_intrinsic_result(ir_tool_result, leg=leg, **forwarded)
        return cls.non_native_intrinsic_result_to_p(
            ir_tool_result, leg=leg, **forwarded
        )

    @classmethod
    def ir_client_call_to_p(cls, ir_tool_call: ToolCallPart, **kwargs: Any) -> Any:
        """IR ToolCallPart → provider tool call, ordinary (client) path.

        Implemented by every dialect; the intrinsic hooks and the template
        above are the only intrinsics-aware part of the write path.

        Args:
            ir_tool_call: IR tool call part.

        Returns:
            Provider tool call.
        """
        raise NotImplementedError(f"{cls.__name__} must implement ir_client_call_to_p")

    @classmethod
    def ir_client_result_to_p(
        cls, ir_tool_result: ToolResultPart, **kwargs: Any
    ) -> Any:
        """IR ToolResultPart → provider tool result, ordinary (client) path.

        Args:
            ir_tool_result: IR tool result part.

        Returns:
            Provider tool result.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement ir_client_result_to_p"
        )

    # ==================== Tool Definition ====================

    @staticmethod
    @abstractmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> Any:
        """IR ToolDefinition → Provider Tool Definition.

        Args:
            ir_tool: IR tool definition.
            **kwargs: Extra arguments.

        Returns:
            Provider tool definition.
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_definition_to_ir(
        provider_tool: Any, **kwargs: Any
    ) -> ToolDefinition | list[ToolDefinition] | None:
        """Provider Tool Definition → IR ToolDefinition.

        Args:
            provider_tool: Provider tool definition.
            **kwargs: Extra arguments.

        Returns:
            IR tool definition(s), or None if the entry cannot be converted.
        """
        pass

    # ==================== Tool Choice ====================

    @staticmethod
    @abstractmethod
    def ir_tool_choice_to_p(ir_tool_choice: ToolChoice, **kwargs: Any) -> Any:
        """IR ToolChoice → Provider Tool Choice Config.

        Args:
            ir_tool_choice: IR tool choice.
            **kwargs: Extra arguments.

        Returns:
            Provider tool choice config.
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        """Provider Tool Choice Config → IR ToolChoice.

        Args:
            provider_tool_choice: Provider tool choice config.
            **kwargs: Extra arguments.

        Returns:
            IR tool choice.
        """
        pass

    # ==================== Tool Call ====================

    @staticmethod
    @abstractmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        """Provider Tool Call → IR ToolCallPart.

        Args:
            provider_tool_call: Provider tool call.
            **kwargs: Extra arguments.

        Returns:
            IR tool call part.
        """
        pass

    # ==================== Tool Result ====================

    @staticmethod
    @abstractmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        """Provider Tool Result → IR ToolResultPart.

        Args:
            provider_tool_result: Provider tool result.
            **kwargs: Extra arguments.

        Returns:
            IR tool result part.
        """
        pass

    # ==================== Tool Config ====================

    @staticmethod
    @abstractmethod
    def ir_tool_config_to_p(ir_tool_config: ToolCallConfig, **kwargs: Any) -> Any:
        """IR ToolCallConfig → Provider Tool Call Config.

        Args:
            ir_tool_config: IR tool call config.
            **kwargs: Extra arguments.

        Returns:
            Provider tool call config.
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        """Provider Tool Call Config → IR ToolCallConfig.

        Args:
            provider_tool_config: Provider tool call config.
            **kwargs: Extra arguments.

        Returns:
            IR tool call config.
        """
        pass
