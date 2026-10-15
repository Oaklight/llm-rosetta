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
from .intrinsic import is_intrinsic_part

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
    def intrinsic_leg(cls, kwargs: Mapping[str, Any]) -> IntrinsicLeg:
        """Which leg an intrinsic part is being written on.

        A caller may state it outright with ``leg=``; otherwise the converter
        maps its own write-path flag through
        :meth:`intrinsic_leg_from_flags`.

        Args:
            kwargs: The keyword arguments the write template was called with.

        Returns:
            ``"history"`` or ``"response"``.
        """
        leg = kwargs.get("leg")
        if leg in _INTRINSIC_LEGS:
            return cast(IntrinsicLeg, leg)
        return cls.intrinsic_leg_from_flags(kwargs)

    @classmethod
    def intrinsic_leg_from_flags(cls, kwargs: Mapping[str, Any]) -> IntrinsicLeg:
        """Map this converter's own leg flag onto :data:`IntrinsicLeg`.

        The default is the history leg: a body is written from IR most often
        to replay a turn.  A converter whose write path already distinguishes
        the two maps its flag here (open_responses: ``for_history``).

        Args:
            kwargs: The keyword arguments the write template was called with.

        Returns:
            ``"history"`` or ``"response"``.
        """
        return "history"

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
    def non_native_intrinsic_call_to_p(
        cls, ir_tool_call: ToolCallPart, *, leg: IntrinsicLeg, **kwargs: Any
    ) -> Any:
        """Write an intrinsic call this dialect does not render natively.

        The default is the parent path: treat it as an ordinary client call,
        which is what a dialect with no intrinsic vocabulary does.  A dialect
        that drops such a call instead (google_generate, google_interactions)
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

        leg = cls.intrinsic_leg(kwargs)
        forwarded = {key: value for key, value in kwargs.items() if key != "leg"}
        native = cls.ir_intrinsic_call_to_p(ir_tool_call, leg=leg, **forwarded)
        if native is not None:
            return native
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

        leg = cls.intrinsic_leg(kwargs)
        forwarded = {key: value for key, value in kwargs.items() if key != "leg"}
        native = cls.ir_intrinsic_result_to_p(ir_tool_result, leg=leg, **forwarded)
        if native is not None:
            return native
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
