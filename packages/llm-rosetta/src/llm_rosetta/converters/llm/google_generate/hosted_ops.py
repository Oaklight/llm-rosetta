"""
LLM-Rosetta - Google GenAI hosted-tool classification

Request-side tool vocabulary and hosted-call classification for the converter.
"""

from typing import TYPE_CHECKING, Any


from ...base.context import (
    ConversionContext,
    StreamContext,
)
from ...base.tools.intrinsic import (
    intrinsic_kind_of,
)
from ...base.helpers.opaque import (
    google_thought_signature,
)

if TYPE_CHECKING:
    from .tool_ops import GoogleGenerateToolOps


class GoogleGenerateHostedOps:
    """Mixin: hosted (server) tool vocabulary and call classification."""

    if TYPE_CHECKING:
        _CONVERTER_TAG: str
        tool_ops: GoogleGenerateToolOps

    @staticmethod
    def _record_request_tool_vocabulary(tools: Any, context: ConversionContext) -> None:
        """Record which server tools and client functions this request enabled.

        Google delivers a hosted tool's call and a client function's call
        through the same ``functionCall`` part, so the response reader needs to
        know what the request asked for.  The request leg is the only leg that
        knows, so it records the two sets on the context for the response leg
        (non-streaming: the same context; streaming: bridged by the pipeline).

        Args:
            tools: The IR request's ``tools`` list (may be ``None``).
            context: The request leg's conversion context.
        """
        kinds: set[str] = set()
        names: set[str] = set()
        for tool in tools or ():
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "intrinsic":
                kind = intrinsic_kind_of(tool)
                if kind:
                    kinds.add(kind)
                continue
            name = tool.get("name")
            if isinstance(name, str) and name:
                names.add(name)
        context.record_request_tool_vocabulary(
            intrinsic_kinds=kinds, function_names=names
        )

    def _classify_hosted_history_calls(self, messages: list[Any], tools: Any) -> None:
        """Promote replayed ``functionCall`` parts to hosted calls.

        The response readers consult the vocabulary the request leg recorded;
        a request *body* being read has no such record, but it declares the
        same tools.  A hosted key enables the kind; a client function declared
        under the same name wins, as it does on the response leg.

        Args:
            messages: IR messages parsed from the body's ``contents``.
            tools: The wire ``tools`` list from the same body.
        """
        enabled_kinds, function_names = self.tool_ops.wire_tool_vocabulary(tools)
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "tool_call":
                    continue
                if part.get("tool_type") == "intrinsic":
                    continue
                kind = self.tool_ops.p_function_call_intrinsic_kind(
                    part.get("tool_name", ""),
                    enabled_kinds=enabled_kinds,
                    client_function_names=function_names,
                )
                if kind:
                    self._promote_to_hosted(part, kind)

    def _hosted_call_kind(
        self,
        tool_name: str,
        context: ConversionContext | StreamContext | None,
    ) -> str:
        """Classify a ``functionCall`` name against what the request enabled.

        With no context, or a context no request leg recorded on (a response
        converted on its own), the name table is the only evidence available;
        :meth:`GoogleGenerateToolOps.p_function_call_intrinsic_kind` states
        that fallback and its limitation.

        Args:
            tool_name: The ``functionCall`` name as it appears on the wire.
            context: The conversion context, or ``None``.

        Returns:
            The canonical intrinsic kind, or ``""`` for a client call.
        """
        if context is None:
            return self.tool_ops.p_function_call_intrinsic_kind(
                tool_name, enabled_kinds=None
            )
        kinds, client_names = context.request_tool_vocabulary()
        return self.tool_ops.p_function_call_intrinsic_kind(
            tool_name, enabled_kinds=kinds, client_function_names=client_names
        )

    def _classify_hosted_calls(
        self,
        message: dict[str, Any],
        context: ConversionContext | StreamContext | None,
    ) -> None:
        """Promote ``functionCall`` parts the request enabled to hosted calls.

        The message parser cannot tell a hosted call from a client function:
        both arrive as ``functionCall``.  The request recorded what it enabled
        (or did not), so a part matching that record is rewritten in place to
        ``tool_type="intrinsic"`` with its canonical kind.  The finish reason
        then reads the turn as complete rather than as owing the client a
        result.

        The native part it arrived as is recorded under
        ``provider_metadata["_native"]``, so a Google target can write the call
        back as the same ``functionCall`` (see
        :meth:`GoogleGenerateToolOps.native_function_call`) while every other
        target still treats it as a server call it has no form for.

        Args:
            message: The IR message parsed from one candidate.
            context: The conversion context the request leg recorded on.
        """
        content = message.get("content")
        if not isinstance(content, list):
            return
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "tool_call":
                continue
            if part.get("tool_type") == "intrinsic":
                continue
            kind = self._hosted_call_kind(part.get("tool_name", ""), context)
            if kind:
                self._promote_to_hosted(part, kind)

    def _promote_to_hosted(self, part: dict[str, Any], kind: str) -> None:
        """Mark *part* as a hosted call and keep the form it arrived as.

        Both readers promote the same way: the native ``functionCall`` is
        recorded under ``provider_metadata["_native"]`` so a Google target can
        write the call back as it came (see
        :meth:`GoogleGenerateToolOps.native_function_call`), while every other
        target still treats it as a server call it has no form for.

        Args:
            part: The IR ``tool_call`` part to promote, rewritten in place.
            kind: The canonical intrinsic kind the call denotes.
        """
        part["tool_type"] = "intrinsic"
        part["intrinsic_kind"] = kind
        native_call: dict[str, Any] = {
            "name": part.get("tool_name", ""),
            "args": part.get("tool_input", {}),
        }
        call_id = part.get("tool_call_id")
        if isinstance(call_id, str) and call_id:
            native_call["id"] = call_id
        metadata = part.get("provider_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            part["provider_metadata"] = metadata
        native_part: dict[str, Any] = {"functionCall": native_call}
        signature = google_thought_signature({"provider_metadata": metadata})
        if signature:
            native_part["thoughtSignature"] = signature
        metadata.setdefault(
            "_native", {"base": self._CONVERTER_TAG, "part": native_part}
        )
