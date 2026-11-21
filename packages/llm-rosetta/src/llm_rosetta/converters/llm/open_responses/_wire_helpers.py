"""Wire-level helpers shared by the Open Responses converter and its stream mixins."""

from typing import Any

from ...base.context import ConversionContext
from ....ir.primitives.finish_reason import reason_for_completed_turn
from ._constants import (
    RESPONSES_INCOMPLETE_REASON_TO_IR,
    RESPONSES_STATUS_TO_REASON,
    ResponsesEventType,
)


def _is_incomplete(response: dict[str, Any], terminal_event: str | None) -> bool:
    """Whether the turn stopped short.

    A ``response.incomplete`` event says so whatever its body's ``status``
    reads.  Otherwise ``status`` decides; when it is absent, an
    ``incomplete_details`` object says the turn is incomplete.
    """
    if terminal_event == ResponsesEventType.RESPONSE_INCOMPLETE:
        return True
    status = response.get("status")
    if status is None:
        return response.get("incomplete_details") is not None
    return status == "incomplete"


def _response_status_reason(
    response: dict[str, Any], terminal_event: str | None = None
) -> str:
    """Map the response status to the IR reason; the parts decide the rest.

    An incomplete response stopped short (see :func:`_is_incomplete`).
    ``incomplete_details.reason`` names why; when it is missing, unknown or
    not a string the turn is a ``length`` stop — the only documented cause of
    ``incomplete`` — and never ``stop``, which the parts would upgrade to
    ``tool_calls`` and send the client to answer a call the model never
    finished (#1612).

    Args:
        response: The response body.
        terminal_event: The stream event that carried it, when there is one.
    """
    if _is_incomplete(response, terminal_event):
        details = response.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        if not isinstance(reason, str):
            reason = ""
        return RESPONSES_INCOMPLETE_REASON_TO_IR.get(reason, "length")
    status = response.get("status") or "completed"
    return RESPONSES_STATUS_TO_REASON.get(status, "stop")


def _stream_finish_reason(
    response: dict[str, Any], terminal_event: str | None = None
) -> str:
    """Decide the streamed terminal reason from the body the frame carries.

    A ``response.completed`` event usually carries the output list, so this
    reader has the parts the JSON reader has.  That list is optional, and it
    is not the last word: the frame's list decides when it names the call,
    and the opened call decides when the list says nothing about it.  The
    base stream finalizer applies the second half from the calls the stream
    opened, so a frame that omits or contradicts them cannot finish a tool
    turn as a normal stop.
    """
    status = response.get("status", "completed")
    calls_the_client = status == "completed" and any(
        isinstance(item, dict)
        and item.get("type") in {"function_call", "custom_tool_call"}
        for item in response.get("output", [])
    )
    return reason_for_completed_turn(
        _response_status_reason(response, terminal_event), calls_the_client
    )


def _response_failure_metadata(
    response: dict[str, Any], context: ConversionContext | None
) -> dict[str, Any]:
    """Keep the response in context and carry only its error through the IR."""
    if context is not None:
        context.metadata["responses_response"] = dict(response)
    error = response.get("error")
    return {"responses_error": dict(error)} if isinstance(error, dict) else {}


def _capture_item_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Extract per-output-item metadata for lossless round-trip."""
    meta: dict[str, Any] = {}
    if "id" in item:
        meta["id"] = item["id"]
    if "status" in item:
        meta["status"] = item["status"]
    item_content = item.get("content", [])
    if isinstance(item_content, list):
        parts_meta: list[dict[str, Any]] = []
        for cp in item_content:
            if not isinstance(cp, dict):
                continue
            pm: dict[str, Any] = {}
            if "annotations" in cp:
                pm["annotations"] = cp["annotations"]
            if "logprobs" in cp:
                pm["logprobs"] = cp["logprobs"]
            parts_meta.append(pm)
        if parts_meta:
            meta["content_meta"] = parts_meta
    return meta


_PASSTHROUGH_REQUEST_FIELDS = (
    "store",
    "background",
    "metadata",
    "user",
    "previous_response_id",
    "safety_identifier",
    "prompt",
    # The spec's ``include`` (which optional payloads the response may carry)
    # and OpenAI's ``conversation`` / ``context_management`` had no reader at
    # all, so a same-format round trip lost them.  They ride here like the rest
    # of the set, which means the owner stamp also keeps them off a body of
    # another format.
    "include",
    "conversation",
    "context_management",
)

#: Request-bound fields an OpenAI *profile* may accept beyond the neutral
#: spec.  The reader records them whenever the client sends them, so a writer
#: of another format can name them in its loss report; the Responses writer
#: emits one only when the **target** declares it (``request_fields`` in the
#: shim's ``provider.yaml``, threaded through ``context.options``), and
#: reports it otherwise.  Recording is not sending: a neutral upstream never
#: receives a field its spec does not define.
#:
#: Measured against the live Responses API: ``moderation`` is accepted,
#: ``prompt_cache_options`` is known but model-gated, ``access_programs`` must
#: be an object, and ``logit_bias`` is not a Responses parameter at all — the
#: OpenAI shim declares it anyway, because forwarding the client's own request
#: reproduces the vendor's answer instead of sampling a different one.
_PROFILE_REQUEST_FIELDS: tuple[str, ...] = (
    "logit_bias",
    "prompt_cache_options",
    "moderation",
    "access_programs",
)


def _forward_passthrough_request_fields(
    provider_request: dict[str, Any], ir_request: dict[str, Any]
) -> None:
    """Forward Responses-specific request fields via ``provider_extensions``.

    These fields have no IR equivalent but must reach the upstream in
    same-format round-trips so the client's values are honoured.  The
    profile-only ones are recorded here too — recording is what lets a writer
    of another format report them as dropped; whether the target may receive
    one is decided by the writer, against the target's declaration.
    """
    for field in (*_PASSTHROUGH_REQUEST_FIELDS, *_PROFILE_REQUEST_FIELDS):
        val = provider_request.get(field)
        if val is not None:
            ir_request.setdefault("provider_extensions", {})[field] = val
    text_obj = provider_request.get("text")
    if isinstance(text_obj, dict) and "verbosity" in text_obj:
        ir_request.setdefault("provider_extensions", {})["_text_verbosity"] = text_obj[
            "verbosity"
        ]


def _declared_tool_types(tools: Any) -> dict[str, str]:
    """Map each declared tool's name to its IR type.

    Used to spell ``allowed_tools`` entries: the wire entry's ``type`` has to
    agree with the declaration, or the API reports the tool as not found.

    Args:
        tools: The IR request's tool list, if it has one.

    Returns:
        Name → IR type, empty when there are no tools to read.
    """
    if not isinstance(tools, list):
        return {}
    types: dict[str, str] = {}
    for tool in tools:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            types[tool["name"]] = str(tool.get("type") or "function")
    return types


def _insert_after_call(
    output: list[dict[str, Any]], call_id: str, item: dict[str, Any]
) -> None:
    """Insert *item* right after the output item that answers *call_id*.

    A result follows its call in a body; appending it at the end would put it
    after the items that came later in the turn.
    """
    for index, existing in enumerate(output):
        if existing.get("call_id") == call_id:
            output.insert(index + 1, item)
            return
    output.append(item)
