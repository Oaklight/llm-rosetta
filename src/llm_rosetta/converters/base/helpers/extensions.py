"""Shared handling for ``IRRequest.provider_extensions``.

Extensions are verbatim provider parameters: the reader of one format records
the fields it could not map into the IR, so the writer of *that same* format can
put them back.  They are not portable, and merging one into another format's
body puts a foreign parameter on the wire.  Measured against live upstreams, a
Messages body carrying ``background`` is answered with ``400
invalid_request_error: background: Extra inputs are not permitted``, and a Chat
Completions body carrying ``previous_response_id`` with ``400
unknown_parameter``.

So the reader stamps the format it recorded the set for (``_owner``), and a
writer merges a set only when the stamp is its own; the rest are reported
through the conversion warnings rather than sent.  An unstamped set -- one a
caller assembled by hand, or carried over from a version that did not stamp --
is left to the writer's judgement, keeping the permissive behaviour for IR that
no reader produced.

Keys beginning with ``_`` are converter-internal markers: they carry state
between a converter's provider→IR and IR→provider halves (e.g.
``_text_verbosity``) and must never be merged into a provider request body.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

_OWNER_KEY = "_owner"

#: Extensions whose loss changes the answer rather than trimming a knob: the
#: request asks an upstream to continue a conversation whose prefix only that
#: upstream holds.
_CONTINUATION_KEYS = frozenset({"previous_response_id", "conversation"})


def own_extensions(ir_request: Any, dialect: str) -> None:
    """Stamp *dialect* as the format this request's extensions were read for.

    Does nothing when the request carries no extensions, or when the set is
    already stamped (the reader that produced it knows better than a later
    stage).
    """
    extensions = ir_request.get("provider_extensions")
    if extensions and _OWNER_KEY not in extensions:
        extensions[_OWNER_KEY] = dialect


def _wire_keys(extensions: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in extensions.items() if not k.startswith("_")}


def wire_extensions(ir_request: Any, dialect: str | None = None) -> dict[str, Any]:
    """Return the provider extensions that may be merged into a request body.

    Drops ``_``-prefixed keys (converter-internal), and drops the whole set when
    it was recorded for another dialect than *dialect*.  Passing ``dialect`` as
    ``None`` keeps the permissive reading of an unstamped set.

    Args:
        ir_request: The IR request.
        dialect: The format of the body being built, usually the converter's
            ``_CONVERTER_TAG``.

    Returns:
        The extensions to merge into the body.
    """
    extensions = ir_request.get("provider_extensions")
    if not extensions:
        return {}
    owner = extensions.get(_OWNER_KEY)
    if dialect is not None and owner is not None and owner != dialect:
        return {}
    return _wire_keys(extensions)


def unowned_extensions(ir_request: Any, dialect: str) -> dict[str, Any]:
    """Return the extensions recorded for some *other* dialect.

    These are the fields the target cannot express; they are reported rather
    than sent, so a caller learns what the route could not carry.
    """
    extensions = ir_request.get("provider_extensions")
    if not extensions:
        return {}
    owner = extensions.get(_OWNER_KEY)
    if owner is None or owner == dialect:
        return {}
    return _wire_keys(extensions)


def extension_loss_warning(dropped: Collection[str], dialect: str) -> str:
    """Describe a set of provider-bound fields the target cannot express.

    Args:
        dropped: The extension keys that had to stay behind.
        dialect: The format that could not express them.

    Returns:
        One warning line for the conversion result.
    """
    fields = ", ".join(sorted(dropped))
    message = f"provider-bound request fields dropped for {dialect}: {fields}"
    continuation = sorted(_CONTINUATION_KEYS.intersection(dropped))
    if continuation:
        message += (
            f"; {', '.join(continuation)} refers to a stored conversation that "
            "only the source upstream holds, so the target answers from this "
            "turn alone"
        )
    return message
