"""Names that were renamed, and what each one became.

This table is an **error table, not an alias table**.  A name listed here is
a spelling an earlier release accepted for a provider type, a shim or a model
type.  Today :mod:`llm_rosetta.provider_names` and
:data:`llm_rosetta.formats.LEGACY_MODEL_TYPES` still translate those spellings
silently.  When an alias table is removed, the name must not fall through to
"unknown type" or "no base_url configured": it must fail with a message that
names its replacement.  This module supplies that message.

The module has no imports from the rest of the package, so the core library,
the gateway and the alias tables can all read it without a cycle.  Core owns
it (rather than the gateway) because the library entry points
(``convert()``, ``ConversionPipeline``, ``get_converter_for_provider``)
receive retired names too, and core must not import the gateway.

The switch
----------

:data:`ALIASES_ENABLED` is the one switch.  While it is ``True`` (the
default) every alias path behaves as before and nothing here raises.  With it
``False`` the alias tables behave as if they were empty, and every boundary
that used to warn about a retired name raises :class:`ValueError` carrying
:func:`retired_name_message` instead.  Removing the alias tables means
deleting the alias code paths and the flag; the table stays.

Retiring a name
---------------

Add one :class:`RetiredName` row.  Never edit a row's meaning and never reuse
a retired name for something new: a row says what the name *used to* mean.
A test keeps every key of the live alias tables covered and rejects a chain
(a replacement that is itself retired).
"""

from __future__ import annotations

from dataclasses import dataclass

#: The one switch.  ``True``: aliases resolve (today's behaviour).  ``False``:
#: a retired name raises a ValueError naming its replacement.
ALIASES_ENABLED: bool = True

#: Release that renamed these names, for the message.
RENAMED_IN = "0.14"

PROVIDER_TYPE = "provider_type"
SHIM = "shim"
MODEL_TYPE = "model_type"

KINDS: tuple[str, ...] = (PROVIDER_TYPE, SHIM, MODEL_TYPE)


@dataclass(frozen=True)
class RetiredName:
    """One retired spelling.

    Attributes:
        name: The retired spelling.
        replacement: What it became.  Never itself a retired name.
        kind: ``provider_type`` (a ``type:`` / format / provider argument),
            ``shim`` (a shim name) or ``model_type``.  The same spelling can
            be retired under two kinds with different replacements
            (``google``).
        preset: For a shim that moved to a standalone package, the
            ``shims.presets`` key that makes the replacement loadable.
    """

    name: str
    replacement: str
    kind: str
    preset: str | None = None


RETIRED_NAMES: tuple[RetiredName, ...] = (
    # -- provider types / formats (provider_names.LEGACY_PROVIDER_ALIASES) --
    RetiredName(name="google", replacement="google_generate", kind=PROVIDER_TYPE),
    RetiredName(name="google-genai", replacement="google_generate", kind=PROVIDER_TYPE),
    RetiredName(name="openai_chat", replacement="chat_completions", kind=PROVIDER_TYPE),
    RetiredName(
        name="openai_responses",
        replacement="openai--open_responses",
        kind=PROVIDER_TYPE,
    ),
    RetiredName(name="anthropic", replacement="messages", kind=PROVIDER_TYPE),
    RetiredName(name="openai-chat", replacement="chat_completions", kind=PROVIDER_TYPE),
    RetiredName(
        name="openai-responses",
        replacement="openai--open_responses",
        kind=PROVIDER_TYPE,
    ),
    RetiredName(
        name="open-responses", replacement="open_responses", kind=PROVIDER_TYPE
    ),
    RetiredName(
        name="google-interactions",
        replacement="google_interactions",
        kind=PROVIDER_TYPE,
    ),
    # -- shim names (provider_names.LEGACY_SHIM_ALIASES) --------------------
    RetiredName(name="openai", replacement="openai--chat_completions", kind=SHIM),
    RetiredName(name="google", replacement="google--google_generate", kind=SHIM),
    RetiredName(
        name="google_generate", replacement="google--google_generate", kind=SHIM
    ),
    RetiredName(
        name="google_interactions",
        replacement="google--google_interactions",
        kind=SHIM,
    ),
    RetiredName(name="anthropic", replacement="anthropic--messages", kind=SHIM),
    RetiredName(
        name="argo--openai_chat",
        replacement="argo--chat_completions",
        kind=SHIM,
        preset="argo",
    ),
    RetiredName(
        name="argo--openai_responses",
        replacement="argo--open_responses",
        kind=SHIM,
        preset="argo",
    ),
    RetiredName(
        name="minimax--openai_chat", replacement="minimax--chat_completions", kind=SHIM
    ),
    RetiredName(
        name="openrouter--openai_chat",
        replacement="openrouter--chat_completions",
        kind=SHIM,
    ),
    RetiredName(
        name="volcengine--openai_chat",
        replacement="volcengine--chat_completions",
        kind=SHIM,
    ),
    RetiredName(
        name="volcengine--openai_responses",
        replacement="volcengine--open_responses",
        kind=SHIM,
    ),
    RetiredName(
        name="argo--anthropic",
        replacement="argo--messages",
        kind=SHIM,
        preset="argo",
    ),
    RetiredName(name="minimax--anthropic", replacement="minimax--messages", kind=SHIM),
    RetiredName(
        name="openrouter--anthropic", replacement="openrouter--messages", kind=SHIM
    ),
    RetiredName(
        name="alcf--metis",
        replacement="alcf_metis--chat_completions",
        kind=SHIM,
        preset="alcf",
    ),
    RetiredName(
        name="alcf--minerva",
        replacement="alcf_minerva--chat_completions",
        kind=SHIM,
        preset="alcf",
    ),
    RetiredName(
        name="alcf--sophia",
        replacement="alcf_sophia--chat_completions",
        kind=SHIM,
        preset="alcf",
    ),
    RetiredName(
        name="asksage--anthropic",
        replacement="asksage--messages",
        kind=SHIM,
        preset="asksage",
    ),
    RetiredName(
        name="asksage--openai_chat",
        replacement="asksage--chat_completions",
        kind=SHIM,
        preset="asksage",
    ),
    RetiredName(
        name="asksage--openai_responses",
        replacement="asksage--open_responses",
        kind=SHIM,
        preset="asksage",
    ),
    # -- model types (formats.LEGACY_MODEL_TYPES) ---------------------------
    RetiredName(name="embedding", replacement="embeddings", kind=MODEL_TYPE),
    RetiredName(name="decision", replacement="decisions", kind=MODEL_TYPE),
)

_BY_KIND: dict[tuple[str, str], RetiredName] = {
    (row.kind, row.name): row for row in RETIRED_NAMES
}


def lookup(name: str, kind: str) -> RetiredName | None:
    """Return the row for *name* under *kind*, or ``None`` if it is not retired.

    Args:
        name: A name as supplied by the caller.
        kind: One of :data:`KINDS`.

    Returns:
        The matching row, or ``None`` (canonical and unknown names alike).
    """
    return _BY_KIND.get((kind, name))


def replacement_for(name: str, kind: str) -> str | None:
    """Return what *name* became under *kind*, or ``None``."""
    row = lookup(name, kind)
    return row.replacement if row is not None else None


def retired_name_message(name: str, kind: str) -> str | None:
    """Return the sentence that says what a retired *name* became.

    Args:
        name: A name as supplied by the caller.
        kind: One of :data:`KINDS`.

    Returns:
        The message, or ``None`` when *name* is not retired under *kind*, so
        a typo never receives a rename hint.
    """
    row = lookup(name, kind)
    if row is None:
        return None
    message = (
        f"{row.name!r} was renamed to {row.replacement!r} in {RENAMED_IN} "
        "and the old spelling is no longer accepted"
    )
    if row.preset is not None:
        message += (
            f" (the shim is provided by the standalone package {row.preset!r}: "
            f"declare shims.presets: ['{row.preset}'] and use {row.replacement!r})"
        )
    else:
        message += f" (use {row.replacement!r})"
    return message


def check_not_retired(name: object, *kinds: str) -> None:
    """Raise if aliases are off and *name* is retired under any of *kinds*.

    A no-op while :data:`ALIASES_ENABLED` is ``True``, and for any name that
    is not retired.

    Args:
        name: The name to check; a non-string is ignored.
        *kinds: The kinds to try, in order; the first match wins.

    Raises:
        ValueError: With :func:`retired_name_message`.
    """
    if ALIASES_ENABLED or not isinstance(name, str):
        return
    for kind in kinds:
        message = retired_name_message(name, kind)
        if message is not None:
            raise ValueError(message)


__all__ = [
    "ALIASES_ENABLED",
    "KINDS",
    "MODEL_TYPE",
    "PROVIDER_TYPE",
    "RETIRED_NAMES",
    "SHIM",
    "RetiredName",
    "check_not_retired",
    "lookup",
    "replacement_for",
    "retired_name_message",
]
