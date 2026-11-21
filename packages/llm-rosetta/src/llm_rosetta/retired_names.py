"""Names that were renamed, and what each one became.

This table is an **error table, not an alias table**.  A name listed here is
a spelling an earlier release accepted for a provider type, a shim or a model
type.  No spelling is translated any more (the alias tables are gone): a
name listed here must not fall through to "unknown type" or "no base_url
configured", so it fails with a message that names its replacement and ends
with :data:`MIGRATE_HINT`: the ``llm-rosetta-migrate`` command when the
gateway distribution (which ships it) is installed, else the migration guide.
This module supplies that message.

The module has no imports from the rest of the package, so the core library
and the gateway can both read it without a cycle.  Core owns it (rather than
the gateway) because the library entry points (``convert()``,
``ConversionPipeline``, ``get_converter_for_provider``) receive retired names
too, and core must not import the gateway.

Every row raises: there is no switch.  :func:`check_not_retired` is the one
call a boundary makes.

Retiring a name
---------------

Add one :class:`RetiredName` row.  Never edit a row's meaning and never reuse
a retired name for something new: a row says what the name *used to* mean.
A test rejects a chain (a replacement that is itself retired).

Removed import paths
--------------------

:data:`RETIRED_MODULES` is a sibling table: a module path that an earlier
release (or the 0.14.0 development line) let user code import, and what to
import instead.  It is data only.  Nothing here hooks the import system; the
reader is ``llm-rosetta-migrate --scan``, which reports an import of a listed
path only when that path does not resolve in the installed core, so a path
that still exists is never reported.  Add one :class:`RetiredModule` row when
a module path is removed or moved.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: Release that renamed these names, for the message.
RENAMED_IN = "0.14"

#: The page that lists every rename and removal of 0.14 with its replacement.
MIGRATION_GUIDE_URL = (
    "https://llm-rosetta.readthedocs.io/en/latest/guide/migrating-to-0.14/"
)


def _migrate_hint(find_spec: Callable[[str], Any] = importlib.util.find_spec) -> str:
    """Return the closing sentence of a retired-name error for this install.

    ``llm-rosetta-migrate`` ships with the gateway distribution, so it is
    named only when ``llm_rosetta_gateway`` is installed.  A core-only user
    (the SDK) gets the migration guide instead.  The check locates the
    package without importing it: core must never import the gateway
    (``tests/test_core_import_boundary.py``).

    Args:
        find_spec: The spec lookup; a test passes a stub.
    """
    try:
        installed = find_spec("llm_rosetta_gateway") is not None
    except (ImportError, ValueError):
        # A finder that refuses the name (a blocked or broken install) means
        # the command cannot run here either.
        installed = False
    if installed:
        return (
            "run 'llm-rosetta-migrate --config <path>' (add --write to apply), "
            f"see the migration guide ({MIGRATION_GUIDE_URL})"
        )
    return f"see the migration guide ({MIGRATION_GUIDE_URL})"


#: The sentence every retired-name error ends with.  With the gateway
#: installed it names ``llm-rosetta-migrate``, the command that rewrites a
#: config or a database; without it, only the migration guide, which explains
#: the full table of renames.  Fixed when this module is first imported.
MIGRATE_HINT = _migrate_hint()

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
    # -- provider types / formats -------------------------------------------
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
    # -- shim names ---------------------------------------------------------
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
    # -- model types ---------------------------------------------------------
    RetiredName(name="embedding", replacement="embeddings", kind=MODEL_TYPE),
    RetiredName(name="decision", replacement="decisions", kind=MODEL_TYPE),
)


@dataclass(frozen=True)
class RetiredModule:
    """One removed import path.

    Attributes:
        path: The dotted module path user code imported.  A submodule of it
            (``<path>.converter``) is covered by the same row.
        replacement: The module to import instead, or ``None`` when there is
            no public replacement.  Never itself a retired path.
        note: What to do instead, for a row with no replacement.
    """

    path: str
    replacement: str | None
    note: str = ""


_CONVERTERS = "llm_rosetta.converters"
_HELPERS = f"{_CONVERTERS}.base.helpers"
_TOOLS = f"{_CONVERTERS}.base.tools"

RETIRED_MODULES: tuple[RetiredModule, ...] = (
    # -- compat converter packages (#1640) -----------------------------------
    RetiredModule(f"{_CONVERTERS}.openai_chat", f"{_CONVERTERS}.llm.chat_completions"),
    RetiredModule(
        f"{_CONVERTERS}.openai_responses", f"{_CONVERTERS}.llm.open_responses"
    ),
    RetiredModule(f"{_CONVERTERS}.anthropic", f"{_CONVERTERS}.llm.messages"),
    RetiredModule(f"{_CONVERTERS}.google_genai", f"{_CONVERTERS}.llm.google_generate"),
    RetiredModule(
        f"{_CONVERTERS}.google_generate", f"{_CONVERTERS}.llm.google_generate"
    ),
    RetiredModule(
        f"{_CONVERTERS}.google_interactions",
        f"{_CONVERTERS}.llm.google_interactions",
    ),
    RetiredModule(f"{_CONVERTERS}.embedding", f"{_CONVERTERS}.embeddings"),
    # -- helper modules and facades (#1647, #1648) ---------------------------
    RetiredModule(
        f"{_HELPERS}.cache_breakpoints", "llm_rosetta.ir.primitives.cache_breakpoints"
    ),
    RetiredModule(f"{_HELPERS}.image_limit", "llm_rosetta.ir.primitives.image_limit"),
    RetiredModule(
        f"{_HELPERS}.system_message_hoist",
        "llm_rosetta.ir.primitives.system_message_hoist",
    ),
    RetiredModule(f"{_HELPERS}.tool_group", "llm_rosetta.ir.primitives.tool_group"),
    RetiredModule(f"{_HELPERS}.truncate", "llm_rosetta.ir.primitives.truncate"),
    RetiredModule(f"{_TOOLS}.call_unwind", "llm_rosetta.ir.primitives.call_unwind"),
    RetiredModule(f"{_TOOLS}.allowed_tools", "llm_rosetta.ir.tools.allowed_tools"),
    RetiredModule(f"{_TOOLS}.batch", "llm_rosetta.ir.tools.batch"),
    RetiredModule(f"{_TOOLS}.call_id", "llm_rosetta.ir.tools.call_id"),
    RetiredModule(f"{_TOOLS}.error_marker", "llm_rosetta.ir.tools.error_marker"),
    RetiredModule(f"{_TOOLS}.intrinsic", "llm_rosetta.ir.tools.intrinsic"),
    RetiredModule(f"{_TOOLS}.orphan_fix", "llm_rosetta.ir.tools.orphan_fix"),
    RetiredModule(f"{_TOOLS}.strict_mode", "llm_rosetta.ir.tools.strict_mode"),
    # -- name aliases and shim declarations (#1656) --------------------------
    RetiredModule(
        "llm_rosetta.provider_names",
        None,
        note=(
            "pass the canonical name; the retired spellings and their "
            "replacements are in llm_rosetta.retired_names.RETIRED_NAMES"
        ),
    ),
    RetiredModule("llm_rosetta.shims.transforms", "llm_rosetta.contracts.transform"),
    # -- extension contracts that moved to llm_rosetta.contracts (#1657) -----
    RetiredModule("llm_rosetta.transforms", "llm_rosetta.contracts.transform"),
    RetiredModule(
        "llm_rosetta.capabilities",
        None,
        note="the capability passes are internal and not part of the public surface",
    ),
    RetiredModule("llm_rosetta.shims.auth_exchange", "llm_rosetta.contracts.auth"),
    RetiredModule("llm_rosetta.shims.quota_probe", "llm_rosetta.contracts.quota"),
    RetiredModule("llm_rosetta.routing", "llm_rosetta.contracts.routing"),
    RetiredModule("llm_rosetta.observability.usage", "llm_rosetta.contracts.usage"),
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
    return f"{message}; {MIGRATE_HINT}"


def retired_key_message(old: str, new: str) -> str:
    """Return the sentence for a config key whose spelling was retired.

    A key such as ``embedding_providers`` is a compound of a retired model
    type, so it has no row of its own; the caller names its replacement.

    Args:
        old: The retired key.
        new: The key that replaces it.
    """
    return (
        f"config key {old!r} was renamed to {new!r} in {RENAMED_IN} "
        f"and the old spelling is no longer accepted (use {new!r}); {MIGRATE_HINT}"
    )


def retired_module(path: str) -> RetiredModule | None:
    """Return the row that covers module *path*, or ``None``.

    A row covers its own path and every submodule of it; the longest
    matching row wins.

    Args:
        path: A dotted module path, as written in an import statement.

    Returns:
        The matching row, or ``None`` when *path* was never retired.
    """
    best: RetiredModule | None = None
    for row in RETIRED_MODULES:
        if path == row.path or path.startswith(row.path + "."):
            if best is None or len(row.path) > len(best.path):
                best = row
    return best


def retired_module_message(path: str) -> str | None:
    """Return the sentence that says what to import instead of *path*.

    Args:
        path: A dotted module path, as written in an import statement.

    Returns:
        The message, or ``None`` when *path* is not covered by a row.
    """
    row = retired_module(path)
    if row is None:
        return None
    message = f"{row.path!r} was removed in {RENAMED_IN}"
    if row.replacement is not None:
        message += f"; import {row.replacement!r} instead"
    else:
        message += f" and has no public replacement ({row.note})"
    return f"{message}; see the migration guide"


def check_not_retired(name: object, *kinds: str) -> None:
    """Raise if *name* is retired under any of *kinds*.

    A name that is not retired is a no-op.

    Args:
        name: The name to check; a non-string is ignored.
        *kinds: The kinds to try, in order; the first match wins.

    Raises:
        ValueError: With :func:`retired_name_message`.
    """
    if not isinstance(name, str):
        return
    for kind in kinds:
        message = retired_name_message(name, kind)
        if message is not None:
            raise ValueError(message)


__all__ = [
    "KINDS",
    "MIGRATE_HINT",
    "MIGRATION_GUIDE_URL",
    "MODEL_TYPE",
    "PROVIDER_TYPE",
    "RETIRED_MODULES",
    "RETIRED_NAMES",
    "SHIM",
    "RetiredModule",
    "RetiredName",
    "check_not_retired",
    "lookup",
    "replacement_for",
    "retired_key_message",
    "retired_module",
    "retired_module_message",
    "retired_name_message",
]
