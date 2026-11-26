"""Names that were renamed, and what each one became.

This table is an **error table, not an alias table**.  A name listed here is
a spelling an earlier release accepted for a provider type, a shim or a model
type.  No spelling is translated any more (the alias tables are gone): a
name listed here must not fall through to "unknown type" or "no base_url
configured", so it fails with a message that names its replacement and ends
with :data:`MIGRATE_HINT`: the ``llm-rosetta migrate`` command when the
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
reader is ``llm-rosetta migrate --scan``, which reports an import of a listed
path only when that path does not resolve in the installed core, so a path
that still exists is never reported.  Add one :class:`RetiredModule` row when
a module path is removed or moved.

Shims that moved out of core
----------------------------

:data:`MOVED_VENDORS` is a third table: a vendor whose shims core shipped and
that now lives in a shim source repository (#1660).  A moved shim keeps its
name, so nothing here is renamed: a config that names it keeps working once
the vendor is installed from that repository, which the gateway does by
itself.  The table is what makes a lookup that finds no such shim say where it
went instead of "unsupported provider" (:func:`moved_shim_message`), and what
``llm-rosetta migrate`` reads to report a moved name as a note.  Add one
:class:`MovedVendor` row when a vendor leaves core.
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

    ``llm-rosetta migrate`` ships with the gateway distribution, so it is
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
            "run 'llm-rosetta migrate --config <path>' (add --write to apply), "
            f"see the migration guide ({MIGRATION_GUIDE_URL})"
        )
    return f"see the migration guide ({MIGRATION_GUIDE_URL})"


#: The sentence every retired-name error ends with.  With the gateway
#: installed it names ``llm-rosetta migrate``, the command that rewrites a
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
        install: For a shim that lives in a standalone package, the
            ``shims.install`` selector that makes the replacement loadable.
    """

    name: str
    replacement: str
    kind: str
    install: str | None = None


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
        install="argonne:argo",
    ),
    RetiredName(
        name="argo--openai_responses",
        replacement="argo--open_responses",
        kind=SHIM,
        install="argonne:argo",
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
        install="argonne:argo",
    ),
    RetiredName(name="minimax--anthropic", replacement="minimax--messages", kind=SHIM),
    RetiredName(
        name="openrouter--anthropic", replacement="openrouter--messages", kind=SHIM
    ),
    RetiredName(
        name="alcf--metis",
        replacement="alcf_metis--chat_completions",
        kind=SHIM,
        install="argonne:alcf",
    ),
    RetiredName(
        name="alcf--minerva",
        replacement="alcf_minerva--chat_completions",
        kind=SHIM,
        install="argonne:alcf",
    ),
    RetiredName(
        name="alcf--sophia",
        replacement="alcf_sophia--chat_completions",
        kind=SHIM,
        install="argonne:alcf",
    ),
    RetiredName(
        name="asksage--anthropic",
        replacement="asksage--messages",
        kind=SHIM,
        install="argonne:asksage",
    ),
    RetiredName(
        name="asksage--openai_chat",
        replacement="asksage--chat_completions",
        kind=SHIM,
        install="argonne:asksage",
    ),
    RetiredName(
        name="asksage--openai_responses",
        replacement="asksage--open_responses",
        kind=SHIM,
        install="argonne:asksage",
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
    if row.install is not None:
        message += (
            " (the shim is provided by a standalone package: "
            f"declare shims.install: ['{row.install}'] and use {row.replacement!r})"
        )
    else:
        message += f" (use {row.replacement!r})"
    moved = moved_vendor(row.replacement) if row.kind == SHIM else None
    if moved is not None:
        message += (
            f"; the shim has since moved to {moved.package}, which the gateway "
            "installs automatically"
        )
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


#: The extension repository the moved vendors live in now (it was
#: ``llm-rosetta-shim-contrib`` before the 0.14.0 extension system).
CONTRIB_REPOSITORY = "https://github.com/Oaklight/llm-rosetta-ext-contrib"


@dataclass(frozen=True)
class MovedVendor:
    """One vendor whose shims moved from core to a shim source repository.

    Attributes:
        vendor: The vendor directory name, which is also the shim-name prefix
            (``opencode`` for ``opencode--chat_completions``).
        shims: The shim names core shipped for the vendor, for the record and
            for ``llm-rosetta migrate --scan``.  A name the repository adds
            later (``<vendor>--<format>``) belongs to the vendor as well.
        probes: The quota probe ids the vendor's ``quota.py`` provides, so a
            ``quota.probe`` that names one is traced to the vendor.
        repository: The repository that publishes the vendor now.
        moved_in: The release that removed the vendor from core.
    """

    vendor: str
    shims: tuple[str, ...]
    probes: tuple[str, ...] = ()
    repository: str = CONTRIB_REPOSITORY
    moved_in: str = RENAMED_IN

    @property
    def package(self) -> str:
        """The repository name, e.g. ``llm-rosetta-ext-contrib``."""
        return self.repository.rstrip("/").rsplit("/", 1)[-1]

    def owns(self, name: str) -> bool:
        """Whether shim *name* belongs to this vendor."""
        return name == self.vendor or name.startswith(f"{self.vendor}--")


def _moved(vendor: str, *formats: str, probes: tuple[str, ...] = ()) -> MovedVendor:
    """A row whose shims are ``<vendor>--<format>`` (``""``: the bare vendor)."""
    names = tuple(f"{vendor}--{fmt}" if fmt else vendor for fmt in formats)
    return MovedVendor(vendor=vendor, shims=names, probes=probes)


#: The vendors that left core in 0.14 (#1660, the placement rule): OAuth and
#: subscription lanes, aggregators and free endpoints, and the model makers
#: that start in contrib before a promotion.  Their shim names are unchanged.
MOVED_VENDORS: tuple[MovedVendor, ...] = (
    _moved("antigravity", "acp"),
    _moved("groq", "chat_completions"),
    _moved("kilo", "chat_completions"),
    _moved("modelscope", "chat_completions"),
    _moved("moonshot", ""),
    _moved("nvidia", "chat_completions"),
    _moved("openai_chatgpt", "open_responses", probes=("openai_codex",)),
    _moved(
        "opencode",
        "chat_completions",
        "messages",
        "open_responses",
        probes=("opencode_go",),
    ),
    _moved("openrouter", "chat_completions", "messages", "open_responses", "rerank"),
    _moved("qwen", ""),
    _moved("siliconflow", "chat_completions", "embedding", "rerank"),
    _moved("volcengine", "chat_completions", "open_responses", "embedding"),
    _moved("xai", ""),
    _moved("zai", "chat_completions", "messages", "embedding", "rerank"),
    _moved("zhipu", "", "embedding", "rerank"),
)


def moved_vendor(name: object) -> MovedVendor | None:
    """Return the moved vendor that shim *name* belongs to, or ``None``.

    A retired spelling is read as the name it became first
    (``openrouter--openai_chat`` belongs to ``openrouter``).

    Args:
        name: A shim name; a non-string is ``None``.
    """
    if not isinstance(name, str) or not name:
        return None
    name = replacement_for(name, SHIM) or name
    return next((row for row in MOVED_VENDORS if row.owns(name)), None)


def moved_probe_vendor(probe_id: object) -> MovedVendor | None:
    """Return the moved vendor whose ``quota.py`` provides *probe_id*, or ``None``."""
    if not isinstance(probe_id, str):
        return None
    return next((row for row in MOVED_VENDORS if probe_id in row.probes), None)


def moved_shim_message(name: object) -> str | None:
    """Return the sentence for a lookup of a moved shim that is not registered.

    The caller decides that the shim is missing (this module reads no
    registry); the sentence says where the vendor went and how to load it.

    Args:
        name: The shim name that did not resolve.

    Returns:
        The message, or ``None`` when *name* is not a moved vendor's shim.
    """
    row = moved_vendor(name)
    if row is None:
        return None
    return (
        f"shim {name!r} is no longer built into llm-rosetta: its vendor "
        f"{row.vendor!r} moved in {row.moved_in} to the extension repository "
        f"{row.package} ({row.repository}) under the same name. The gateway "
        "installs it automatically from its 'contrib' source; a library "
        f"user installs it with `llm-rosetta ext install {row.vendor}` "
        "(llm_rosetta loads installed extensions on import; call "
        "llm_rosetta.extensions.load_extensions() to pick up a later install)"
    )


__all__ = [
    "CONTRIB_REPOSITORY",
    "KINDS",
    "MIGRATE_HINT",
    "MIGRATION_GUIDE_URL",
    "MODEL_TYPE",
    "MOVED_VENDORS",
    "PROVIDER_TYPE",
    "RETIRED_MODULES",
    "RETIRED_NAMES",
    "SHIM",
    "MovedVendor",
    "RetiredModule",
    "RetiredName",
    "check_not_retired",
    "lookup",
    "moved_probe_vendor",
    "moved_shim_message",
    "moved_vendor",
    "replacement_for",
    "retired_key_message",
    "retired_module",
    "retired_module_message",
    "retired_name_message",
]
