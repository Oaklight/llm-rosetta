"""The tier of each public module, stated once (#1657, C3).

This module is data only: no imports, so a tool can read it from a source
tree without importing ``llm_rosetta`` (``scripts/dev/public_api_diff.py``
loads it by file path).

Tiers:

- ``USER`` — the library API: converting requests and responses, the
  converter classes, the IR types, the pipelines.  A breaking change follows
  the deprecation policy.
- ``EXTENSION`` — the surface a host or a shim author builds on:
  :mod:`llm_rosetta.api` and :mod:`llm_rosetta.contracts` with its
  submodules.  Within a minor release it only gains names
  (``contracts.SURFACE_VERSION``).
- ``COMPAT`` — deprecated paths kept importable for a release that shipped
  them; each names its replacement and its removal version.
- ``REMOVED`` — modules a final release shipped and this tree no longer has.
  They are listed so the public-API diff keeps comparing them, which makes
  every removal an acknowledged entry in
  ``scripts/dev/public_api_allowlist.json`` instead of a silent drop.

Dropping a module from these tables is reported by the public-API diff
against a base that ships ``_tiers.py``, i.e. from 0.14.0 on: the base is
measured with its own table.  The 0.13.0 base has no table and is measured
with the head's list, so it is covered by the frozen module list in
``tests/test_public_api_diff.py`` (``V0_13_0_PUBLIC_MODULES``) instead.

The diff compares module-level names and constructor signatures only.  An
attribute removed from a class (the three ``ProviderShim`` properties of
0.14.0) is outside it; such removals are recorded in ``REMOVED_ATTRIBUTES`` in
``tests/test_no_deprecation_bounds.py``.

Every module not listed is **internal** and may move in any release.
``tests/test_tiers.py`` checks that every module of the first three tiers
imports and that no ``REMOVED`` module does.
"""

USER: tuple[str, ...] = (
    "llm_rosetta",
    "llm_rosetta.auto_detect",
    "llm_rosetta.converters",
    "llm_rosetta.converters.base",
    "llm_rosetta.converters.decisions",
    "llm_rosetta.converters.embeddings",
    "llm_rosetta.converters.llm.chat_completions",
    "llm_rosetta.converters.llm.google_generate",
    "llm_rosetta.converters.llm.google_interactions",
    "llm_rosetta.converters.llm.messages",
    "llm_rosetta.converters.llm.open_responses",
    "llm_rosetta.converters.registry",
    "llm_rosetta.converters.rerank",
    "llm_rosetta.fidelity",
    "llm_rosetta.ir.model",
    "llm_rosetta.pipeline",
    "llm_rosetta.retired_names",
    "llm_rosetta.shims",
    "llm_rosetta.tool_ops",
    "llm_rosetta.types",
)

EXTENSION: tuple[str, ...] = (
    "llm_rosetta.api",
    "llm_rosetta.contracts",
    "llm_rosetta.contracts.auth",
    "llm_rosetta.contracts.cli_login",
    "llm_rosetta.contracts.command",
    "llm_rosetta.contracts.extension",
    "llm_rosetta.contracts.quota",
    "llm_rosetta.contracts.resolver",
    "llm_rosetta.contracts.routing",
    "llm_rosetta.contracts.shim",
    "llm_rosetta.contracts.shim_manifest",
    "llm_rosetta.contracts.transform",
    "llm_rosetta.contracts.transport",
    "llm_rosetta.contracts.usage",
)

COMPAT: tuple[str, ...] = (
    # Forwards to llm_rosetta_gateway (deprecated).
    "llm_rosetta.gateway",
    # The 0.13 observability package: an alias finder for the telemetry that
    # moved to llm_rosetta_gateway.observability, plus the usage names (now in
    # llm_rosetta.contracts.usage).
    "llm_rosetta.observability",
    # The shim registry stays here; its re-exports of the data classes in
    # llm_rosetta.contracts.shim are removed in 0.15.0.
    "llm_rosetta.shims.provider_shim",
    # 0.13.0 held the shim loader here; the names it exported forward to
    # llm_rosetta.shims._loader (use llm_rosetta.api), warning on access;
    # removed in 0.15.0.
    "llm_rosetta.shims.providers",
    # 0.13.0 flat module, forwards to llm_rosetta.contracts.routing; removed
    # in 0.15.0.
    "llm_rosetta.routing",
    # 0.13.0 flat module, forwards the passes and the shim types to their
    # current homes; removed in 0.15.0.
    "llm_rosetta.capabilities",
    # 0.13.0 package, forwards to llm_rosetta.ir.model; removed in 0.15.0.
    "llm_rosetta.types.ir",
    # 0.13.0 module (already deprecated then), forwards to
    # llm_rosetta.ir.model.extensions_experimental; removed in 0.15.0.
    "llm_rosetta.types.ir.extensions",
)

REMOVED: tuple[str, ...] = ()
