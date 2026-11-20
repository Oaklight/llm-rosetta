"""Versioned config migration framework.

Each migration is a function that transforms a raw config dict from one
schema version to the next.  Migrations are registered in ``_MIGRATIONS``
and run sequentially by :func:`migrate`.  The ``config_version`` key
tracks which migrations have been applied.

To add a new migration:
  1. Write a function ``_migrate_vN_to_vM(raw)`` that mutates *raw* in-place.
  2. Append ``(N, _migrate_vN_to_vM)`` to ``_MIGRATIONS``.
  3. Bump ``CURRENT_VERSION`` to ``M``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("llm-rosetta-gateway")

CURRENT_VERSION = 5


def _migrate_v0_to_v1(raw: dict[str, Any]) -> None:
    """Merge legacy rerank/embedding separate-pool config into unified providers.

    Before (v0):
        rerank_providers:   {name: {api_key, base_url, format, rerank_path}}
        rerank_models:      {model: provider | {provider}}
        embedding_providers: {name: {api_key, base_url, format, embedding_path}}
        embedding_models:    {model: provider | {provider}}
        default_rerank_format: str
        default_embedding_format: str

    After (v1):
        providers.{name}.rerank_format / rerank_path  (merged onto existing or new)
        providers.{name}.embedding_format / embedding_path
        models.{model}.type = "rerank" | "embedding"
    """
    providers = raw.setdefault("providers", {})
    models = raw.setdefault("models", {})

    default_rerank_fmt = raw.get("default_rerank_format", "jina")
    default_embed_fmt = raw.get("default_embedding_format", "openai")

    # --- Merge rerank_providers ---
    for name, cfg in raw.get("rerank_providers", {}).items():
        if not isinstance(cfg, dict):
            continue
        if name in providers:
            prov = providers[name]
            prov.setdefault("rerank_format", cfg.get("format", default_rerank_fmt))
            prov.setdefault("rerank_path", cfg.get("rerank_path", "/v1/rerank"))
        else:
            providers[name] = {
                "api_key": cfg.get("api_key", ""),
                "base_url": cfg.get("base_url", ""),
                "rerank_format": cfg.get("format", default_rerank_fmt),
                "rerank_path": cfg.get("rerank_path", "/v1/rerank"),
            }
            if cfg.get("enabled") is False:
                providers[name]["enabled"] = False

    # --- Merge embedding_providers ---
    for name, cfg in raw.get("embedding_providers", {}).items():
        if not isinstance(cfg, dict):
            continue
        if name in providers:
            prov = providers[name]
            prov.setdefault("embedding_format", cfg.get("format", default_embed_fmt))
            prov.setdefault(
                "embedding_path", cfg.get("embedding_path", "/v1/embeddings")
            )
        else:
            providers[name] = {
                "api_key": cfg.get("api_key", ""),
                "base_url": cfg.get("base_url", ""),
                "embedding_format": cfg.get("format", default_embed_fmt),
                "embedding_path": cfg.get("embedding_path", "/v1/embeddings"),
            }
            if cfg.get("enabled") is False:
                providers[name]["enabled"] = False

    # --- Merge rerank_models ---
    for model_name, value in raw.get("rerank_models", {}).items():
        if model_name in models:
            continue
        prov = value if isinstance(value, str) else value.get("provider", "")
        models[model_name] = {"provider": prov, "type": "rerank"}

    # --- Merge embedding_models ---
    for model_name, value in raw.get("embedding_models", {}).items():
        if model_name in models:
            continue
        prov = value if isinstance(value, str) else value.get("provider", "")
        models[model_name] = {"provider": prov, "type": "embedding"}

    # --- Remove legacy keys ---
    for key in (
        "rerank_providers",
        "rerank_models",
        "default_rerank_format",
        "embedding_providers",
        "embedding_models",
        "default_embedding_format",
    ):
        raw.pop(key, None)


def _merge_section(raw: dict[str, Any], old: str, new: str) -> None:
    """Rename the top-level pool section *old* to *new*, keeping both.

    A per-type pool section (``embedding_providers`` / ``embedding_models``)
    is read by the same parser under either spelling, so moving it is
    lossless.  When both spellings are present the canonical section wins
    entry by entry — the precedence the loader applies — and entries only the
    legacy section has are carried over rather than dropped.
    """
    section = raw.pop(old, None)
    if section is None:
        return
    target = raw.get(new)
    if target is None:
        raw[new] = section
    elif isinstance(target, dict) and isinstance(section, dict):
        for key, value in section.items():
            target.setdefault(key, value)


def _migrate_v1_to_v2(raw: dict[str, Any]) -> None:
    """Rename the ``embedding`` model type to ``embeddings``.

    Before (v1):
        providers.{name}.embedding_format / embedding_path / embedding_shim
        models.{model}.type = "embedding"
        default_embedding_format: str
        embedding_providers / embedding_models sections

    After (v2):
        providers.{name}.embeddings_format / embeddings_path / embeddings_shim
        models.{model}.type = "embeddings"
        default_embeddings_format: str
        embeddings_providers / embeddings_models sections

    The category is spelled like the endpoint it serves (``/v1/embeddings``),
    matching the shim directory and the route.  The old spelling is still
    accepted at runtime by the parser, so a config that cannot be written
    back (a read-only mount) keeps working.
    """
    renames = {
        "embedding_format": "embeddings_format",
        "embedding_path": "embeddings_path",
        "embedding_shim": "embeddings_shim",
        "default_embedding_format": "default_embeddings_format",
    }
    for old, new in renames.items():
        if old in raw:
            raw.setdefault(new, raw.pop(old))

    # The pool sections carry the same shape under either spelling (the
    # loader parses ``embeddings_providers`` with ``format_key="format"``,
    # exactly as it parsed ``embedding_providers``).  Left behind, they
    # would keep the old spelling in the document — the admin save
    # round-trips what it loaded, so the file would say both.
    _merge_section(raw, "embedding_providers", "embeddings_providers")
    _merge_section(raw, "embedding_models", "embeddings_models")

    providers = raw.get("providers")
    if isinstance(providers, dict):
        for cfg in providers.values():
            if not isinstance(cfg, dict):
                continue
            for old, new in renames.items():
                if old in cfg:
                    cfg.setdefault(new, cfg.pop(old))

    models = raw.get("models")
    if isinstance(models, dict):
        for cfg in models.values():
            if isinstance(cfg, dict) and cfg.get("type") == "embedding":
                cfg["type"] = "embeddings"


def _migrate_v2_to_v3(raw: dict[str, Any]) -> None:
    """Rename the ``decision`` model type to ``decisions``.

    Before (v2):
        providers.{name}.decision_format / decision_path / decision_shim
        models.{model}.type = "decision"
        default_decision_format: str
        decision_providers / decision_models sections

    After (v3):
        providers.{name}.decisions_format / decisions_path / decisions_shim
        models.{model}.type = "decisions"
        default_decisions_format: str
        decisions_providers / decisions_models sections

    Same rule as v2: the category is spelled like the endpoint it serves
    (``/v1/decisions``).  The old spelling is still read at runtime.
    """
    renames = {
        "decision_format": "decisions_format",
        "decision_path": "decisions_path",
        "decision_shim": "decisions_shim",
        "default_decision_format": "default_decisions_format",
    }
    for old, new in renames.items():
        if old in raw:
            raw.setdefault(new, raw.pop(old))

    # The pool sections carry the same shape under either spelling (the
    # loader parses ``decisions_providers`` with ``format_key="format"``,
    # exactly as it parsed ``decision_providers``).  Left behind, they
    # would keep the old spelling in the document — the admin save
    # round-trips what it loaded, so the file would say both.
    _merge_section(raw, "decision_providers", "decisions_providers")
    _merge_section(raw, "decision_models", "decisions_models")

    providers = raw.get("providers")
    if isinstance(providers, dict):
        for cfg in providers.values():
            if not isinstance(cfg, dict):
                continue
            for old, new in renames.items():
                if old in cfg:
                    cfg.setdefault(new, cfg.pop(old))

    models = raw.get("models")
    if isinstance(models, dict):
        for cfg in models.values():
            if isinstance(cfg, dict) and cfg.get("type") == "decision":
                cfg["type"] = "decisions"


def _in_current_spelling(cfg: dict[str, Any]) -> dict[str, Any]:
    """A copy of a model entry with a retired type spelling read as its successor.

    A v3 document spells a capability ``embedding``.  The loader rejects that
    spelling now, so the migration that upgrades such a document reads it
    through the retired-names table instead of through the loader's parser.
    """
    from llm_rosetta.api import retired_names

    def current(name: Any) -> Any:
        if not isinstance(name, str):
            return name
        return retired_names.replacement_for(name, retired_names.MODEL_TYPE) or name

    entry = dict(cfg)
    if "type" in entry:
        entry["type"] = current(entry["type"])
    caps = entry.get("capabilities")
    if isinstance(caps, list):
        entry["capabilities"] = [current(cap) for cap in caps]
    return entry


def _migrate_v3_to_v4(raw: dict[str, Any]) -> None:
    """Give a legacy capability-only model an explicit ``type``.

    Before (v3):
        models.{model}.capabilities = ["embedding"]   (no ``type``)

    After (v4):
        models.{model}.type = "embeddings"

    The admin console has always rendered such an entry as that type, and the
    parser reads it the same way (:func:`config.parse.declared_model_type`);
    writing the field makes the document say what both already do, so a
    reader without the capability fallback cannot change the answer.

    The type is written only when the entry's provider can serve that
    endpoint — the same pool parse the distribution uses.  A provider without
    an ``embeddings_format`` (or the matching shim) leaves the entry in the
    chat pool at runtime, so writing the type there would make the file
    disagree with the process, and with a reader that trusts ``type``.

    **One exception, measured.**  :func:`_serving_providers` counts an entry
    that *declares* the endpoint even when its credential has no lane to share
    (:func:`~.config.typed.parse_typed_config` with
    ``refuse_unowned_minted=False``), because membership is a document
    question.  So a legacy ``embeddings_providers`` entry carrying its own
    ``token_command`` is written as serving — while the runtime parses it,
    disables the endpoint with a reason, and logs ``Model '<m>' has
    type=embeddings but provider '<p>' lacks embeddings_format — skipped``.
    The window is narrow (it needs a v3 document, a capability-only model, and
    a legacy minting provider at once) and the runtime says what happened; the
    alternative — writing no type — would leave the document silent about a
    model the file otherwise routes, which is the disagreement this migration
    exists to remove.
    """
    models = raw.get("models")
    if not isinstance(models, dict):
        return
    from ..config.parse import declared_model_type, declared_provider_names

    serving: dict[str, set[str]] = {}
    for cfg in models.values():
        if not isinstance(cfg, dict) or cfg.get("type"):
            continue
        model_type = declared_model_type(_in_current_spelling(cfg))
        if model_type == "llm":
            continue
        if not serving:
            serving.update(_serving_providers(raw))
        names = declared_provider_names(cfg)
        if not any(name in serving.get(model_type, ()) for name in names):
            continue
        cfg["type"] = model_type


#: The spellings the versioned migrations upgrade a document *from*.  These are
#: the document's history, fixed in the migrations that rewrite them; they are
#: not a vocabulary the loader accepts (a current document that spells them is
#: rejected with the retired-names message).
_PRE_RENAME_TYPE = dict(embeddings="embedding", decisions="decision")


def _serving_providers(raw: dict[str, Any]) -> dict[str, set[str]]:
    """The providers each non-LLM type would route to, from the raw document.

    Uses the same ``parse_typed_config`` calls the runtime pools make, so the
    migration's idea of "this provider serves the type" cannot drift from the
    distribution's.  Only membership is read, so an endpoint this pass has no
    credential owner for is still counted: whether it can *serve* is a runtime
    question (it is disabled there with a reason), while whether the document
    routes the type at it is a document question.
    """
    from ..config.typed import _typed_descriptors, parse_typed_config

    raw_providers = raw.get("providers") or {}
    proxy = (raw.get("server") or {}).get("proxy")
    serving: dict[str, set[str]] = {}
    for descriptor in _typed_descriptors():
        unified = parse_typed_config(
            raw_providers,
            {},
            descriptor=descriptor,
            require_declared_format=True,
            map_models=False,
            global_proxy=proxy,
            refuse_unowned_minted=False,
        )[0]
        old_name = _PRE_RENAME_TYPE.get(descriptor.name)
        legacy_raw = raw.get(f"{descriptor.name}_providers") or (
            raw.get(f"{old_name}_providers", {}) if old_name else {}
        )
        legacy = parse_typed_config(
            legacy_raw,
            {},
            descriptor=descriptor,
            format_key="format",
            global_proxy=proxy,
            refuse_unowned_minted=False,
        )[0]
        serving[descriptor.name] = set(unified) | set(legacy)
    return serving


def _migrate_v4_to_v5(raw: dict[str, Any]) -> None:
    """Fold a provider entry's flat endpoint keys into its face blocks.

    Before (v4):
        providers.{name}.embeddings_shim / embeddings_format /
        embeddings_path / embeddings_enabled

    After (v5):
        providers.{name}.embeddings.shim / format / path / enabled

    The flat keys stay readable (``config/typed.py::face_value`` falls back to
    them), so this changes no behaviour — it makes the document say which
    facts belong to one endpoint.  The nesting is what lets an endpoint carry
    its own ``base_url``, which a flat ``embeddings_base_url`` would otherwise
    have had to add a fourth key for.

    The legacy ``<type>_providers`` sections are left alone: each of those is
    one face by construction, and its keys are read with ``format_key="format"``
    rather than the descriptor's name.

    A key whose value is empty is moved as well — the form writes ``""`` to
    mean "cleared", and leaving the flat twin behind would make the entry
    carry two spellings of one setting that disagree.
    """
    from ..config.typed import _typed_descriptors

    providers = raw.get("providers")
    if not isinstance(providers, dict):
        return
    suffixes = ("shim", "format", "path", "enabled")
    for cfg in providers.values():
        if not isinstance(cfg, dict):
            continue
        for descriptor in _typed_descriptors():
            name = descriptor.name
            keys = [f"{name}_{suffix}" for suffix in suffixes]
            if not any(key in cfg for key in keys):
                continue
            block = cfg.get(name)
            if block is not None and not isinstance(block, dict):
                # Leave a block the parser will reject for the parser to
                # report, rather than silently swapping it for a new one.
                continue
            block = dict(block or {})
            for suffix in suffixes:
                flat = f"{name}_{suffix}"
                if flat in cfg and suffix not in block:
                    block[suffix] = cfg[flat]
                cfg.pop(flat, None)
            if block:
                cfg[name] = block


_MIGRATIONS: list[tuple[int, Any]] = [
    (0, _migrate_v0_to_v1),
    (1, _migrate_v1_to_v2),
    (2, _migrate_v2_to_v3),
    (3, _migrate_v3_to_v4),
    (4, _migrate_v4_to_v5),
]


def _declared_version(raw: dict[str, Any]) -> int:
    """The version this document declares, as an integer.

    ``config_version`` is this module's bookkeeping, not something an operator
    writes — the scaffold ``llm-rosetta-gateway init`` produces does not carry
    the key at all — so a value we cannot read is not a reason to refuse the
    document.  A quoted number (a template that renders ``"1"``) and a JSON
    float are the same version; anything else is "no version recorded", which
    means the migrations run.  Warning about the odd value is the operator's
    business; failing the load over our own field is nobody's.
    """
    declared = raw.get("config_version")
    if declared is None:
        return 0
    if not isinstance(declared, bool):
        if isinstance(declared, int):
            return declared
        if isinstance(declared, float) and declared.is_integer():
            return int(declared)
        if isinstance(declared, str):
            try:
                return int(declared.strip())
            except ValueError:
                pass
    logger.warning(
        "config: config_version is %r, which is not a version number; treating "
        "the document as having none recorded and bringing it forward",
        declared,
    )
    return 0


def migrate(raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Apply all pending migrations to *raw* config dict.

    Returns:
        Tuple of (migrated_dict, changed) where *changed* is True if any
        migration was applied.
    """
    initial_version = _declared_version(raw)
    if initial_version > CURRENT_VERSION:
        # Written by a release that knows migrations this one does not.  Serve
        # it as it is — running the migrations we do know could take apart a
        # shape they were never written for, and stamping our own version on it
        # would record work that did not happen.
        logger.warning(
            "config: this document declares config_version %d, newer than this "
            "release knows (%d); serving it as it is",
            initial_version,
            CURRENT_VERSION,
        )
        return raw, False
    if initial_version == CURRENT_VERSION:
        return raw, False

    changed = False
    for from_version, migration_fn in _MIGRATIONS:
        if initial_version <= from_version:
            logger.info(
                "config: applying migration v%d → v%d", from_version, from_version + 1
            )
            migration_fn(raw)
            changed = True

    raw["config_version"] = CURRENT_VERSION
    return raw, changed
