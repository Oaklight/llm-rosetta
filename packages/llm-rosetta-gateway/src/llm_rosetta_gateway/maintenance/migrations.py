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

CURRENT_VERSION = 2


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


_MIGRATIONS: list[tuple[int, Any]] = [
    (0, _migrate_v0_to_v1),
    (1, _migrate_v1_to_v2),
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
