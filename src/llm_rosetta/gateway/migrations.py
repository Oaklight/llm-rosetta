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

CURRENT_VERSION = 1


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

    # --- Merge rerank_providers ---
    for name, cfg in raw.get("rerank_providers", {}).items():
        if not isinstance(cfg, dict):
            continue
        if name in providers:
            prov = providers[name]
            prov.setdefault("rerank_format", cfg.get("format", "jina"))
            prov.setdefault("rerank_path", cfg.get("rerank_path", "/v1/rerank"))
        else:
            providers[name] = {
                "api_key": cfg.get("api_key", ""),
                "base_url": cfg.get("base_url", ""),
                "rerank_format": cfg.get("format", "jina"),
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
            prov.setdefault("embedding_format", cfg.get("format", "openai"))
            prov.setdefault(
                "embedding_path", cfg.get("embedding_path", "/v1/embeddings")
            )
        else:
            providers[name] = {
                "api_key": cfg.get("api_key", ""),
                "base_url": cfg.get("base_url", ""),
                "embedding_format": cfg.get("format", "openai"),
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


_MIGRATIONS: list[tuple[int, Any]] = [
    (0, _migrate_v0_to_v1),
]


def migrate(raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Apply all pending migrations to *raw* config dict.

    Returns:
        Tuple of (migrated_dict, changed) where *changed* is True if any
        migration was applied.
    """
    version = raw.get("config_version", 0)
    if version >= CURRENT_VERSION:
        return raw, False

    changed = False
    for from_version, migration_fn in _MIGRATIONS:
        if version <= from_version:
            logger.info(
                "config: applying migration v%d → v%d", from_version, from_version + 1
            )
            migration_fn(raw)
            changed = True

    raw["config_version"] = CURRENT_VERSION
    return raw, changed
