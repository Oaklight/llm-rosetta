"""Bridge between the gateway config and declarative shim sources.

The top-level ``shims`` config section declares extra local directories and
remote archives to load provider shims from.  This module resolves that section
against the config file location and drives
:func:`llm_rosetta.shims.rebuild_shims`.

Shims must be registered **before** :class:`~llm_rosetta.gateway.config.GatewayConfig`
is constructed: its provider-type resolution looks shims up in the global
registry.  The CLI therefore calls :func:`apply_shim_sources` between
``load_config`` and ``GatewayConfig``; the admin reload path does the same
before rebuilding the config.

Programmatic embedders holding a raw config dict should call
:func:`apply_shim_sources` before ``GatewayConfig(raw)``; ``create_app`` also
applies the section automatically when none was supplied.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llm_rosetta.shims import ShimLoadReport, rebuild_shims
from llm_rosetta.shims.sources import (
    DEFAULT_CACHE_DIRNAME,
    REMOTE_TTL_SECONDS,
    HttpGet,
)

logger = logging.getLogger("llm-rosetta-gateway")


def shims_signature(raw: Mapping[str, Any] | None) -> str:
    """Return a stable hash of the ``shims`` section.

    Used by the admin reload path to skip a full shim rebuild (and any remote
    fetch) when an unrelated part of the config changed.  Key order inside the
    section does not affect the result.
    """
    section = raw.get("shims") if isinstance(raw, Mapping) else None
    if not section:
        return ""
    try:
        blob = json.dumps(section, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = repr(section)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def resolve_source_config(
    section: Mapping[str, Any] | None, config_path: str | None
) -> tuple[list[str], list[dict[str, Any]], str]:
    """Resolve a ``shims`` section into absolute dirs, remotes and cache root.

    Relative ``paths`` entries and a relative ``cache_dir`` are anchored to the
    directory containing the config file, mirroring ``server.data_dir``.

    Args:
        section: The ``shims`` mapping (or ``None``/empty).
        config_path: Path to the config file, used as the anchoring base.

    Returns:
        ``(dirs, remotes, cache_root)``.

    Raises:
        ValueError: If the section or one of its fields has the wrong shape.
    """
    config_dir = os.path.dirname(config_path) if config_path else None
    if not section:
        return [], [], _cache_root(None, config_dir)
    if not isinstance(section, Mapping):
        raise ValueError("config: 'shims' must be an object")

    raw_paths = section.get("paths") or []
    if not isinstance(raw_paths, list):
        raise ValueError("config: 'shims.paths' must be an array")
    dirs: list[str] = []
    for entry in raw_paths:
        if not isinstance(entry, str) or not entry:
            raise ValueError("config: 'shims.paths' entries must be non-empty strings")
        dirs.append(_anchor(entry, config_dir))

    raw_remotes = section.get("remotes") or []
    if not isinstance(raw_remotes, list):
        raise ValueError("config: 'shims.remotes' must be an array")
    remotes: list[dict[str, Any]] = []
    for entry in raw_remotes:
        if not isinstance(entry, Mapping):
            raise ValueError("config: 'shims.remotes' entries must be objects")
        remotes.append(dict(entry))

    cache_dir = section.get("cache_dir")
    if cache_dir is not None and not isinstance(cache_dir, str):
        raise ValueError("config: 'shims.cache_dir' must be a string")

    remote_ttl = section.get("remote_ttl")
    if remote_ttl is not None:
        try:
            int(remote_ttl)
        except (TypeError, ValueError):
            raise ValueError("config: 'shims.remote_ttl' must be an integer") from None

    strict = section.get("strict")
    if strict is not None and not isinstance(strict, bool):
        raise ValueError("config: 'shims.strict' must be a boolean")

    return dirs, remotes, _cache_root(cache_dir, config_dir)


def apply_shim_sources(
    raw: Mapping[str, Any],
    config_path: str | None = None,
    *,
    strict: bool | None = None,
    http_get: HttpGet | None = None,
    proxy: str | None = None,
    force_refresh: bool = False,
) -> ShimLoadReport:
    """Apply ``raw['shims']`` to the global shim registry.

    Always performs a full rebuild, so calling it also *drops* shims whose
    source was removed from the config.  Callers that want to avoid a rebuild
    when nothing changed should compare :func:`shims_signature` first.

    Args:
        raw: The parsed config mapping.
        config_path: Path to the config file (for path anchoring).
        strict: Override ``shims.strict``; ``None`` reads it from the section.
        http_get: Injectable HTTP getter (tests).
        proxy: Upstream proxy; defaults to ``server.proxy``.
        force_refresh: Fetch every remote regardless of its cache/refresh mode.

    Returns:
        The :class:`ShimLoadReport` for the rebuild.
    """
    section = raw.get("shims") if isinstance(raw, Mapping) else None
    if proxy is None and isinstance(raw, Mapping):
        server = raw.get("server")
        if isinstance(server, Mapping):
            proxy = server.get("proxy")
    return apply_shim_section(
        section,
        config_path,
        strict=strict,
        http_get=http_get,
        proxy=proxy,
        force_refresh=force_refresh,
    )


def apply_shim_section(
    section: Mapping[str, Any] | None,
    config_path: str | None = None,
    *,
    strict: bool | None = None,
    http_get: HttpGet | None = None,
    proxy: str | None = None,
    force_refresh: bool = False,
) -> ShimLoadReport:
    """Apply a ``shims`` section (already extracted) to the registry."""
    effective = section if isinstance(section, Mapping) else {}
    dirs, remotes, cache_root = resolve_source_config(effective, config_path)
    if strict is None:
        strict = bool(effective.get("strict", False))
    ttl = _int_setting(effective, "remote_ttl", REMOTE_TTL_SECONDS)
    if force_refresh:
        remotes = [dict(spec, refresh="always") for spec in remotes]

    report = rebuild_shims(
        dirs,
        remotes,
        cache_root=cache_root,
        strict=bool(strict),
        http_get=http_get,
        ttl=ttl,
        proxy=proxy,
    )
    if report.sources:
        logger.info(
            "Shim sources: %d source(s), %d shim(s), %d failure(s)",
            len(report.sources),
            report.total_shims,
            len(report.failures()),
        )
    return report


def _anchor(path: str, config_dir: str | None) -> str:
    """Anchor a relative path to the config directory."""
    if os.path.isabs(path) or not config_dir:
        return path
    return os.path.join(config_dir, path)


def _cache_root(cache_dir: str | None, config_dir: str | None) -> str:
    """Resolve the remote-archive cache directory (config-adjacent)."""
    if cache_dir:
        return _anchor(cache_dir, config_dir)
    if config_dir:
        return os.path.join(config_dir, DEFAULT_CACHE_DIRNAME)
    return str(Path("~/.cache/llm-rosetta/shims").expanduser())


def _int_setting(section: Mapping[str, Any], key: str, default: int) -> int:
    """Read an integer setting, falling back to *default*."""
    value = section.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("config: 'shims.%s' must be an integer; using %d", key, default)
        return default


__all__ = [
    "apply_shim_section",
    "apply_shim_sources",
    "resolve_source_config",
    "shims_signature",
]
