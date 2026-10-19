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

from llm_rosetta.shims import (
    SHIM_SOURCE_PACKAGES,
    ShimLoadReport,
    get_source_package,
    rebuild_shims,
)
from llm_rosetta.shims.removals import ShimRemovalAdvisory, detect_shim_removals
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


def _resolve_presets(raw_presets: list[Any]) -> list[dict[str, Any]]:
    """Expand ``shims.presets`` keys into their catalogue remotes.

    Duplicates collapse to the first mention, so the same archive is not
    listed twice.
    """
    remotes: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for key in raw_presets:
        if not isinstance(key, str) or not key:
            raise ValueError(
                "config: 'shims.presets' entries must be non-empty strings"
            )
        package = get_source_package(key)
        if package is None:
            known = ", ".join(entry.key for entry in SHIM_SOURCE_PACKAGES)
            raise ValueError(
                f"config: unknown shim preset {key!r}; known presets: {known}"
            )
        remote = package.remote
        url = remote.get("url")
        if url and url not in seen_urls:
            remotes.append(remote)
            seen_urls.add(url)
    return remotes


def _resolve_paths(raw_paths: Any, config_dir: str | None) -> list[str]:
    """Validate and anchor the ``shims.paths`` entries."""
    if not isinstance(raw_paths, list):
        raise ValueError("config: 'shims.paths' must be an array")
    dirs: list[str] = []
    for entry in raw_paths:
        if not isinstance(entry, str) or not entry:
            raise ValueError("config: 'shims.paths' entries must be non-empty strings")
        dirs.append(_anchor(entry, config_dir))
    return dirs


def _remote_url_key(url: Any) -> str:
    """Normalize a remote URL so a preset and a spelling of it still match.

    A trailing slash and a ``.git`` suffix address the same repository; the
    catalogue and an operator's config rarely spell it the same way.
    """
    if not isinstance(url, str):
        return ""
    key = url.strip().rstrip("/")
    if key.endswith(".git"):
        key = key[:-4].rstrip("/")
    return key


def _resolve_remotes(
    raw_remotes: Any, remotes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply explicit remotes on top of the preset-expanded *remotes*.

    An explicit entry with a preset's URL is **merged onto** that preset, with
    its own keys winning: that is what lets the documented ``presets: ["argo"]``
    plus ``{url: ..., ref: "v1"}`` pattern pin a ref without dropping the
    catalogue's ``path`` (and any ``token_env``).  Explicit entries keep their
    previous append-only behaviour among themselves.
    """
    if not isinstance(raw_remotes, list):
        raise ValueError("config: 'shims.remotes' must be an array")
    preset_keys = {
        _remote_url_key(remote.get("url")) for remote in remotes if remote.get("url")
    }
    for entry in raw_remotes:
        if not isinstance(entry, Mapping):
            raise ValueError("config: 'shims.remotes' entries must be objects")
        spec = dict(entry)
        key = _remote_url_key(spec.get("url"))
        preset = None
        if key and key in preset_keys:
            preset = next(
                item for item in remotes if _remote_url_key(item.get("url")) == key
            )
        if preset is not None:
            # Layer the explicit keys over the preset instead of replacing it:
            # a pin (`ref`/`sha256`/`refresh`) must not lose `path`.
            spec = {**preset, **spec}
            remotes = [item for item in remotes if item is not preset]
            preset_keys.discard(key)
        remotes.append(spec)
    return remotes


def _validate_knobs(section: Mapping[str, Any]) -> None:
    """Validate the scalar ``shims`` knobs in place."""
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


def resolve_source_config(
    section: Mapping[str, Any] | None, config_path: str | None
) -> tuple[list[str], list[dict[str, Any]], str]:
    """Resolve a ``shims`` section into absolute dirs, remotes and cache root.

    Relative ``paths`` entries and a relative ``cache_dir`` are anchored to the
    directory containing the config file, mirroring ``server.data_dir``.

    ``presets`` names packages from
    :data:`llm_rosetta.shims.SHIM_SOURCE_PACKAGES` (``"argo"``, ``"alcf"``,
    ``"asksage"``); each is expanded to its catalogue ``remote`` and applied
    like an explicit entry.  Presets come first, so an explicit ``remotes``
    entry with the same URL is merged onto its preset (explicit keys win) —
    a ``ref`` can be pinned without losing the catalogue's ``path``.

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

    dirs = _resolve_paths(section.get("paths") or [], config_dir)
    raw_presets = section.get("presets") or []
    if not isinstance(raw_presets, list):
        raise ValueError("config: 'shims.presets' must be an array")
    remotes = _resolve_remotes(
        section.get("remotes") or [], _resolve_presets(raw_presets)
    )
    _validate_knobs(section)
    return dirs, remotes, _cache_root(section.get("cache_dir"), config_dir)


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


def shim_removal_advisories(
    providers: Mapping[str, Any] | None,
    report: ShimLoadReport | None = None,
) -> list[ShimRemovalAdvisory]:
    """Advisories for providers still on a built-in shim that is going away.

    A shim name supplied by a successfully loaded extra source is *not* reported:
    a declared source of the same name overrides the built-in at rebuild time, so
    that provider is already on the shim that is staying.  A source that failed
    overrode nothing, so it does not suppress anything either.

    Args:
        providers: The raw ``providers`` section.  Pass the **unfiltered** one —
            a disabled provider is still worth a warning, because its config
            breaks the moment it is switched back on.
        report: The shim-source load report, when the section was applied.

    Returns:
        One advisory per provider, sorted by provider name.
    """
    supplied = {
        name
        for status in (report.sources if report else ())
        if status.ok
        for name in status.shim_names
    }
    return detect_shim_removals(providers, external_shims=supplied)


def report_shim_removals(
    providers: Mapping[str, Any] | None,
    report: ShimLoadReport | None = None,
) -> list[ShimRemovalAdvisory]:
    """Log one warning per provider on a built-in shim that is going away.

    Args:
        providers: The raw ``providers`` section (see
            :func:`shim_removal_advisories`).
        report: The shim-source load report, when the section was applied.

    Returns:
        The advisories that were logged.
    """
    advisories = shim_removal_advisories(providers, report)
    for advisory in advisories:
        preset = next(
            (
                package.key
                for package in SHIM_SOURCE_PACKAGES
                if package.repo == advisory.remote.get("url")
            ),
            None,
        )
        if preset is not None:
            logger.warning(
                "Provider %r uses built-in shim %r, removed in %s; declare the "
                "source instead: shims.presets += [%r] (or shims.remotes += [%s])",
                advisory.provider,
                advisory.shim,
                advisory.removed_in,
                preset,
                json.dumps(dict(advisory.remote)),
            )
        else:
            logger.warning(
                "Provider %r uses built-in shim %r, removed in %s; declare the "
                "source instead: shims.remotes += [%s]",
                advisory.provider,
                advisory.shim,
                advisory.removed_in,
                json.dumps(dict(advisory.remote)),
            )
    return advisories


__all__ = [
    "apply_shim_section",
    "apply_shim_sources",
    "report_shim_removals",
    "resolve_source_config",
    "shim_removal_advisories",
    "shims_signature",
]
