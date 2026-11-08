"""Bridge between the gateway config and declarative shim sources.

The top-level ``shims`` config section declares extra local directories and
remote archives to load provider shims from.  This module resolves that section
against the config file location and drives
:func:`llm_rosetta.shims.rebuild_shims`.

Shims must be registered **before** :class:`~llm_rosetta_gateway.config.GatewayConfig`
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .shims import (
    SHIM_SOURCE_PACKAGES,
    ShimLoadReport,
    get_source_package,
    rebuild_shims,
)
from .shims.removals import ShimRemovalAdvisory, detect_shim_removals
from .shims.sources import (
    DEFAULT_CACHE_DIRNAME,
    REMOTE_TTL_SECONDS,
    HttpGet,
    normalize_subpath,
    normalize_mirror,
)

logger = logging.getLogger("llm-rosetta-gateway")


def shims_signature(raw: Mapping[str, Any] | None) -> str:
    """Return a stable hash of the ``shims`` section.

    Used by the admin reload path to skip a full shim rebuild (and any remote
    fetch) when an unrelated part of the config changed.  Key order inside the
    section does not affect the result.

    The hash covers the section **as the document spells it**, not the parsed
    :class:`ShimSources`: the loader stores this value and the reload path
    recomputes it from the document it has just read, so both sides must hash
    the same kind of value.  A parsed section has already anchored its paths
    and expanded its presets, which would make the two disagree.
    """
    section = raw.get("shims") if isinstance(raw, Mapping) else None
    if not section:
        return ""
    try:
        blob = json.dumps(section, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = repr(section)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _known_preset_keys() -> str:
    """The preset keys this build ships, for an error or a warning."""
    return ", ".join(entry.key for entry in SHIM_SOURCE_PACKAGES)


def _unknown_preset_error(key: str) -> ValueError:
    """The one error for a preset key this build does not ship.

    One constructor for both the parse path and the already-parsed path, so
    the two cannot word the same refusal differently.
    """
    return ValueError(
        f"config: unknown shim preset {key!r}; known presets: {_known_preset_keys()}"
    )


def _resolve_presets(
    raw_presets: list[Any], *, on_unknown: str = "raise"
) -> list[dict[str, Any]]:
    """Expand ``shims.presets`` keys into their catalogue remotes.

    Duplicates collapse to the first mention, so the same archive is not
    listed twice.

    Args:
        raw_presets: The declared keys.
        on_unknown: ``"raise"`` (the admin save path, where the operator can
            correct the body) or ``"skip"`` (startup, where a key this build
            does not ship must not keep the gateway from running — it may come
            from a newer build, and the panel that would fix it is part of the
            same process).

    Raises:
        ValueError: On a non-string key, or an unknown key when *on_unknown*
            is ``"raise"``.
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
            if on_unknown == "skip":
                logger.warning(
                    "config: unknown shim preset %r is skipped; this build ships: %s",
                    key,
                    _known_preset_keys(),
                )
                continue
            raise _unknown_preset_error(key)
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


def _validate_remote_specs(remotes: Sequence[Mapping[str, Any]]) -> None:
    """Validate the per-remote fields the config layer owns.

    The loader checks the primary URL shape and the cache paths at fetch time.
    A ``path`` and a ``mirror`` are checked here, at parse, because they are
    declarations this layer can judge without a network: startup and the admin
    save path share this parser, so a bad value is refused before it is
    persisted or served.  ``path`` is normalized by the same rule the loader
    applies (``mirror.path`` included) — refusing it here rather than at the
    fetch keeps a document a load rejected from ever being written.

    Raises:
        ValueError: If a remote's ``path`` or ``mirror`` declaration is invalid.
    """
    for index, entry in enumerate(remotes):
        try:
            normalize_subpath(entry.get("path"), where="'path'")
        except ValueError as exc:
            raise ValueError(
                f"config: 'shims.remotes[{index}].path' is invalid: {exc}"
            ) from None
        try:
            mirror = normalize_mirror(entry.get("mirror"))
        except ValueError as exc:
            raise ValueError(
                f"config: 'shims.remotes[{index}].mirror' is invalid: {exc}"
            ) from None
        if mirror is not None and entry.get("sha256") and mirror[1] is None:
            # The mirror is the path an attacker can force (make the primary
            # unreachable); without its own digest it would serve bytes the
            # primary's pin never approved.
            raise ValueError(
                f"config: 'shims.remotes[{index}].mirror' needs its own 'sha256' "
                "when the remote pins its primary: the two legs serve different "
                "bytes, and an unpinned mirror would bypass the pin"
            )


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


@dataclass(frozen=True)
class ShimSources:
    """The ``shims`` section, parsed and validated once.

    One parser feeds the config loader, :func:`apply_shim_sources` and the
    admin save path, so a document a load accepts is a document an apply
    accepts too — they cannot drift apart the way a loader that kept the raw
    mapping and an apply that validated it did (#1330).

    Attributes:
        presets: The declared catalogue keys, as written.
        remotes: The remotes to load: presets expanded, explicit entries
            merged over a preset with the same URL.
        paths: Local shim directories, anchored to the config file.
        disabled: Shim names to keep out of the registry.
        cache_root: Where remote archives are cached, anchored the same way.
        remote_ttl: Seconds a remote archive may be reused without a refetch.
        strict: Whether an unknown preset or a failed source stops the load.
        declared: Whether the document carried a non-empty ``shims``
            section.  Only this makes the object truthy — a parsed empty
            section still resolves a cache root, which is not a declaration.
    """

    presets: tuple[str, ...] = ()
    remotes: tuple[Mapping[str, Any], ...] = ()
    paths: tuple[str, ...] = ()
    disabled: tuple[str, ...] = ()
    cache_root: str | None = None
    remote_ttl: int = REMOTE_TTL_SECONDS
    strict: bool = False
    declared: bool = False

    def __bool__(self) -> bool:
        """Whether the config declared this section."""
        return self.declared


def parse_shims_section(
    section: Mapping[str, Any] | None,
    config_path: str | None = None,
    *,
    on_unknown: str | None = None,
) -> ShimSources:
    """Parse and validate a ``shims`` section.

    Args:
        section: The ``shims`` mapping (or ``None``/empty).
        config_path: Path to the config file, the anchoring base for relative
            ``paths`` and ``cache_dir`` entries.
        on_unknown: ``"raise"`` or ``"skip"`` for a ``presets`` key this build
            does not ship.  ``None`` reads it from ``shims.strict``: a strict
            document raises, a tolerant one skips with a warning (see
            :func:`_resolve_presets`).

    Returns:
        The parsed section.  Falsy when the document carried none.

    Raises:
        ValueError: If the section or one of its fields has the wrong shape.
    """
    config_dir = os.path.dirname(config_path) if config_path else None
    if not section:
        return ShimSources(cache_root=_cache_root(None, config_dir))
    if not isinstance(section, Mapping):
        raise ValueError("config: 'shims' must be an object")

    mode = on_unknown
    if mode is None:
        mode = "raise" if section.get("strict") is True else "skip"
    dirs = _resolve_paths(section.get("paths") or [], config_dir)
    raw_presets = section.get("presets") or []
    if not isinstance(raw_presets, list):
        raise ValueError("config: 'shims.presets' must be an array")
    remotes = _resolve_remotes(
        section.get("remotes") or [], _resolve_presets(raw_presets, on_unknown=mode)
    )
    _validate_remote_specs(remotes)
    _validate_knobs(section)
    return ShimSources(
        presets=tuple(raw_presets),
        remotes=tuple(remotes),
        paths=tuple(dirs),
        disabled=tuple(resolve_disabled(section)),
        cache_root=_cache_root(section.get("cache_dir"), config_dir),
        remote_ttl=_int_setting(section, "remote_ttl", REMOTE_TTL_SECONDS),
        strict=bool(section.get("strict", False)),
        declared=True,
    )


def resolve_source_config(
    section: Mapping[str, Any] | None,
    config_path: str | None,
    *,
    on_unknown: str = "raise",
) -> tuple[list[str], list[dict[str, Any]], str]:
    """Resolve a ``shims`` section into absolute dirs, remotes and cache root.

    A thin view over :func:`parse_shims_section`, kept for callers that want
    the three values rather than the object.

    Args:
        section: The ``shims`` mapping (or ``None``/empty).
        config_path: Path to the config file, used as the anchoring base.
        on_unknown: What to do with a ``presets`` key this build does not ship;
            defaults to raising, because every caller of this view is a path
            where the caller can still correct what it sent.

    Returns:
        ``(dirs, remotes, cache_root)``.

    Raises:
        ValueError: If the section or one of its fields has the wrong shape.
    """
    sources = parse_shims_section(section, config_path, on_unknown=on_unknown)
    return (
        list(sources.paths),
        [dict(remote) for remote in sources.remotes],
        sources.cache_root or _cache_root(None, None),
    )


def resolve_disabled(section: Mapping[str, Any] | None) -> list[str]:
    """Return the shim names ``shims.disabled`` keeps out of the registry.

    The switch is a declaration, not a deletion: the shim's source stays
    declared, so enabling the name again puts the same definition back.

    Args:
        section: The ``shims`` mapping (or ``None``/empty).

    Returns:
        The disabled names, in declaration order.

    Raises:
        ValueError: If the value is not an array of strings.
    """
    if not isinstance(section, Mapping):
        return []
    raw = section.get("disabled") or []
    if not isinstance(raw, list) or any(not isinstance(n, str) for n in raw):
        raise ValueError("config: 'shims.disabled' must be an array of strings")
    return [name for name in raw if name]


def describe_shim_sources(report: ShimLoadReport | None) -> list[str]:
    """One line per declared remote source, naming the leg it loaded from.

    Local ``paths`` entries are left out: they have no legs to choose between.
    The lines are what the startup output prints under the banner, next to the
    matching log line.

    Args:
        report: The applied load report, or ``None`` when no section applied.

    Returns:
        The lines, in declaration order.
    """
    if report is None:
        return []
    lines: list[str] = []
    for status in report.sources:
        if status.kind != "remote":
            continue
        if not status.ok:
            lines.append(f"{status.location}: failed")
            continue
        leg = status.leg or "primary"
        origin = "cache" if status.from_cache else "network"
        if status.stale:
            origin += ", stale"
        lines.append(f"{status.location}: loaded via {leg} ({origin})")
    return lines


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
    section: ShimSources | Mapping[str, Any] | None,
    config_path: str | None = None,
    *,
    strict: bool | None = None,
    http_get: HttpGet | None = None,
    proxy: str | None = None,
    force_refresh: bool = False,
) -> ShimLoadReport:
    """Apply a parsed ``shims`` section to the registry.

    Accepts the parsed object (what the config loader stores) or a raw
    mapping, which is parsed with the same validator the loader uses — so a
    section that reached this far has already been checked, and the two
    paths cannot disagree about its shape.

    A *strict* argument given here also governs the parse: an unknown preset
    is refused when it is ``True``, skipped when it is ``False``, and read from
    ``shims.strict`` when it is ``None``.  A section that was already parsed
    cannot be re-judged — the loader parsed it with the document's own
    ``strict`` — so ``strict=True`` over such a section **raises** when it
    carries a preset this build does not ship, rather than reporting success
    for a rule it never applied; ``strict=False`` accepts what the parse
    already skipped.

    Raises:
        ValueError: If the section or one of its fields has the wrong shape,
            or if a parsed section contradicts an explicit ``strict=True``.
    """
    if isinstance(section, ShimSources):
        sources = section
        if strict is True:
            unknown = [
                key for key in sources.presets if get_source_package(key) is None
            ]
            if unknown:
                raise _unknown_preset_error(unknown[0])
    else:
        sources = parse_shims_section(
            section,
            config_path,
            on_unknown=(None if strict is None else ("raise" if strict else "skip")),
        )
    if strict is None:
        strict = sources.strict
    remotes = [dict(spec) for spec in sources.remotes]
    if force_refresh:
        remotes = [dict(spec, refresh="always") for spec in remotes]

    report = rebuild_shims(
        list(sources.paths),
        remotes,
        cache_root=sources.cache_root or _cache_root(None, None),
        strict=bool(strict),
        http_get=http_get,
        ttl=sources.remote_ttl,
        proxy=proxy,
        disabled=list(sources.disabled),
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
    "ShimSources",
    "apply_shim_section",
    "apply_shim_sources",
    "describe_shim_sources",
    "parse_shims_section",
    "report_shim_removals",
    "resolve_source_config",
    "shim_removal_advisories",
    "shims_signature",
]
