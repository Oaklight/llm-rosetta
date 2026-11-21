"""Install shim vendors from a source's versioned ``manifest.json`` (#1660).

A shim source repository publishes ``dist/manifest.json`` with one record per
released vendor version (the format is
:mod:`llm_rosetta.contracts.shim_manifest`).  The gateway's ``shims.sources``
names the manifests to read and ``shims.install`` the vendors to take from
them.  Per selected vendor this module:

1. picks the newest version whose ``requires`` this core satisfies
   (:func:`~llm_rosetta.contracts.shim_manifest.select_release`) and logs a
   WARNING for each newer version it passed over;
2. keeps the installed version when it is still compatible and not older than
   that choice — a gateway upgrades only when a newer installable version
   appears, never because the manifest was re-read;
3. otherwise downloads ``<dist root>/<archive>``, checks its sha256 against
   the record and installs it into the shim cache atomically.

When the manifest cannot be fetched the last fetched copy is used, and when a
download fails the last good install is kept (marked ``stale``), the same
warm-cache behaviour a plain remote has.  A digest mismatch never installs.

Freshness guard: a manifest URL can answer 200 with an old copy (a CDN that
has not caught up with a publish).  When the URL that answered serves a
manifest whose ``generated_at`` is older than the cached one, or that lacks a
vendor the selectors expect (an explicit ``<source>:<vendor>``, a mapped
preset's package vendors, a moved vendor), the other URL is fetched once and
the manifest with the newer ``generated_at`` is used (the first on a tie).  A
cached manifest is never replaced by an older one: it is kept and marked
``stale`` until a fetch returns one at least as new.

Precedence (S3): core wins per shim — a shim name core already registers is
not loaded from the source, and the ignored names are logged with the source
and vendor; the vendor's other shims still install.  A vendor whose installable
record says ``promoted_to: core@X`` is not installed once this core is at
``X`` or later (a dev or pre-release of ``X`` is before it) and core really
registers the record's shims.

The release record is authoritative for what registers: an archive without a
``shims.yaml`` loads only the shim names its record lists.

Like any shim source, an installed vendor may ship a ``transforms.py`` that
runs on load: a manifest source is code execution at the trust level of the
repository that publishes it.
"""

from __future__ import annotations

import hashlib
import os
import json
import posixpath
import re
import shutil
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

from llm_rosetta.contracts.shim_manifest import (
    ShimRequirement,
    SourceManifest,
    VendorRelease,
    parse_manifest,
    promoted_core_version,
    requirement_problem,
    select_release,
    version_key,
)

from ._archive import _extract_archive
from ._cache import _atomic_replace, _file_lock, _read_stamp, _write_stamp
from ._legs import _auth_headers, _is_unreachable, _same_origin
from ._spec import _validate_url_scheme
from ._types import (
    HTTP_TIMEOUT,
    MAX_ARCHIVE_BYTES,
    HttpGet,
    ShimArchiveMismatch,
    ShimSourceError,
    ShimSourceStatus,
    ShimSourceUnreachable,
    logger,
)

#: The cache subdirectory manifest installs live under.
MANIFEST_CACHE_DIRNAME = "manifests"

_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_VENDOR_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*(:[a-z0-9][a-z0-9_.-]*)?$")


# ---------------------------------------------------------------------------
# The plan: which manifests, which vendors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstallSelector:
    """One ``shims.install`` entry, parsed.

    Attributes:
        raw: The entry as written.
        source: The ``<source>:`` qualifier, or ``None`` for every source.
        kind: ``"all"`` (``*``), ``"tag"`` (``tag:<t>``) or ``"vendor"``.
        value: The tag or the vendor name; ``""`` for ``"all"``.
        include: Shim-name patterns to keep from the selected vendors (only a
            legacy preset carries one, from ``shims.include``).
        exclude: Shim-name patterns to drop.
        origin: ``"install"``, or ``"presets:<key>"`` for a mapped preset.
        fallback_remote: The ``shims.remotes`` entry a mapped preset loaded
            before manifests existed.  Used only while the source has never
            answered (no manifest fetched, none cached), so a gateway is not
            left without the package before its source publishes one.
        expected: Vendors the source is expected to publish for this
            selector, for the freshness guard only (a mapped preset's package
            vendors); what installs is still decided by ``kind``/``value``.
    """

    raw: str
    source: str | None
    kind: Literal["all", "tag", "vendor"]
    value: str
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    origin: str = "install"
    fallback_remote: Mapping[str, Any] | None = None
    expected: tuple[str, ...] = ()

    def applies_to(self, source_id: str | None) -> bool:
        """Whether this selector reads the source with *source_id*."""
        return self.source is None or self.source == source_id

    def matches(self, vendor: str, tags: Collection[str]) -> bool:
        """Whether this selector picks *vendor* (whose newest record has *tags*)."""
        if self.kind == "all":
            return True
        if self.kind == "tag":
            return self.value in tags
        return self.value == vendor


def parse_selector(text: Any, **extra: Any) -> InstallSelector:
    """Parse one ``shims.install`` entry.

    Accepted spellings: ``<vendor>``, ``tag:<tag>``, ``*``, and each of them
    behind a ``<source>:`` qualifier (``argonne:alcf``, ``contrib:tag:tier:free``,
    ``argonne:*``).

    Args:
        text: The entry.
        **extra: Fields to set on the result (``include``, ``origin``, ...).

    Returns:
        The selector.

    Raises:
        ValueError: On a non-string or malformed entry.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("config: 'shims.install' entries must be non-empty strings")
    raw = text.strip()
    source: str | None = None
    rest = raw
    if not rest.startswith("tag:") and ":" in rest:
        source, rest = rest.split(":", 1)
        if not _SOURCE_ID_RE.match(source):
            raise ValueError(f"config: 'shims.install' entry {raw!r}: bad source id")
    if rest == "*":
        return InstallSelector(raw=raw, source=source, kind="all", value="", **extra)
    if rest.startswith("tag:"):
        tag = rest[4:]
        if not _TAG_RE.match(tag):
            raise ValueError(f"config: 'shims.install' entry {raw!r}: bad tag")
        return InstallSelector(raw=raw, source=source, kind="tag", value=tag, **extra)
    if not _VENDOR_RE.match(rest):
        raise ValueError(
            f"config: 'shims.install' entry {raw!r} is not a vendor name, "
            "'tag:<tag>', '*' or '<source>:' followed by one of them"
        )
    return InstallSelector(raw=raw, source=source, kind="vendor", value=rest, **extra)


@dataclass(frozen=True)
class ManifestSourceSpec:
    """One ``shims.sources`` entry, resolved to URLs.

    Attributes:
        key: The source id when the entry named a catalogue source; ``None``
            for a URL, whose id is the manifest's own ``source`` field.
        url: The primary manifest URL.
        fallback_url: A second manifest URL tried when the primary fails.
        token_env: Environment variable holding a bearer token, sent only to
            the primary URL's origin.
    """

    key: str | None
    url: str
    fallback_url: str | None = None
    token_env: str | None = None

    @property
    def label(self) -> str:
        """How logs name the source."""
        return self.key or self.url


@dataclass(frozen=True)
class ManifestPlan:
    """The manifest half of the ``shims`` section: sources and selectors."""

    sources: tuple[ManifestSourceSpec, ...] = ()
    selectors: tuple[InstallSelector, ...] = ()

    def __bool__(self) -> bool:
        """Whether anything is to be installed."""
        return bool(self.sources and self.selectors)


def source_spec_from_catalogue(source: Any) -> ManifestSourceSpec:
    """The spec for a catalogue :class:`~.catalog.ShimSource`."""
    return ManifestSourceSpec(
        key=source.key, url=source.manifest_url, fallback_url=source.fallback_url
    )


def parse_source_entry(entry: Any) -> ManifestSourceSpec:
    """Parse one ``shims.sources`` entry.

    A string is a catalogue id (``contrib``) or a manifest URL; an object
    carries ``url`` and optional ``fallback`` and ``token_env``.

    Raises:
        ValueError: On an unknown id, a malformed entry or a non-HTTPS URL.
    """
    from .catalog import SHIM_SOURCES, get_shim_source

    if isinstance(entry, str) and "://" not in entry:
        source = get_shim_source(entry)
        if source is None:
            known = ", ".join(s.key for s in SHIM_SOURCES)
            raise ValueError(
                f"config: unknown shim source {entry!r}; known sources: {known} "
                "(or give a manifest URL)"
            )
        return source_spec_from_catalogue(source)
    if isinstance(entry, str):
        entry = {"url": entry}
    if not isinstance(entry, Mapping):
        raise ValueError(
            "config: 'shims.sources' entries must be a source id, a manifest "
            "URL, or an object with 'url'"
        )
    unknown = set(entry) - {"url", "fallback", "token_env"}
    if unknown:
        raise ValueError(
            "config: unknown 'shims.sources' key(s): " + ", ".join(sorted(unknown))
        )
    url = entry.get("url")
    fallback = entry.get("fallback")
    token_env = entry.get("token_env")
    if not isinstance(url, str) or not url:
        raise ValueError("config: a 'shims.sources' object needs a non-empty 'url'")
    for value in (url, fallback):
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError("config: 'shims.sources' URLs must be strings")
        try:
            _validate_url_scheme(value)
        except ShimSourceError as exc:
            raise ValueError(f"config: 'shims.sources': {exc}") from None
    if token_env is not None and (not isinstance(token_env, str) or not token_env):
        raise ValueError("config: 'shims.sources[].token_env' must be a variable name")
    return ManifestSourceSpec(
        key=None, url=url, fallback_url=fallback or None, token_env=token_env
    )


# ---------------------------------------------------------------------------
# Installing
# ---------------------------------------------------------------------------


@dataclass
class _ManifestState:
    """What :func:`_load_manifest` learned about one source."""

    manifest: SourceManifest | None
    fetched_at: float | None = None
    stale: bool = False
    from_cache: bool = False
    leg_url: str | None = None
    error: str | None = None
    unreachable: bool = False
    stale_reason: str | None = None


def install_from_manifests(
    plan: ManifestPlan,
    *,
    cache_root: str | os.PathLike[str],
    http_get: HttpGet,
    now: Callable[[], float],
    ttl: int,
    proxy: str | None = None,
    force_refresh: bool = False,
    core_names: Collection[str] = (),
    core_version: str | None = None,
) -> tuple[list[ShimSourceStatus], list[Mapping[str, Any]]]:
    """Fetch every planned manifest and install the selected vendors.

    The registry is not touched: each returned status carries the directory
    to scan, and :func:`scan_manifest_status` loads it.

    Args:
        plan: The sources and the selectors.
        cache_root: The shim cache directory.
        http_get: ``(url, *, headers, timeout, proxy) -> bytes``.
        now: Clock.
        ttl: Seconds a fetched manifest is reused without a refetch.
        proxy: Upstream proxy URL.
        force_refresh: Refetch every manifest regardless of its age.
        core_names: The shim names core registers; a vendor shipping one is
            not downloaded (the scan step re-checks against the live set).
        core_version: The running core's version, for ``promoted_to``;
            :data:`llm_rosetta.api.__version__` by default.

    Returns:
        ``(statuses, fallback_remotes)``: one status per selected vendor (or
        per failed source), and the legacy remotes to load instead for mapped
        presets whose source has never answered.
    """
    if core_version is None:
        from llm_rosetta.api import __version__ as core_version
    statuses: list[ShimSourceStatus] = []
    fallbacks: list[Mapping[str, Any]] = []
    claimed: dict[str, str] = {}
    matched: set[str] = set()
    answered: list[tuple[str, ManifestSourceSpec]] = []
    fell_back: set[str] = set()
    for spec in plan.sources:
        root = _source_root(cache_root, spec)
        root.mkdir(parents=True, exist_ok=True)
        with _file_lock(root.parent / f"{root.name}.lock"):
            state = _load_manifest(
                spec,
                root,
                http_get=http_get,
                now=now,
                ttl=ttl,
                proxy=proxy,
                force=force_refresh,
                selectors=plan.selectors,
            )
            if state.manifest is None:
                statuses_for, fallback_for = _unanswered_source(spec, state, plan)
                statuses.extend(statuses_for)
                fallbacks.extend(fallback_for)
                fell_back.update(
                    s.raw
                    for s in plan.selectors
                    if s.fallback_remote and s.applies_to(spec.key)
                )
                continue
            source_id = spec.key or state.manifest.source
            answered.append((source_id, spec))
            for vendor, selector in _select_vendors(
                state.manifest, source_id, plan.selectors, claimed, matched, root
            ):
                status = _install_vendor(
                    spec,
                    source_id,
                    root,
                    state,
                    vendor,
                    selector,
                    http_get=http_get,
                    now=now,
                    proxy=proxy,
                    core_names=core_names,
                    core_version=core_version,
                )
                if state.stale_reason and status.ok and not status.skipped:
                    status = replace(
                        status, stale=True, error=status.error or state.stale_reason
                    )
                statuses.append(status)
    statuses.extend(
        _unmatched_rows(plan.selectors, matched, answered, fell_back, claimed)
    )
    return statuses, fallbacks


def _unmatched_rows(
    selectors: Sequence[InstallSelector],
    matched: set[str],
    answered: Sequence[tuple[str, ManifestSourceSpec]],
    fell_back: set[str],
    claimed: Mapping[str, str] | None = None,
) -> list[ShimSourceStatus]:
    """Warn about, and report, each selector that matched no vendor.

    A mapped legacy preset is no exception: a source that answers without the
    vendor (a staged publish, a renamed tag) must not drop the operator's
    shims silently.  Its old remote is *not* loaded in that case — the source
    answered, and an old remote carries code this core may not load.  The
    status row (``skipped="not in the manifest"``) is what the panel shows.
    A vendor the manifest lists but whose every record was invalid is not
    unmatched: :func:`_install_vendor` reports it as ``no valid release``.

    A selector the gateway added for a vendor that moved out of core
    (``origin="moved:<vendor>"``) is satisfied when any selector installed
    that vendor: a declared tag or ``*`` selector may cover it first.
    """
    rows: list[ShimSourceStatus] = []
    for selector in selectors:
        if selector.raw in matched or selector.raw in fell_back:
            continue
        if selector.origin.startswith("moved:") and selector.value in (claimed or {}):
            continue
        logger.warning(
            "Shim install %r (%s) matched no vendor in the reachable sources",
            selector.raw,
            selector.origin,
        )
        target = next((a for a in answered if selector.applies_to(a[0])), None)
        if target is None:
            continue
        source_id, spec = target
        rows.append(
            ShimSourceStatus(
                kind="remote",
                location=f"{spec.url}#{selector.raw}",
                name=selector.raw if selector.source else f"{source_id}:{selector.raw}",
                source=source_id,
                vendor=selector.value if selector.kind == "vendor" else None,
                skipped="not in the manifest",
            )
        )
    return rows


def _source_root(cache_root: str | os.PathLike[str], spec: ManifestSourceSpec) -> Path:
    digest = hashlib.sha256(spec.url.encode()).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", spec.key or "source").strip("-.")
    return Path(cache_root) / MANIFEST_CACHE_DIRNAME / f"{slug or 'source'}-{digest}"


def _unanswered_source(
    spec: ManifestSourceSpec, state: _ManifestState, plan: ManifestPlan
) -> tuple[list[ShimSourceStatus], list[Mapping[str, Any]]]:
    """Statuses and fallbacks for a source with no manifest at all.

    A mapped preset falls back to the remote it used before manifests: the
    source has never answered, so there is no install to keep, and leaving the
    package out would break a config that worked yesterday.
    """
    mine = [s for s in plan.selectors if s.applies_to(spec.key)]
    fallbacks = [s.fallback_remote for s in mine if s.fallback_remote]
    for selector in mine:
        if selector.fallback_remote:
            logger.warning(
                "Shim source %s: no manifest (%s); %s loads its legacy remote %s",
                spec.label,
                state.error,
                selector.origin,
                selector.fallback_remote.get("url"),
            )
    if fallbacks and all(s.fallback_remote for s in mine):
        return [], fallbacks
    status = ShimSourceStatus(
        kind="remote",
        location=spec.url,
        name=spec.key,
        ok=False,
        error=state.error or "manifest unavailable",
        unreachable=state.unreachable,
        source=spec.key,
    )
    return [status], fallbacks


def _select_vendors(
    manifest: SourceManifest,
    source_id: str,
    selectors: Sequence[InstallSelector],
    claimed: dict[str, str],
    matched: set[str],
    root: Path | None = None,
) -> list[tuple[str, InstallSelector]]:
    """The vendors of *manifest* the selectors pick, each with its selector.

    The first selector that matches a vendor speaks for it.  A vendor an
    earlier source already provided is not installed twice: the first source
    in ``shims.sources`` order wins, and the other is logged.

    A vendor the manifest lists but whose every record was invalid
    (``manifest.dropped``) is selected too, its tags read from the install
    stamp under *root*: :func:`_install_vendor` then keeps the last good
    install rather than letting the vendor vanish because a record went bad.
    """
    out: list[tuple[str, InstallSelector]] = []
    for vendor in sorted(set(manifest.vendors) | set(manifest.dropped)):
        releases = manifest.vendors.get(vendor, ())
        # The record that would be installed speaks for the tags: one newer
        # release this core cannot run must not drop the vendor from a tag.
        installable, _ = select_release(releases)
        record = installable or (releases[0] if releases else None)
        if record is None and root is not None:
            record = _installed_release(
                _read_stamp(root / f"{vendor}.stamp.json"), vendor
            )
        tags = record.tags if record is not None else ()
        selector = next(
            (
                s
                for s in selectors
                if s.applies_to(source_id) and s.matches(vendor, tags)
            ),
            None,
        )
        if selector is None:
            continue
        matched.add(selector.raw)
        if vendor in claimed:
            logger.warning(
                "Shim source %s: vendor %r is also in %s, which comes first in "
                "shims.sources; this source's copy is ignored",
                source_id,
                vendor,
                claimed[vendor],
            )
            continue
        claimed[vendor] = source_id
        out.append((vendor, selector))
    return out


def _narrow(
    patterns: Sequence[str], shims: Sequence[str], selector: InstallSelector
) -> tuple[str, ...]:
    """The *patterns* of *selector* this vendor's scan should apply.

    A selector that names one vendor hands its patterns over whole, so a
    pattern that matches nothing is reported by the scan as before.  A tag or
    ``*`` selector spans several vendors (the legacy ``free`` preset), so each
    vendor receives only the patterns that can match one of its own shims;
    otherwise every other vendor would report them as unmatched.
    """
    if selector.kind == "vendor":
        return tuple(patterns)
    return tuple(p for p in patterns if any(fnmatch(s, p) for s in shims))


def _install_vendor(
    spec: ManifestSourceSpec,
    source_id: str,
    root: Path,
    state: _ManifestState,
    vendor: str,
    selector: InstallSelector,
    *,
    http_get: HttpGet,
    now: Callable[[], float],
    proxy: str | None,
    core_names: Collection[str],
    core_version: str,
) -> ShimSourceStatus:
    """Decide on one vendor and install it when needed."""
    assert state.manifest is not None
    releases = state.manifest.vendors.get(vendor, ())
    dropped = state.manifest.dropped.get(vendor)
    base = ShimSourceStatus(
        kind="remote",
        location=f"{spec.url}#{vendor}",
        name=f"{source_id}:{vendor}",
        source=source_id,
        vendor=vendor,
        fetched_at=state.fetched_at,
        include=selector.include,
        exclude=selector.exclude,
    )

    chosen, passed_over = select_release(releases)
    for release, why in passed_over:
        logger.warning(
            "Shim source %s: %s %s is not installable here: %s",
            source_id,
            vendor,
            release.version,
            why,
        )

    if chosen is not None:
        skip = _skip_reason(chosen, selector, source_id, core_names, core_version)
        if skip:
            return replace(base, skipped=skip)

    vendor_dir = root / vendor
    stamp_path = root / f"{vendor}.stamp.json"
    stamp = _read_stamp(stamp_path)
    installed = _installed_release(stamp, vendor) if vendor_dir.is_dir() else None
    usable = installed is not None and requirement_problem(installed.requires) is None
    if installed is not None and installed.version not in {r.version for r in releases}:
        logger.warning(
            "Shim source %s: manifest no longer lists %s %s; keeping the install",
            source_id,
            vendor,
            installed.version,
        )

    if dropped is not None:
        # Every record of the vendor is invalid: keep the last good install
        # (stale, like an unreachable source) and never uninstall for it.
        why = f"no valid release: {dropped}"
        logger.warning(
            "Shim source %s: vendor %r has %s; %s",
            source_id,
            vendor,
            why,
            f"keeping the installed {installed.version}"
            if usable and installed is not None
            else "it is not installed",
        )
        if usable:
            assert installed is not None
            status = _installed_status(
                base, vendor_dir, installed, stamp, state, selector
            )
            return replace(status, stale=True, error=why)
        return replace(base, skipped=why)

    if chosen is None:
        if usable:
            assert installed is not None
            return _installed_status(
                base, vendor_dir, installed, stamp, state, selector
            )
        reason = "; ".join(why for _, why in passed_over) or "no release"
        return replace(
            base,
            ok=False,
            error=f"no version of {vendor} is installable on this core ({reason})",
        )

    if usable and installed is not None:
        if version_key(installed.version) >= version_key(chosen.version):
            return _installed_status(
                base, vendor_dir, installed, stamp, state, selector
            )

    try:
        archive_url = _download_release(
            spec, state, chosen, vendor_dir, http_get=http_get, proxy=proxy
        )
    except ShimSourceError as exc:
        mismatch = isinstance(exc, ShimArchiveMismatch)
        logger.warning(
            "Shim source %s: cannot install %s %s: %s%s",
            source_id,
            vendor,
            chosen.version,
            exc,
            f"; keeping {installed.version}" if usable and installed else "",
        )
        if usable and installed is not None:
            status = _installed_status(
                base, vendor_dir, installed, stamp, state, selector
            )
            return replace(
                status,
                stale=True,
                error=str(exc),
                checksum_ok=False if mismatch else None,
            )
        return replace(
            base,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            unreachable=isinstance(exc, ShimSourceUnreachable),
            checksum_ok=False if mismatch else None,
        )

    record = {
        **chosen.to_dict(),
        "vendor": vendor,
        "source": source_id,
        "archive_url": archive_url,
        "installed_at": now(),
    }
    _write_stamp(stamp_path, record)
    if installed is not None:
        logger.info(
            "Shim source %s: %s upgraded %s -> %s",
            source_id,
            vendor,
            installed.version,
            chosen.version,
        )
    return replace(
        base,
        resolved_dir=str(_scan_root(vendor_dir)),
        resolved_ref=chosen.commit,
        checksum_ok=True,
        archive_url=archive_url,
        leg="primary" if archive_url.startswith(_dist_root(spec.url)) else "mirror",
        release=chosen.to_dict(),
        include=_narrow(selector.include, chosen.shims, selector),
        exclude=_narrow(selector.exclude, chosen.shims, selector),
    )


def _skip_reason(
    chosen: VendorRelease,
    selector: InstallSelector,
    source_id: str,
    core_names: Collection[str],
    core_version: str,
) -> str | None:
    """Why the chosen release is not installed at all, or ``None``.

    A promotion this core honours, a record every shim of which core already
    registers, or a legacy include that names none of the vendor's shims.  A
    *partial* clash is not a reason: the vendor installs and the scan refuses
    the core names one by one (core wins per shim).
    """
    promoted = _promotion(chosen, core_version, core_names, source_id)
    if promoted:
        return f"promoted to {promoted}"
    clash = sorted(set(chosen.shims) & set(core_names))
    if clash and set(clash) == set(chosen.shims):
        logger.warning(
            "Shim source %s: vendor %r is ignored; core already provides %s",
            source_id,
            chosen.vendor,
            ", ".join(clash),
        )
        return "core provides " + ", ".join(clash)
    if clash:
        logger.warning(
            "Shim source %s: vendor %r: core already provides %s; those "
            "shims are ignored and the rest install",
            source_id,
            chosen.vendor,
            ", ".join(clash),
        )
    if (
        selector.kind != "vendor"
        and selector.include
        and not _narrow(selector.include, chosen.shims, selector)
    ):
        # A legacy include spans several vendors and names none of this
        # one's shims: the vendor is not part of the selection at all.
        return "not selected by include"
    return None


def _installed_status(
    base: ShimSourceStatus,
    vendor_dir: Path,
    installed: VendorRelease,
    stamp: Mapping[str, Any],
    state: _ManifestState,
    selector: InstallSelector,
) -> ShimSourceStatus:
    """The status for a vendor served from its existing install."""
    archive_url = stamp.get("archive_url")
    return replace(
        base,
        resolved_dir=str(_scan_root(vendor_dir)),
        resolved_ref=installed.commit,
        from_cache=True,
        stale=state.stale,
        checksum_ok=True,
        archive_url=archive_url,
        release=installed.to_dict(),
        include=_narrow(selector.include, installed.shims, selector),
        exclude=_narrow(selector.exclude, installed.shims, selector),
    )


def _core_version_key(version: str) -> tuple[int, int, int, int]:
    """A PEP 440-aware sort key for a core version: ``(major, minor, patch, phase)``.

    The phase orders the releases of one ``X.Y.Z``: a dev release (0) before
    a pre-release (1) before the final release (2) before a post release (3),
    so ``0.14.0.dev35`` has not reached ``0.14.0``.
    """
    match = re.match(r"^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(.*)$", version.strip())
    if match is None:
        return (0, 0, 0, 0)
    major, minor, patch, rest = match.groups()
    rest = rest.lower()
    if "post" in rest:
        phase = 3
    elif "dev" in rest:
        phase = 0
    elif re.search(r"(a|b|c|rc|alpha|beta|pre|preview)\d*", rest):
        phase = 1
    else:
        phase = 2
    return int(major), int(minor or 0), int(patch or 0), phase


def _promotion(
    release: VendorRelease,
    core_version: str,
    core_names: Collection[str],
    source_id: str,
) -> str | None:
    """The ``promoted_to`` claim of *release*, when this core honours it.

    The claim is read from the record that would be installed, so a later
    release without it un-promotes the vendor.  It holds only when this core
    is at or after the named version and core registers every shim the record
    lists; a claim that is due but not borne out is logged and ignored.
    """
    target = promoted_core_version(release)
    if target is None:
        return None
    if _core_version_key(core_version) < (*target, 2):
        return None
    missing = sorted(set(release.shims) - set(core_names))
    if missing:
        logger.warning(
            "Shim source %s: %s %s says promoted_to %s, but this core does not "
            "provide %s; installing from the source",
            source_id,
            release.vendor,
            release.version,
            release.promoted_to,
            ", ".join(missing),
        )
        return None
    logger.info(
        "Shim source %s: %s %s says promoted_to %s; core provides it",
        source_id,
        release.vendor,
        release.version,
        release.promoted_to,
    )
    return release.promoted_to


def _installed_release(stamp: Mapping[str, Any], vendor: str) -> VendorRelease | None:
    """The release an install stamp records, or ``None`` when unreadable."""
    try:
        requires = stamp["requires"]
        return VendorRelease(
            vendor=vendor,
            version=str(stamp["version"]),
            commit=str(stamp["commit"]),
            archive=str(stamp["archive"]),
            sha256=str(stamp["sha256"]),
            requires=ShimRequirement(int(requires["surface"]), int(requires["ir"])),
            shims=tuple(stamp.get("shims") or ()),
            maturity=str(stamp.get("maturity") or ""),
            tags=tuple(stamp.get("tags") or ()),
            since=stamp.get("since"),
            conformance=stamp.get("conformance"),
            promoted_to=stamp.get("promoted_to"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _scan_root(vendor_dir: Path) -> Path:
    """The directory to scan inside an installed vendor archive.

    The archive holds ``providers/<vendor>/...``; an archive laid out from the
    vendor directory itself is scanned from its root.
    """
    providers = vendor_dir / "providers"
    return providers if providers.is_dir() else vendor_dir


def _dist_root(manifest_url: str) -> str:
    """The URL prefix a manifest's relative archive paths resolve against."""
    return manifest_url.rsplit("/", 1)[0] + "/"


def _manifest_legs(spec: ManifestSourceSpec, first: str | None) -> list[str]:
    legs = [spec.url] + ([spec.fallback_url] if spec.fallback_url else [])
    if first in legs and legs[0] != first:
        legs.remove(first)
        legs.insert(0, first)
    return legs


def _headers_for(spec: ManifestSourceSpec, url: str) -> dict[str, str]:
    """Request headers; the token rides only to the primary URL's origin."""
    token_env = spec.token_env if _same_origin(url, spec.url) else None
    return _auth_headers(token_env)


def _load_manifest(
    spec: ManifestSourceSpec,
    root: Path,
    *,
    http_get: HttpGet,
    now: Callable[[], float],
    ttl: int,
    proxy: str | None,
    force: bool,
    selectors: Sequence[InstallSelector] = (),
) -> _ManifestState:
    """Return the source's manifest: cached while fresh, else fetched.

    A fetch that fails on every URL — unreachable, an HTTP error, or a body
    that is not a valid manifest — falls back to the last fetched copy, marked
    stale.

    A URL that answers is checked for freshness (:func:`_freshness_doubt`):
    when its manifest is older than the cached one or lacks a vendor the
    selectors expect, the next URL is fetched once and the newer of the two
    is used.  Whatever URL served it, a manifest older than the cached one
    never replaces it: the cached copy is used, marked stale, and the stamp
    records why, so every later apply refetches (TTL or not) and the mark
    stays until a fetch returns a manifest at least as new as the cache.

    A cached copy within the TTL that lacks an expected vendor is refetched
    rather than trusted.  A fetch that still ends without that vendor starts
    a :data:`VENDOR_RECHECK_SECONDS` cooldown for that set of vendors, so a
    selector naming a vendor no source publishes costs at most one check per
    window, not one per apply; *force* ignores the cooldown.

    The URL that served last is tried first while its stamp is within the
    TTL; after that the primary URL is tried first again.

    Args:
        spec: The source.
        root: The source's cache directory.
        http_get: ``(url, *, headers, timeout, proxy) -> bytes``.
        now: Clock.
        ttl: Seconds a fetched manifest is reused without a refetch.
        proxy: Upstream proxy URL.
        force: Refetch regardless of the cached copy's age or a cooldown.
        selectors: The install selectors; see :func:`_expected_vendors`.
    """
    doc_path = root / "manifest.json"
    stamp_path = root / "manifest.stamp.json"
    stamp = _read_stamp(stamp_path)
    cached = _read_cached_manifest(doc_path, spec)
    fetched_at = stamp.get("fetched_at")
    known_at = fetched_at if isinstance(fetched_at, (int, float)) else None
    within_ttl = known_at is not None and now() - known_at <= ttl
    if cached is not None and not force and within_ttl:
        missing = _missing_vendors(cached, spec, selectors)
        if not stamp.get("stale_reason") and (
            not missing or _cooling(stamp, missing, now())
        ):
            return _ManifestState(
                cached, fetched_at=known_at, from_cache=True, leg_url=stamp.get("url")
            )

    failures: list[str] = []
    all_unreachable = True
    legs = _manifest_legs(spec, stamp.get("url") if within_ttl else None)
    for index, url in enumerate(legs):
        try:
            document, manifest = _fetch_leg(spec, url, http_get=http_get, proxy=proxy)
        except Exception as exc:  # noqa: BLE001 - every leg failure is reported
            all_unreachable = all_unreachable and _is_unreachable(exc)
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            continue
        cooling = not force and _cooling(
            stamp, _missing_vendors(manifest, spec, selectors), now()
        )
        doubt = _freshness_doubt(
            manifest, cached, spec, selectors, check_missing=not cooling
        )
        if doubt and index + 1 < len(legs):
            url, document, manifest = _second_opinion(
                spec,
                (url, document, manifest),
                legs[index + 1],
                doubt,
                http_get=http_get,
                proxy=proxy,
            )
        if spec.key and manifest.source != spec.key:
            logger.warning(
                "Shim source %s: the manifest at %s calls itself %r",
                spec.key,
                url,
                manifest.source,
            )
        if cached is not None and _older(manifest, cached):
            return _keep_newer_cache(
                spec, url, manifest, cached, stamp, stamp_path, known_at
            )
        tmp = doc_path.with_name(f"manifest.json.tmp-{uuid.uuid4().hex}")
        tmp.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        tmp.replace(doc_path)
        when = now()
        record: dict[str, Any] = {"fetched_at": when, "url": url}
        missing = _missing_vendors(manifest, spec, selectors)
        if missing:
            record["vendor_check"] = (
                stamp["vendor_check"] if cooling else {"missing": missing, "at": when}
            )
        _write_stamp(stamp_path, record)
        return _ManifestState(manifest, fetched_at=when, leg_url=url)

    error = "; ".join(failures)
    if cached is not None:
        logger.warning(
            "Shim source %s: manifest fetch failed (%s); using the cached manifest",
            spec.label,
            error,
        )
        return _ManifestState(
            cached,
            fetched_at=known_at,
            stale=True,
            from_cache=True,
            leg_url=stamp.get("url"),
            error=error,
        )
    logger.warning("Shim source %s: manifest unavailable: %s", spec.label, error)
    return _ManifestState(None, error=error, unreachable=all_unreachable)


#: Seconds a vendor-presence check that came back empty is not repeated for
#: the same source and vendors (see :func:`_load_manifest`).
VENDOR_RECHECK_SECONDS = 600


def _read_cached_manifest(
    doc_path: Path, spec: ManifestSourceSpec
) -> SourceManifest | None:
    """The cached manifest, or ``None`` when there is none or it is unusable."""
    if not doc_path.is_file():
        return None
    try:
        return _read_manifest(
            json.loads(doc_path.read_text(encoding="utf-8")), spec, warn=False
        )
    except (OSError, ValueError) as exc:
        logger.warning("Shim source %s: cached manifest unusable: %s", spec.label, exc)
        return None


def _cooling(stamp: Mapping[str, Any], missing: Sequence[str], when: float) -> bool:
    """Whether a recent check already found *missing* absent (the cooldown)."""
    record = stamp.get("vendor_check")
    if not missing or not isinstance(record, Mapping):
        return False
    at, seen = record.get("at"), record.get("missing")
    if not isinstance(at, (int, float)) or not isinstance(seen, list):
        return False
    return set(missing) <= set(seen) and 0 <= when - at < VENDOR_RECHECK_SECONDS


def _keep_newer_cache(
    spec: ManifestSourceSpec,
    url: str,
    manifest: SourceManifest,
    cached: SourceManifest,
    stamp: Mapping[str, Any],
    stamp_path: Path,
    known_at: float | None,
) -> _ManifestState:
    """Keep *cached* over the older *manifest*, and make the next apply ask again.

    The stamp keeps its time and URL but records the reason, which bypasses
    the TTL: the source stays ``stale`` until a fetch returns a manifest at
    least as new as the cache.
    """
    reason = (
        f"{url} served a manifest generated_at {manifest.generated_at}, "
        f"older than the cached one (generated_at {cached.generated_at}); "
        "keeping the cached manifest"
    )
    logger.warning("Shim source %s: %s", spec.label, reason)
    _write_stamp(stamp_path, {**stamp, "stale_reason": reason})
    return _ManifestState(
        cached,
        fetched_at=known_at,
        stale=True,
        from_cache=True,
        leg_url=stamp.get("url"),
        error=reason,
        stale_reason=reason,
    )


def _fetch_leg(
    spec: ManifestSourceSpec, url: str, *, http_get: HttpGet, proxy: str | None
) -> tuple[Any, SourceManifest]:
    """Fetch and parse the manifest at *url*: ``(document, manifest)``."""
    data = http_get(
        url, headers=_headers_for(spec, url), timeout=HTTP_TIMEOUT, proxy=proxy
    )
    document = json.loads(bytes(data).decode("utf-8"))
    return document, _read_manifest(document, spec)


_RFC3339_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[Tt ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?"
    r"(Z|z|[+-]\d{2}:\d{2})$"
)


def _generated_at(manifest: SourceManifest) -> datetime | None:
    """The manifest's ``generated_at`` as an aware UTC datetime, or ``None``.

    Parsed by hand: ``datetime.fromisoformat`` on Python 3.10 refuses ``Z``
    and fractions that are not 3 or 6 digits, both of which RFC 3339 allows.
    """
    match = _RFC3339_RE.match(str(manifest.generated_at).strip())
    if match is None:
        return None
    day, clock, fraction, offset = match.groups()
    micros = (fraction or "0")[:6].ljust(6, "0")
    offset = "+00:00" if offset in ("Z", "z") else offset
    try:
        moment = datetime.fromisoformat(f"{day}T{clock}.{micros}{offset}")
    except ValueError:
        return None
    return moment.astimezone(timezone.utc)


def _older(manifest: SourceManifest, than: SourceManifest) -> bool:
    """Whether *manifest* was generated before *than* (an unknown time: no)."""
    mine, theirs = _generated_at(manifest), _generated_at(than)
    return mine is not None and theirs is not None and mine < theirs


def _expected_vendors(selectors: Sequence[InstallSelector], source_id: str) -> set[str]:
    """The vendors the selectors expect this source to publish.

    An explicit ``<source>:<vendor>`` (a moved vendor's auto-install is one)
    expects its vendor, and a mapped legacy preset expects the vendors its
    package shipped (:attr:`InstallSelector.expected`), even when it installs
    by tag.  A plain tag, ``*`` or an unqualified selector expects nothing:
    another source may serve it, so its absence here proves nothing.
    """
    out: set[str] = set()
    for s in selectors:
        if s.source != source_id:
            continue
        if s.kind == "vendor":
            out.add(s.value)
        out.update(s.expected)
    return out


def _missing_vendors(
    manifest: SourceManifest,
    spec: ManifestSourceSpec,
    selectors: Sequence[InstallSelector],
) -> list[str]:
    """The expected vendors (:func:`_expected_vendors`) *manifest* lacks.

    A vendor listed only in ``manifest.dropped`` (every record invalid) counts
    as lacking: an old copy can carry a record a newer producer has fixed.
    """
    expected = _expected_vendors(selectors, spec.key or manifest.source)
    return sorted(expected - set(manifest.vendors))


def _freshness_doubt(
    manifest: SourceManifest,
    cached: SourceManifest | None,
    spec: ManifestSourceSpec,
    selectors: Sequence[InstallSelector],
    *,
    check_missing: bool = True,
) -> str | None:
    """Why a manifest that answered may be an old copy, or ``None``.

    Args:
        check_missing: Whether a missing expected vendor counts; ``False``
            while a recent check already found the same vendors absent.
    """
    if cached is not None and _older(manifest, cached):
        return (
            f"its generated_at {manifest.generated_at} is older than the cached "
            f"{cached.generated_at}"
        )
    missing = _missing_vendors(manifest, spec, selectors) if check_missing else []
    if missing:
        return "it does not list " + ", ".join(missing)
    return None


def _second_opinion(
    spec: ManifestSourceSpec,
    first: tuple[str, Any, SourceManifest],
    other_url: str,
    doubt: str,
    *,
    http_get: HttpGet,
    proxy: str | None,
) -> tuple[str, Any, SourceManifest]:
    """Fetch *other_url* once and return the newer of the two manifests.

    The first manifest wins a tie, an unknown ``generated_at`` and a failed
    second fetch (logged as a WARNING) — even when it is the copy that lacks
    the expected vendor: only a newer ``generated_at`` replaces it, because
    two copies of one generation that disagree say nothing about which is
    current.
    """
    url, _, manifest = first
    try:
        other = _fetch_leg(spec, other_url, http_get=http_get, proxy=proxy)
    except Exception as exc:  # noqa: BLE001 - the first answer still stands
        logger.warning(
            "Shim source %s: %s answered but %s; %s failed (%s: %s); using %s",
            spec.label,
            url,
            doubt,
            other_url,
            type(exc).__name__,
            exc,
            url,
        )
        return first
    if _older(manifest, other[1]):
        logger.info(
            "Shim source %s: using %s (generated_at %s): %s answered but %s, "
            "and served generated_at %s",
            spec.label,
            other_url,
            other[1].generated_at,
            url,
            doubt,
            manifest.generated_at,
        )
        return other_url, other[0], other[1]
    logger.info(
        "Shim source %s: using %s (generated_at %s): it answered but %s, and "
        "%s is not newer (generated_at %s)",
        spec.label,
        url,
        manifest.generated_at,
        doubt,
        other_url,
        other[1].generated_at,
    )
    return first


def _download_release(
    spec: ManifestSourceSpec,
    state: _ManifestState,
    release: VendorRelease,
    vendor_dir: Path,
    *,
    http_get: HttpGet,
    proxy: str | None,
) -> str:
    """Download, verify and install *release* into *vendor_dir*.

    The archive is resolved against the dist root of each manifest URL, the
    one that served the manifest first.  A digest mismatch on one URL moves on
    to the next (a stale CDN copy), but never installs.

    Returns:
        The archive URL that was installed.

    Raises:
        ShimArchiveMismatch: Every URL that answered served other bytes.
        ShimSourceUnreachable: No URL answered.
        ShimSourceError: Another failure (HTTP error, unusable archive).
    """
    failures: list[str] = []
    mismatch = False
    all_unreachable = True
    for manifest_url in _manifest_legs(spec, state.leg_url):
        url = _dist_root(manifest_url) + posixpath.normpath(release.archive)
        try:
            data = bytes(
                http_get(
                    url,
                    headers=_headers_for(spec, url),
                    timeout=HTTP_TIMEOUT,
                    proxy=proxy,
                )
            )
        except Exception as exc:  # noqa: BLE001 - the next URL is tried
            all_unreachable = all_unreachable and _is_unreachable(exc)
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            continue
        all_unreachable = False
        if len(data) > MAX_ARCHIVE_BYTES:
            failures.append(f"{url}: archive exceeds {MAX_ARCHIVE_BYTES} bytes")
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest != release.sha256:
            mismatch = True
            failures.append(
                f"{url}: sha256 mismatch: expected {release.sha256}, got {digest}"
            )
            continue
        try:
            _install_tree(data, vendor_dir, url)
        except ShimSourceError as exc:
            failures.append(f"{url}: {exc}")
            continue
        return url
    detail = "; ".join(failures)
    if mismatch:
        raise ShimArchiveMismatch(detail)
    if all_unreachable:
        raise ShimSourceUnreachable(detail)
    raise ShimSourceError(detail)


def _install_tree(data: bytes, vendor_dir: Path, url: str) -> None:
    """Extract *data* next to *vendor_dir* and swap it in atomically."""
    tmp_root = vendor_dir.parent / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_root / f"extract-{uuid.uuid4().hex}"
    try:
        tmp_dir.mkdir()
        # ``providers`` as the declared path keeps a ``providers/`` root from
        # being stripped as the archive's single top-level directory.
        _extract_archive(
            data, tmp_dir, archive="tar.gz", url=url, declared_path="providers"
        )
        _atomic_replace(tmp_dir, vendor_dir)
    except OSError as exc:
        raise ShimSourceError(f"cannot install locally: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Scanning (inside the registry rebuild)
# ---------------------------------------------------------------------------


def scan_manifest_status(
    status: ShimSourceStatus, core_names: Collection[str]
) -> ShimSourceStatus:
    """Load one installed vendor into the registry being built.

    *core_names* is the authoritative set (built-ins and plugins, just
    registered): those names are refused by name (core wins) and the vendor's
    other shims load; a vendor all of whose shims core registers is skipped.
    When the archive ships no ``shims.yaml``, only the names its release record
    lists may register, and any other is refused with a WARNING.
    """
    from ._scan import _scan_dir
    from .manifest import MANIFEST_FILENAME

    if not status.ok or status.skipped or not status.resolved_dir:
        return status
    record_shims = (status.release or {}).get("shims") or ()
    clash = sorted(set(record_shims) & set(core_names))
    if clash and set(clash) == set(record_shims):
        logger.warning(
            "Shim source %s: vendor %r is ignored; core already provides %s",
            status.source,
            status.vendor,
            ", ".join(clash),
        )
        return replace(status, skipped="core provides " + ", ".join(clash))
    scanned = _scan_dir(
        Path(status.resolved_dir),
        kind="remote",
        location=status.location,
        name=status.name,
        from_cache=status.from_cache,
        stale=status.stale,
        fetched_at=status.fetched_at,
        resolved_ref=status.resolved_ref,
        include=status.include,
        exclude=status.exclude,
        refuse=core_names,
        only=(
            None
            if (Path(status.resolved_dir) / MANIFEST_FILENAME).is_file()
            else frozenset(record_shims)
        ),
    )
    return replace(
        scanned,
        leg=status.leg,
        archive_url=status.archive_url,
        checksum_ok=status.checksum_ok,
        error=scanned.error or status.error,
        source=status.source,
        vendor=status.vendor,
        release=status.release,
    )


def check_manifest_updates(
    statuses: Sequence[ShimSourceStatus],
    *,
    http_get: HttpGet,
    proxy: str | None = None,
) -> list[dict[str, Any]]:
    """Compare each installed vendor with its manifest's newest installable version.

    One manifest fetch per source.  A vendor counts as having an update only
    when a newer version is installable on this core — the rule the install
    itself follows.

    Returns:
        One entry per vendor status: ``location``, ``current`` and ``head``
        (versions), ``update`` and ``error``.
    """
    from .catalog import SHIM_SOURCES

    fallbacks = {s.manifest_url: s.fallback_url for s in SHIM_SOURCES}
    manifests: dict[str, SourceManifest | str] = {}
    out: list[dict[str, Any]] = []
    for status in statuses:
        current = (status.release or {}).get("version")
        entry: dict[str, Any] = {
            "location": status.location,
            "current": current,
            "head": None,
            "update": False,
            "error": None,
        }
        url = status.location.split("#", 1)[0]
        if url not in manifests:
            spec = ManifestSourceSpec(
                key=None, url=url, fallback_url=fallbacks.get(url)
            )
            manifests[url] = _fetch_manifest_once(spec, http_get=http_get, proxy=proxy)
        manifest = manifests[url]
        if isinstance(manifest, str):
            entry["error"] = manifest
        elif status.vendor not in manifest.vendors:
            entry["error"] = f"{status.vendor} is no longer in the manifest"
        else:
            chosen, _ = select_release(manifest.vendors[status.vendor])
            if chosen is None:
                entry["error"] = "no version is installable on this core"
            else:
                entry["head"] = chosen.version
                entry["update"] = bool(current) and version_key(
                    chosen.version
                ) > version_key(str(current))
        out.append(entry)
    return out


def _read_manifest(
    document: Any, spec: ManifestSourceSpec, *, warn: bool = True
) -> SourceManifest:
    """Parse a published manifest record by record.

    A broken top level fails the source (:class:`ShimManifestInvalid`, a
    ``ValueError``).  An invalid vendor entry or release record is left out
    with one WARNING naming the vendor, the version and the JSON-path reason,
    and the rest of the manifest is used: a record written by a newer
    producer must not take every other vendor of the source down with it,
    and selection only ever sees valid records.

    Args:
        document: The parsed JSON value.
        spec: The source, for the log line.
        warn: Whether to log the skipped records; the cached copy and the
            update check re-read a document whose fetch already logged them.
    """
    manifest = parse_manifest(document, invalid_records="skip")
    if warn:
        for line in manifest.skipped:
            logger.warning("Shim source %s: manifest record %s", spec.label, line)
    return manifest


def _fetch_manifest_once(
    spec: ManifestSourceSpec, *, http_get: HttpGet, proxy: str | None
) -> SourceManifest | str:
    """Fetch and parse a manifest without the cache; an error is returned as text."""
    failures = []
    for url in _manifest_legs(spec, None):
        try:
            data = http_get(
                url, headers=_headers_for(spec, url), timeout=HTTP_TIMEOUT, proxy=proxy
            )
            return _read_manifest(
                json.loads(bytes(data).decode("utf-8")), spec, warn=False
            )
        except Exception as exc:  # noqa: BLE001 - a check must not break the page
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
    return "; ".join(failures)


__all__ = [
    "InstallSelector",
    "ManifestPlan",
    "ManifestSourceSpec",
    "check_manifest_updates",
    "install_from_manifests",
    "parse_selector",
    "parse_source_entry",
    "scan_manifest_status",
    "source_spec_from_catalogue",
]
