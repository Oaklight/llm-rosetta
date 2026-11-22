"""Freshness: fetching a source's manifest and keeping it trustworthy.

Fetch order, the warm cache, and the stale-CDN guard (second opinion from the
other URL when the first answered with an old or incomplete copy).
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_rosetta.contracts.shim_manifest import (
    SourceManifest,
    parse_manifest,
)

from .._cache import _read_stamp, _write_stamp
from .._legs import _is_unreachable
from .._types import (
    HTTP_TIMEOUT,
    HttpGet,
    logger,
)
from .plan import (
    InstallSelector,
    ManifestSourceSpec,
    _headers_for,
    _manifest_legs,
)


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
    expects its vendor.  A plain tag, ``*`` or an unqualified selector expects
    nothing: another source may serve it, so its absence here proves nothing.
    """
    out: set[str] = set()
    for s in selectors:
        if s.source != source_id:
            continue
        if s.kind == "vendor" and not s.negate:
            out.add(s.value)
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
