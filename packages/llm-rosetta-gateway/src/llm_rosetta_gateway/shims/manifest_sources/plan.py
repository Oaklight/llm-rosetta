"""The plan: which manifests to read and which vendors to take from them.

Parses ``shims.install`` entries and ``shims.sources`` entries, and locates a
source's cache directory, dist root and manifest URLs.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .._legs import _auth_headers, _same_origin
from .._spec import _validate_url_scheme
from .._types import (
    ShimSourceError,
)

#: The cache subdirectory manifest installs live under.
MANIFEST_CACHE_DIRNAME = "manifests"


_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


_VENDOR_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*(:[a-z0-9][a-z0-9_.-]*)?$")


_VERSION_RE = re.compile(r"^[0-9][0-9A-Za-z.+-]*$")


@dataclass(frozen=True)
class InstallSelector:
    """One ``shims.install`` entry, parsed.

    Attributes:
        raw: The entry as written.
        source: The ``<source>:`` qualifier, or ``None`` for every source.
        kind: ``"all"`` (``*``), ``"tag"`` (``tag:<t>``) or ``"vendor"``.
        value: The tag or the vendor name; ``""`` for ``"all"``.
        negate: An exclusion (``!<vendor>``, ``!<source>:<vendor>``): the
            vendor is not installed whatever a tag, ``*`` or explicit
            entry says.  An exclusion always wins, so the order of the list
            never decides it.
        version: The pinned release (``<source>:<vendor>@<version>``): that
            version is installed and nothing newer replaces it.  ``None``
            follows the newest installable release.
        origin: ``"install"``, or ``"moved:<vendor>"`` for a selector the
            gateway added for a vendor that moved out of core.
    """

    raw: str
    source: str | None
    kind: Literal["all", "tag", "vendor"]
    value: str
    origin: str = "install"
    negate: bool = False
    version: str | None = None
    #: Every pin of the vendor when more than one names it with different
    #: versions; set by :func:`pick_selector`, never by the parser.
    pin_conflict: tuple[str, ...] = ()

    def applies_to(self, source_id: str | None) -> bool:
        """Whether this selector reads the source with *source_id*."""
        return self.source is None or self.source == source_id

    def matches(self, vendor: str, tags: Collection[str]) -> bool:
        """Whether this selector names *vendor* (whose newest record has *tags*).

        This is the pattern alone: an exclusion *matches* the vendor it
        excludes.  Which selector installs a vendor is :func:`pick_selector`.
        """
        if self.kind == "all":
            return True
        if self.kind == "tag":
            return self.value in tags
        return self.value == vendor


def pick_selector(
    selectors: Sequence[InstallSelector],
    source_id: str | None,
    vendor: str,
    tags: Collection[str],
) -> InstallSelector | None:
    """The selector that installs *vendor* from *source_id*, or ``None``.

    The rule, deterministic and independent of list order where it matters:

    1. an exclusion (``!<vendor>`` / ``!<source>:<vendor>``) that applies
       wins over everything — the vendor is not installed;
    2. an explicit positive ``<vendor>`` entry beats a tag or ``*`` entry, and
       a pinned one beats an unpinned one, so a pin is never shadowed by an
       entry listed before it;
    3. two pins of one vendor with different versions are not decided by
       order: the selector comes back with :attr:`InstallSelector.pin_conflict`
       naming both, and the installer refuses to choose (WARNING, status row);
    4. otherwise the first matching entry speaks, as before.
    """
    applicable = [
        s for s in selectors if s.applies_to(source_id) and s.matches(vendor, tags)
    ]
    if any(s.negate for s in applicable):
        return None
    positive = [s for s in applicable if not s.negate]
    explicit = [s for s in positive if s.kind == "vendor"]
    pins = [s for s in explicit if s.version is not None]
    if len({s.version for s in pins}) > 1:
        return replace(pins[0], pin_conflict=tuple(s.raw for s in pins))
    chosen = (pins or explicit or positive or [None])[0]
    return chosen


def _overridden_reason(
    selectors: Sequence[InstallSelector],
    source_id: str | None,
    vendor: str,
    tags: Collection[str],
    winner: InstallSelector | None,
) -> dict[str, str]:
    """Why each other entry that names *vendor* did not install it.

    An entry an exclusion overrode, or one another entry beat, is still in the
    manifest; its status row says which entry won instead.
    """
    exclusion = excluded_by(selectors, source_id, vendor)
    out: dict[str, str] = {}
    for s in selectors:
        if s.negate or not (s.applies_to(source_id) and s.matches(vendor, tags)):
            continue
        if winner is not None and s.raw == winner.raw:
            continue
        if exclusion is not None:
            out[s.raw] = (
                f"excluded by shims.install ({exclusion.raw!r}); "
                "removing that entry brings it back"
            )
        elif winner is not None and s.raw in winner.pin_conflict:
            out[s.raw] = f"pinned more than once ({', '.join(winner.pin_conflict)})"
        elif winner is not None:
            out[s.raw] = f"already selected by {winner.raw}"
    return out


def excluded_by(
    selectors: Sequence[InstallSelector], source_id: str | None, vendor: str
) -> InstallSelector | None:
    """The exclusion that keeps *vendor* of *source_id* out, if any."""
    return next(
        (
            s
            for s in selectors
            if s.negate and s.applies_to(source_id) and s.value == vendor
        ),
        None,
    )


def parse_selector(text: Any, **extra: Any) -> InstallSelector:
    """Parse one ``shims.install`` entry.

    Accepted spellings: ``<vendor>``, ``tag:<tag>``, ``*``, and each of them
    behind a ``<source>:`` qualifier (``argonne:alcf``, ``contrib:tag:tier:free``,
    ``argonne:*``).  A vendor entry may also be an exclusion, ``!<vendor>`` or
    ``!<source>:<vendor>``, or a pin, ``<vendor>@<version>`` or
    ``<source>:<vendor>@<version>``.  Only a vendor can be excluded or pinned:
    ``!tag:…``, ``!*`` and a pin on a tag are refused, and so is a pinned
    exclusion.

    Args:
        text: The entry.
        **extra: Fields to set on the result (``origin``).

    Returns:
        The selector.

    Raises:
        ValueError: On a non-string or malformed entry.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("config: 'shims.install' entries must be non-empty strings")
    raw = text.strip()
    source: str | None = None
    negate = raw.startswith("!")
    rest = raw[1:].strip() if negate else raw
    version: str | None = None
    if "@" in rest:
        rest, version = rest.split("@", 1)
        if negate or not _VERSION_RE.match(version):
            raise ValueError(
                f"config: 'shims.install' entry {raw!r}: "
                + ("an exclusion takes no version" if negate else "bad version")
            )
    if not rest.startswith("tag:") and ":" in rest:
        source, rest = rest.split(":", 1)
        if not _SOURCE_ID_RE.match(source):
            raise ValueError(f"config: 'shims.install' entry {raw!r}: bad source id")
    if (negate or version) and (rest == "*" or rest.startswith("tag:")):
        raise ValueError(
            f"config: 'shims.install' entry {raw!r}: only a vendor can be "
            + ("excluded" if negate else "pinned")
        )
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
    return InstallSelector(
        raw=raw,
        source=source,
        kind="vendor",
        value=rest,
        negate=negate,
        version=version,
        **extra,
    )


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
    from ..catalog import SHIM_SOURCES, get_shim_source

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


def _source_root(cache_root: str | os.PathLike[str], spec: ManifestSourceSpec) -> Path:
    digest = hashlib.sha256(spec.url.encode()).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", spec.key or "source").strip("-.")
    return Path(cache_root) / MANIFEST_CACHE_DIRNAME / f"{slug or 'source'}-{digest}"


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
