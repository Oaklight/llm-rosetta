"""Selection: which release of a selected vendor to install, or why none.

Picks the newest compatible version, honours pins and promotions to core, and
reads back what is already installed.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from llm_rosetta.contracts.shim_manifest import (
    ShimRequirement,
    SourceManifest,
    VendorRelease,
    promoted_core_version,
    requirement_problem,
    select_release,
    version_key,
)

from .._cache import _read_stamp
from .._types import (
    ShimSourceStatus,
    logger,
)
from .plan import (
    InstallSelector,
    ManifestSourceSpec,
    _overridden_reason,
    excluded_by,
    pick_selector,
)


def _unmatched_rows(
    selectors: Sequence[InstallSelector],
    matched: set[str],
    answered: Sequence[tuple[str, ManifestSourceSpec]],
    fell_back: set[str],
    claimed: Mapping[str, str] | None = None,
    overridden: Mapping[str, str] | None = None,
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
        if selector.negate:
            continue
        if selector.raw in matched or selector.raw in fell_back:
            continue
        if selector.origin.startswith("moved:") and selector.value in (claimed or {}):
            continue
        why = (overridden or {}).get(selector.raw)
        if why and selector.origin.startswith("moved:"):
            # A configured provider names this vendor's shim, and the
            # operator's own exclusion keeps it out: say so once, loudly.
            logger.warning(
                "Shim install: a configured provider uses vendor %r, but it is "
                "%s; that provider stays disabled until then",
                selector.value,
                why,
            )
        elif why:
            # The vendor is in the manifest: another entry decided it.
            logger.info("Shim install %r (%s): %s", selector.raw, selector.origin, why)
        else:
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
                skipped=why or "not in the manifest",
            )
        )
    return rows


def _select_vendors(
    manifest: SourceManifest,
    source_id: str,
    selectors: Sequence[InstallSelector],
    claimed: dict[str, str],
    matched: set[str],
    root: Path | None = None,
    overridden: dict[str, str] | None = None,
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
        selector = pick_selector(selectors, source_id, vendor, tags)
        if overridden is not None:
            for raw, why in _overridden_reason(
                selectors, source_id, vendor, tags, selector
            ).items():
                overridden.setdefault(raw, why)
        if selector is None:
            if excluded_by(selectors, source_id, vendor) is not None:
                logger.info(
                    "Shim source %s: vendor %r is excluded by shims.install",
                    source_id,
                    vendor,
                )
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


def _choose_release(
    releases: Sequence[VendorRelease],
    selector: InstallSelector,
    source_id: str,
    vendor: str,
) -> tuple[VendorRelease | None, list[tuple[VendorRelease, str]], str | None]:
    """The release to install: the pinned one, or the newest installable.

    Returns:
        ``(chosen, passed_over, pin_problem)``; each problem is logged here.
    """
    if selector.pin_conflict:
        logger.warning(
            "Shim source %s: %s is pinned more than once (%s); keep one",
            source_id,
            vendor,
            ", ".join(selector.pin_conflict),
        )
        return None, [], "pinned more than once"
    if selector.version is not None:
        chosen, problem = _pinned_release(releases, selector.version)
        if problem:
            logger.warning(
                "Shim source %s: %s is pinned to %s, which %s",
                source_id,
                vendor,
                selector.version,
                problem,
            )
        return chosen, [], problem
    chosen, passed_over = select_release(releases)
    for release, why in passed_over:
        logger.warning(
            "Shim source %s: %s %s is not installable here: %s",
            source_id,
            vendor,
            release.version,
            why,
        )
    return chosen, list(passed_over), None


def _keeps(
    installed: VendorRelease, chosen: VendorRelease, selector: InstallSelector
) -> bool:
    """Whether the installed release stays.

    Unpinned, a gateway never downgrades; pinned, it holds exactly the pinned
    version, so a pin below the installed one installs it.
    """
    if selector.version is not None:
        return installed.version == chosen.version
    return version_key(installed.version) >= version_key(chosen.version)


def _pinned_release(
    releases: Sequence[VendorRelease], version: str
) -> tuple[VendorRelease | None, str | None]:
    """The release a pin names, or why it cannot be installed here."""
    release = next((r for r in releases if r.version == version), None)
    if release is None:
        return None, "the manifest does not list"
    problem = requirement_problem(release.requires)
    if problem:
        return None, f"is not installable on this core ({problem})"
    return release, None


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


def _no_release_error(
    vendor: str,
    selector: InstallSelector,
    pin_problem: str | None,
    passed_over: Sequence[tuple[VendorRelease, str]],
) -> str:
    """Why nothing can be installed.

    A pin naming a release this core cannot take keeps what runs (stale, with
    this reason) and never falls back to a version the operator did not choose.
    """
    if selector.pin_conflict:
        return f"{vendor} is pinned more than once: {', '.join(selector.pin_conflict)}"
    if pin_problem:
        return f"pinned {vendor}@{selector.version} {pin_problem}"
    reason = "; ".join(why for _, why in passed_over) or "no release"
    return f"no version of {vendor} is installable on this core ({reason})"


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
