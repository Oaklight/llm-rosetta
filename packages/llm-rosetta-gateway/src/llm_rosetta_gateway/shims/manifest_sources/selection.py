"""Selection: which release of a selected vendor to install, or why none.

Picks the newest compatible version, honours pins, and
reads back what is already installed.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_rosetta.api import ExtensionIndex, IndexEntry, choose_release, version_key

from .._cache import _read_stamp
from .._types import (
    ShimSourceStatus,
    logger,
)
from .releases import release_from_dict, requirement_problem
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
    claimed: Mapping[str, str] | None = None,
    overridden: Mapping[str, str] | None = None,
) -> list[ShimSourceStatus]:
    """Warn about, and report, each selector that matched no vendor.

    A source that answers without the vendor (a staged publish, a renamed tag)
    must not drop the operator's shims silently; the status row
    (``skipped="not in the manifest"``) is what the panel shows.
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
        if selector.raw in matched:
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
    manifest: ExtensionIndex,
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

    A vendor with no release in the index is not selected.  *root* is
    where a vendor's last install stamp lives; its tags speak for a vendor
    whose releases are all unreadable here.
    """
    out: list[tuple[str, InstallSelector]] = []
    for vendor in sorted(manifest):
        releases = manifest.get(vendor, ())
        # The record that would be installed speaks for the tags: one newer
        # release this core cannot run must not drop the vendor from a tag.
        installable, _ = choose_release(releases, prefer_stable=True)
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


def _choose_release(
    releases: Sequence[IndexEntry],
    selector: InstallSelector,
    source_id: str,
    vendor: str,
) -> tuple[IndexEntry | None, list[tuple[IndexEntry, str]], str | None]:
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
    chosen, passed_over = choose_release(releases, prefer_stable=True)
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
    installed: IndexEntry, chosen: IndexEntry, selector: InstallSelector
) -> bool:
    """Whether the installed release stays.

    Unpinned, a gateway never downgrades; pinned, it holds exactly the pinned
    version, so a pin below the installed one installs it.
    """
    if selector.version is not None:
        return installed.version == chosen.version
    return version_key(installed.version) >= version_key(chosen.version)


def _pinned_release(
    releases: Sequence[IndexEntry], version: str
) -> tuple[IndexEntry | None, str | None]:
    """The release a pin names, or why it cannot be installed here."""
    release = next((r for r in releases if r.version == version), None)
    if release is None:
        return None, "the index does not list"
    problem = requirement_problem(release)
    if problem:
        return None, f"is not installable on this core ({problem})"
    return release, None


def _provided_by(names: Collection[str]) -> str:
    """Name who already registers *names*: core, a plugin, or both.

    A plugin shim is registered before the manifest install, so it looks like
    a core name to the catalog.  Naming the plugin keeps a skipped vendor from
    being reported as a core feature the operator cannot find in the tree.
    """
    from llm_rosetta.api import plugin_source

    sources = {plugin_source(name) for name in names}
    plugins = sorted(source for source in sources if source)
    if not plugins:
        return "core"
    joined = ", ".join(plugins)
    if None not in sources:
        if len(plugins) == 1:
            return f"installed plugin {plugins[0]}"
        return "installed plugins " + joined
    return "core and installed plugin(s) " + joined


def _skip_reason(
    chosen: IndexEntry,
    selector: InstallSelector,
    source_id: str,
    core_names: Collection[str],
) -> str | None:
    """Why the chosen release is not installed at all, or ``None``.

    A release every shim of which core or an installed plugin already
    registers.  A *partial* clash is not a reason: the vendor installs
    and the extension loader decides: it refuses an extension whose declared
    shim names another owner holds (the holder wins), and the status says so.
    """
    clash = sorted(set(chosen.shims) & set(core_names))
    if clash and set(clash) == set(chosen.shims):
        reason = f"already provided by {_provided_by(clash)}: " + ", ".join(clash)
        logger.warning(
            "Shim source %s: vendor %r is ignored; %s",
            source_id,
            chosen.id,
            reason,
        )
        return reason
    if clash:
        logger.warning(
            "Shim source %s: vendor %r: already provided by %s: %s; the "
            "loader refuses an extension whose shim names are held, so it "
            "will not load",
            source_id,
            chosen.id,
            _provided_by(clash),
            ", ".join(clash),
        )
    return None


def _no_release_error(
    vendor: str,
    selector: InstallSelector,
    pin_problem: str | None,
    passed_over: Sequence[tuple[IndexEntry, str]],
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


def _installed_release(stamp: Mapping[str, Any], vendor: str) -> IndexEntry | None:
    """The release an install stamp records, or ``None`` when unreadable."""
    return release_from_dict(vendor, stamp)
