"""The shim source manifest: what a shim repository publishes, and what installs.

A shim source repository (``llm-rosetta-shim-contrib``,
``llm-rosetta-shim-argonne``) publishes one ``manifest.json`` on its ``dist``
branch, with one record per released vendor version (#1660).  A host such as
the gateway reads it, picks per vendor the newest version this core can run,
downloads that version's archive and checks its digest.

The format is stated once, in ``shim_manifest.schema.json`` beside this module
(JSON Schema draft 2020-12).  :func:`validate_manifest` interprets the subset
of keywords that schema uses, so the schema file is the rule and this module
adds no second copy of it; no third-party validator is needed.

The compatibility rule (:func:`requirement_problem`):

- ``requires.surface`` must equal :data:`llm_rosetta.contracts.SURFACE_VERSION`
  — a rise of the surface level means a removal or a changed meaning, so a
  shim built for another level is not run;
- ``requires.ir`` must not exceed :data:`llm_rosetta.contracts.IR_VERSION`.

This module imports only :mod:`llm_rosetta.contracts`, so a producer's CI can
validate a manifest without loading any converter.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION

#: The manifest format level this core reads.  A document at another level is
#: refused rather than guessed at.
MANIFEST_VERSION = 1

#: The schema file, shipped beside this module.
SCHEMA_FILENAME = "shim_manifest.schema.json"

#: The maturity ladder below core, lowest first.
MATURITY_LEVELS: tuple[str, ...] = ("experimental", "beta", "stable")

_VERSION_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?$"
)
_PROMOTED_RE = re.compile(r"^core@(\d+)\.(\d+)(?:\.(\d+))?$")


class ShimManifestInvalid(ValueError):
    """A manifest document does not follow the schema.

    Attributes:
        problems: One sentence per violation, each naming its JSON path.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        shown = "; ".join(self.problems[:5])
        more = len(self.problems) - 5
        if more > 0:
            shown += f"; and {more} more"
        super().__init__(f"invalid shim manifest: {shown}")


@dataclass(frozen=True)
class ShimRequirement:
    """What a release needs from the core that runs it.

    Attributes:
        surface: The extension surface level it was built for.
        ir: The lowest IR level it reads.
    """

    surface: int
    ir: int


@dataclass(frozen=True)
class VendorRelease:
    """One released version of one vendor.

    Attributes:
        vendor: The vendor directory name (the manifest's key).
        version: The per-vendor semantic version.
        commit: The source commit the archive was built from.
        archive: Archive path relative to the manifest's directory.
        sha256: Lowercase hex digest of the archive.
        requires: The compatibility floor.
        shims: The shim names the archive registers.
        tags: Selection tags (``tier:free`` and so on).
        maturity: ``experimental``, ``beta`` or ``stable``.
        since: The date the maturity was reached, when stated.
        conformance: The last conformance result, as published.
        promoted_to: ``core@X.Y[.Z]`` once the vendor moved into core.
    """

    vendor: str
    version: str
    commit: str
    archive: str
    sha256: str
    requires: ShimRequirement
    shims: tuple[str, ...]
    maturity: str
    tags: tuple[str, ...] = ()
    since: str | None = None
    conformance: Mapping[str, Any] | None = None
    promoted_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the record in the manifest's own shape."""
        return {
            "version": self.version,
            "commit": self.commit,
            "archive": self.archive,
            "sha256": self.sha256,
            "requires": {"surface": self.requires.surface, "ir": self.requires.ir},
            "shims": list(self.shims),
            "tags": list(self.tags),
            "maturity": self.maturity,
            "since": self.since,
            "conformance": dict(self.conformance) if self.conformance else None,
            "promoted_to": self.promoted_to,
        }


@dataclass(frozen=True)
class SourceManifest:
    """A parsed manifest.

    Attributes:
        source: The source id.
        repo: The repository that publishes it.
        generated_at: When it was generated.
        vendors: Every vendor's releases, newest first by semantic version.
        manifest_version: The format level (always :data:`MANIFEST_VERSION`).
    """

    source: str
    repo: str
    generated_at: str
    vendors: Mapping[str, tuple[VendorRelease, ...]] = field(default_factory=dict)
    manifest_version: int = MANIFEST_VERSION


def load_manifest_schema() -> dict[str, Any]:
    """Return the manifest JSON Schema shipped with this core.

    Returns:
        The parsed schema document.
    """
    path = Path(__file__).with_name(SCHEMA_FILENAME)
    return json.loads(path.read_text(encoding="utf-8"))


_SCHEMA_CACHE: dict[str, Any] = {}


def _schema() -> dict[str, Any]:
    if not _SCHEMA_CACHE:
        _SCHEMA_CACHE.update(load_manifest_schema())
    return _SCHEMA_CACHE


# ---------------------------------------------------------------------------
# The schema interpreter
# ---------------------------------------------------------------------------

#: The keywords :func:`validate_manifest` understands.  Annotation keywords are
#: read and ignored.  ``tests/contracts`` checks that the schema uses nothing
#: else, so a keyword added to the schema cannot be silently skipped.
SUPPORTED_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "const",
        "enum",
        "pattern",
        "minimum",
        "minItems",
        "uniqueItems",
        "required",
        "properties",
        "additionalProperties",
        "propertyNames",
        "items",
        "$ref",
    }
)
ANNOTATION_KEYWORDS: frozenset[str] = frozenset(
    {"$schema", "$id", "$defs", "title", "description", "default"}
)

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, Mapping),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _resolve_ref(ref: str, root: Mapping[str, Any]) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"only local $ref is supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _check(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
    root: Mapping[str, Any],
    problems: list[str],
) -> None:
    """Append every violation of *schema* by *value* to *problems*."""
    if "$ref" in schema:
        _check(value, _resolve_ref(schema["$ref"], root), path, root, problems)
    if "type" in schema:
        types = schema["type"]
        types = [types] if isinstance(types, str) else list(types)
        if not any(_TYPE_CHECKS[t](value) for t in types):
            problems.append(f"{path}: expected {' or '.join(types)}")
            return
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(x) for x in schema["enum"])
        problems.append(f"{path}: must be one of {allowed}")
    if isinstance(value, str) and "pattern" in schema:
        if re.search(schema["pattern"], value) is None:
            problems.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if (
        "minimum" in schema
        and _TYPE_CHECKS["number"](value)
        and value < schema["minimum"]
    ):
        problems.append(f"{path}: must be >= {schema['minimum']}")
    if isinstance(value, list):
        _check_array(value, schema, path, root, problems)
    if isinstance(value, Mapping):
        _check_object(value, schema, path, root, problems)


def _check_array(
    value: list[Any],
    schema: Mapping[str, Any],
    path: str,
    root: Mapping[str, Any],
    problems: list[str],
) -> None:
    if len(value) < schema.get("minItems", 0):
        problems.append(f"{path}: needs at least {schema['minItems']} item(s)")
    if schema.get("uniqueItems"):
        seen: list[Any] = []
        for item in value:
            if item in seen:
                problems.append(f"{path}: duplicate item {item!r}")
            seen.append(item)
    if "items" in schema:
        for index, item in enumerate(value):
            _check(item, schema["items"], f"{path}[{index}]", root, problems)


def _check_object(
    value: Mapping[str, Any],
    schema: Mapping[str, Any],
    path: str,
    root: Mapping[str, Any],
    problems: list[str],
) -> None:
    for key in schema.get("required", ()):
        if key not in value:
            problems.append(f"{path}: missing required key {key!r}")
    properties = schema.get("properties", {})
    names = schema.get("propertyNames")
    extra = schema.get("additionalProperties", True)
    for key, item in value.items():
        where = f"{path}.{key}"
        if names is not None:
            _check(key, names, f"{path} key {key!r}", root, problems)
        if key in properties:
            _check(item, properties[key], where, root, problems)
        elif extra is False:
            problems.append(f"{path}: unknown key {key!r}")
        elif isinstance(extra, Mapping):
            _check(item, extra, where, root, problems)


def validate_manifest(document: Any) -> list[str]:
    """Check *document* against the manifest schema.

    Beyond the schema, a vendor may not list the same version twice.

    Args:
        document: The parsed JSON value.

    Returns:
        One sentence per violation; empty when the document is valid.
    """
    root = _schema()
    problems: list[str] = []
    _check(document, root, "$", root, problems)
    if problems or not isinstance(document, Mapping):
        return problems
    for vendor, entry in document["vendors"].items():
        versions = [release["version"] for release in entry["versions"]]
        for version in sorted({v for v in versions if versions.count(v) > 1}):
            problems.append(f"$.vendors.{vendor}: version {version} is listed twice")
    return problems


def parse_manifest(document: Any) -> SourceManifest:
    """Validate *document* and return it as a :class:`SourceManifest`.

    Args:
        document: The parsed JSON value.

    Returns:
        The manifest, each vendor's releases ordered newest first.

    Raises:
        ShimManifestInvalid: When :func:`validate_manifest` reports a problem.
    """
    problems = validate_manifest(document)
    if problems:
        raise ShimManifestInvalid(problems)
    vendors: dict[str, tuple[VendorRelease, ...]] = {}
    for vendor, entry in document["vendors"].items():
        releases = [_release(vendor, raw) for raw in entry["versions"]]
        releases.sort(key=lambda r: version_key(r.version), reverse=True)
        vendors[vendor] = tuple(releases)
    return SourceManifest(
        source=document["source"],
        repo=document["repo"],
        generated_at=document["generated_at"],
        vendors=vendors,
        manifest_version=document["manifest_version"],
    )


def _release(vendor: str, raw: Mapping[str, Any]) -> VendorRelease:
    requires = raw["requires"]
    return VendorRelease(
        vendor=vendor,
        version=raw["version"],
        commit=raw["commit"],
        archive=raw["archive"],
        sha256=raw["sha256"],
        requires=ShimRequirement(surface=requires["surface"], ir=requires["ir"]),
        shims=tuple(raw["shims"]),
        maturity=raw["maturity"],
        tags=tuple(raw.get("tags") or ()),
        since=raw.get("since"),
        conformance=raw.get("conformance"),
        promoted_to=raw.get("promoted_to"),
    )


# ---------------------------------------------------------------------------
# Versions and compatibility
# ---------------------------------------------------------------------------


def version_key(version: str) -> tuple[int, int, int, int, tuple[Any, ...]]:
    """A sort key that orders semantic versions as SemVer 2.0 does.

    A pre-release sorts below its release (``1.0.0-rc1 < 1.0.0``); its
    dot-separated identifiers compare numerically when numeric.

    Args:
        version: A ``MAJOR.MINOR.PATCH[-PRE]`` string.

    Returns:
        The key.

    Raises:
        ValueError: When *version* is not a semantic version.
    """
    match = _VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"not a semantic version: {version!r}")
    major, minor, patch, pre = match.groups()
    if pre is None:
        return int(major), int(minor), int(patch), 1, ()
    parts = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))
    return int(major), int(minor), int(patch), 0, parts


def requirement_problem(
    requires: ShimRequirement,
    *,
    surface: int | None = None,
    ir: int | None = None,
) -> str | None:
    """Why a release with *requires* cannot run on this core, or ``None``.

    Args:
        requires: The release's requirement.
        surface: The surface level to check against; this core's by default.
        ir: The IR level to check against; this core's by default.

    Returns:
        The sentence to log, or ``None`` when the release is installable.
    """
    surface = SURFACE_VERSION if surface is None else surface
    ir = IR_VERSION if ir is None else ir
    if requires.surface != surface:
        return (
            f"requires surface {requires.surface}, this core provides surface {surface}"
        )
    if requires.ir > ir:
        return f"requires IR {requires.ir}, this core reads up to IR {ir}"
    return None


def is_prerelease(version: str) -> bool:
    """Whether *version* carries a pre-release suffix (``1.0.0-rc.1``)."""
    return "-" in version


def select_release(
    releases: Sequence[VendorRelease],
    *,
    surface: int | None = None,
    ir: int | None = None,
) -> tuple[VendorRelease | None, tuple[tuple[VendorRelease, str], ...]]:
    """Pick the release this core should install.

    The newest installable **stable** release wins.  A pre-release is chosen
    only when no stable release is installable, so one pre-release published
    to a source does not upgrade every gateway that reads it.

    Args:
        releases: One vendor's releases, in any order.
        surface: The surface level to check against; this core's by default.
        ir: The IR level to check against; this core's by default.

    Returns:
        ``(chosen, skipped)``: the chosen release (``None`` when nothing is
        installable), and every release *newer* than it that was refused as
        incompatible, with the reason.  Older incompatible releases, and newer
        pre-releases passed over for a stable one, are not reported.
    """
    ordered = sorted(releases, key=lambda r: version_key(r.version), reverse=True)
    incompatible: list[tuple[VendorRelease, str]] = []
    prerelease: VendorRelease | None = None
    chosen: VendorRelease | None = None
    for release in ordered:
        problem = requirement_problem(release.requires, surface=surface, ir=ir)
        if problem is not None:
            incompatible.append((release, problem))
        elif not is_prerelease(release.version):
            chosen = release
            break
        elif prerelease is None:
            prerelease = release
    chosen = chosen or prerelease
    if chosen is None:
        return None, tuple(incompatible)
    floor = version_key(chosen.version)
    return chosen, tuple(
        (release, why)
        for release, why in incompatible
        if version_key(release.version) > floor
    )


def promoted_core_version(release: VendorRelease) -> tuple[int, int, int] | None:
    """The core version a release says its vendor moved into, if any.

    Args:
        release: A release record.

    Returns:
        ``(major, minor, patch)`` from ``promoted_to: core@X.Y[.Z]``, or
        ``None`` when the vendor has not been promoted.
    """
    if not release.promoted_to:
        return None
    match = _PROMOTED_RE.match(release.promoted_to)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


__all__ = [
    "MANIFEST_VERSION",
    "MATURITY_LEVELS",
    "SCHEMA_FILENAME",
    "ShimManifestInvalid",
    "ShimRequirement",
    "SourceManifest",
    "VendorRelease",
    "is_prerelease",
    "load_manifest_schema",
    "parse_manifest",
    "promoted_core_version",
    "requirement_problem",
    "select_release",
    "validate_manifest",
    "version_key",
]
