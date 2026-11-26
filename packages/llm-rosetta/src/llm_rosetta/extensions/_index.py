"""The extension index: how ``ext install <id>`` and ``ext update`` find a wheel.

An extension repository publishes each wheel as a release asset and writes one
``index.json`` beside them (its CI owns both, and the file is generated from the
extensions' own files: nothing in it is written by hand).  The installer reads
the index, picks the newest version this interpreter and core accept, downloads
that wheel and checks its ``sha256`` before anything is written.  Version 1 of
the file::

    {
      "index_version": 1,
      "source": "contrib",
      "generated_at": "2026-11-26T08:00:00Z",
      "extensions": {
        "<id>": [
          {
            "version": "0.4.1",
            "url": "llm_rosetta_ext_<id>-0.4.1-py3-none-any.whl",
            "sha256": "<64 hex digits>",
            "surface": 1,
            "ir": 1,
            "requires_python": ">=3.10",
            "requires_extensions": ["<other id>"],
            "name": "Kilo (Free)",
            "maturity": "beta",
            "tags": ["tier:free"],
            "shims": ["kilo_free--chat_completions"],
            "logo": "https://…/kilocode.svg",
            "commit": "<git sha>"
          }
        ]
      }
    }

``source`` names the repository (``contrib``, ``argonne``) and ``generated_at``
is when the index was written, so a host can refuse an index older than the one
it holds.  ``url`` may be relative to the index.  Installing needs ``version``,
``url`` and ``sha256``; ``surface``, ``ir``, ``requires_python`` and
``requires_extensions`` say whether this host can run the wheel (a missing one is
not checked here: the wheel is checked again when it is installed).  ``name``,
``maturity``, ``tags``, ``shims`` and ``logo`` are what a listing of extensions
that are not installed yet shows; ``commit`` is where the wheel was built from.  The
index location is ``--index``, else ``$LLM_ROSETTA_EXTENSION_INDEX``, else
:data:`DEFAULT_INDEX_URL`; it is an ``https`` URL or a local file path.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta.contracts.extension import MATURITY_LEVELS

from ._dirs import ExtensionInstallError
from ._install import (
    InstalledExtension,
    fetch_bytes,
    install_wheel,
    installed_extensions,
    python_requirement_problem,
)

#: The environment variable that names the index.
INDEX_ENV = "LLM_ROSETTA_EXTENSION_INDEX"

#: The official extension repository's index.
DEFAULT_INDEX_URL = (
    "https://github.com/Oaklight/llm-rosetta-ext-contrib/"
    "releases/latest/download/index.json"
)

#: The index format this installer reads.
INDEX_VERSION = 1

#: The largest index accepted.
MAX_INDEX_BYTES = 4 * 1024 * 1024

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION_RE = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_SOURCE_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$"
)


@dataclass(frozen=True)
class IndexEntry:
    """One published wheel.

    Attributes:
        id: The extension id.
        version: The wheel's version.
        url: Where the wheel is: an absolute URL or file path.
        sha256: The wheel's hex digest.
        requires_python: The wheel's ``Requires-Python``, if the index says.
        surface: The extension surface level it was built for, if stated.
        ir: The lowest IR level it reads, if stated.
        requires_extensions: Extensions it needs installed too.
        name: The extension's display name, for a listing.
        maturity: ``experimental``, ``beta`` or ``stable``, if the index says.
        tags: Free-form tags (``tier:free``), for selecting and filtering.
        shims: The shim names it provides.
        logo: A URL of the vendor's mark, if it has one.
        commit: The git commit the wheel was built from, if known.
    """

    id: str
    version: str
    url: str
    sha256: str
    requires_python: str | None = None
    surface: int | None = None
    ir: int | None = None
    requires_extensions: tuple[str, ...] = ()
    name: str | None = None
    maturity: str | None = None
    tags: tuple[str, ...] = ()
    shims: tuple[str, ...] = ()
    logo: str | None = None
    commit: str | None = None


class ExtensionIndex(dict):  # type: ignore[type-arg]
    """The entries of each extension (``{id: [IndexEntry, ...]}``) and the header.

    A plain mapping to every caller that only looks entries up; the header
    fields are attributes.

    Attributes:
        source: The repository's short name (``contrib``).
        generated_at: When the index was written (RFC 3339), or ``""``.
    """

    source: str = ""
    generated_at: str = ""


@dataclass(frozen=True)
class UpdatePlan:
    """What ``ext update`` would do for one installed extension.

    Attributes:
        id: The extension id.
        current: The installed version, or ``None`` when it could not be read.
        status: ``"update"``, ``"current"``, ``"not_in_index"`` or
            ``"no_compatible"``.
        target: The entry to install when ``status`` is ``"update"``.
        detail: One sentence for the other statuses.
    """

    id: str
    current: str | None
    status: str
    target: IndexEntry | None = None
    detail: str | None = None


def version_key(
    version: str,
) -> tuple[int, int, int, int, tuple[tuple[int, int, str], ...]]:
    """Order ``MAJOR.MINOR.PATCH[-pre]`` strings as SemVer 2.0 does.

    A pre-release sorts below its final (``1.0.0-rc.1 < 1.0.0``), and its
    dot-separated identifiers compare numerically when they are numbers
    (``rc.2 < rc.10``) and as text otherwise; a number sorts below text.

    Args:
        version: An index entry's version.

    Returns:
        A tuple that compares in release order.

    Raises:
        ValueError: *version* is not ``MAJOR.MINOR.PATCH`` with an optional
            pre-release.
    """
    match = _VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"{version!r} is not MAJOR.MINOR.PATCH[-pre]")
    major, minor, patch, pre = match.groups()
    if not pre:
        return (int(major), int(minor), int(patch), 1, ())
    parts = tuple(
        (0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre[1:].split(".")
    )
    return (int(major), int(minor), int(patch), 0, parts)


def index_location(explicit: str | None = None) -> str:
    """Return where the index is: *explicit*, else the environment, else the default."""
    return explicit or os.environ.get(INDEX_ENV) or DEFAULT_INDEX_URL


def _is_url(text: str) -> bool:
    return urllib.parse.urlsplit(text).scheme.lower() in ("http", "https")


def _locate(base: str, url: str) -> str:
    """Resolve an entry's ``url`` against the index location."""
    if _is_url(url) or os.path.isabs(url):
        return url
    if _is_url(base):
        return urllib.parse.urljoin(base, url)
    return str(Path(base).parent / url)


def _string_list(value: Any, where: str, label: str, pattern: re.Pattern[str] | None):
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item and (pattern is None or pattern.match(item))
        for item in value
    ):
        raise ExtensionInstallError(f"{where} has a bad {label}")
    return tuple(value)


def _optional_text(raw: Mapping[str, Any], key: str, where: str) -> str | None:
    value = raw.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ExtensionInstallError(f"{where} has a bad {key}")
    return value


def _levels(raw: Mapping[str, Any], where: str) -> dict[str, int | None]:
    numbers: dict[str, int | None] = {}
    for key in ("surface", "ir"):
        value = raw.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ExtensionInstallError(f"{where} has a bad {key}")
        numbers[key] = value
    return numbers


def _display(raw: Mapping[str, Any], where: str) -> dict[str, Any]:
    """The fields a listing shows; each is optional and checked when present."""
    maturity = _optional_text(raw, "maturity", where)
    if maturity is not None and maturity not in MATURITY_LEVELS:
        raise ExtensionInstallError(f"{where} has a bad maturity {maturity!r}")
    commit = _optional_text(raw, "commit", where)
    if commit is not None and not _COMMIT_RE.match(commit.lower()):
        raise ExtensionInstallError(f"{where} has a bad commit")
    return {
        "name": _optional_text(raw, "name", where),
        "maturity": maturity,
        "tags": _string_list(raw.get("tags", []), where, "tags", None),
        "shims": _string_list(raw.get("shims", []), where, "shims", None),
        "logo": _optional_text(raw, "logo", where),
        "commit": commit.lower() if commit else None,
    }


def _parse_entry(ext_id: str, position: int, raw: Any, base: str) -> IndexEntry:
    where = f"index entry {ext_id}[{position}]"
    if not isinstance(raw, Mapping):
        raise ExtensionInstallError(f"{where} is not an object")
    version, url, sha = raw.get("version"), raw.get("url"), raw.get("sha256")
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        raise ExtensionInstallError(f"{where} has a bad version {version!r}")
    if not isinstance(url, str) or not url:
        raise ExtensionInstallError(f"{where} has no url")
    if not isinstance(sha, str) or not _SHA_RE.match(sha.lower()):
        raise ExtensionInstallError(f"{where} has no valid sha256")
    spec = raw.get("requires_python")
    if spec is not None and not isinstance(spec, str):
        raise ExtensionInstallError(f"{where} has a bad requires_python")
    deps = _string_list(
        raw.get("requires_extensions", []), where, "requires_extensions", _ID_RE
    )
    numbers = _levels(raw, where)
    return IndexEntry(
        id=ext_id,
        version=version,
        url=_locate(base, url) if base else url,
        sha256=sha.lower(),
        requires_python=spec,
        surface=numbers["surface"],
        ir=numbers["ir"],
        requires_extensions=deps,
        **_display(raw, where),
    )


def _header(document: Mapping[str, Any]) -> tuple[str, str]:
    source = document.get("source", "")
    if not isinstance(source, str) or (source and not _SOURCE_RE.match(source)):
        raise ExtensionInstallError(f"the index has a bad source {source!r}")
    stamp = document.get("generated_at", "")
    if not isinstance(stamp, str) or (stamp and not _RFC3339_RE.match(stamp.strip())):
        raise ExtensionInstallError(f"the index has a bad generated_at {stamp!r}")
    return source, stamp


def parse_index(document: Any, *, base: str = "") -> ExtensionIndex:
    """Validate an index document and list its entries by extension id.

    Args:
        document: The parsed JSON value.
        base: The index's own location, so a relative ``url`` can be resolved.

    Returns:
        The entries of each extension, in the order the file lists them, with
        the header's ``source`` and ``generated_at`` as attributes.

    Raises:
        ExtensionInstallError: The document is not a version-1 index, or an
            entry is malformed (the message names it).
    """
    if not isinstance(document, Mapping):
        raise ExtensionInstallError("the index is not a JSON object")
    if document.get("index_version") != INDEX_VERSION:
        raise ExtensionInstallError(
            f"the index has index_version {document.get('index_version')!r}; "
            f"this installer reads {INDEX_VERSION}"
        )
    extensions = document.get("extensions")
    if not isinstance(extensions, Mapping):
        raise ExtensionInstallError("the index has no 'extensions' object")
    result = ExtensionIndex()
    result.source, result.generated_at = _header(document)
    for ext_id, entries in extensions.items():
        if not isinstance(ext_id, str) or not _ID_RE.match(ext_id):
            raise ExtensionInstallError(
                f"the index names a bad extension id {ext_id!r}"
            )
        if not isinstance(entries, list):
            raise ExtensionInstallError(f"index entry {ext_id!r} is not a list")
        result[ext_id] = [
            _parse_entry(ext_id, position, raw, base)
            for position, raw in enumerate(entries)
        ]
    return result


def load_index(location: str | None = None, *, timeout: float = 30.0) -> ExtensionIndex:
    """Read and validate the index.

    Args:
        location: An ``https`` URL or a local file path; see :func:`index_location`.
        timeout: Seconds for each network operation.

    Returns:
        The entries of each extension (see :func:`parse_index`).

    Raises:
        ExtensionInstallError: The index cannot be read, is too large, is not
            JSON, or is not a valid index.
    """
    where = index_location(location)
    try:
        if _is_url(where):
            raw = fetch_bytes(where, limit=MAX_INDEX_BYTES, timeout=timeout)
        else:
            path = Path(where).expanduser()
            if path.stat().st_size > MAX_INDEX_BYTES:
                raise ExtensionInstallError(
                    f"the index exceeds the {MAX_INDEX_BYTES} byte cap"
                )
            raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except ExtensionInstallError:
        raise
    except OSError as exc:
        raise ExtensionInstallError(f"cannot read the index {where}: {exc}") from exc
    except ValueError as exc:
        raise ExtensionInstallError(f"the index {where} is not JSON: {exc}") from exc
    return parse_index(document, base=where)


def entry_problems(entry: IndexEntry) -> list[str]:
    """List why this interpreter and core cannot use *entry*; empty when they can."""
    problems: list[str] = []
    if entry.requires_python:
        text = python_requirement_problem(entry.requires_python)
        if text:
            problems.append(text)
    if entry.surface is not None and entry.surface != SURFACE_VERSION:
        problems.append(
            f"needs extension surface {entry.surface}, this core is {SURFACE_VERSION}"
        )
    if entry.ir is not None and entry.ir > IR_VERSION:
        problems.append(f"needs IR level {entry.ir}, this core has {IR_VERSION}")
    return problems


def select_entry(
    index: Mapping[str, list[IndexEntry]], ext_id: str, *, version: str | None = None
) -> IndexEntry:
    """Pick the newest entry of *ext_id* this interpreter and core accept.

    Args:
        index: From :func:`load_index`.
        ext_id: The extension id.
        version: Only this version, when given.

    Returns:
        The chosen entry.

    Raises:
        ExtensionInstallError: The id is not in the index, the version is not
            listed, or no listed version is compatible (each reason is named).
    """
    entries = index.get(ext_id)
    if not entries:
        raise ExtensionInstallError(f"the index has no extension {ext_id!r}")
    if version is not None:
        entries = [entry for entry in entries if entry.version == version]
        if not entries:
            raise ExtensionInstallError(
                f"the index has no version {version} of {ext_id!r}"
            )
    usable = [entry for entry in entries if not entry_problems(entry)]
    if not usable:
        reasons = "; ".join(
            f"{entry.version}: {', '.join(entry_problems(entry))}"
            for entry in sorted(entries, key=lambda e: version_key(e.version))
        )
        raise ExtensionInstallError(
            f"no version of {ext_id!r} can run here ({reasons})"
        )
    return max(usable, key=lambda entry: version_key(entry.version))


def install_entry(
    entry: IndexEntry, dest: str | os.PathLike[str] | None = None
) -> InstalledExtension:
    """Download (or read) one entry's wheel and install it, checking its digest.

    Args:
        entry: From :func:`select_entry`.
        dest: The extensions directory; :func:`extensions_dir` by default.

    Returns:
        The installed extension.

    Raises:
        ExtensionInstallError: The wheel cannot be fetched or is refused.
    """
    if _is_url(entry.url):
        return install_wheel(fetch_bytes(entry.url), sha256=entry.sha256, dest=dest)
    try:
        return install_wheel(entry.url, sha256=entry.sha256, dest=dest)
    except OSError as exc:
        raise ExtensionInstallError(f"cannot read {entry.url}: {exc}") from exc


def install_from_index(
    ext_id: str,
    index: Mapping[str, list[IndexEntry]],
    *,
    version: str | None = None,
    dest: str | os.PathLike[str] | None = None,
    _seen: set[str] | None = None,
) -> list[InstalledExtension]:
    """Install *ext_id* from the index, and the extensions it requires first.

    A required extension that is already installed is left alone.

    Args:
        ext_id: The extension id.
        index: From :func:`load_index`.
        version: Only this version, when given.
        dest: The extensions directory; :func:`extensions_dir` by default.

    Returns:
        What was installed, dependencies first.

    Raises:
        ExtensionInstallError: See :func:`select_entry` and
            :func:`install_wheel`; also a requirement cycle.
    """
    seen = _seen if _seen is not None else set()
    if ext_id in seen:
        raise ExtensionInstallError(
            f"extension requirements form a cycle at {ext_id!r}"
        )
    seen.add(ext_id)
    entry = select_entry(index, ext_id, version=version)
    have = {item.id for item in installed_extensions(dest)}
    done: list[InstalledExtension] = []
    for dependency in entry.requires_extensions:
        if dependency not in have:
            done.extend(install_from_index(dependency, index, dest=dest, _seen=seen))
    done.append(install_entry(entry, dest))
    return done


def plan_updates(
    index: Mapping[str, list[IndexEntry]],
    installed: Iterable[InstalledExtension],
) -> list[UpdatePlan]:
    """Compare what is installed with the index.

    Args:
        index: From :func:`load_index`.
        installed: From :func:`installed_extensions`.

    Returns:
        One plan per installed extension, in the order given.
    """
    plans: list[UpdatePlan] = []
    for item in installed:
        if item.id not in index:
            plans.append(UpdatePlan(item.id, item.version, "not_in_index"))
            continue
        try:
            best = select_entry(index, item.id)
        except ExtensionInstallError as exc:
            plans.append(
                UpdatePlan(item.id, item.version, "no_compatible", detail=str(exc))
            )
            continue
        if item.version is None or version_key(best.version) > version_key(
            item.version
        ):
            plans.append(UpdatePlan(item.id, item.version, "update", target=best))
        else:
            plans.append(UpdatePlan(item.id, item.version, "current"))
    return plans
