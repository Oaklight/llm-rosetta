"""Vendor manifests: the connection defaults every leaf in a vendor inherits.

``<vendor>/provider.yaml`` without an identity is a *manifest*: it declares a
``connection:`` block that each leaf below it inherits, field by field, with
``headers`` merging per key.  It is not a shim — it has no name, registers
nothing and never appears in a provider picker — so it has no row in the shim
inventory of :mod:`.shim_files`; it is listed beside the leaves that inherit
it instead.

The panel shows which leaf inherits each field a manifest declares and which
leaf overrides it.  The merge rule is the loader's, not a second copy: an
inheritance view that disagreed with the loader would be worse than none.  A
manifest is read-only here, like a shim (:func:`.shim_files._read_only_reason`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_rosetta.api import leaf_connection, manifest_vocabulary

from llm_rosetta_gateway.admin_api import catalogue
from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from .._shared import _qp
from .shim_files import (
    _read_only_reason,
    _shim_roots,
    _yaml_meta,
)


def _identity_keys() -> tuple[str, ...]:
    """Keys that make a ``provider.yaml`` a shim rather than a manifest.

    The loader's own tuple: a file is a manifest here exactly when it is one
    there, so the panel cannot present as a manifest something the loader reads
    as a shim.
    """
    return manifest_vocabulary().identity_keys


def _leaf_connection(cfg: Any) -> dict[str, Any]:
    """A leaf's own connection values, in the loader's canonical spelling."""
    return leaf_connection(cfg)


def _read_definition(path: Path) -> dict[str, Any] | None:
    """A ``provider.yaml`` as a mapping, or ``None`` when it is not one.

    A comment-only file is an empty mapping: a manifest may exist only to hold
    comments beside the vendor's ``auth.py``.
    """
    try:
        cfg = catalogue().yaml_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if cfg is None:
        return {}
    return cfg if isinstance(cfg, dict) else None


def _real(path: Path | None) -> Path | None:
    """The resolved path, or ``None`` when it cannot be resolved."""
    if path is None:
        return None
    try:
        return path.resolve()
    except OSError:
        return None


def _walk_definitions(
    directory: Path, *, skip: Path | None = None, prune: bool = True
) -> list[Path]:
    """Every ``provider.yaml`` below *directory*, following directory links.

    Two rules, both taken from the loader:

    - It follows a symlinked directory (``iterdir`` and ``is_dir`` do), so a
      walk that stopped at real directories answered "nobody inherits from
      this manifest" for a leaf the loader does load and merge.  ``seen``
      holds the resolved directories, so a link back to an ancestor cannot
      loop.
    - A definition owns its subtree: the loader does not nest, so a nested
      manifest's leaves belong to it and not to an outer one.  That is what
      *prune* stops at.

    Args:
        directory: The directory to walk.
        skip: A definition to leave out of the result — the one being
            described.  It does not stop the walk at its own level.
        prune: Stop at each definition found.  False is for the inventory's
            own scan, which wants a nested manifest listed as well.

    Returns:
        The definition paths, sorted.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    skip_real = _real(skip)
    stack = [directory]
    while stack:
        current = stack.pop()
        real = _real(current)
        if real is None or real in seen:
            continue
        seen.add(real)
        definition = current / "provider.yaml"
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        if definition.is_file() and _real(definition) != skip_real:
            found.append(definition)
            if prune:
                continue
        for entry in reversed(entries):
            if entry.is_dir():
                stack.append(entry)
    return sorted(found)


def _definitions_below(directory: Path, *, skip: Path) -> list[Path]:
    """Every definition below *directory*, stopping at the first one on a path.

    The manifest's own leaves: a nested vendor's manifest owns everything
    under it, so this is the set the loader would merge *skip* into.
    """
    return _walk_definitions(directory, skip=skip)


def _leaves_below(path: Path) -> list[Path]:
    """The leaf definitions strictly below the directory holding *path*."""
    return [
        child
        for child in _definitions_below(path.parent, skip=path)
        if _yaml_meta(child) is not None
    ]


def _root_manifests(origin: str, location: str, root: Path) -> list[dict[str, Any]]:
    """Every vendor manifest one source root defines."""
    entries: list[dict[str, Any]] = []
    if not root.is_dir():
        return entries
    for path in _walk_definitions(root, prune=False):
        cfg = _read_definition(path)
        if cfg is None or any(key in cfg for key in _identity_keys()):
            continue
        if not _leaves_below(path):
            # Nothing inherits from it: it is not a manifest yet.  A vendor
            # directory with a stray file must not gain an editor.
            continue
        entry: dict[str, Any] = {
            "path": str(path),
            "vendor": path.parent.name,
            "origin": origin,
            "source": location,
            "root": str(root),
        }
        entries.append(entry)
    return entries


def _manifest_entries(request: Any) -> list[dict[str, Any]]:
    """Return one entry per vendor manifest the declared sources define.

    A ``provider.yaml`` is a manifest when it declares no identity and at
    least one leaf ``provider.yaml`` sits below it.  The walk is the same one
    :func:`._shim_index` makes, so a manifest and its leaves always agree on
    which source they belong to.  Cached on the app like that index, keyed by
    the source report's last rebuild.
    """
    report = getattr(request.app, "shim_source_report", None)
    stamp = getattr(report, "rebuilt_at", 0.0) if report is not None else 0.0
    cached = getattr(request.app, "_shim_manifest_index", None)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    entries: list[dict[str, Any]] = []
    for origin, location, root in _shim_roots(request):
        entries.extend(_root_manifests(origin, location, root))

    entries.sort(key=lambda item: item["vendor"])
    request.app._shim_manifest_index = (stamp, entries)
    return entries


def _row(entry: dict[str, Any]) -> dict[str, Any]:
    """The summary row the inventory renders, with the load-time read-only rule."""
    row = dict(entry)
    reason = _read_only_reason(row)
    row["read_only_reason"] = reason
    return row


def _leaf_entries(manifest_path: Path) -> list[dict[str, Any]]:
    """Every leaf below *manifest_path*, with the connection values it declares.

    Read from the files rather than the registry: a leaf the loader skipped
    still inherits nothing, and showing it is how the operator finds out.  The
    walk stops at a nested definition, because the loader would attribute its
    leaves to it rather than to this manifest.
    """
    leaves: list[dict[str, Any]] = []
    for path in _definitions_below(manifest_path.parent, skip=manifest_path):
        meta = _yaml_meta(path)
        if meta is None:
            continue
        try:
            cfg = catalogue().yaml_load(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = {}
        leaves.append(
            {
                "name": meta["name"],
                "path": str(path),
                "own": _leaf_connection(cfg),
            }
        )
    return leaves


def _fields(connection: dict[str, Any], leaves: list[dict[str, Any]]) -> dict[str, Any]:
    """Which leaves inherit each field the manifest declares, and which override.

    ``headers`` is one level deeper, because it merges per key: a leaf that
    declares its own ``x-api-version`` still inherits every other header.
    """
    fields: dict[str, Any] = {}
    for key, value in connection.items():
        if key == "headers" and isinstance(value, dict):
            per_header: dict[str, Any] = {}
            for header in value:
                per_header[header] = {
                    "inherited_by": [
                        leaf["name"]
                        for leaf in leaves
                        if header not in (leaf["own"].get("headers") or {})
                    ],
                    "overridden_by": [
                        leaf["name"]
                        for leaf in leaves
                        if header in (leaf["own"].get("headers") or {})
                    ],
                }
            fields["headers"] = per_header
            continue
        fields[key] = {
            "inherited_by": [leaf["name"] for leaf in leaves if key not in leaf["own"]],
            "overridden_by": [leaf["name"] for leaf in leaves if key in leaf["own"]],
        }
    return fields


def _connection_of(text: str) -> dict[str, Any]:
    """The ``connection:`` block of a manifest text, or an empty mapping."""
    try:
        cfg = catalogue().yaml_load(text)
    except Exception:  # noqa: BLE001 - the caller validated the text first
        return {}
    if not isinstance(cfg, dict):
        return {}
    connection = cfg.get("connection")
    return dict(connection) if isinstance(connection, dict) else {}


def _detail(entry: dict[str, Any]) -> dict[str, Any]:
    """One manifest with its text, its leaves and the per-field inheritance."""
    path = Path(entry["path"])
    text = ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    leaves = _leaf_entries(path)
    connection = _connection_of(text)
    detail = _row(entry)
    detail.update(
        {
            "yaml": text,
            "connection": connection,
            "leaves": [
                {
                    "name": leaf["name"],
                    "path": leaf["path"],
                    "overrides": sorted(leaf["own"]),
                    "header_overrides": sorted(
                        (leaf["own"].get("headers") or {}).keys()
                    ),
                }
                for leaf in leaves
            ],
            "fields": _fields(connection, leaves),
        }
    )
    return detail


def _find_entry(request: Any, path: str | None) -> dict[str, Any] | None:
    """Resolve a manifest by path, or ``None``.

    The path comes from the client, so it is matched against the manifests the
    sources actually define: a caller cannot name an arbitrary file and have
    the panel rewrite it.
    """
    if not path:
        return None
    wanted = Path(path).expanduser()
    for entry in _manifest_entries(request):
        if Path(entry["path"]) == wanted:
            return entry
    return None


def manifest_rows(request: Any) -> list[dict[str, Any]]:
    """One row per vendor manifest, naming the leaves that inherit from it.

    The shim inventory carries these beside its shim rows, so the panel can
    put a vendor above the leaves it feeds without a second request.  A
    manifest is never an entry of the inventory's ``shims`` list: that list is
    what a picker would resolve, and a manifest has no shim name to resolve.
    """
    rows = []
    for entry in _manifest_entries(request):
        row = _row(entry)
        row["leaves"] = [leaf["name"] for leaf in _leaf_entries(Path(entry["path"]))]
        rows.append(row)
    return rows


async def get_shim_manifests(request: Any) -> Response:
    """List every vendor manifest, with the leaves that inherit its fields."""
    return JSONResponse({"manifests": manifest_rows(request)})


async def get_shim_manifest(request: Any) -> Response:
    """Return one manifest: its text, its leaves and the inheritance map."""
    entry = _find_entry(request, _qp(request, "path"))
    if entry is None:
        return JSONResponse({"error": "No such vendor manifest"}, status_code=404)
    return JSONResponse(_detail(entry))
