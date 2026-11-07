"""Vendor manifests: the connection defaults every leaf in a vendor inherits.

``<vendor>/provider.yaml`` without an identity is a *manifest*: it declares a
``connection:`` block that each leaf below it inherits, field by field, with
``headers`` merging per key.  It is not a shim — it has no name, registers
nothing and never appears in a provider picker — so it has no row in the shim
inventory of :mod:`.shim_files`; it is listed beside the leaves that inherit
it instead.

Editing one is a vendor-wide change, so the panel shows the blast radius
before saving: which leaf inherits each field the edit touches, and which leaf
overrides it and is therefore unaffected.  The merge and the read-only rules
are the loader's and :func:`.shim_files._read_only_reason`'s, not second
copies: an inheritance view that disagreed with the loader would be worse than
none.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from llm_rosetta._vendor.yaml import load as yaml_load
from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from .._shared import _qp, parse_json_body
from .shim_files import (
    _read_only_reason,
    _rebuild_shims,
    _shim_roots,
    _yaml_meta,
)

_UNKNOWN_KEY_REASON = (
    "A vendor manifest may declare only a 'connection:' block; found %s. "
    "Identity belongs in a leaf's provider.yaml."
)


def _connection_keys() -> tuple[str, ...]:
    """The ``connection:`` keys a manifest may declare, from the core."""
    from llm_rosetta.shims.providers import _CONNECTION_KEYS

    return _CONNECTION_KEYS


def _identity_keys() -> tuple[str, ...]:
    """Keys that make a ``provider.yaml`` a shim rather than a manifest.

    The loader's own tuple: a file is a manifest here exactly when it is one
    there, so the panel cannot offer an editor for something the loader reads
    as a shim.
    """
    from llm_rosetta.shims.providers import _IDENTITY_KEYS

    return _IDENTITY_KEYS


def _leaf_connection(cfg: Any) -> dict[str, Any]:
    """A leaf's own connection values, in the loader's canonical spelling."""
    from llm_rosetta.shims.providers import _leaf_connection as reader

    return reader(cfg) if isinstance(cfg, dict) else {}


def _merged_connection(
    connection: dict[str, Any], leaf: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    """The effective connection of a leaf: its own values over the manifest's."""
    from llm_rosetta.shims.provider_shim import VendorManifest
    from llm_rosetta.shims.providers import _merged_connection as merge

    manifest = VendorManifest(
        vendor=manifest_path.parent.name,
        path=manifest_path,
        connection=connection,
    )
    merged, _resolve = merge(leaf, manifest)
    return merged


def _manifest_schema() -> dict[str, Any]:
    """The manifest shape of ``provider.schema.json`` and its ``connection``."""
    from llm_rosetta.shims import providers as _providers

    path = Path(_providers.__file__).parent / "provider.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    definitions = schema.get("definitions") or {}
    return {
        "manifest": definitions.get("vendorManifest") or {},
        "connection": definitions.get("connection") or {},
    }


def _list_problem(value: Any, spec: dict[str, Any], key: str) -> str | None:
    """Why *value* is not the list one ``connection`` property allows."""
    if not isinstance(value, list):
        return f"connection.{key} must be a list"
    minimum = spec.get("minItems")
    if minimum and len(value) < minimum:
        return f"connection.{key} needs at least {minimum} item(s)"
    items = spec.get("items") or {}
    if items.get("type") == "string" and not all(
        isinstance(item, str) for item in value
    ):
        return f"connection.{key} must be a list of strings"
    return None


def _mapping_problem(value: Any, spec: dict[str, Any], key: str) -> str | None:
    """Why *value* is not the mapping one ``connection`` property allows."""
    if not isinstance(value, dict):
        return f"connection.{key} must be a mapping"
    values = spec.get("additionalProperties")
    if isinstance(values, dict) and values.get("type") == "string":
        bad = [name for name, item in value.items() if not isinstance(item, str)]
        if bad:
            return (
                f"connection.{key} must map names to strings; "
                f"{', '.join(sorted(bad))} is not a string"
            )
    return None


def _value_problem(value: Any, spec: dict[str, Any], key: str) -> str | None:
    """Why *value* does not match one ``connection`` property, or ``None``."""
    kind = spec.get("type")
    if kind == "string":
        if not isinstance(value, str) or isinstance(value, bool):
            return f"connection.{key} must be a string"
        allowed = spec.get("enum")
        if allowed and value not in allowed:
            return f"connection.{key} must be one of: {', '.join(allowed)}"
        return None
    if kind == "boolean":
        if isinstance(value, bool):
            return None
        return f"connection.{key} must be true or false"
    if kind == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"connection.{key} must be an integer"
        minimum = spec.get("minimum")
        if minimum is not None and value < minimum:
            return f"connection.{key} must be at least {minimum}"
        return None
    if kind == "array":
        return _list_problem(value, spec, key)
    if kind == "object":
        return _mapping_problem(value, spec, key)
    return None


def manifest_problem(text: str) -> str | None:
    """Why *text* is not a valid vendor manifest, or ``None``.

    Validates against the manifest shape — the ``vendorManifest`` half of the
    ``oneOf`` in ``provider.schema.json`` — not the leaf shape, so a manifest
    that accidentally grows a ``name`` is refused here rather than being
    silently read as a shim by the loader.
    """
    try:
        cfg = yaml_load(text)
    except Exception as exc:  # noqa: BLE001 - the parser's message is the point
        return f"Not valid YAML: {exc}"
    if cfg is None:
        # A comment-only manifest is a legitimate marker: the vendor's shared
        # facts are then its ``auth.py`` and the comments say why.
        return None
    if not isinstance(cfg, dict):
        return "The file must be a YAML mapping"
    identity = [key for key in _identity_keys() if key in cfg]
    if identity:
        return (
            "A vendor manifest declares no identity ("
            + ", ".join(f"'{key}'" for key in identity)
            + "): a file with 'name' or 'format' is a shim, and identity "
            "belongs in the leaf's provider.yaml"
        )

    schema = _manifest_schema()
    allowed = set(schema["manifest"].get("properties") or {})
    unknown = set(cfg) - allowed
    if unknown:
        return _UNKNOWN_KEY_REASON % ", ".join(sorted(unknown))

    connection = cfg.get("connection")
    if connection is None:
        return None
    if not isinstance(connection, dict):
        return "connection must be a mapping"
    properties = schema["connection"].get("properties") or {}
    unknown = set(connection) - set(properties)
    if unknown:
        return (
            f"unknown connection key(s) {', '.join(sorted(unknown))}; "
            f"a vendor manifest may declare {', '.join(_connection_keys())}"
        )
    for key, value in connection.items():
        problem = _value_problem(value, properties.get(key) or {}, key)
        if problem is not None:
            return problem
    return None


def _read_definition(path: Path) -> dict[str, Any] | None:
    """A ``provider.yaml`` as a mapping, or ``None`` when it is not one.

    A comment-only file is an empty mapping: a manifest may exist only to hold
    comments beside the vendor's ``auth.py``.
    """
    try:
        cfg = yaml_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if cfg is None:
        return {}
    return cfg if isinstance(cfg, dict) else None


def _definitions_below(directory: Path, *, skip: Path) -> list[Path]:
    """Every ``provider.yaml`` below *directory*, not descending past one.

    A definition owns its subtree, because the loader does not nest: a leaf
    resolves against the manifest in its own vendor directory, and a
    directory holding a ``provider.yaml`` is a leaf or a manifest, never a
    level to walk through.  A plain ``rglob`` ignored that, so an outer
    manifest claimed a nested vendor's leaves — an inheritance view of
    something the loader never does.

    Args:
        directory: The directory to walk (the manifest's own directory).
        skip: The definition being described.  It is not a member of the
            result and does not stop the walk at its own level.

    Returns:
        The definition paths, sorted.
    """
    found: list[Path] = []
    for current, dirnames, filenames in os.walk(directory):
        dirnames.sort()
        if "provider.yaml" not in filenames:
            continue
        definition = Path(current) / "provider.yaml"
        if definition == skip:
            continue
        found.append(definition)
        dirnames[:] = []
    return sorted(found)


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
    for path in sorted(root.rglob("provider.yaml")):
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
        entry["at_source_root"] = _at_source_root(entry)
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

    entries.sort(key=lambda item: (item["origin"] != "path", item["vendor"]))
    request.app._shim_manifest_index = (stamp, entries)
    return entries


def _at_source_root(entry: dict[str, Any]) -> bool:
    """Whether the manifest's directory *is* the declared source root.

    Mirrors :func:`.shim_files._at_source_root` for a file one level up from a
    shim: such a manifest is read-only for the same reason (its directory is
    the whole source).
    """
    root, path = entry.get("root"), entry.get("path")
    if not root or not path:
        return False
    try:
        return Path(path).parent.resolve() == Path(root).resolve()
    except OSError:
        return False


def _row(entry: dict[str, Any]) -> dict[str, Any]:
    """The summary row the inventory renders, with the load-time read-only rule."""
    row = dict(entry)
    reason = _read_only_reason(row)
    row["editable"] = reason is None
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
            cfg = yaml_load(path.read_text(encoding="utf-8"))
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


def _headers_of(connection: dict[str, Any]) -> dict[str, Any]:
    """The ``headers:`` mapping of a connection block, or an empty one."""
    headers = connection.get("headers")
    return headers if isinstance(headers, dict) else {}


def _units(
    old: dict[str, Any], new: dict[str, Any]
) -> list[tuple[str, str | None, bool]]:
    """The ``(field, header, removed)`` units an edit to the manifest touches."""
    units: list[tuple[str, str | None, bool]] = []
    for key in sorted((set(old) | set(new)) - {"headers"}):
        if old.get(key) != new.get(key):
            units.append((key, None, key not in new))
    old_headers = _headers_of(old)
    new_headers = _headers_of(new)
    for name in sorted(set(old_headers) | set(new_headers)):
        if old_headers.get(name) != new_headers.get(name):
            units.append(("headers", name, name not in new_headers))
    return units


def _affected_leaves(
    units: list[tuple[str, str | None, bool]], leaves: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The leaves whose effective connection each touched unit changes.

    A leaf that declares the field itself overrides the manifest and is
    unaffected — the one case an operator can otherwise only find by reading
    every leaf.
    """
    changes: list[dict[str, Any]] = []
    for field, header, removed in units:
        affected: list[str] = []
        for leaf in leaves:
            own = leaf["own"]
            if field == "headers":
                if header not in (own.get("headers") or {}):
                    affected.append(leaf["name"])
            elif field not in own:
                affected.append(leaf["name"])
        changes.append(
            {"field": field, "header": header, "removed": removed, "leaves": affected}
        )
    return changes


def _connection_of(text: str) -> dict[str, Any]:
    """The ``connection:`` block of a manifest text, or an empty mapping."""
    try:
        cfg = yaml_load(text)
    except Exception:  # noqa: BLE001 - the caller validated the text first
        return {}
    if not isinstance(cfg, dict):
        return {}
    connection = cfg.get("connection")
    return dict(connection) if isinstance(connection, dict) else {}


def _changed_leaves(
    before: dict[str, Any],
    after: dict[str, Any],
    leaves: list[dict[str, Any]],
    manifest_path: Path,
) -> list[str]:
    """The leaves whose *effective* connection differs between two manifests.

    The effective value is the loader's merge, so this names a leaf only when
    the save actually changes what it would use.
    """
    changed: list[str] = []
    for leaf in leaves:
        was = _merged_connection(before, leaf["own"], manifest_path)
        now = _merged_connection(after, leaf["own"], manifest_path)
        if was != now:
            changed.append(leaf["name"])
    return changed


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


async def post_preview_shim_manifest(request: Any) -> Response:
    """Say which leaves an unsaved edit would change, without writing anything.

    The editor calls this while the operator types, so a vendor-wide change is
    visible before Save: ``changes`` names the field and the leaves that
    inherit it, and ``changed_leaves`` is the union.
    """
    body, err = parse_json_body(request)
    if err:
        return err
    if not isinstance(body, dict) or not isinstance(body.get("yaml"), str):
        return JSONResponse({"error": "Body must be {'yaml': <text>}"}, status_code=400)
    entry = _find_entry(request, body.get("path"))
    if entry is None:
        return JSONResponse({"error": "No such vendor manifest"}, status_code=404)

    problem = manifest_problem(body["yaml"])
    if problem is not None:
        return JSONResponse({"error": problem}, status_code=400)

    path = Path(entry["path"])
    try:
        before = path.read_text(encoding="utf-8")
    except OSError as exc:
        return JSONResponse({"error": f"Cannot read {path}: {exc}"}, status_code=500)
    leaves = _leaf_entries(path)
    old, new = _connection_of(before), _connection_of(body["yaml"])
    changes = _affected_leaves(_units(old, new), leaves)
    changed = _changed_leaves(old, new, leaves, path)
    return JSONResponse(
        {
            "ok": True,
            "changes": changes,
            "changed_leaves": changed,
            "unchanged_leaves": [
                leaf["name"] for leaf in leaves if leaf["name"] not in changed
            ],
        }
    )


async def put_shim_manifest(request: Any) -> Response:
    """Save an edited vendor manifest, restoring it if the reload rejects it.

    The reload is the loader's own verdict: a manifest that makes a leaf fail
    to register is rolled back, exactly as :func:`.shim_files.put_loaded_shim`
    does for a leaf.  The response names the leaves whose effective connection
    the save changed.
    """
    body, err = parse_json_body(request)
    if err:
        return err
    if not isinstance(body, dict) or not isinstance(body.get("yaml"), str):
        return JSONResponse({"error": "Body must be {'yaml': <text>}"}, status_code=400)

    entry = _find_entry(request, body.get("path"))
    if entry is None:
        return JSONResponse({"error": "No such vendor manifest"}, status_code=404)
    row = _row(entry)
    if not row["editable"]:
        return JSONResponse({"error": row["read_only_reason"]}, status_code=403)
    problem = manifest_problem(body["yaml"])
    if problem is not None:
        return JSONResponse({"error": problem}, status_code=400)

    path = Path(entry["path"])
    try:
        backup = path.read_text(encoding="utf-8")
    except OSError as exc:
        return JSONResponse({"error": f"Cannot read {path}: {exc}"}, status_code=500)

    text = body["yaml"]
    if text and not text.endswith("\n"):
        text += "\n"
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        return JSONResponse({"error": f"Cannot write {path}: {exc}"}, status_code=500)

    complaint = await _rebuild_shims(request)
    if complaint is not None:
        path.write_text(backup, encoding="utf-8")
        await _rebuild_shims(request)
        return JSONResponse({"error": complaint}, status_code=400)

    leaves = _leaf_entries(path)
    changed = _changed_leaves(
        _connection_of(backup), _connection_of(text), leaves, path
    )
    return JSONResponse(
        {
            "ok": True,
            "manifest": _detail({**entry, "path": str(path)}),
            "changed_leaves": changed,
            "unchanged_leaves": [
                leaf["name"] for leaf in leaves if leaf["name"] not in changed
            ],
        }
    )
