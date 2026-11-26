"""Loaded shims: where each one is defined, and the switch that turns one off.

A shim is a ``provider.yaml`` (plus optional ``transforms.py``) in a directory,
so the admin panel can show every loaded shim and the text of its definition.
Every definition is read-only here: a built-in ships inside the package, and an
installed extension is a fetched artifact that the next update would overwrite.
The enable switch is a config declaration (``shims.disabled``), not an edit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_rosetta_gateway._vendor.yaml import load as yaml_load
from llm_rosetta.api import list_shims
from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from .._shared import (
    _get_config_io,
    _get_config_path,
    config_mutate,
    parse_json_body,
)

#: Why a definition is read-only in the panel, keyed by origin.
_READ_ONLY_REASON = {
    "builtin": (
        "A built-in shim ships inside the package: edit it in the core repository."
    ),
    "remote": (
        "An installed extension is a fetched artifact: the next update would "
        "overwrite an edit. Edit it in its own repository."
    ),
    "plugin": (
        "A plugin shim is registered by an installed package: edit it in that package."
    ),
}


def _plugin_source(name: str) -> str | None:
    """Return the distribution that registered plugin shim *name*, if any.

    A plugin shim has no ``provider.yaml`` for the index to walk to, so this
    is the only way the inventory can say where its code came from.
    """
    from llm_rosetta.api import plugin_source

    return plugin_source(name)


def _builtin_root() -> Path:
    """Return the directory the bundled shims are loaded from."""
    from llm_rosetta.api import builtin_providers_dir

    return builtin_providers_dir()


def _yaml_meta(path: Path) -> dict[str, Any] | None:
    """Return the fields the inventory needs from a ``provider.yaml``."""
    try:
        cfg = yaml_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict) or not isinstance(cfg.get("name"), str):
        return None
    model_type = cfg.get("model_type") or "llm"
    group = cfg.get("group")
    return {
        "name": cfg["name"],
        "model_type": model_type if isinstance(model_type, str) else "llm",
        "format": cfg.get("format") or "",
        "display_name": cfg.get("display_name"),
        # A manifest-backed shim names the vendor it belongs to; a disabled
        # one is read from this index alone, and without the group it would
        # surface as a vendor of its own.
        "group": group if isinstance(group, str) else None,
        # The mark and the group title identify the vendor, and a declined shim
        # is only ever read from this index — without them its card would fall
        # back to initials while the loaded sibling beside it shows the mark.
        "group_label": cfg.get("group_label"),
        "logo": cfg.get("logo"),
        "free_source": bool(cfg.get("free_source")),
    }


def _shim_roots(request: Any) -> list[tuple[str, str, Path]]:
    """Every root a definition may live under, in the order the loader reads.

    Built-ins first, then the declared sources in report order, so a source
    that shadows a built-in name wins — the registry keeps the later
    registration, and the panel must show the file that is actually in force.
    """
    report = getattr(request.app, "shim_source_report", None)
    roots: list[tuple[str, str, Path]] = [("builtin", "", _builtin_root())]
    for status in getattr(report, "sources", []) or []:
        resolved = getattr(status, "resolved_dir", None)
        if not getattr(status, "ok", False) or not resolved:
            continue
        roots.append((status.kind, status.location, Path(resolved)))
    return roots


def _shim_index(request: Any) -> dict[str, dict[str, Any]]:
    """Map every shim name to the file that defines it.

    Cached on the app, keyed by the last rebuild time: the walk reads one file
    per shim, and the report only changes when the sources are rebuilt.
    """
    report = getattr(request.app, "shim_source_report", None)
    stamp = getattr(report, "rebuilt_at", 0.0) if report is not None else 0.0
    cached = getattr(request.app, "_shim_file_index", None)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    index: dict[str, dict[str, Any]] = {}
    for origin, location, root in _shim_roots(request):
        if not root.is_dir():
            continue
        for yaml_path in sorted(root.rglob("provider.yaml")):
            meta = _yaml_meta(yaml_path)
            if meta:
                index[meta["name"]] = {
                    "origin": origin,
                    "source": location,
                    "root": str(root),
                    "path": str(yaml_path),
                    **{k: v for k, v in meta.items() if k != "name"},
                }
    request.app._shim_file_index = (stamp, index)
    return index


def _shim_usage(config: Any) -> dict[str, list[str]]:
    """Map shim name to the configured providers that name it.

    Reads ``provider_shim_names``: ``provider_types`` holds the wire format,
    which a bare-format provider shares with every vendor shim of that format
    and so cannot say which shim it uses.
    """
    usage: dict[str, list[str]] = {}
    for provider, shim in (getattr(config, "provider_shim_names", {}) or {}).items():
        if isinstance(shim, str) and shim:
            usage.setdefault(shim, []).append(provider)
    return usage


def _disabled_names(request: Any) -> list[str]:
    """Return ``shims.disabled`` as it stands on disk, or an empty list."""
    try:
        raw = _get_config_io(request).load_raw(_get_config_path(request))
    except Exception:  # noqa: BLE001 - the inventory must still render
        return []
    section = raw.get("shims") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        return []
    names = section.get("disabled")
    if not isinstance(names, list):
        return []
    return [name for name in names if isinstance(name, str)]


def _read_only_reason(row: dict[str, Any]) -> str:
    """Why a row's definition cannot be edited here."""
    return _READ_ONLY_REASON.get(row["origin"]) or "This shim cannot be edited here."


def _manifest_above(entry: dict[str, Any]) -> str | None:
    """The vendor manifest a leaf inherits from, when one sits above it.

    A disabled shim is not in the registry, so its row cannot read
    :attr:`ProviderShim.manifest`; the file walk answers the same question.
    A ``provider.yaml`` with no ``name`` is a manifest, not a shim.
    """
    path, root = entry.get("path"), entry.get("root")
    if not path or not root:
        return None
    try:
        leaf = Path(path).resolve()
        root_path = Path(root).resolve()
    except OSError:
        return None
    for parent in leaf.parents:
        if parent == root_path:
            break
        candidate = parent / "provider.yaml"
        if candidate == leaf or not candidate.is_file():
            continue
        if _yaml_meta(candidate) is None:
            return str(candidate)
    return None


def _shim_rows(request: Any) -> list[dict[str, Any]]:
    """Return one row per shim the sources define, sources first.

    A disabled shim is not in the registry, so its metadata comes from the
    ``provider.yaml`` the index found: it has to stay visible, because the
    switch that hid it lives here.
    """
    index = _shim_index(request)
    usage = _shim_usage(getattr(request.app, "gateway_config", None))
    disabled = set(_disabled_names(request))
    rows: dict[str, dict[str, Any]] = {}
    for shim in list_shims():
        entry = index.get(shim.name) or {}
        plugin = "" if entry else (_plugin_source(shim.name) or "")
        origin = entry.get("origin", "plugin" if plugin else "builtin")
        rows[shim.name] = {
            "name": shim.name,
            "display_name": shim.display_name,
            "group": shim.group,
            "group_label": shim.group_label,
            "logo": shim.logo,
            "model_type": shim.model_type,
            "format": shim.format,
            "origin": origin,
            "source": entry.get("source") or plugin,
            "path": entry.get("path"),
            "manifest_path": (
                str(shim.manifest.path) if shim.manifest is not None else None
            ),
            "enabled": shim.name not in disabled,
            "free_source": bool(shim.free_source),
            "used_by": sorted(usage.get(shim.name, [])),
        }
    for name, entry in index.items():
        if name in rows:
            continue
        origin = entry.get("origin", "unknown")
        rows[name] = {
            "name": name,
            "display_name": entry.get("display_name"),
            "group": entry.get("group"),
            "group_label": entry.get("group_label"),
            "logo": entry.get("logo"),
            "model_type": entry.get("model_type", "llm"),
            "format": entry.get("format", ""),
            "origin": origin,
            "source": entry.get("source", ""),
            "path": entry.get("path"),
            "manifest_path": _manifest_above(entry),
            "enabled": False,
            "free_source": bool(entry.get("free_source")),
            "used_by": sorted(usage.get(name, [])),
        }
    return sorted(rows.values(), key=lambda row: row["name"])


def _find_row(request: Any, name: str) -> dict[str, Any] | None:
    """Return the row for *name*, or ``None`` when no such shim is loaded."""
    for row in _shim_rows(request):
        if row["name"] == name:
            return row
    return None


def _read_yaml(path: str | None) -> str | None:
    """Return a shim definition's text, or ``None`` when it cannot be read."""
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


async def post_shim_enabled(request: Any, name: str) -> Response:
    """Turn one shim off or on by declaring it in ``shims.disabled``.

    A switch rather than a deletion: the source keeps providing the shim, so
    enabling the name again restores the same definition.  Refused while a
    configured provider still names the shim, for the same reason deletion is.
    """
    body, err = parse_json_body(request)
    if err:
        return err
    if not isinstance(body, dict) or not isinstance(body.get("enabled"), bool):
        return JSONResponse(
            {"error": "Body must be {'enabled': <bool>}"}, status_code=400
        )
    enabled = body["enabled"]

    row = _find_row(request, name)
    if row is None:
        return JSONResponse({"error": f"Shim {name!r} is not known"}, status_code=404)
    if not enabled and row["used_by"]:
        return JSONResponse(
            {
                "error": "Providers still use this shim: " + ", ".join(row["used_by"]),
                "providers": row["used_by"],
            },
            status_code=409,
        )

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error
        section = ctx.data.get("shims")
        if not isinstance(section, dict):
            section = {}
        current = [
            n
            for n in (section.get("disabled") or [])
            if isinstance(n, str) and n != name
        ]
        if not enabled:
            current.append(name)
        if current:
            section["disabled"] = sorted(set(current))
        else:
            section.pop("disabled", None)
        if section:
            ctx.data["shims"] = section
        else:
            ctx.data.pop("shims", None)
        ctx.commit()

    if ctx.error:
        return ctx.error
    return JSONResponse({"ok": True, "shim": _find_row(request, name)})


async def get_loaded_shims(request: Any) -> Response:
    """List every loaded shim with its origin and the providers that use it.

    Vendor manifests ride along under their own key: they have no shim name,
    so they are not rows of ``shims`` — that list is what resolves a shim —
    but the inventory renders them above the leaves they feed.
    """
    from .shim_manifests import manifest_rows

    return JSONResponse(
        {"shims": _shim_rows(request), "manifests": manifest_rows(request)}
    )


def release_requires_problem(release: dict[str, Any]) -> str | None:
    """Why an installed release record does not fit this core.

    ``None`` when it fits.  The rule is core's own
    (:func:`llm_rosetta.api.entry_problems`).
    """
    from llm_rosetta.api import IndexEntry, entry_problems

    try:
        entry = IndexEntry(
            id="installed",
            version=str(release["version"]),
            url="",
            sha256="",
            requires_python=release.get("requires_python"),
            surface=release.get("surface"),
            ir=release.get("ir"),
        )
    except (KeyError, TypeError, ValueError):
        return "the installed release names no readable requirement"
    problems = entry_problems(entry)
    return "; ".join(problems) if problems else None


async def get_loaded_shim(request: Any, name: str) -> Response:
    """Return one loaded shim, including the text of its definition."""
    row = _find_row(request, name)
    if row is None:
        return JSONResponse({"error": f"Shim {name!r} is not loaded"}, status_code=404)
    detail = dict(row)
    detail["yaml"] = _read_yaml(row.get("path"))
    detail["read_only_reason"] = _read_only_reason(row)
    return JSONResponse(detail)
