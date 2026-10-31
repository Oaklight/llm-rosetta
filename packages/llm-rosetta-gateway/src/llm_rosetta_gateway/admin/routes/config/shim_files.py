"""Loaded shims: where each one is defined, and editing the ones we own.

A shim is a ``provider.yaml`` (plus optional ``transforms.py``) in a directory,
so the admin panel can show every loaded shim and, for the ones living in a
*declared local directory*, edit that file or delete its directory — the
sources the operator already vouched for by listing them in ``shims.paths``.

A built-in shim ships inside the package and a remote one is a fetched
artifact: both are read-only here, because an edit would be lost on upgrade or
on the next refetch.

Deleting a shim removes its directory, so the route refuses while a configured
provider still names the shim.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from llm_rosetta._vendor.yaml import load as yaml_load
from llm_rosetta.shims import list_shims
from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from .._shared import (
    _get_config_io,
    _get_config_path,
    _reload_gateway_config,
    config_mutate,
    parse_json_body,
)

logger = logging.getLogger("llm-rosetta-gateway")

#: Why a shim cannot be edited or deleted from the panel, keyed by origin.
_READ_ONLY_REASON = {
    "builtin": (
        "A built-in shim ships inside the package: edit it in the core "
        "repository, or declare your own copy as a local shim source."
    ),
    "remote": (
        "A remote shim is a fetched artifact: the next refetch would overwrite "
        "an edit. Declare a local copy instead."
    ),
}

#: Logger the shim loader reports a skipped directory on, captured while a
#: saved file is reloaded so the panel can say *why* it did not load.
_LOADER_LOGGER = "llm_rosetta.shims.providers"


def _builtin_root() -> Path:
    """Return the directory the bundled shims are loaded from."""
    from llm_rosetta.shims import providers as _providers

    return Path(_providers.__file__).parent


def _yaml_meta(path: Path) -> dict[str, Any] | None:
    """Return the fields the inventory needs from a ``provider.yaml``."""
    try:
        cfg = yaml_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict) or not isinstance(cfg.get("name"), str):
        return None
    model_type = cfg.get("model_type") or "llm"
    return {
        "name": cfg["name"],
        "model_type": model_type if isinstance(model_type, str) else "llm",
        "format": cfg.get("format") or cfg.get("base") or "",
        "display_name": cfg.get("display_name"),
    }


def _shim_index(request: Any) -> dict[str, dict[str, Any]]:
    """Map every shim name to the file that defines it.

    Built-ins first, then the declared sources in report order, so a source
    that shadows a built-in name wins — the registry keeps the later
    registration, and the panel must show the file that is actually in force.

    Cached on the app, keyed by the last rebuild time: the walk reads one file
    per shim, and the report only changes when the sources are rebuilt.
    """
    report = getattr(request.app, "shim_source_report", None)
    stamp = getattr(report, "rebuilt_at", 0.0) if report is not None else 0.0
    cached = getattr(request.app, "_shim_file_index", None)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    roots: list[tuple[str, str, Path]] = [("builtin", "", _builtin_root())]
    for status in getattr(report, "sources", []) or []:
        resolved = getattr(status, "resolved_dir", None)
        if not getattr(status, "ok", False) or not resolved:
            continue
        roots.append((status.kind, status.location, Path(resolved)))

    index: dict[str, dict[str, Any]] = {}
    for origin, location, root in roots:
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


SOURCE_ROOT_REASON = (
    "This definition sits at the root of a declared source, where the loader "
    "does not look for shims. Move it into its own directory to edit or "
    "delete it here."
)


def _at_source_root(entry: dict[str, Any]) -> bool:
    """Whether a definition's directory *is* the declared source root.

    A file there is not a shim directory: the loader wants
    ``<source>/<name>/provider.yaml``.  Deleting "its directory" would remove
    the whole source, siblings included.
    """
    root, path = entry.get("root"), entry.get("path")
    if not root or not path:
        return False
    try:
        return Path(path).parent.resolve() == Path(root).resolve()
    except OSError:
        return False


def _read_only_reason(row: dict[str, Any]) -> str | None:
    """Why a row cannot be edited or deleted here, or ``None`` when it can."""
    if row.get("at_source_root"):
        return SOURCE_ROOT_REASON
    if row["origin"] not in ("path",):
        return (
            _READ_ONLY_REASON.get(row["origin"]) or "This shim cannot be edited here."
        )
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
        origin = entry.get("origin", "builtin")
        at_root = _at_source_root(entry)
        writable = origin == "path" and not at_root
        rows[shim.name] = {
            "name": shim.name,
            "display_name": shim.display_name,
            "group": shim.group,
            "model_type": shim.model_type,
            "format": shim.format,
            "origin": origin,
            "source": entry.get("source", ""),
            "path": entry.get("path"),
            "enabled": shim.name not in disabled,
            "editable": writable,
            "deletable": writable,
            "at_source_root": at_root,
            "used_by": sorted(usage.get(shim.name, [])),
        }
    for name, entry in index.items():
        if name in rows:
            continue
        origin = entry.get("origin", "unknown")
        at_root = _at_source_root(entry)
        writable = origin == "path" and not at_root
        rows[name] = {
            "name": name,
            "display_name": entry.get("display_name"),
            "group": None,
            "model_type": entry.get("model_type", "llm"),
            "format": entry.get("format", ""),
            "origin": origin,
            "source": entry.get("source", ""),
            "path": entry.get("path"),
            "enabled": False,
            "editable": writable,
            "deletable": writable,
            "at_source_root": at_root,
            "used_by": sorted(usage.get(name, [])),
        }
    ordered = sorted(
        rows.values(), key=lambda row: (row["origin"] != "path", row["name"])
    )
    return ordered


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


def _yaml_problem(text: str, name: str) -> str | None:
    """Return why *text* is not a valid definition for *name*, or ``None``.

    Only the checks the loader would make anyway, so a saved file fails here
    with a message rather than half-registering.
    """
    try:
        cfg = yaml_load(text)
    except Exception as exc:  # noqa: BLE001 - the parser's message is the point
        return f"Not valid YAML: {exc}"
    if not isinstance(cfg, dict):
        return "The file must be a YAML mapping"
    if not isinstance(cfg.get("name"), str):
        return "The file must declare a 'name'"
    if cfg["name"] != name:
        return (
            f"The file declares name {cfg['name']!r}; rename a shim by creating "
            "a new directory instead of editing this one"
        )
    if "format" not in cfg and "base" not in cfg:
        return "The file must declare a 'format'"
    return None


class _CaptureWarnings(logging.Handler):
    """Collect the loader's warnings while a saved file is reloaded."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:  # noqa: BLE001 - a broken record must not break a save
            pass


async def _rebuild_shims(request: Any) -> str | None:
    """Reload the shim sources and the config; return the loader's complaint.

    Returns ``None`` on success, or the last warning the loader emitted — a
    shim that failed to parse is skipped with a warning rather than raised, so
    the warning is the only signal that the save did not take.
    """
    capture = _CaptureWarnings()
    loader_logger = logging.getLogger(_LOADER_LOGGER)
    loader_logger.addHandler(capture)
    try:
        await _reload_gateway_config(
            request, _get_config_path(request), force_shims=True
        )
    except Exception as exc:  # noqa: BLE001 - reported to the admin panel
        return str(exc)
    finally:
        loader_logger.removeHandler(capture)
    return capture.messages[-1] if capture.messages else None


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
    """List every loaded shim with its origin and the providers that use it."""
    return JSONResponse({"shims": _shim_rows(request)})


async def get_loaded_shim(request: Any, name: str) -> Response:
    """Return one loaded shim, including the text of its definition."""
    row = _find_row(request, name)
    if row is None:
        return JSONResponse({"error": f"Shim {name!r} is not loaded"}, status_code=404)
    detail = dict(row)
    detail["yaml"] = _read_yaml(row.get("path"))
    detail["read_only_reason"] = _read_only_reason(row)
    return JSONResponse(detail)


async def put_loaded_shim(request: Any, name: str) -> Response:
    """Save an edited definition for a shim that lives in a local source.

    The previous file is restored when the edited one does not come back as a
    registered shim, so a typo cannot leave the gateway without the shim.
    """
    body, err = parse_json_body(request)
    if err:
        return err
    if not isinstance(body, dict) or not isinstance(body.get("yaml"), str):
        return JSONResponse({"error": "Body must be {'yaml': <text>}"}, status_code=400)

    row = _find_row(request, name)
    if row is None:
        return JSONResponse({"error": f"Shim {name!r} is not loaded"}, status_code=404)
    if not row["editable"]:
        return JSONResponse({"error": _read_only_reason(row)}, status_code=403)
    problem = _yaml_problem(body["yaml"], name)
    if problem is not None:
        return JSONResponse({"error": problem}, status_code=400)

    path = Path(row["path"])
    try:
        backup = path.read_text(encoding="utf-8")
    except OSError as exc:
        return JSONResponse({"error": f"Cannot read {path}: {exc}"}, status_code=500)

    text = body["yaml"]
    if not text.endswith("\n"):
        text += "\n"
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        return JSONResponse({"error": f"Cannot write {path}: {exc}"}, status_code=500)

    complaint = await _rebuild_shims(request)
    if complaint is None and name in {shim.name for shim in list_shims()}:
        return JSONResponse({"ok": True, "shim": _find_row(request, name)})

    path.write_text(backup, encoding="utf-8")
    await _rebuild_shims(request)
    return JSONResponse(
        {
            "error": complaint
            or "The edited file did not register this shim; the previous file was restored."
        },
        status_code=400,
    )


async def delete_loaded_shim(request: Any, name: str) -> Response:
    """Remove a shim that lives in a local source, with its directory.

    Refused while a configured provider still names the shim: the provider
    would stop resolving, and the operator has to choose a replacement anyway.
    """
    row = _find_row(request, name)
    if row is None:
        return JSONResponse({"error": f"Shim {name!r} is not loaded"}, status_code=404)
    if not row["deletable"]:
        return JSONResponse({"error": _read_only_reason(row)}, status_code=403)
    if row["used_by"]:
        return JSONResponse(
            {
                "error": "Providers still use this shim: " + ", ".join(row["used_by"]),
                "providers": row["used_by"],
            },
            status_code=409,
        )

    directory = Path(row["path"]).parent
    try:
        shutil.rmtree(directory)
    except OSError as exc:
        return JSONResponse(
            {"error": f"Cannot remove {directory}: {exc}"}, status_code=500
        )

    complaint = await _rebuild_shims(request)
    if complaint is not None:
        logger.warning("Shim rebuild after deleting %s: %s", name, complaint)
    return JSONResponse({"ok": True, "status": _shim_rows(request)})
