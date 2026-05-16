"""API key management route handlers."""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ...config import GatewayConfig, load_config_raw, write_config
from ._shared import _get_config_path, _mask_api_key, _reload_gateway_config


async def get_api_keys(request: Any) -> Response:
    """List all gateway API keys (values masked)."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    server = data.get("server", {})
    keys = list(server.get("api_keys", []))
    # Backward compat: expose legacy single key as a synthetic entry
    if not keys and server.get("api_key"):
        keys = [
            {
                "id": "default",
                "key": server["api_key"],
                "label": "default",
                "created": "",
            }
        ]

    masked = [{**entry, "key": _mask_api_key(entry.get("key", ""))} for entry in keys]
    return JSONResponse({"keys": masked})


async def create_api_key(request: Any) -> Response:
    """Create a new gateway API key."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    try:
        body = request.json()
    except Exception:
        body = {}

    label = body.get("label", "")
    manual_key = body.get("key")
    key_value = manual_key if manual_key else f"rsk-{secrets.token_hex(16)}"

    entry = {
        "id": uuid.uuid4().hex[:8],
        "key": key_value,
        "label": label,
        "created": datetime.now(timezone.utc).isoformat(),
    }

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    server = data.setdefault("server", {})

    # Migrate legacy single key → api_keys array
    if "api_key" in server and "api_keys" not in server:
        old_key = server.pop("api_key")
        server["api_keys"] = [
            {"id": "default", "key": old_key, "label": "default", "created": ""}
        ]

    server.setdefault("api_keys", []).append(entry)

    try:
        write_config(config_path, data)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to write config: {exc}"}, status_code=500
        )

    try:
        _reload_gateway_config(request, config_path)
    except Exception as exc:
        return JSONResponse(
            {
                "error": f"Config saved but reload failed: {exc}",
                "saved": True,
                "reloaded": False,
            },
            status_code=500,
        )

    # Return the full key exactly once so the user can copy it
    return JSONResponse({"ok": True, "key": entry})


async def update_api_key(request: Any, **kwargs: Any) -> Response:
    """Update an API key's label."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    key_id = request.path_params["key_id"]

    try:
        body = request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    keys = data.get("server", {}).get("api_keys", [])
    target = None
    for entry in keys:
        if entry.get("id") == key_id:
            target = entry
            break

    if target is None:
        return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)

    if "label" in body:
        target["label"] = body["label"]

    try:
        write_config(config_path, data)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to write config: {exc}"}, status_code=500
        )

    try:
        _reload_gateway_config(request, config_path)
    except Exception as exc:
        return JSONResponse(
            {
                "error": f"Config saved but reload failed: {exc}",
                "saved": True,
                "reloaded": False,
            },
            status_code=500,
        )

    return JSONResponse({"ok": True, "id": key_id, "label": target["label"]})


async def delete_api_key(request: Any, **kwargs: Any) -> Response:
    """Delete a gateway API key."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    key_id = request.path_params["key_id"]

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    keys = data.get("server", {}).get("api_keys", [])
    original_len = len(keys)
    keys[:] = [e for e in keys if e.get("id") != key_id]

    if len(keys) == original_len:
        return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)

    try:
        write_config(config_path, data)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to write config: {exc}"}, status_code=500
        )

    try:
        _reload_gateway_config(request, config_path)
    except Exception as exc:
        return JSONResponse(
            {
                "error": f"Config saved but reload failed: {exc}",
                "saved": True,
                "reloaded": False,
            },
            status_code=500,
        )

    return JSONResponse({"ok": True, "deleted": key_id})


async def reveal_api_key(request: Any, **kwargs: Any) -> Response:
    """Return the raw (unmasked) API key value."""
    config: GatewayConfig = request.app.gateway_config
    if not config.credential_visible:
        return JSONResponse(
            {"error": "Credential visibility is disabled"}, status_code=403
        )
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    key_id = request.path_params["key_id"]

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    keys = data.get("server", {}).get("api_keys", [])
    for entry in keys:
        if entry.get("id") == key_id:
            return JSONResponse({"key": entry.get("key", "")})

    return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)


async def get_internal_token(request: Any) -> Response:
    """Return the ephemeral internal token for admin panel test requests."""
    token = getattr(request.app, "internal_token", None)
    if not token:
        return JSONResponse({"error": "No internal token available"}, status_code=500)
    return JSONResponse({"token": token})
