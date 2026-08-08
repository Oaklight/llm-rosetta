"""API key management route handlers (SQLite keystore)."""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ...keystore import KeyStore


def _get_keystore(request: Any) -> KeyStore:
    ks = getattr(request.app, "keystore", None)
    if ks is None:
        raise RuntimeError("KeyStore not configured on this application")
    return ks


async def get_api_keys(request: Any) -> Response:
    """List all gateway API keys (no secrets returned)."""
    keystore = _get_keystore(request)
    return JSONResponse({"keys": keystore.list_keys()})


async def create_api_key(request: Any) -> Response:
    """Create a new gateway API key."""
    keystore = _get_keystore(request)

    try:
        body = request.json()
    except Exception:
        body = {}

    label = body.get("label", "")
    manual_key = body.get("key")
    allowed_shims = body.get("allowed_shims")

    try:
        key_id, raw_key = keystore.create(
            label=label,
            allowed_shims=allowed_shims,
            manual_key=manual_key,
        )
    except Exception as exc:
        return JSONResponse({"error": f"Failed to create key: {exc}"}, status_code=500)

    entry = keystore.list_keys()
    created_entry = next((k for k in entry if k["id"] == key_id), {"id": key_id})
    created_entry["key"] = raw_key
    return JSONResponse({"ok": True, "key": created_entry})


async def update_api_key(request: Any, **kwargs: Any) -> Response:
    """Update an API key's label and/or allowed_shims."""
    keystore = _get_keystore(request)
    key_id = request.path_params["key_id"]

    try:
        body = request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    label = body.get("label")
    allowed_shims = body.get("allowed_shims")

    if not keystore.update(key_id, label=label, allowed_shims=allowed_shims):
        return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)

    result: dict[str, Any] = {"ok": True, "id": key_id}
    if label is not None:
        result["label"] = label
    if allowed_shims is not None:
        result["allowed_shims"] = allowed_shims
    return JSONResponse(result)


async def delete_api_key(request: Any, **kwargs: Any) -> Response:
    """Delete a gateway API key."""
    keystore = _get_keystore(request)
    key_id = request.path_params["key_id"]

    if not keystore.delete(key_id):
        return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)

    return JSONResponse({"ok": True, "deleted": key_id})


async def rotate_api_key(request: Any, **kwargs: Any) -> Response:
    """Rotate an API key: generate a new value, keep the same id and label."""
    keystore = _get_keystore(request)
    key_id = request.path_params["key_id"]

    new_key = keystore.rotate(key_id)
    if new_key is None:
        return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)

    return JSONResponse({"ok": True, "id": key_id, "key": new_key})


async def get_internal_token(request: Any) -> Response:
    """Return the ephemeral internal token for admin panel test requests."""
    token = getattr(request.app, "internal_token", None)
    if not token:
        return JSONResponse({"error": "No internal token available"}, status_code=500)
    return JSONResponse({"token": token})
