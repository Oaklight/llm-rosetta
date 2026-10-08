"""API key management route handlers (SQLite keystore)."""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ...keystore import KeyStore
from ._shared import parse_json_body


def _get_keystore(request: Any) -> KeyStore:
    ks = getattr(request.app, "keystore", None)
    if ks is None:
        raise RuntimeError("KeyStore not configured on this application")
    return ks


# _log_key_event removed — replaced by OpsKey* in gateway/ops/keys.py


async def get_api_keys(request: Any) -> Response:
    """List all gateway API keys (no secrets returned)."""
    keystore = _get_keystore(request)
    return JSONResponse({"keys": await keystore.list_keys()})


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
        key_id, raw_key = await keystore.create_key(
            label=label,
            allowed_shims=allowed_shims,
            manual_key=manual_key,
        )
    except Exception as exc:
        return JSONResponse({"error": f"Failed to create key: {exc}"}, status_code=500)

    entry = await keystore.list_keys()
    created_entry = next((k for k in entry if k["id"] == key_id), {"id": key_id})
    created_entry["key"] = raw_key
    from llm_rosetta.gateway.ops.keys import OpsKeyCreate

    _ops_ctx = getattr(request.app, "ops_ctx", None)
    if _ops_ctx is not None:
        await OpsKeyCreate(_ops_ctx, key_id=key_id, label=label or "").execute()
    return JSONResponse({"ok": True, "key": created_entry})


async def update_api_key(request: Any, **kwargs: Any) -> Response:
    """Update an API key's label and/or allowed_shims."""
    keystore = _get_keystore(request)
    key_id = request.path_params["key_id"]

    body, err = parse_json_body(request)
    if err:
        return err

    label = body.get("label")
    allowed_shims = body.get("allowed_shims")

    if not await keystore.update(key_id, label=label, allowed_shims=allowed_shims):
        return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)

    result: dict[str, Any] = {"ok": True, "id": key_id}
    if label is not None:
        result["label"] = label
    if allowed_shims is not None:
        result["allowed_shims"] = allowed_shims
    from llm_rosetta.gateway.ops.keys import OpsKeyUpdate

    changed = [k for k in ("label", "allowed_shims") if body.get(k) is not None]
    _ops_ctx = getattr(request.app, "ops_ctx", None)
    if _ops_ctx is not None:
        await OpsKeyUpdate(_ops_ctx, key_id=key_id, changed_fields=changed).execute()
    return JSONResponse(result)


async def delete_api_key(request: Any, **kwargs: Any) -> Response:
    """Delete a gateway API key."""
    keystore = _get_keystore(request)
    key_id = request.path_params["key_id"]

    # Capture label before deletion (gone afterward)
    entries = await keystore.list_keys()
    entry = next((k for k in entries if k["id"] == key_id), None)
    label = entry.get("label") if entry else None

    if not await keystore.delete(key_id):
        return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)

    from llm_rosetta.gateway.ops.keys import OpsKeyDelete

    _ops_ctx = getattr(request.app, "ops_ctx", None)
    if _ops_ctx is not None:
        await OpsKeyDelete(_ops_ctx, key_id=key_id, label=label).execute()
    return JSONResponse({"ok": True, "deleted": key_id})


async def rotate_api_key(request: Any, **kwargs: Any) -> Response:
    """Rotate an API key: generate a new value, keep the same id and label."""
    keystore = _get_keystore(request)
    key_id = request.path_params["key_id"]

    entries = await keystore.list_keys()
    entry = next((k for k in entries if k["id"] == key_id), None)
    label = entry.get("label") if entry else None

    new_key = await keystore.rotate(key_id)
    if new_key is None:
        return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)

    from llm_rosetta.gateway.ops.keys import OpsKeyRotate

    _ops_ctx = getattr(request.app, "ops_ctx", None)
    if _ops_ctx is not None:
        await OpsKeyRotate(_ops_ctx, key_id=key_id, label=label).execute()
    return JSONResponse({"ok": True, "id": key_id, "key": new_key})


async def backfill_keys_last_used(request: Any, **kwargs: Any) -> Response:
    """Backfill last_used from request log for keys missing the value."""
    keystore = _get_keystore(request)
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    updated = await keystore.backfill_last_used(persistence.db_path)
    return JSONResponse({"updated": updated})


async def get_internal_token(request: Any) -> Response:
    """Return the ephemeral internal token for admin panel test requests."""
    token = getattr(request.app, "internal_token", None)
    if not token:
        return JSONResponse({"error": "No internal token available"}, status_code=500)
    return JSONResponse({"token": token})
