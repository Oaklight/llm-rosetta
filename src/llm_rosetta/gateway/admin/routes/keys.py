"""API key management route handlers (SQLite keystore)."""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response
from llm_rosetta.observability.persistence import (
    USAGE_TOKEN_COLUMNS,
    USAGE_WINDOW_LABELS,
)

from ...keystore import INTERNAL_LABEL, INTERNAL_USAGE_BUCKET, KeyStore
from ._shared import parse_json_body


# Shown for traffic whose row carries no key label.  Either the request
# presented no key at all and was served in open mode, or the key that made it
# was created without a label and has since been deleted.
_NO_LABEL = "(no credential)"


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


def _zero_windows() -> dict[str, dict[str, int]]:
    """A full window map with every counter at zero."""
    empty = {"request_count": 0, **dict.fromkeys(USAGE_TOKEN_COLUMNS, 0)}
    return {name: dict(empty) for name in USAGE_WINDOW_LABELS}


def _add_windows(
    target: dict[str, dict[str, int]], source: dict[str, dict[str, int]]
) -> None:
    """Add *source* windows into *target* in place."""
    for name, counts in source.items():
        slot = target.setdefault(
            name, {"request_count": 0, **dict.fromkeys(USAGE_TOKEN_COLUMNS, 0)}
        )
        for counter, value in counts.items():
            slot[counter] = slot.get(counter, 0) + value


async def get_key_usage(request: Any) -> Response:
    """Return per-key token usage over the rolling usage windows.

    Keys are attributed by credential bucket, so a renamed key keeps its
    usage and a rotated key starts fresh.  Every live key is present,
    zero-filled when it has no traffic.

    The report accounts for every request the dashboard counts, not only the
    live keys.  Usage belonging to the admin panel's own token is reported
    under ``internal``, and everything no live key owns — a deleted or
    rotated-away secret, a request logged before the bucket column existed,
    an anonymous request — is summed into ``unattributed`` with its per-label
    split, so the panel rows always add up to the dashboard.

    An anonymous request can only be logged at all when the gateway runs with
    ``server.open_on_no_keys`` and no keys configured: with keys configured, a
    missing one is refused by the auth hook, which writes no log row.

    A key whose label another live key also carries is flagged
    ``shared_label``: the row can only ever read zero, because the label on a
    log row cannot say which of the two spent the tokens, and their traffic
    sits in the remainder.
    """
    keystore = _get_keystore(request)
    persistence = getattr(request.app, "persistence", None)

    keys = await keystore.usage_keys()
    live_buckets = {key["usage_bucket"] for key in keys if key["usage_bucket"]}
    # A live key labelled "internal" owns that label: rows carrying it that
    # predate the bucket column are that key's, not the admin token's.
    internal_claimed = any(key["label"] == INTERNAL_LABEL for key in keys)

    by_bucket: dict[str, dict[str, dict[str, int]]] = {}
    internal: dict[str, dict[str, int]] = _zero_windows()
    internal_seen = False
    unattributed = _zero_windows()
    by_label: dict[str, dict[str, dict[str, int]]] = {}

    if persistence is not None:
        report = await persistence.query_token_usage_by_key()
        for group in report["groups"]:
            bucket, label, windows = group["bucket"], group["label"], group["windows"]
            if bucket in live_buckets:
                _add_windows(by_bucket.setdefault(bucket, _zero_windows()), windows)
            elif bucket == INTERNAL_USAGE_BUCKET or (
                bucket is None and label == INTERNAL_LABEL and not internal_claimed
            ):
                _add_windows(internal, windows)
                internal_seen = True
            else:
                _add_windows(unattributed, windows)
                _add_windows(
                    by_label.setdefault(label or _NO_LABEL, _zero_windows()), windows
                )

    label_counts: dict[str, int] = {}
    for key in keys:
        label = key["label"]
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1

    result: dict[str, Any] = {}
    for key in keys:
        bucket = key["usage_bucket"]
        label = key["label"]
        entry: dict[str, Any] = {
            "label": label,
            "windows": (by_bucket.get(bucket) if bucket else None) or _zero_windows(),
        }
        if label and label_counts[label] > 1:
            entry["shared_label"] = True
        result[key["id"]] = entry

    payload: dict[str, Any] = {
        "windows": list(USAGE_WINDOW_LABELS),
        "keys": result,
    }
    if internal_seen:
        payload["internal"] = {"windows": internal}
    if any(unattributed[name]["request_count"] for name in USAGE_WINDOW_LABELS):
        payload["unattributed"] = {"windows": unattributed, "by_label": by_label}
    return JSONResponse(payload)


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
