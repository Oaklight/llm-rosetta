"""API key management route handlers (SQLite keystore)."""

from __future__ import annotations

import re
import time

from collections.abc import Mapping
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta_gateway._vendor.ratelimit import parse_quota
from llm_rosetta.observability.usage import UsageReport

from ...keystore import INTERNAL_LABEL, INTERNAL_USAGE_BUCKET, KeyStore
from ._shared import parse_json_body


def _get_keystore(request: Any) -> KeyStore:
    ks = getattr(request.app, "keystore", None)
    if ks is None:
        raise RuntimeError("KeyStore not configured on this application")
    return ks


# _log_key_event removed — replaced by OpsKey* in gateway/ops/keys.py

#: What a model name may look like: the vendor spellings in use, no spaces.
_MODEL_NAME = re.compile(r"^[A-Za-z0-9._:\-/@+]+$")


def _key_models(raw: Any) -> tuple[list[str] | None, str | None]:
    """A key's allowed models, deduplicated, or the reason they are refused.

    Both input forms are read — the admin sends a list, a hand-written request
    may send the comma string the field shows — so one rule covers both.
    """
    if raw is None:
        return None, None
    if isinstance(raw, str):
        parts: list[Any] = re.split(r"[,\s]+", raw)
    elif isinstance(raw, list):
        parts = raw
    else:
        return None, "allowed_models must be a list of names"
    models: list[str] = []
    for part in parts:
        name = str(part).strip()
        if not name or name in models:
            continue
        if name == "*":
            # The stored spelling of "every model".  The panel sends it back
            # unchanged when the operator picks nothing, so it has to be one
            # of the values this validator accepts.
            return ["*"], None
        if not _MODEL_NAME.match(name):
            return None, f"Invalid model name '{name}'"
        models.append(name)
    # An empty value is "no model", the state a damaged record reads as; the
    # way to say "every model" is `*`, which the panel sends.  Omitting the
    # field leaves the stored value alone (the `None` branch above).
    return models, None


def _key_quota(raw: Any) -> tuple[str, str | None]:
    """A key's own rate limit, normalized to the settings fields' form.

    ``"60/m"`` or ``"30/m, 500/h"`` — the same strings the settings rate
    limit fields take, validated here so a bad spec is never stored.
    """
    if raw is None:
        return "", None
    if isinstance(raw, list):
        raw = ", ".join(str(part) for part in raw)
    text = str(raw).strip()
    if not text:
        return "", None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    for part in parts:
        try:
            parsed = parse_quota(part)
        except ValueError:
            return text, (
                f"Invalid rate limit '{part}': use N/unit, e.g. 60/m or 30/m, 500/h"
            )
        if parsed["limit"] < 1:
            # A window of zero is not "no limit" — it refuses every request —
            # and the fixed-window limiter refuses to be built with it.
            return text, (
                f"Rate limit '{part}' must allow at least one request; "
                "leave the field blank for no limit"
            )
    return ", ".join(parts), None


async def get_api_keys(request: Any) -> Response:
    """List all gateway API keys (no secrets returned)."""
    keystore = _get_keystore(request)
    return JSONResponse({"keys": await keystore.list_keys()})


async def get_key_usage(request: Any) -> Response:
    """Return per-key token usage over the rolling usage windows.

    The classification — which rows are attributable to a live key, which are
    the admin panel's own traffic, and what nothing owns — lives in
    :class:`~llm_rosetta.observability.usage.UsageReport`, because a library
    user with no gateway needs the same answer.  This handler supplies the two
    things only the gateway knows: the credential bucket of every live key, and
    the sentinel its own token authenticates with.

    Keys are attributed by credential bucket, so a renamed key keeps its usage
    and a rotated key starts fresh.  Every live key is present, zero-filled when
    it has no traffic, and the report's rows always add up to the dashboard's
    total — see the report for what each class means.
    """
    keystore = _get_keystore(request)
    persistence = getattr(request.app, "persistence", None)

    # The panel asks for this on tab paint and again on every window switch,
    # and the answer is one pass over the widest usage window.  Five seconds of
    # memory makes the switch free without making the numbers stale to the eye
    # (#1522); the cache lives on the app, so it cannot outlive its store.
    cached = getattr(request.app, "_usage_payload", None)
    if cached and time.monotonic() - cached[0] < 5.0:
        return JSONResponse(cached[1])

    keys = await keystore.usage_keys()
    subjects = [
        (key["usage_bucket"], key["label"]) for key in keys if key["usage_bucket"]
    ]
    key_ids = {key["usage_bucket"]: key["id"] for key in keys if key["usage_bucket"]}

    groups: list[Mapping[str, Any]] = []
    if persistence is not None:
        groups = (await persistence.query_usage_groups())["groups"]

    report = UsageReport.build(
        groups=groups,
        subjects=subjects,
        internal_subject=INTERNAL_USAGE_BUCKET,
        internal_label=INTERNAL_LABEL,
    )
    payload = report.to_payload(key_ids)
    request.app._usage_payload = (time.monotonic(), payload)
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
    allowed_models, models_err = _key_models(body.get("allowed_models"))
    rate_limit, quota_err = _key_quota(body.get("rate_limit"))
    for field, error in (("allowed_models", models_err), ("rate_limit", quota_err)):
        if error:
            return JSONResponse({"error": error, "field": field}, status_code=400)

    try:
        key_id, raw_key = await keystore.create_key(
            label=label,
            allowed_shims=allowed_shims,
            allowed_models=allowed_models,
            rate_limit=rate_limit,
            manual_key=manual_key,
        )
    except Exception as exc:
        return JSONResponse({"error": f"Failed to create key: {exc}"}, status_code=500)

    entry = await keystore.list_keys()
    created_entry = next((k for k in entry if k["id"] == key_id), {"id": key_id})
    created_entry["key"] = raw_key
    from llm_rosetta_gateway.ops.keys import OpsKeyCreate

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
    allowed_models, models_err = _key_models(body.get("allowed_models"))
    rate_limit, quota_err = _key_quota(body.get("rate_limit"))
    for field, error in (("allowed_models", models_err), ("rate_limit", quota_err)):
        if error:
            return JSONResponse({"error": error, "field": field}, status_code=400)

    if not await keystore.update(
        key_id,
        label=label,
        allowed_shims=allowed_shims,
        allowed_models=allowed_models,
        rate_limit=None if body.get("rate_limit") is None else rate_limit,
    ):
        return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)

    result: dict[str, Any] = {"ok": True, "id": key_id}
    if label is not None:
        result["label"] = label
    if allowed_shims is not None:
        result["allowed_shims"] = allowed_shims
    if allowed_models is not None:
        # What the store now holds, not a spelling of it: `[]` and `["*"]` are
        # different keys, and a client that echoes this body back must not be
        # handed the wide one for a key that allows nothing.
        result["allowed_models"] = allowed_models
    if body.get("rate_limit") is not None:
        result["rate_limit"] = rate_limit
    from llm_rosetta_gateway.ops.keys import OpsKeyUpdate

    changed = [
        k
        for k in ("label", "allowed_shims", "allowed_models", "rate_limit")
        if body.get(k) is not None
    ]
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

    from llm_rosetta_gateway.ops.keys import OpsKeyDelete

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

    from llm_rosetta_gateway.ops.keys import OpsKeyRotate

    _ops_ctx = getattr(request.app, "ops_ctx", None)
    if _ops_ctx is not None:
        await OpsKeyRotate(_ops_ctx, key_id=key_id, label=label).execute()
    return JSONResponse({"ok": True, "id": key_id, "key": new_key})


async def get_internal_token(request: Any) -> Response:
    """Return the ephemeral internal token for admin panel test requests."""
    token = getattr(request.app, "internal_token", None)
    if not token:
        return JSONResponse({"error": "No internal token available"}, status_code=500)
    return JSONResponse({"token": token})
