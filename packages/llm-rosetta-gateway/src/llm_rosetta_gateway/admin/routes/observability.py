"""Observability route handlers: metrics, request log, network diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from llm_rosetta_gateway._vendor.httpclient import AsyncClient, Response as HttpResponse
from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from ...config import GatewayConfig
from ._shared import _qp, parse_json_body


def _normalise_time_bound(value: str) -> str:
    """Normalise an ISO 8601 query bound to UTC ``isoformat()``.

    Stored log timestamps are ``datetime.now(timezone.utc).isoformat()``
    strings, so the bounds are converted to the same representation
    before they are compared lexicographically.  A trailing ``Z`` is
    accepted because that is what JavaScript's ``toISOString()`` emits.

    A value without a UTC offset is rejected rather than assumed to be
    UTC: guessing would silently shift the window by the caller's local
    offset, and the error names the expected shape instead.

    Raises:
        ValueError: If *value* is not a parseable ISO 8601 timestamp with
            an explicit offset.
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid ISO 8601 timestamp {value!r} (expected e.g. 2026-10-19T07:00:00Z)"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError(
            f"Timestamp {value!r} has no UTC offset; add Z or an explicit "
            f"offset (e.g. 2026-10-19T07:00:00Z)"
        )
    return parsed.astimezone(timezone.utc).isoformat()


def _bounded_int(
    request: Any, key: str, default: int, *, minimum: int, maximum: int
) -> int:
    """Parse an integer query parameter and clamp it to a closed range.

    ``LIMIT -1`` in SQLite means "no limit", so a negative page size must
    not reach the query.  The clamp also keeps a hand-edited URL from
    asking for the whole table in one response.

    Raises:
        ValueError: If the value is not an integer; callers turn this into
            a 400 rather than letting it surface as a 500.
    """
    raw = _qp(request, key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc
    return max(minimum, min(value, maximum))


def _detect_host_ip() -> dict[str, Any]:
    """Detect the Docker host IP from the default gateway in /proc/net/route.

    This is a synchronous, microsecond-level operation that reads a
    single procfs file — safe to call on every page load.

    Returns:
        Dict with ``ok``, ``ip`` (on success) or ``error`` (on failure).
    """
    try:
        with open("/proc/net/route") as f:
            for line in f:
                fields = line.strip().split()
                if fields[1] == "00000000":  # default route
                    gw = int(fields[2], 16)
                    host_ip = (
                        f"{gw & 0xFF}.{(gw >> 8) & 0xFF}"
                        f".{(gw >> 16) & 0xFF}.{(gw >> 24) & 0xFF}"
                    )
                    return {"ok": True, "ip": host_ip}
        return {"ok": False, "error": "No default route"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _persistence_snapshot(persistence: Any) -> dict[str, Any] | None:
    """Build the persistence sub-block for the metrics snapshot.

    Returns a dict with on-disk byte sizes and per-class entry counts
    plus retention caps, or ``None`` when persistence is not configured
    (e.g. in tests without a config file).
    """
    if persistence is None:
        return None
    try:
        sizes = persistence.db_file_sizes()
        return {
            **sizes,
            "log_entries": await persistence.count_log_entries(),
            "log_success_entries": await persistence.count_success_entries(),
            "log_error_entries": await persistence.count_error_entries(),
            "log_max_success": persistence.success_max,
            "dump_entries": await persistence.count_error_dumps(),
            "dump_max": persistence.dump_max,
        }
    except Exception:
        # Persistence introspection is purely informational; never let
        # it block the metrics endpoint.
        return None


async def get_metrics(request: Any) -> Response:
    """Return a full metrics snapshot."""
    metrics = request.app.metrics
    seconds = int(_qp(request, "seconds", "60"))
    seconds = max(1, min(seconds, 300))
    snap = metrics.snapshot(series_seconds=seconds)

    persistence = getattr(request.app, "persistence", None)
    persistence_snap = await _persistence_snapshot(persistence)
    if persistence_snap is not None:
        snap["persistence"] = persistence_snap

    if persistence is not None:
        snap.update(await persistence.query_rolling_24h_tokens())

    # Include circuit breaker states when the feature is enabled
    config: GatewayConfig | None = getattr(request.app, "gateway_config", None)
    if config is not None and config.circuit_breaker_registry.enabled:
        snap["circuit_breakers"] = config.circuit_breaker_registry.get_all_states()

    return JSONResponse(snap)


async def get_request_histogram(request: Any) -> Response:
    """Count requests per time bucket over the persisted request log.

    Query params:
        minutes: Window length (default 300, at most 10080).
        buckets: Number of buckets (default 60, at most 240).

    The in-memory series stops at five minutes, so the dashboard's longer
    view reads the log instead.
    """
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "Persistence is disabled"}, status_code=404)
    try:
        minutes = int(_qp(request, "minutes", "300"))
        buckets = int(_qp(request, "buckets", "60"))
    except ValueError:
        return JSONResponse(
            {"error": "minutes and buckets must be integers"}, status_code=400
        )
    return JSONResponse(await persistence.query_request_histogram(minutes, buckets))


async def get_circuit_breaker_states(request: Any) -> Response:
    """Return per-provider circuit breaker states.

    Returns an empty dict when circuit breaking is disabled globally.
    """
    config: GatewayConfig | None = getattr(request.app, "gateway_config", None)
    if config is None or not config.circuit_breaker_registry.enabled:
        return JSONResponse({"enabled": False, "providers": {}})
    states = config.circuit_breaker_registry.get_all_states()
    return JSONResponse({"enabled": True, "providers": states})


async def get_rate_limit_status(request: Any) -> Response:
    """Return rate limiter introspection: enabled state and per-dimension status.

    Query params:
        key: Limiter key to peek (default: ``__global__``).
    """
    rate_limit_state = getattr(request.app, "rate_limit_state", None)
    if rate_limit_state is None:
        return JSONResponse({"enabled": False, "dimensions": {}})
    key = _qp(request, "key", "__global__")
    return JSONResponse(rate_limit_state.snapshot(key=key))


# _rebuild_counters_after_mutation removed — now handled by Ops._record()


async def rebuild_metrics(request: Any) -> Response:
    """Rebuild metrics counters from request log entries.

    Useful after fixing a counter bug or when persisted counters
    have drifted from the actual request log data.

    Returns both the pre-rebuild and post-rebuild counter snapshots
    so the admin UI can display what changed (sanity check).
    """
    from llm_rosetta_gateway.ops.data import OpsRebuildMetrics

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None or ctx.persistence is None:
        return JSONResponse(
            {"error": "No persistence configured (in-memory mode)"},
            status_code=400,
        )
    result = await OpsRebuildMetrics(ctx).execute()
    return JSONResponse({"ok": True, **(result or {})})


async def get_token_usage(request: Any) -> Response:
    """Daily token usage, optionally filtered by key label and model type."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"days": [], "totals": {}})
    days = int(_qp(request, "days", "7"))
    days = max(1, min(days, 365))
    api_key_label = _qp(request, "api_key_label")
    model_type = _qp(request, "model_type")
    return JSONResponse(
        await persistence.query_token_usage_by_day(
            days=days, api_key_label=api_key_label, model_type=model_type
        )
    )


async def get_request_key_labels(request: Any) -> Response:
    """Return API key labels seen in request logs."""
    log = request.app.request_log
    return JSONResponse({"labels": await log.get_api_key_labels()})


async def get_requests(request: Any) -> Response:
    """Return paginated, filtered request log entries.

    ``since`` and ``until`` accept ISO 8601 timestamps (``Z`` or an
    explicit offset); ``until`` is exclusive so adjacent windows do not
    double-count the boundary instant.
    """
    log = request.app.request_log
    model = _qp(request, "model")
    provider = _qp(request, "provider")
    status = _qp(request, "status")
    api_key_label = _qp(request, "api_key_label")

    try:
        limit = _bounded_int(request, "limit", 50, minimum=1, maximum=500)
        offset = _bounded_int(request, "offset", 0, minimum=0, maximum=10_000_000)
        since_raw = _qp(request, "since")
        until_raw = _qp(request, "until")
        since = _normalise_time_bound(since_raw) if since_raw else None
        until = _normalise_time_bound(until_raw) if until_raw else None
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    # Resolve provider name → provider type so the query can match old
    # entries that only have target_provider (the API type) without a
    # target_provider_name backfill.  Use the raw config so that even
    # disabled providers (missing API key, etc.) are resolvable.
    provider_type: str | None = None
    if provider:
        config: GatewayConfig | None = getattr(request.app, "gateway_config", None)
        if config is not None:
            provider_type = config.provider_types.get(provider)
        if provider_type is None:
            # Fallback: read from raw config file (covers disabled providers)
            from ._shared import _get_config_path
            from ._shared import _get_config_io

            try:
                config_path = _get_config_path(request)
                raw = _get_config_io(request).load_raw(config_path)
                raw_prov = raw.get("providers", {}).get(provider, {})
                raw_type = raw_prov.get("type") or raw_prov.get("shim")
                if raw_type:
                    from llm_rosetta.provider_names import canonical_provider_name
                    from llm_rosetta.shims import resolve_format

                    provider_type = canonical_provider_name(resolve_format(raw_type))
            except Exception:
                pass

    entries, total = await log.get_entries(
        limit=limit,
        offset=offset,
        model=model,
        provider=provider,
        provider_type=provider_type,
        status=status,
        api_key_label=api_key_label,
        since=since,
        until=until,
    )
    return JSONResponse({"entries": entries, "total": total})


async def get_request_by_id(request: Any, **kwargs: Any) -> Response:
    """Return a single request log entry by ID."""
    log = request.app.request_log
    entry_id = kwargs.get("entry_id") or request.path_params["entry_id"]
    entry = await log.get_entry(entry_id)
    if entry is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(entry)


async def clear_requests(request: Any) -> Response:
    """Clear the request log."""
    from llm_rosetta_gateway.ops.data import OpsClearData

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None:
        return JSONResponse({"error": "No ops context configured"}, status_code=400)
    result = await OpsClearData(ctx, table="request_log").execute()
    return JSONResponse({"ok": True, **(result or {})})


async def get_provider_key(request: Any, **kwargs: Any) -> Response:
    """Return the raw (unmasked) API key for a single provider."""
    config: GatewayConfig = request.app.gateway_config
    if not request.app.auth_state.admin_password:
        return JSONResponse(
            {"error": "Credential reveal requires an admin password"},
            status_code=403,
        )
    if not config.credential_visible:
        return JSONResponse(
            {"error": "Credential visibility is disabled"}, status_code=403
        )
    from ._shared import _get_config_path
    from ._shared import _get_config_io

    config_path = _get_config_path(request)

    name = request.path_params["name"]

    try:
        data = _get_config_io(request).load_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    provider = data.get("providers", {}).get(name)
    if not provider:
        return JSONResponse({"error": f"Provider '{name}' not found"}, status_code=404)

    return JSONResponse({"api_key": provider.get("api_key", "")})


async def network_diagnostics(request: Any) -> Response:
    """Run basic network diagnostics: IP geolocation and Google connectivity.

    Uses the gateway's configured global proxy (if any) so the diagnostics
    reflect the actual outbound path of API requests.
    """
    # Resolve the global proxy from current gateway config
    gw_config: GatewayConfig | None = getattr(request.app, "gateway_config", None)
    proxy_url = gw_config.proxy if gw_config else None

    client_kwargs: dict[str, Any] = {"timeout": 15.0}
    if proxy_url:
        client_kwargs["proxy"] = proxy_url

    results: dict[str, Any] = {}
    if proxy_url:
        results["proxy"] = proxy_url

    # IP geolocation via ip-api.com (no key required, JSON by default)
    try:
        async with AsyncClient(**client_kwargs) as client:
            resp = await client.get(
                "http://ip-api.com/json/?fields=query,country,city,isp"
            )
            assert isinstance(resp, HttpResponse)
            if resp.status_code == 200:
                data = resp.json()
                results["ip"] = {
                    "ok": True,
                    "ip": data.get("query", ""),
                    "country": data.get("country", ""),
                    "city": data.get("city", ""),
                    "isp": data.get("isp", ""),
                }
            else:
                results["ip"] = {"ok": False, "error": f"HTTP {resp.status_code}"}
    except Exception as exc:
        results["ip"] = {"ok": False, "error": str(exc)}

    results["host"] = _detect_host_ip()

    # Google connectivity
    try:
        async with AsyncClient(**client_kwargs) as client:
            resp = await client.get("https://www.google.com/generate_204")
            results["google"] = {
                "ok": resp.status_code == 204,
                "status": resp.status_code,
            }
    except Exception as exc:
        results["google"] = {"ok": False, "error": str(exc)}

    return JSONResponse(results)


async def get_host_ip(request: Any) -> Response:
    """Return the detected Docker host IP (lightweight, no network calls)."""
    return JSONResponse(_detect_host_ip())


# ------------------------------------------------------------------
# Error dumps
# ------------------------------------------------------------------


def _dump_status_bounds(request: Any) -> tuple[int | None, int | None]:
    """Parse ``status`` into an inclusive code range.

    ``4xx``/``5xx`` select a class; a three-digit code selects itself.  The
    toolbar offers both, and the two used to be resolved in the browser over
    whatever page had loaded.

    Raises:
        ValueError: If the value is neither a class nor a code.
    """
    raw = (_qp(request, "status") or "").strip()
    if not raw:
        return None, None
    lowered = raw.lower()
    if len(lowered) == 3 and lowered[1:] == "xx" and lowered[0] in "12345":
        base = int(lowered[0]) * 100
        return base, base + 99
    try:
        code = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"status must be a class like 4xx or a code like 429, got {raw!r}"
        ) from exc
    if not 100 <= code <= 599:
        raise ValueError(f"status must be between 100 and 599, got {code}")
    return code, code


def _dump_filters(request: Any) -> dict[str, Any]:
    """Read the error-dump filters shared by the list and id reads.

    The tar.gz export route parses ``start``/``end`` itself: it is a
    date-range download that also walks the body blobs, so it is not one of
    the reads below.

    Raises:
        ValueError: If a bound is not parseable; callers answer 400.
    """
    status_min, status_max = _dump_status_bounds(request)
    start = (_qp(request, "start") or "").strip()
    end = (_qp(request, "end") or "").strip()
    return {
        "model": (_qp(request, "model") or None),
        "model_search": (_qp(request, "model_search") or None),
        "error_phase": (_qp(request, "error_phase") or None),
        "provider": (_qp(request, "provider") or None),
        "status_min": status_min,
        "status_max": status_max,
        "start": _normalise_time_bound(start) if start else None,
        "end": _normalise_time_bound(end) if end else None,
        "q": (_qp(request, "q") or None),
    }


def _dump_cursor(request: Any) -> tuple[str, str] | None:
    """Read the ``(timestamp, id)`` keyset cursor, or ``None`` when absent.

    The pair is what makes a page boundary stable while the table is being
    written; either half alone does not name a row position.

    Raises:
        ValueError: If only one of the two halves is present, or the
            timestamp carries no offset.
    """
    timestamp = (_qp(request, "before_timestamp") or "").strip()
    dump_id = (_qp(request, "before_id") or "").strip()
    if not timestamp and not dump_id:
        return None
    if not timestamp or not dump_id:
        raise ValueError("before_timestamp and before_id must be given together")
    return _normalise_time_bound(timestamp), dump_id


#: One ``ids=`` read may carry at most this many ids: SQLite binds each one as
#: a separate variable and refuses a statement past its variable limit.  The
#: panel chunks a larger selection to match.
MAX_DUMP_IDS = 500


def _dump_ids(request: Any) -> list[str] | None:
    """Read the ``ids`` parameter as a list, or ``None`` when absent.

    Raises:
        ValueError: If more ids are named than one statement can bind.
    """
    raw = _qp(request, "ids")
    if raw is None or not raw.strip():
        return None
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    if len(ids) > MAX_DUMP_IDS:
        raise ValueError(f"ids accepts at most {MAX_DUMP_IDS} entries, got {len(ids)}")
    return ids


async def get_error_dumps(request: Any) -> Response:
    """Return paginated, filtered error dump entries."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    try:
        limit = _bounded_int(request, "limit", 50, minimum=1, maximum=500)
        offset = _bounded_int(request, "offset", 0, minimum=0, maximum=10_000_000)
        filters = _dump_filters(request)
        ids = _dump_ids(request)
        before = _dump_cursor(request)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    entries, total = await persistence.query_error_dumps(
        limit=limit, offset=offset, ids=ids, before=before, **filters
    )
    return JSONResponse({"entries": entries, "total": total})


async def get_error_dump_ids(request: Any) -> Response:
    """Return the ids of every dump the current filters keep.

    *Select all filtered* and *invert* mean the whole filtered set, not the
    page on screen; this route is how the panel reaches it without loading
    the rows.
    """
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    try:
        filters = _dump_filters(request)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    ids = await persistence.query_error_dump_ids(**filters)
    return JSONResponse({"ids": ids, "total": len(ids)})


async def get_error_dump_facets(request: Any) -> Response:
    """Return the provider and model values present in the dump store."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    return JSONResponse(await persistence.error_dump_facets())


async def get_error_dump_detail(request: Any, **kwargs: Any) -> Response:
    """Return a single error dump with decompressed bodies."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    dump_id = request.path_params["dump_id"]
    entry = await persistence.get_error_dump(dump_id)
    if entry is None:
        return JSONResponse({"error": "Not found"}, status_code=404)

    from llm_rosetta_gateway.observability.error_dump import decompress_body

    # Decompress request body if present
    if "body_hash" in entry:
        body_data = await persistence.get_dump_body(entry["body_hash"])
        if body_data:
            try:
                entry["request_body"] = decompress_body(body_data)
            except Exception:
                entry["request_body"] = None

    # Decompress converted body if present
    if "converted_body_hash" in entry:
        conv_data = await persistence.get_dump_body(entry["converted_body_hash"])
        if conv_data:
            try:
                entry["converted_body"] = decompress_body(conv_data)
            except Exception:
                entry["converted_body"] = None

    return JSONResponse(entry)


async def get_error_dump_body(request: Any, **kwargs: Any) -> Response:
    """Download the raw decompressed request body JSON for an error dump."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    dump_id = request.path_params["dump_id"]
    entry = await persistence.get_error_dump(dump_id)
    if entry is None:
        return JSONResponse({"error": "Not found"}, status_code=404)

    body_hash = entry.get("body_hash")
    if not body_hash:
        return JSONResponse(
            {"error": "No request body stored for this dump"}, status_code=404
        )

    body_data = await persistence.get_dump_body(body_hash)
    if not body_data:
        return JSONResponse({"error": "Body data not found"}, status_code=404)

    import zlib

    raw_json = zlib.decompress(body_data)
    return Response(
        body=raw_json,
        status_code=200,
        content_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="error-dump-{dump_id}.json"'
        },
    )


async def clear_error_dumps(request: Any) -> Response:
    """Delete all error dumps."""
    from llm_rosetta_gateway.ops.data import OpsClearData

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None or ctx.persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    await OpsClearData(ctx, table="error_dumps").execute()
    return JSONResponse({"ok": True})


async def delete_error_dump(request: Any, **kwargs: Any) -> Response:
    """Delete a single error dump by ID."""
    from llm_rosetta_gateway.ops.data import OpsDeleteErrorDump

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None or ctx.persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    dump_id = request.path_params["dump_id"]
    found = await OpsDeleteErrorDump(ctx, dump_id=dump_id).execute()
    if found:
        return JSONResponse({"ok": True})
    return JSONResponse({"error": "Not found"}, status_code=404)


async def db_cleanup(request: Any) -> Response:
    """Delete records older than max_age_days and vacuum the database."""
    from llm_rosetta_gateway.ops.data import OpsCleanupData

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None or ctx.persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    body, err = parse_json_body(request)
    if err:
        return err

    max_age_days = body.get("max_age_days", 90)
    if not isinstance(max_age_days, int) or max_age_days < 1:
        return JSONResponse(
            {"error": "max_age_days must be a positive integer"}, status_code=400
        )

    result = await OpsCleanupData(ctx, mode="age", max_age_days=max_age_days).execute()
    return JSONResponse({"ok": True, **result})


async def db_vacuum(request: Any) -> Response:
    """Run VACUUM on the database to reclaim disk space."""
    from llm_rosetta_gateway.ops.data import OpsVacuum

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None or ctx.persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    result = await OpsVacuum(ctx).execute()
    return JSONResponse({"ok": True, **result})


async def cleanup_requests_by_age(request: Any) -> Response:
    """Delete request log entries older than max_age_days."""
    from llm_rosetta_gateway.ops.data import OpsCleanupData

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None or ctx.persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    body, err = parse_json_body(request)
    if err:
        return err

    max_age_days = body.get("max_age_days", 90)
    if not isinstance(max_age_days, int) or max_age_days < 1:
        return JSONResponse(
            {"error": "max_age_days must be a positive integer"}, status_code=400
        )

    result = await OpsCleanupData(
        ctx, mode="age", tables=("request_log",), max_age_days=max_age_days
    ).execute()
    return JSONResponse({"ok": True, **result})


async def cleanup_error_dumps_by_age(request: Any) -> Response:
    """Delete error dumps older than max_age_days."""
    from llm_rosetta_gateway.ops.data import OpsCleanupData

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None or ctx.persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    body, err = parse_json_body(request)
    if err:
        return err

    max_age_days = body.get("max_age_days", 90)
    if not isinstance(max_age_days, int) or max_age_days < 1:
        return JSONResponse(
            {"error": "max_age_days must be a positive integer"}, status_code=400
        )

    result = await OpsCleanupData(
        ctx, mode="age", tables=("error_dumps",), max_age_days=max_age_days
    ).execute()
    return JSONResponse({"ok": True, **result})


async def export_error_dumps(request: Any) -> Response:
    """Export error dumps in a date range as tar.gz."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    start = _qp(request, "start")
    end = _qp(request, "end")

    data = await persistence.export_error_dumps(start=start, end=end)

    return Response(
        body=data,
        status_code=200,
        content_type="application/gzip",
        headers={"Content-Disposition": "attachment; filename=error-dumps.tar.gz"},
    )


# ------------------------------------------------------------------
# Ops log
# ------------------------------------------------------------------


async def get_ops_log(request: Any) -> Response:
    """Return paginated, filtered ops log entries."""
    ops_log = getattr(request.app, "ops_log", None)
    if ops_log is None:
        return JSONResponse({"entries": [], "total": 0})
    try:
        limit = _bounded_int(request, "limit", 50, minimum=1, maximum=500)
        offset = _bounded_int(request, "offset", 0, minimum=0, maximum=10_000_000)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    event_type = _qp(request, "event_type")
    severity = _qp(request, "severity")
    source = _qp(request, "source")
    entries, total = await ops_log.get_entries(
        limit=limit,
        offset=offset,
        event_type=event_type,
        severity=severity,
        source=source,
    )
    return JSONResponse({"entries": entries, "total": total})


async def clear_ops_log(request: Any) -> Response:
    """Clear the ops log, recording the clear action itself."""
    from llm_rosetta_gateway.ops.data import OpsClearOpsLog

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None:
        return JSONResponse({"ok": True})
    result = await OpsClearOpsLog(ctx).execute()
    return JSONResponse({"ok": True, **(result or {})})


async def get_ops_log_event_types(request: Any) -> Response:
    """Return the list of known ops log event types."""
    from llm_rosetta_gateway.observability.ops_log import ALL_EVENT_TYPES

    return JSONResponse({"event_types": ALL_EVENT_TYPES})


async def get_ops_log_sources(request: Any) -> Response:
    """Return the list of known ops log source subsystems."""
    from llm_rosetta_gateway.observability.ops_log import ALL_SOURCES

    return JSONResponse({"sources": ALL_SOURCES})


async def cleanup_ops_log_by_age(request: Any) -> Response:
    """Delete ops log entries older than max_age_days."""
    from llm_rosetta_gateway.ops.data import OpsCleanupData

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None or ctx.persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    body, err = parse_json_body(request)
    if err:
        return err

    max_age_days = body.get("max_age_days", 90)
    if not isinstance(max_age_days, int) or max_age_days < 1:
        return JSONResponse(
            {"error": "max_age_days must be a positive integer"}, status_code=400
        )

    result = await OpsCleanupData(
        ctx, mode="age", tables=("ops_log",), max_age_days=max_age_days
    ).execute()
    return JSONResponse({"ok": True, **result})


_TARGET_TABLES: dict[str, tuple[str, ...]] = {
    "logs": ("request_log",),
    "errors": ("error_dumps",),
    "ops": ("ops_log",),
    "all": ("request_log", "error_dumps", "ops_log"),
}


async def db_cleanup_before(request: Any) -> Response:
    """Delete rows with timestamp < *before* from selected tables."""
    from llm_rosetta_gateway.ops.data import OpsCleanupData

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None or ctx.persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    body, err = parse_json_body(request)
    if err:
        return err

    before = body.get("before", "")
    if not isinstance(before, str) or not before:
        return JSONResponse(
            {"error": "before must be a non-empty ISO-8601 string"}, status_code=400
        )

    tables = _TARGET_TABLES.get(body.get("target", "all"), _TARGET_TABLES["all"])
    result = await OpsCleanupData(
        ctx, mode="before", tables=tables, before=before
    ).execute()
    return JSONResponse({"ok": True, **result})


async def db_cleanup_range(request: Any) -> Response:
    """Delete rows with *start* <= timestamp < *end* from selected tables."""
    from llm_rosetta_gateway.ops.data import OpsCleanupData

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None or ctx.persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    body, err = parse_json_body(request)
    if err:
        return err

    start = body.get("start", "")
    end = body.get("end", "")
    if not isinstance(start, str) or not start:
        return JSONResponse(
            {"error": "start must be a non-empty ISO-8601 string"}, status_code=400
        )
    if not isinstance(end, str) or not end:
        return JSONResponse(
            {"error": "end must be a non-empty ISO-8601 string"}, status_code=400
        )

    tables = _TARGET_TABLES.get(body.get("target", "all"), _TARGET_TABLES["all"])
    result = await OpsCleanupData(
        ctx, mode="range", tables=tables, start=start, end=end
    ).execute()
    return JSONResponse({"ok": True, **result})


async def db_trim(request: Any) -> Response:
    """Trim all tables to their configured retention caps."""
    from llm_rosetta_gateway.ops.data import OpsTrimData

    ctx = getattr(request.app, "ops_ctx", None)
    if ctx is None or ctx.persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    result = await OpsTrimData(ctx).execute()
    return JSONResponse({"ok": True, **result})
