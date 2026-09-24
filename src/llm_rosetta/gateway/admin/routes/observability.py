"""Observability route handlers: metrics, request log, network diagnostics."""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpclient import AsyncClient, Response as HttpResponse
from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ...config import GatewayConfig
from ._shared import _qp, parse_json_body


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


def _persistence_snapshot(persistence: Any) -> dict[str, Any] | None:
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
            "log_entries": persistence.count_log_entries(),
            "log_success_entries": persistence.count_success_entries(),
            "log_error_entries": persistence.count_error_entries(),
            "log_max_success": persistence.success_max,
            "dump_entries": persistence.count_error_dumps(),
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
    persistence_snap = _persistence_snapshot(persistence)
    if persistence_snap is not None:
        snap["persistence"] = persistence_snap

    if persistence is not None:
        snap.update(persistence.query_rolling_24h_tokens())

    # Include circuit breaker states when the feature is enabled
    config: GatewayConfig | None = getattr(request.app, "gateway_config", None)
    if config is not None and config.circuit_breaker_registry.enabled:
        snap["circuit_breakers"] = config.circuit_breaker_registry.get_all_states()

    return JSONResponse(snap)


async def get_circuit_breaker_states(request: Any) -> Response:
    """Return per-provider circuit breaker states.

    Returns an empty dict when circuit breaking is disabled globally.
    """
    config: GatewayConfig | None = getattr(request.app, "gateway_config", None)
    if config is None or not config.circuit_breaker_registry.enabled:
        return JSONResponse({"enabled": False, "providers": {}})
    states = config.circuit_breaker_registry.get_all_states()
    return JSONResponse({"enabled": True, "providers": states})


def _rebuild_counters_after_mutation(request: Any) -> None:
    """Rebuild in-memory counters from request_log after admin-initiated deletion."""
    metrics = getattr(request.app, "metrics", None)
    persistence = getattr(request.app, "persistence", None)
    if metrics is None or persistence is None:
        return
    metrics.rebuild_counters(persistence.iter_log_rows_for_rebuild())
    persistence.save_metrics(metrics.export_counters())


async def rebuild_metrics(request: Any) -> Response:
    """Rebuild metrics counters from request log entries.

    Useful after fixing a counter bug or when persisted counters
    have drifted from the actual request log data.

    Returns both the pre-rebuild and post-rebuild counter snapshots
    so the admin UI can display what changed (sanity check).
    """
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse(
            {"error": "No persistence configured (in-memory mode)"},
            status_code=400,
        )

    metrics = request.app.metrics
    before = metrics.export_counters()
    count = metrics.rebuild_counters(persistence.iter_log_rows_for_rebuild())
    after = metrics.export_counters()

    # Persist the rebuilt counters immediately
    persistence.save_metrics(after)

    return JSONResponse(
        {
            "ok": True,
            "rebuilt_from": count,
            "before": before,
            "counters": after,
        }
    )


async def get_token_usage(request: Any) -> Response:
    """Return daily-aggregated token usage, optionally filtered by API key."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"days": [], "totals": {}})
    days = int(_qp(request, "days", "7"))
    days = max(1, min(days, 365))
    api_key_label = _qp(request, "api_key_label")
    return JSONResponse(
        persistence.query_token_usage_by_day(days=days, api_key_label=api_key_label)
    )


async def get_request_key_labels(request: Any) -> Response:
    """Return API key labels seen in request logs."""
    log = request.app.request_log
    return JSONResponse({"labels": log.get_api_key_labels()})


async def get_requests(request: Any) -> Response:
    """Return paginated, filtered request log entries."""
    log = request.app.request_log
    limit = int(_qp(request, "limit", "50"))
    offset = int(_qp(request, "offset", "0"))
    model = _qp(request, "model")
    provider = _qp(request, "provider")
    status = _qp(request, "status")
    api_key_label = _qp(request, "api_key_label")

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
                    from llm_rosetta.shims import resolve_base

                    provider_type = resolve_base(raw_type)
            except Exception:
                pass

    entries, total = log.get_entries(
        limit=limit,
        offset=offset,
        model=model,
        provider=provider,
        provider_type=provider_type,
        status=status,
        api_key_label=api_key_label,
    )
    return JSONResponse({"entries": entries, "total": total})


async def backfill_dump_log_ids(request: Any, **kwargs: Any) -> Response:
    """Backfill NULL request_log_id in error_dumps by matching timestamps."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    config = getattr(request.app, "gateway_config", None)
    aliases = config.model_upstream_names if config else {}
    updated = persistence.backfill_error_dump_log_ids(model_aliases=aliases)
    return JSONResponse({"updated": updated})


async def get_request_by_id(request: Any, **kwargs: Any) -> Response:
    """Return a single request log entry by ID."""
    log = request.app.request_log
    entry_id = kwargs.get("entry_id") or request.path_params["entry_id"]
    entry = log.get_entry(entry_id)
    if entry is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(entry)


async def clear_requests(request: Any) -> Response:
    """Clear the request log."""
    log = request.app.request_log
    log.clear()
    _rebuild_counters_after_mutation(request)
    return JSONResponse({"ok": True})


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


async def get_error_dumps(request: Any) -> Response:
    """Return paginated, filtered error dump entries."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    limit = int(_qp(request, "limit", "50"))
    offset = int(_qp(request, "offset", "0"))
    model = _qp(request, "model")
    error_phase = _qp(request, "error_phase")
    provider = _qp(request, "provider")

    entries, total = persistence.query_error_dumps(
        limit=limit,
        offset=offset,
        model=model,
        error_phase=error_phase,
        provider=provider,
    )
    return JSONResponse({"entries": entries, "total": total})


async def get_error_dump_detail(request: Any, **kwargs: Any) -> Response:
    """Return a single error dump with decompressed bodies."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    dump_id = request.path_params["dump_id"]
    entry = persistence.get_error_dump(dump_id)
    if entry is None:
        return JSONResponse({"error": "Not found"}, status_code=404)

    from llm_rosetta.observability.error_dump import decompress_body

    # Decompress request body if present
    if "body_hash" in entry:
        body_data = persistence.get_dump_body(entry["body_hash"])
        if body_data:
            try:
                entry["request_body"] = decompress_body(body_data)
            except Exception:
                entry["request_body"] = None

    # Decompress converted body if present
    if "converted_body_hash" in entry:
        conv_data = persistence.get_dump_body(entry["converted_body_hash"])
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
    entry = persistence.get_error_dump(dump_id)
    if entry is None:
        return JSONResponse({"error": "Not found"}, status_code=404)

    body_hash = entry.get("body_hash")
    if not body_hash:
        return JSONResponse(
            {"error": "No request body stored for this dump"}, status_code=404
        )

    body_data = persistence.get_dump_body(body_hash)
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
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    persistence.clear_error_dumps()
    return JSONResponse({"ok": True})


async def delete_error_dump(request: Any, **kwargs: Any) -> Response:
    """Delete a single error dump by ID."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    dump_id = request.path_params["dump_id"]
    if persistence.delete_error_dump(dump_id):
        return JSONResponse({"ok": True})
    return JSONResponse({"error": "Not found"}, status_code=404)


async def db_cleanup(request: Any) -> Response:
    """Delete records older than max_age_days and vacuum the database."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    body, err = parse_json_body(request)
    if err:
        return err

    max_age_days = body.get("max_age_days", 90)
    if not isinstance(max_age_days, int) or max_age_days < 1:
        return JSONResponse(
            {"error": "max_age_days must be a positive integer"}, status_code=400
        )

    result = persistence.cleanup_by_age(max_age_days)
    _rebuild_counters_after_mutation(request)
    return JSONResponse({"ok": True, **result})


async def db_vacuum(request: Any) -> Response:
    """Run VACUUM on the database to reclaim disk space."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    result = persistence.vacuum()
    return JSONResponse({"ok": True, **result})


async def cleanup_requests_by_age(request: Any) -> Response:
    """Delete request log entries older than max_age_days."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    body, err = parse_json_body(request)
    if err:
        return err

    max_age_days = body.get("max_age_days", 90)
    if not isinstance(max_age_days, int) or max_age_days < 1:
        return JSONResponse(
            {"error": "max_age_days must be a positive integer"}, status_code=400
        )

    result = persistence.cleanup_logs_by_age(max_age_days)
    _rebuild_counters_after_mutation(request)
    return JSONResponse({"ok": True, **result})


async def cleanup_error_dumps_by_age(request: Any) -> Response:
    """Delete error dumps older than max_age_days."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    body, err = parse_json_body(request)
    if err:
        return err

    max_age_days = body.get("max_age_days", 90)
    if not isinstance(max_age_days, int) or max_age_days < 1:
        return JSONResponse(
            {"error": "max_age_days must be a positive integer"}, status_code=400
        )

    result = persistence.cleanup_error_dumps_by_age(max_age_days)
    return JSONResponse({"ok": True, **result})


async def export_error_dumps(request: Any) -> Response:
    """Export error dumps in a date range as tar.gz."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    start = _qp(request, "start")
    end = _qp(request, "end")

    data = persistence.export_error_dumps(start=start, end=end)

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
    limit = min(int(_qp(request, "limit", "50")), 500)
    offset = int(_qp(request, "offset", "0"))
    event_type = _qp(request, "event_type")
    severity = _qp(request, "severity")
    source = _qp(request, "source")
    entries, total = ops_log.get_entries(
        limit=limit,
        offset=offset,
        event_type=event_type,
        severity=severity,
        source=source,
    )
    return JSONResponse({"entries": entries, "total": total})


async def clear_ops_log(request: Any) -> Response:
    """Clear the ops log, recording the clear action itself."""
    ops_log = getattr(request.app, "ops_log", None)
    if ops_log is None:
        return JSONResponse({"ok": True})

    from llm_rosetta.observability.ops_log import (
        EVENT_OPS_LOG_CLEARED,
        OpsLogEntry,
        SEVERITY_INFO,
        SOURCE_ADMIN,
    )

    count = ops_log.clear()
    ops_log.add(
        OpsLogEntry.create(
            event_type=EVENT_OPS_LOG_CLEARED,
            severity=SEVERITY_INFO,
            message=f"Ops log cleared ({count} entries removed)",
            details={"cleared_count": count},
            source=SOURCE_ADMIN,
        )
    )
    return JSONResponse({"ok": True, "cleared": count})


async def get_ops_log_event_types(request: Any) -> Response:
    """Return the list of known ops log event types."""
    from llm_rosetta.observability.ops_log import ALL_EVENT_TYPES

    return JSONResponse({"event_types": ALL_EVENT_TYPES})


async def get_ops_log_sources(request: Any) -> Response:
    """Return the list of known ops log source subsystems."""
    from llm_rosetta.observability.ops_log import ALL_SOURCES

    return JSONResponse({"sources": ALL_SOURCES})


async def cleanup_ops_log_by_age(request: Any) -> Response:
    """Delete ops log entries older than max_age_days."""
    persistence = getattr(request.app, "persistence", None)
    if persistence is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)

    body, err = parse_json_body(request)
    if err:
        return err

    max_age_days = body.get("max_age_days", 90)
    if not isinstance(max_age_days, int) or max_age_days < 1:
        return JSONResponse(
            {"error": "max_age_days must be a positive integer"}, status_code=400
        )

    result = persistence.cleanup_ops_log_by_age(max_age_days)
    return JSONResponse({"ok": True, **result})
