"""Server settings: rate limits, log format, log retention."""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from .._shared import (
    _sanitize_server_section,
    config_mutate,
    parse_json_body,
)

logger = logging.getLogger("llm-rosetta-gateway")


def _apply_rate_limit_settings(
    server: dict[str, Any], rl_body: dict[str, Any]
) -> Response | None:
    """Validate and apply rate_limit settings; return error Response or None."""
    from llm_rosetta_gateway.middleware.ratelimit import parse_quota

    rl_cfg = server.setdefault("rate_limit", {})
    if "enabled" in rl_body:
        rl_cfg["enabled"] = bool(rl_body["enabled"])
    if "algorithm" in rl_body:
        algo = rl_body["algorithm"]
        valid = {"token_bucket", "fixed_window", "sliding_window", "gcra"}
        if algo not in valid:
            return JSONResponse(
                {
                    "error": f"Invalid algorithm: {algo!r} (expected one of {sorted(valid)})"
                },
                status_code=400,
            )
        rl_cfg["algorithm"] = algo
    for quota_key in ("global", "per_ip", "per_key", "per_model"):
        if quota_key in rl_body:
            val = rl_body[quota_key]
            if val:
                quotas = val if isinstance(val, list) else [val]
                for q in quotas:
                    try:
                        parse_quota(q)
                    except ValueError as exc:
                        return JSONResponse(
                            {"error": f"Invalid {quota_key} quota: {exc}"},
                            status_code=400,
                        )
                rl_cfg[quota_key] = val
            else:
                rl_cfg.pop(quota_key, None)
    return None


def _apply_log_format(debug: dict, raw_value: Any) -> Response | None:
    """Validate and apply a log_format value; return error Response or None."""
    fmt = str(raw_value).strip().lower()
    if fmt not in ("json", "text", "auto"):
        return JSONResponse(
            {
                "error": f"Invalid log_format: {raw_value!r} "
                "(expected 'json', 'text', or 'auto')"
            },
            status_code=400,
        )
    debug["log_format"] = fmt
    return None


_RL_FIELDS = (("success_max", 50000), ("error_dump_max", 1000), ("max_age_days", 1))


_OL_FIELDS = (("info_max", 100), ("warn_max", 100), ("max_age_days", 1))


def _apply_log_retention(body: dict[str, Any], server: dict[str, Any]) -> None:
    """Apply request_log and ops_log retention settings from *body*."""
    for section, fields in (("request_log", _RL_FIELDS), ("ops_log", _OL_FIELDS)):
        src = body.get(section) or {}
        if not src:
            continue
        cfg = server.setdefault(section, {})
        for key, floor in fields:
            if key in src:
                cfg[key] = max(floor, int(src[key]))


async def put_server_settings(request: Any) -> Response:  # noqa: C901
    """Update server settings (e.g. global proxy)."""
    body, err = parse_json_body(request)
    if err:
        return err

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        server = ctx.data.setdefault("server", {})

        # Update proxy — empty string removes it
        if "proxy" in body:
            proxy = body["proxy"]
            if proxy:
                server["proxy"] = proxy
            else:
                server.pop("proxy", None)

        # Credential visibility toggle
        if "credential_visible" in body:
            want_visible = bool(body["credential_visible"])
            if want_visible and not request.app.auth_state.admin_password:
                return JSONResponse(
                    {"error": "Credential reveal requires an admin password"},
                    status_code=403,
                )
            server["credential_visible"] = want_visible

        # Log retention caps (floor prevents accidental wipe)
        _apply_log_retention(body, server)

        # Debug / log level
        debug = ctx.data.setdefault("debug", {})
        if "verbose" in body:
            debug["verbose"] = bool(body["verbose"])
        if "log_bodies" in body:
            debug["log_bodies"] = bool(body["log_bodies"])
        if "error_dumps" in body:
            debug["error_dumps"] = bool(body["error_dumps"])
        if "log_format" in body:
            fmt_err = _apply_log_format(debug, body["log_format"])
            if fmt_err is not None:
                return fmt_err

        # Rate limiting
        if "rate_limit" in body:
            rl_err = _apply_rate_limit_settings(server, body["rate_limit"])
            if rl_err is not None:
                return rl_err

        ctx.commit()

    if ctx.error:
        return ctx.error

    return JSONResponse(
        {"ok": True, "server": _sanitize_server_section(ctx.data.get("server", {}))}
    )
