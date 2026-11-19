"""Provider connectivity test handler."""

from __future__ import annotations

from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta_gateway.providers import default_models_path
from llm_rosetta_gateway.transport.provider_info import VERSION_SUFFIXES

from ...config.typed import face_value
from ._shared import _resolve_models_path


def _resolve_base_url(provider_cfg: dict, config: Any, name: str) -> str:
    """Return the base_url from raw config, falling back to shim defaults."""
    url = provider_cfg.get("base_url", "").rstrip("/")
    if not url:
        pinfo = config.providers.get(name) if hasattr(config, "providers") else None
        if pinfo and pinfo.base_url:
            url = pinfo.base_url.rstrip("/")
    return url


async def test_provider_connectivity(request: Any, name: str) -> Response:
    """Probe a provider's base_url and endpoint paths for reachability.

    Checks:
    1. Base URL reachability (GET with short timeout)
    2. Each configured endpoint (models list, embedding, rerank)
    3. Double version prefix detection
    """
    config = getattr(request.app, "config", None) or getattr(
        request.app, "gateway_config", None
    )
    if config is None:
        return JSONResponse({"error": "No config loaded"}, status_code=500)

    raw_providers = getattr(config, "_raw_providers", {})
    provider_cfg = raw_providers.get(name)
    if provider_cfg is None:
        return JSONResponse({"error": f"Provider '{name}' not found"}, status_code=404)

    base_url = _resolve_base_url(provider_cfg, config, name)
    if not base_url:
        return JSONResponse(
            {"error": "Provider has no base_url configured"}, status_code=400
        )

    # A lane that is not reached over HTTP has no host to dial: its
    # ``base_url`` is a placeholder the transport contract wants, and probing
    # it would report a DNS failure about a lane that is working.  The
    # transport answers this question itself when it can.
    lane = str(provider_cfg.get("transport") or "")
    pinfo = config.providers.get(name) if hasattr(config, "providers") else None
    if not lane and pinfo is not None:
        lane = getattr(pinfo, "transport", "http")
    if lane and lane != "http":
        return JSONResponse(
            {
                "provider": name,
                "base_url": base_url,
                "transport": lane,
                "endpoints": {},
                "warnings": [
                    f"This lane is served over {lane!r}, not HTTP: there is no host "
                    "to dial and no HTTP endpoint to probe. Its reachability is "
                    "whatever the local agent answers on the next request."
                ],
            }
        )

    from llm_rosetta_gateway._vendor.httpclient import AsyncClient

    proxy = provider_cfg.get("proxy") or getattr(config, "proxy", None)
    timeout = float(provider_cfg.get("timeout", 10))
    async with AsyncClient(timeout=min(timeout, 10), proxy=proxy) as client:
        # Build auth headers from provider config (same as fetch-models)
        pinfo = config.providers.get(name) if hasattr(config, "providers") else None
        # A token_command provider still holding the pending sentinel would
        # send that placeholder as its bearer and get a 401 that says nothing
        # about reachability, so the authenticated probes are skipped until it
        # has been seeded.  A provider that is merely rate-limited or spent is
        # not pending, and its endpoints are still worth probing.
        auth_ready = pinfo is None or not pinfo.credentials_pending
        auth_headers = {**pinfo.extra_headers, **pinfo.auth_headers()} if pinfo else {}

        results: dict[str, Any] = {
            "provider": name,
            "base_url": base_url,
            "endpoints": {},
            "warnings": [],
        }
        if not auth_ready:
            results["warnings"].append(
                "Provider is still starting up — its API key has not been "
                "fetched yet; authenticated endpoint checks were skipped."
            )

        # 1. Probe base URL — deliberately unauthenticated. This only answers
        # "can we reach the host at all", so an auth rejection would be noise;
        # any HTTP response, including 401/404, proves reachability.
        await _probe_base(client, base_url, results)

        # 2. Check models endpoint (type-aware URL, same as fetch_upstream_models)
        await _probe_models(
            client,
            config,
            name,
            provider_cfg,
            base_url,
            auth_ready,
            auth_headers,
            results,
        )

        # 3–5. Check optional POST endpoints (embedding, rerank, decision)
        for ep_name, fmt_key, path_key, default_path in (
            ("embeddings", "embeddings_format", "embeddings_path", "/v1/embeddings"),
            ("rerank", "rerank_format", "rerank_path", "/v1/rerank"),
            ("decisions", "decisions_format", "decisions_path", "/v1/systemone"),
        ):
            # The endpoint's own block first, the flat key after it: the v5 fold
            # moves these, and a reader that only knew the flat key reported a
            # migrated provider as having no typed endpoints at all (#1595
            # review).  A shim declares the endpoint just as a format does.
            if not (
                face_value(provider_cfg, fmt_key, ep_name)
                or face_value(provider_cfg, f"{ep_name}_shim", ep_name)
            ):
                continue
            # The face's own host and path as written — what the operator has
            # to fix — and the URL the runtime builds from them, which is what
            # the probe asks for.
            face_host = (
                str(
                    face_value(provider_cfg, f"{ep_name}_base_url", ep_name) or ""
                ).rstrip("/")
                or base_url
            )
            face_path = face_value(provider_cfg, path_key, ep_name) or default_path
            _check_double_prefix(face_host, face_path, ep_name, results)
            url = _endpoint_url(config, name, ep_name, face_host, face_path)
            if url is None:
                continue
            if auth_ready:
                await _check_post_endpoint(client, url, auth_headers, ep_name, results)
            else:
                results["endpoints"][ep_name] = {
                    "url": url,
                    "status": None,
                    "ok": False,
                    "skipped": "provider not ready",
                }

    return JSONResponse(results)


async def _probe_base(client: Any, base_url: str, results: dict[str, Any]) -> None:
    """Record whether the entry's own host answers at all."""
    try:
        resp = await client.get(base_url)
        results["reachable"] = True
        results["base_status"] = resp.status_code
    except Exception as exc:
        results["reachable"] = False
        results["base_status"] = None
        results["base_error"] = str(exc)


async def _probe_models(
    client: Any,
    config: Any,
    name: str,
    provider_cfg: dict,
    base_url: str,
    auth_ready: bool,
    auth_headers: dict[str, str],
    results: dict[str, Any],
) -> None:
    """Record whether the entry's models listing answers."""
    ptype = (
        config.provider_types.get(name, "unknown")
        if hasattr(config, "provider_types")
        else "unknown"
    )
    explicit_path = _resolve_models_path(provider_cfg, config, name)
    if explicit_path and explicit_path.startswith(("https://", "http://")):
        models_url = explicit_path
    elif explicit_path:
        models_url = f"{base_url}{explicit_path}"
    else:
        models_url = f"{base_url}{default_models_path(ptype)}"
    if not auth_ready:
        results["endpoints"]["models"] = {
            "url": models_url,
            "status": None,
            "ok": False,
            "skipped": "provider not ready",
        }
        return
    try:
        resp = await client.get(models_url, headers=auth_headers)
        results["endpoints"]["models"] = {
            "url": models_url,
            "status": resp.status_code,
            "ok": resp.status_code < 400,
        }
    except Exception as exc:
        results["endpoints"]["models"] = {
            "url": models_url,
            "status": None,
            "ok": False,
            "error": str(exc),
        }


def _endpoint_url(
    config: Any, name: str, ep_name: str, face_host: str, face_path: str
) -> str | None:
    """The URL a typed endpoint's requests actually reach.

    Asked of the pool the runtime routes through rather than rebuilt here: the
    path is the shim's, the descriptor's per-format one and the operator's
    override in a precedence this module would have to copy, and a copy drifts
    (#1595 review — the probe posted to ``…/v2/v1/embeddings`` while the
    gateway called ``…/v2/embed``).  A provider the pool did not parse falls
    back to the face's own host and path, so the probe still names something
    when the endpoint is configured but disabled.
    """
    info = (getattr(config, f"{ep_name}_provider_infos", None) or {}).get(name)
    if info is not None:
        return info.upstream_url(
            _sample_model(getattr(config, f"{ep_name}_models", {}), name)
        )
    if not face_host:
        return None
    return f"{face_host}{face_path}"


def _sample_model(models: dict[str, str], name: str) -> str:
    """One model this provider serves, for a path template that needs one.

    Google's embedding endpoint carries the model in the URL
    (``/v1beta/models/{model}:embedContent``), so probing with an empty model
    asks for ``…/models/:embedContent`` — a URL no request ever uses, drawn as
    a broken endpoint by a gateway that knows the model (#1595 review).  An
    endpoint with no model mapped keeps the empty value, and its template needs
    none.
    """
    for model, provider in models.items():
        if provider == name:
            return model
    return ""


async def _check_post_endpoint(
    client: Any,
    url: str,
    auth_headers: dict[str, str],
    ep_name: str,
    results: dict[str, Any],
) -> None:
    """Probe a POST endpoint for reachability."""
    try:
        resp = await client.post(url, headers=auth_headers, json={})
        results["endpoints"][ep_name] = {
            "url": url,
            "status": resp.status_code,
            "ok": resp.status_code != 404,
        }
    except Exception as exc:
        results["endpoints"][ep_name] = {
            "url": url,
            "status": None,
            "ok": False,
            "error": str(exc),
        }


def _check_double_prefix(
    base_url: str, path: str, endpoint_name: str, results: dict
) -> None:
    """Warn if base_url and path share a version prefix."""
    m = VERSION_SUFFIXES.search(base_url)
    if m and path.startswith(m.group()):
        results["warnings"].append(
            f"{endpoint_name}: base_url ends with '{m.group()}' and "
            f"{endpoint_name}_path starts with '{m.group()}' — "
            f"this would cause a double prefix but is auto-corrected by "
            f"the gateway's base_url normalization."
        )
