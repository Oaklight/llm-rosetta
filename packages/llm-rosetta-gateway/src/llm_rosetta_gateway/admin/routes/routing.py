"""The Routing panel's data: what the selector decided, and why.

One endpoint, one model, and every number the panel shows comes from here.
The rule that shapes it is the one the quota card follows: the panel must not
re-derive a routing decision in the browser, because a second derivation is a
second answer.  So this returns the facts selection itself read — the
eligibility answer per candidate, the weight it serves at, the block's end —
plus the model's routing block and the last few decisions from the request
log.

The block itself is written through the ordinary model save
(``PUT /admin/api/config/models/<model>``), which rebuilds the config and
therefore runs the same parser: a value this endpoint reports is a value the
save will accept, and a bad one is refused with the parser's own message.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ..._vendor.httpserver import JSONResponse, Response
from ...quota.service import provider_allowance, provider_cooldown
from ...routing import parse_model_routing, provider_eligibility
from ._shared import _qp, config_mutate, parse_json_body
from .config.payload import _get_gateway_config

#: How many recent decisions the panel shows beside the board.
RECENT_DECISIONS = 20
#: How many recent rows the session table folds, so a long-running model does
#: not turn a tab switch into a scan.
SESSION_SAMPLE = 200

#: The windows the panel's model list counts traffic over.  ``None`` is "all
#: that the log still holds", not "no bound on the query".
ROUTING_WINDOWS: dict[str, int | None] = {"5h": 5, "24h": 24, "7d": 24 * 7, "all": None}


def _state(answer: Any) -> str:
    """The word the panel prints for one provider."""
    if answer is None:
        return "unknown"
    if answer.eligible:
        return "sunk" if answer.sunk else "serving"
    return answer.reason or "ineligible"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


def _expected_shares(
    route: Any, lanes: list[tuple[Any, Any, float, bool]]
) -> list[float]:
    """What each lane is expected to serve, in the strategy's own arithmetic.

    The Share column is not "the weight the selector was handed": for
    ``ordered`` the first eligible lane serves every request and the rest are
    fallbacks, and for ``rotate`` and ``least_used`` the configured weight is
    ignored.  Reporting the raw weight for those would tell an operator their
    primary lane serves a fifth of the traffic when it serves all of it.

    Args:
        route: The model's route; its block names the strategy.
        lanes: ``(entry, answer, headroom factor, eligible)`` per candidate,
            in the model's order.

    Returns:
        One relative share per lane; the panel normalises over the sum.
    """
    strategy = route.routing.strategy
    shares = [0.0] * len(lanes)
    eligible = [index for index, lane in enumerate(lanes) if lane[3]]
    if not eligible:
        return shares
    # A sunk lane is a fallback, not a share: selection only offers one when
    # nothing un-sunk is left, so counting it here would advertise traffic it
    # cannot get.  That demotion is the block's own switch — a model with
    # ``sink: false`` keeps its sunk lanes in the rotation.
    demote = getattr(route.routing, "sink", True)
    standing = [
        index for index in eligible if not (demote and _sunk(lanes[index]))
    ] or eligible
    if strategy == "ordered":
        shares[standing[0]] = 1.0
        return shares
    if strategy in ("rotate", "least_used"):
        for index in standing:
            shares[index] = 1.0
        return shares
    for index in standing:
        entry, _answer, factor, _ok = lanes[index]
        shares[index] = factor if strategy == "pace" else entry.weight * factor
    return shares


def _sunk(lane: tuple[Any, Any, float, bool]) -> bool:
    """Whether a lane is eligible only as a fallback."""
    answer = lane[1]
    return bool(answer is not None and getattr(answer, "sunk", False))


def _sessions(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold recent log rows into one line per client identity.

    The gateway's session id is derived from the conversation and not stored
    (see ``routing/session.py``), so the identity it can show is the one the
    request log keeps: the key label or client address, and the User-Agent.
    The rows arrive newest first, so the first one seen for an identity names
    the lane it last reached.

    Args:
        entries: Log rows, newest first.

    Returns:
        One dict per identity, in the order the identities were last seen.
    """
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        client = entry.get("api_key_label") or entry.get("client_ip") or ""
        agent = entry.get("user_agent") or ""
        key = (client, agent)
        row = grouped.get(key)
        if row is None:
            grouped[key] = {
                "client": client or None,
                "user_agent": agent or None,
                "provider": entry.get("target_provider_name")
                or entry.get("target_provider"),
                "requests": 1,
                "last": entry.get("timestamp"),
            }
        else:
            row["requests"] += 1
    return list(grouped.values())


def _decision(entry: dict[str, Any]) -> dict[str, Any]:
    """One request-log row, reduced to what the decision stream shows.

    The failover lives in the row's profile, not at its top level: the proxy
    records it there (``proxy/failover.py``) so a reader can see that the
    answer came from a second provider.
    """
    profile = entry.get("profile") or {}
    failover = profile.get("failover") if isinstance(profile, dict) else None
    return {
        "time": entry.get("timestamp"),
        "model": entry.get("model"),
        "request_id": entry.get("request_id"),
        "client": entry.get("api_key_label") or entry.get("client_ip"),
        "user_agent": entry.get("user_agent"),
        "provider": entry.get("target_provider_name") or entry.get("target_provider"),
        "status": entry.get("status_code"),
        "duration_ms": entry.get("duration_ms"),
        "failover": failover,
    }


def _window_since(window: str) -> str:
    """The ISO lower bound a window stands for.

    Args:
        window: One of :data:`ROUTING_WINDOWS`.

    Returns:
        An ISO timestamp; the epoch for ``all``.
    """
    hours = ROUTING_WINDOWS[window]
    if hours is None:
        return "0000-01-01T00:00:00+00:00"
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


async def list_routable_models(request: Any, **kwargs: Any) -> Response:
    """Return the models this gateway can route, with a one-line summary each.

    The panel's list shows these, not the config document's ``models`` keys:
    a model whose provider was skipped at load has nothing to route to, and
    offering it would turn a click into a 404.

    The ``window`` query parameter (``5h`` by default) bounds the request
    count each row carries; ``all`` counts everything the log still holds.
    Each summary also carries the lanes' states and their relative shares, so
    the list can draw the same picture the board does without a request per
    model.
    """
    config = _get_gateway_config(request)
    if config is None:
        return JSONResponse({"error": "Gateway config not loaded"}, status_code=500)
    window = _qp(request, "window", "5h") or "5h"
    if window not in ROUTING_WINDOWS:
        return JSONResponse({"error": f"Unknown window: {window!r}"}, status_code=400)

    # The counts are the log's, not the panel's: a reader that fails leaves
    # them unknown rather than zero, so the roster does not report an idle
    # gateway when what broke was the reading.
    activity: dict[str, dict[str, Any]] | None = {}
    log = getattr(request.app, "request_log", None)
    if log is not None:
        try:
            activity = await log.activity_by_model(_window_since(window))
        except Exception:  # noqa: BLE001 - the roster must render without a log
            activity = None

    routes = getattr(config, "models", {})
    summary: dict[str, dict[str, Any]] = {}
    for name, route in routes.items():
        lanes: list[tuple[Any, Any, float, bool]] = []
        for entry in route.providers:
            pinfo = config.providers.get(entry.name)
            answer = (
                provider_eligibility(pinfo, entry.name) if pinfo is not None else None
            )
            factor = answer.weight_factor if answer is not None else 1.0
            lanes.append(
                (entry, answer, factor, bool(answer is not None and answer.eligible))
            )
        shares = _expected_shares(route, lanes)
        row = (activity or {}).get(name, {})
        summary[name] = {
            "providers": len(lanes),
            "serving": sum(1 for lane in lanes if lane[3]),
            "blocked": sum(1 for lane in lanes if not lane[3]),
            "shares": [round(value, 4) for value in shares],
            # The lane behind each share, so the list's bar can name what it
            # draws: a full bar beside "1 blocked" does not say which lane
            # the 100 % belongs to.
            "lanes": [
                {
                    "name": entry.name,
                    "share": round(shares[index], 4),
                    "eligible": lanes[index][3],
                    # The lane's own state, so the list can colour a slice the
                    # way the board colours its dot: amber for a lane that is
                    # only sunk, red for one that cannot serve at all.
                    "state": _state(lanes[index][1]),
                }
                for index, (entry, _answer, _factor, _ok) in enumerate(lanes)
            ],
            "requests": int(row.get("requests", 0)),
            "last_seen": row.get("last_seen"),
        }

    # Most recently requested first: the list answers "what was just routed",
    # and the panel selects the top row when the tab opens.  A model with no
    # traffic in the window follows, by name.
    seen = [name for name in routes if summary[name]["last_seen"]]
    seen.sort(key=lambda name: summary[name]["last_seen"], reverse=True)
    unseen = sorted(name for name in routes if not summary[name]["last_seen"])
    return JSONResponse(
        {
            "models": seen + unseen,
            "summary": summary,
            "window": window,
            # ``False`` says the counts could not be read: every row is zero
            # because nothing answered, not because nothing ran.
            "activity_known": activity is not None,
        }
    )


def _lane_row(config: Any, entry: Any, answer: Any, share: float) -> dict[str, Any]:
    """One candidate as the board draws it: what the selector will do with it."""
    pinfo = config.providers.get(entry.name)
    allowance = provider_allowance(pinfo) if pinfo is not None else None
    return {
        "name": entry.name,
        "weight": entry.weight,
        "share": share,
        "headroom": answer.headroom if answer is not None else None,
        "eligible": bool(answer is not None and answer.eligible),
        "state": _state(answer),
        "reason": answer.reason if answer is not None else None,
        "until": _iso(answer.until) if answer is not None else None,
        "sunk": bool(answer is not None and answer.sunk),
        "block": provider_cooldown(entry.name, pinfo) if pinfo is not None else None,
        # Whether the lane may still spend, and every window the vendor
        # reports, tightest first: 0.00 alone does not say which window is
        # spent.
        "has_allowance": allowance.has_allowance if allowance else True,
        "windows": list(allowance.windows) if allowance else [],
        # The lane's wire format and the shim that supplies it, so the panel
        # can say where a provider actually goes.
        "format": getattr(config, "provider_types", {}).get(entry.name),
        "shim": getattr(config, "provider_shim_names", {}).get(entry.name),
        "upstream_model": entry.upstream_model,
        # Whether this lane can report an allowance at all: a probe that has
        # not answered yet and a lane with no probe are both ``headroom:
        # None``, and the panel words them differently.
        "probe": bool(pinfo is not None and getattr(pinfo, "quota_probe", None)),
        # A lane the document declares but the runtime does not route.  The
        # board shows it read-only, after the serving ones.
        "excluded": False,
    }


def _excluded_lane_row(config: Any, lane: dict[str, Any]) -> dict[str, Any]:
    """One declared lane the selector cannot offer, and why.

    A lane leaves the route in three ways, and the row has to say which one:
    the provider is switched off everywhere; the gateway disabled it itself on
    a load-time fault (the marker says what it saw); or this model's own entry
    turns it off, which is not the provider's switch at all.

    Args:
        config: The gateway config, for the provider's own entry.
        lane: ``{"name", "weight", "enabled"}`` from ``declared_model_lanes``.

    Returns:
        A row shaped like :func:`_lane_row`, with no traffic figures.
    """
    name = lane["name"]
    cfg = (getattr(config, "raw_providers", {}) or {}).get(name) or {}
    marker = cfg.get("_auto_disabled")
    if lane.get("enabled") is False:
        kind, reason = "entry", "switched off on this model"
    elif isinstance(marker, dict) and marker.get("reason"):
        kind, reason = "auto", str(marker["reason"])
    elif cfg.get("enabled", True) is False:
        kind, reason = "provider", "switched off in the providers section"
    else:
        kind, reason = "missing", "provider is not declared in the providers section"
    return {
        "name": name,
        "weight": lane.get("weight", 1),
        "share": 0,
        "headroom": None,
        "eligible": False,
        "state": "off",
        "reason": reason,
        "reason_kind": kind,
        "until": None,
        "sunk": False,
        "block": None,
        "has_allowance": False,
        "windows": [],
        "format": cfg.get("type"),
        "shim": cfg.get("shim"),
        "upstream_model": None,
        "probe": False,
        "excluded": True,
    }


async def get_model_routing(request: Any, **kwargs: Any) -> Response:
    """Return one model's routing facts, block and recent decisions."""
    model = request.path_params["model"]
    config = _get_gateway_config(request)
    if config is None:
        return JSONResponse({"error": "Gateway config not loaded"}, status_code=500)
    route = getattr(config, "models", {}).get(model)
    if route is None:
        # A model whose every lane is off has no route at all, but it is still
        # in the document.  Answering 404 would take away the one page where
        # the lane could be switched back on.
        declared = getattr(config, "declared_model_lanes", {}).get(model)
        if declared:
            return JSONResponse(
                {
                    "model": model,
                    "strategy": None,
                    "routing": None,
                    "providers": [
                        _excluded_lane_row(config, lane) for lane in declared
                    ],
                    "decisions": [],
                    "sessions": [],
                }
            )
        configured = ", ".join(sorted(getattr(config, "models", {})))
        return JSONResponse(
            {
                "error": f"Unknown model: '{model}'. Configured models: {configured}",
            },
            status_code=404,
        )

    lanes: list[tuple[Any, Any, float, bool]] = []
    for entry in route.providers:
        pinfo = config.providers.get(entry.name)
        answer = provider_eligibility(pinfo, entry.name) if pinfo is not None else None
        factor = answer.weight_factor if answer is not None else 1.0
        lanes.append(
            (entry, answer, factor, bool(answer is not None and answer.eligible))
        )
    shares = _expected_shares(route, lanes)

    providers = [
        _lane_row(config, entry, answer, shares[index])
        for index, (entry, answer, _factor, _eligible) in enumerate(lanes)
    ]
    # A lane the document declares but the route does not carry is still the
    # operator's lane: append it after the serving ones, marked ``off``.
    routed = {row["name"] for row in providers}
    for lane in getattr(config, "declared_model_lanes", {}).get(model, []):
        if lane["name"] not in routed:
            providers.append(_excluded_lane_row(config, lane))

    decisions: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    log = getattr(request.app, "request_log", None)
    if log is not None:
        try:
            entries, _total = await log.get_entries(
                limit=SESSION_SAMPLE, offset=0, model=model
            )
            decisions = [_decision(entry) for entry in entries[:RECENT_DECISIONS]]
            sessions = _sessions(entries)
        except Exception:  # noqa: BLE001 - the board must render without a log
            decisions = []
            sessions = []

    return JSONResponse(
        {
            "model": model,
            "strategy": type(route.strategy).__name__,
            "routing": route.routing.describe(),
            "providers": providers,
            "decisions": decisions,
            "sessions": sessions,
        }
    )


async def put_model_routing(request: Any, **kwargs: Any) -> Response:
    """Write one model's routing block.

    The write is the raw document edit plus the ordinary rebuild, so the block
    is validated by the same parser the load uses: a value the panel shows is a
    value this accepts, and a bad one comes back as the parser's own message.

    A single-provider model stored as a bare string is promoted to the object
    form the block needs; the provider it named is kept.
    """
    model = request.path_params["model"]
    body, err = parse_json_body(request)
    if err is not None:
        return err
    if "routing" not in body:
        return JSONResponse({"error": "'routing' is required"}, status_code=400)
    routing = body["routing"]
    if routing not in (None, {}):
        # The block is written to the document as-is, so it must be a block
        # the loader reads back: an unknown key here would not mislabel the
        # panel, it would stop the next start.
        try:
            parse_model_routing(routing, label=f"models.{model}.routing")
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error
        models = ctx.data.setdefault("models", {})
        entry = models.get(model)
        if entry is None:
            return JSONResponse({"error": f"Unknown model: '{model}'"}, status_code=404)
        if isinstance(entry, str):
            # The bare-string form has nowhere to put a block.
            entry = {"provider": entry}
            models[model] = entry
        if not isinstance(entry, dict):
            return JSONResponse(
                {"error": f"Model '{model}' is not an object entry"},
                status_code=400,
            )
        if routing in (None, {}):
            entry.pop("routing", None)
        else:
            entry["routing"] = routing
        ctx.commit()

    if ctx.error:
        return ctx.error
    return JSONResponse({"ok": True, "model": model})
