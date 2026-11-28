"""One trusted-proxy policy, resolved once per request (#1830 A11).

The rule, in one place:

* ``server.trusted_proxies`` lists CIDRs.  Empty (the default) trusts nobody,
  and the client is the TCP peer.
* A peer outside the set is the client; every forwarding header is ignored.
* A trusted peer lets ``X-Forwarded-For`` speak.  The chain is walked from the
  right, skipping trusted addresses; the first untrusted address is the client.
  The left of the chain is whatever the original caller wrote, so it only
  counts when every hop to its right is a proxy we trust.
* A chain that is all trusted, empty, or holds an entry that is not an IP
  address names no client: the peer is used.
* ``X-Real-IP`` is read only for a trusted peer that sent no usable
  ``X-Forwarded-For``; it never overrides a conclusion drawn from the chain.
* ``X-Forwarded-Proto: https`` counts only from a trusted peer.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta_gateway.admin.routes.auth import admin_login
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.middleware.auth import AuthState
from llm_rosetta_gateway.middleware.ratelimit import (
    RateLimitState,
    create_rate_limit_hook,
)
from llm_rosetta_gateway.middleware.request_context import (
    ProxyPolicy,
    TrustedProxies,
    request_context_var,
    resolve_client_ip,
    setup_request_context,
)
from llm_rosetta_gateway.observability.request_log import RequestLog
from llm_rosetta_gateway.ops import OpsContext
from llm_rosetta_gateway.ops.proxy import record_proxy_request
from llm_rosetta_gateway.ops.rejected import record_read_rejection


class _Request:
    def __init__(
        self,
        peer: str | None = "10.0.0.9",
        headers: dict[str, str] | None = None,
        path: str = "/v1/chat/completions",
    ) -> None:
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.client_addr = (peer, 4000) if peer else None
        self.path = path
        self.method = "POST"
        self.cookies: dict[str, str] = {}
        self.state = SimpleNamespace()
        self.app: Any = None

    def json(self) -> dict[str, Any]:
        return {"password": "wrong"}


def _trusted(*cidrs: str) -> TrustedProxies:
    return TrustedProxies.parse(list(cidrs))


def _client(trusted: TrustedProxies, **kwargs: Any) -> str:
    return resolve_client_ip(_Request(**kwargs), trusted)


PROXIES = ("10.0.0.0/8", "172.16.0.0/12", "127.0.0.1/32")


# -- the resolution rule ------------------------------------------------------


def test_with_no_trusted_set_the_peer_is_the_client() -> None:
    headers = {"X-Forwarded-For": "1.2.3.4", "X-Real-IP": "5.6.7.8"}
    assert _client(TrustedProxies.parse(None), peer="9.9.9.9", headers=headers) == (
        "9.9.9.9"
    )


def test_a_forged_leftmost_entry_cannot_change_the_client_from_an_untrusted_peer() -> (
    None
):
    trusted = _trusted(*PROXIES)
    for forged in ("1.1.1.1", "1.1.1.1, 2.2.2.2", "10.0.0.5"):
        got = _client(trusted, peer="203.0.113.9", headers={"X-Forwarded-For": forged})
        assert got == "203.0.113.9"


def test_the_first_untrusted_entry_from_the_right_is_the_client() -> None:
    trusted = _trusted(*PROXIES)
    chain = "9.9.9.9, 198.51.100.7, 10.1.1.1"  # the left entry is forged
    assert _client(trusted, peer="10.0.0.2", headers={"X-Forwarded-For": chain}) == (
        "198.51.100.7"
    )


def test_a_forged_entry_between_the_client_and_the_proxies_is_not_skipped() -> None:
    trusted = _trusted(*PROXIES)
    chain = "203.0.113.1, 6.6.6.6, 10.3.3.3, 10.4.4.4"
    assert _client(trusted, peer="10.0.0.2", headers={"X-Forwarded-For": chain}) == (
        "6.6.6.6"
    )


def test_several_trusted_hops_are_all_skipped() -> None:
    trusted = _trusted(*PROXIES)
    chain = "203.0.113.9, 10.1.1.1, 172.16.5.5, 127.0.0.1"
    assert _client(trusted, peer="10.0.0.2", headers={"X-Forwarded-For": chain}) == (
        "203.0.113.9"
    )


def test_an_all_trusted_chain_or_no_header_falls_back_to_the_peer() -> None:
    trusted = _trusted(*PROXIES)
    all_trusted = {"X-Forwarded-For": "10.1.1.1, 10.2.2.2"}
    assert _client(trusted, peer="10.0.0.2", headers=all_trusted) == "10.0.0.2"
    assert _client(trusted, peer="10.0.0.2") == "10.0.0.2"
    assert _client(trusted, peer="10.0.0.2", headers={"X-Forwarded-For": " "}) == (
        "10.0.0.2"
    )


def test_an_entry_that_is_not_an_address_names_no_client() -> None:
    trusted = _trusted(*PROXIES)
    for chain in ("unknown", "203.0.113.9, garbage", "203.0.113.9:4000"):
        got = _client(trusted, peer="10.0.0.2", headers={"X-Forwarded-For": chain})
        assert got == "10.0.0.2"


def test_ipv6_peers_chains_and_ranges() -> None:
    trusted = _trusted("fd00::/8", "10.0.0.0/8")
    chain = {"X-Forwarded-For": "2001:DB8::5, fd00::2"}
    assert _client(trusted, peer="fd00::1", headers=chain) == "2001:db8::5"
    assert _client(trusted, peer="2001:db8::77", headers=chain) == "2001:db8::77"


def test_an_ipv4_mapped_peer_matches_an_ipv4_range() -> None:
    trusted = _trusted("10.0.0.0/8")
    headers = {"X-Forwarded-For": "203.0.113.9"}
    assert _client(trusted, peer="::ffff:10.0.0.2", headers=headers) == "203.0.113.9"


def test_real_ip_is_read_only_for_a_trusted_peer_without_a_chain() -> None:
    trusted = _trusted(*PROXIES)
    real = {"X-Real-IP": "198.51.100.8"}
    assert _client(trusted, peer="10.0.0.2", headers=real) == "198.51.100.8"
    assert _client(trusted, peer="203.0.113.9", headers=real) == "203.0.113.9"
    assert _client(trusted, peer="10.0.0.2", headers={"X-Real-IP": "nope"}) == (
        "10.0.0.2"
    )
    assert _client(trusted, peer="10.0.0.2", headers={"X-Real-IP": "10.9.9.9"}) == (
        "10.0.0.2"
    )


def test_real_ip_never_overrides_the_chain() -> None:
    trusted = _trusted(*PROXIES)
    both = {"X-Forwarded-For": "198.51.100.7", "X-Real-IP": "6.6.6.6"}
    assert _client(trusted, peer="10.0.0.2", headers=both) == "198.51.100.7"
    all_trusted = {"X-Forwarded-For": "10.1.1.1", "X-Real-IP": "6.6.6.6"}
    assert _client(trusted, peer="10.0.0.2", headers=all_trusted) == "10.0.0.2"


def test_a_request_without_a_peer_is_unknown() -> None:
    assert _client(_trusted(*PROXIES), peer=None) == "unknown"


def test_forwarded_proto_counts_only_from_a_trusted_peer() -> None:
    policy = ProxyPolicy(_trusted("10.0.0.0/8"))
    https = {"X-Forwarded-Proto": "https"}
    assert policy.is_https(_Request(peer="10.0.0.2", headers=https)) is True
    assert policy.is_https(_Request(peer="203.0.113.9", headers=https)) is False
    assert policy.is_https(_Request(peer="10.0.0.2")) is False


# -- the configured set -------------------------------------------------------


def test_the_set_accepts_cidrs_bare_addresses_and_ipv6() -> None:
    parsed = TrustedProxies.parse(["10.0.0.0/8", "127.0.0.1", "fd00::/8"])
    assert "10.200.0.1" in parsed and "127.0.0.1" in parsed and "fd00::9" in parsed
    assert "11.0.0.1" not in parsed
    assert bool(parsed) is True
    assert bool(TrustedProxies.parse([])) is False
    assert bool(TrustedProxies.parse(None)) is False


@pytest.mark.parametrize(
    ("raw", "needle"),
    [
        (["10.0.0.0/8", "not-a-cidr"], "server.trusted_proxies[1]"),
        (["10.0.0.0/33"], "server.trusted_proxies[0]"),
        ([42], "server.trusted_proxies[0]"),
        ("10.0.0.0/8", "must be a list"),
    ],
)
def test_a_bad_set_names_the_item(raw: Any, needle: str) -> None:
    with pytest.raises(ValueError, match="config:") as caught:
        TrustedProxies.parse(raw)
    assert needle in str(caught.value)


def _config(tmp_path: Any, **server: Any) -> dict[str, Any]:
    return {
        "providers": {
            "Mock": {
                "type": "chat_completions",
                "base_url": "http://127.0.0.1:1/v1",
                "api_key": "x",
            }
        },
        "models": {"m": {"provider": "Mock"}},
        "server": {
            "data_dir": str(tmp_path / "data"),
            "open_on_no_keys": True,
            **server,
        },
    }


def test_the_config_reads_server_trusted_proxies(tmp_path: Any) -> None:
    config = GatewayConfig(_config(tmp_path, trusted_proxies=["10.0.0.0/8"]))
    assert "10.1.2.3" in config.trusted_proxies
    assert not GatewayConfig(_config(tmp_path)).trusted_proxies


def test_an_invalid_cidr_fails_the_load(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match=r"server\.trusted_proxies\[0\]"):
        GatewayConfig(_config(tmp_path, trusted_proxies=["bogus"]))


def test_the_retired_rate_limit_key_is_refused_with_its_replacement(
    tmp_path: Any,
) -> None:
    raw = _config(tmp_path)
    raw["server"]["rate_limit"] = {"trust_proxy": True}
    with pytest.raises(ValueError, match=r"server\.trusted_proxies"):
        GatewayConfig(raw)


# -- every consumer reads the one value --------------------------------------


def _hooked(trusted: TrustedProxies) -> Any:
    return setup_request_context(ProxyPolicy(trusted))


def test_the_hook_computes_the_client_once_and_stores_it() -> None:
    hook = _hooked(_trusted("10.0.0.0/8"))
    request = _Request(peer="10.0.0.2", headers={"X-Forwarded-For": "198.51.100.7"})

    async def run() -> Any:
        await hook(request)
        return request_context_var.get()

    ctx = asyncio.run(run())
    assert ctx.client_ip == "198.51.100.7"


def test_the_api_rate_limit_counts_the_resolved_client() -> None:
    state = RateLimitState()
    config = SimpleNamespace(
        rate_limit_enabled=True,
        rate_limit_algorithm="sliding_window",
        rate_limit_global=None,
        rate_limit_per_ip="1/m",
        rate_limit_per_key=None,
        rate_limit_per_model=None,
        rate_limit_exclude=["/health"],
    )
    state.rebuild(config)  # ty: ignore[invalid-argument-type]
    limit = create_rate_limit_hook(state)

    async def status(trusted: TrustedProxies, forged: str) -> int | None:
        request = _Request(peer="203.0.113.9", headers={"X-Forwarded-For": forged})
        await _hooked(trusted)(request)
        denied = await limit(request)
        return denied.status_code if denied else None

    async def run() -> list[int | None]:
        none = TrustedProxies.parse(None)
        return [await status(none, f"1.1.1.{i}") for i in range(3)]

    assert asyncio.run(run()) == [None, 429, 429]


def test_the_login_lockout_counts_the_resolved_client() -> None:
    state = AuthState(None, None, admin_password="right")
    app = SimpleNamespace(auth_state=state)

    async def attempt(forged: str) -> int:
        request = _Request(peer="203.0.113.9", headers={"X-Forwarded-For": forged})
        request.app = app
        await _hooked(TrustedProxies.parse(None))(request)
        return (await admin_login(request)).status_code

    async def run() -> list[int]:
        return [await attempt(f"1.1.1.{i}") for i in range(7)]

    assert asyncio.run(run()) == [401] * 5 + [429] * 2


def test_the_login_lockout_separates_clients_behind_a_trusted_proxy() -> None:
    state = AuthState(None, None, admin_password="right")
    app = SimpleNamespace(auth_state=state)
    trusted = _trusted("10.0.0.0/8")

    async def attempt(client: str) -> int:
        request = _Request(peer="10.0.0.2", headers={"X-Forwarded-For": client})
        request.app = app
        await _hooked(trusted)(request)
        return (await admin_login(request)).status_code

    async def run() -> list[int]:
        out = [await attempt("198.51.100.1") for _ in range(6)]
        out.append(await attempt("198.51.100.2"))
        return out

    assert asyncio.run(run()) == [401] * 5 + [429, 401]


def test_the_request_log_row_carries_the_resolved_client() -> None:
    log = RequestLog()
    app = SimpleNamespace(ops_ctx=OpsContext(request_log=log))
    hook = _hooked(_trusted("10.0.0.0/8"))
    request = _Request(peer="10.0.0.2", headers={"X-Forwarded-For": "198.51.100.7"})

    async def run() -> Any:
        await hook(request)
        await record_proxy_request(
            app,
            model="m",
            source_provider="openai",
            target_provider="openai",
            provider_name="p",
            is_stream=False,
            status_code=200,
            duration_ms=1.0,
            error_detail=None,
        )
        entries, _ = await log.get_entries(limit=1)
        return entries[0]

    assert asyncio.run(run())["client_ip"] == "198.51.100.7"


def test_the_rejected_log_row_carries_the_resolved_client() -> None:
    rows: list[Any] = []

    class Log:
        async def add(self, entry: Any) -> None:
            rows.append(entry)

    app = SimpleNamespace(
        ops_ctx=OpsContext(request_log=Log()),
        proxy_policy=ProxyPolicy(_trusted("10.0.0.0/8")),
    )
    rejection = SimpleNamespace(
        path="/v1/chat/completions",
        method="POST",
        headers={"x-forwarded-for": "6.6.6.6, 198.51.100.7, 10.1.1.1"},
        client_addr=("10.0.0.2", 1),
        exception=None,
        reason="body too short",
        content_length=10,
        bytes_received=1,
        status_code=400,
        request_id="rid",
    )
    asyncio.run(record_read_rejection(app, rejection))
    assert rows[0].client_ip == "198.51.100.7"
