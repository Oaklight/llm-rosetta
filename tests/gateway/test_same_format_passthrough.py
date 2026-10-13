"""Tests for same-format provider affinity routing.

Covers ``server.prefer_same_format``: when a model is served by providers
in more than one API standard, prefer the one that already speaks the
client's format.  Conversion still always runs through IR.
"""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.config import GatewayConfig


@pytest.fixture(autouse=True)
def _load_shims():
    """Ensure built-in shims are registered before config resolution.

    The legacy ``openai_responses`` config type resolves to its base through the
    ``openai--open_responses`` shim, so the registry must be populated (other
    test modules reset it and do not always restore the built-ins).
    """
    from llm_rosetta.shims.providers import load_providers

    load_providers()


PROVIDERS = {
    "chat_a": {
        "type": "openai_chat",
        "api_key": "sk-a",
        "base_url": "https://a.example.com",
    },
    "chat_b": {
        "type": "openai_chat",
        "api_key": "sk-b",
        "base_url": "https://b.example.com",
    },
    "resp_a": {
        "type": "openai_responses",
        "api_key": "sk-c",
        "base_url": "https://c.example.com",
    },
    "resp_b": {
        "type": "openai_responses",
        "api_key": "sk-d",
        "base_url": "https://d.example.com",
    },
    "claude": {
        "type": "messages",
        "api_key": "sk-e",
        "base_url": "https://e.example.com",
    },
}


def _make_config(
    models: dict,
    *,
    prefer_same_format: bool | None = None,
    providers: dict | None = None,
) -> GatewayConfig:
    """Build a GatewayConfig, optionally enabling same-format affinity."""
    server: dict = {}
    if prefer_same_format is not None:
        server["prefer_same_format"] = prefer_same_format
    return GatewayConfig(
        {
            "providers": providers if providers is not None else PROVIDERS,
            "models": models,
            "server": server,
        }
    )


def _make_unready(config: GatewayConfig, *names: str) -> None:
    """Swap providers for ones whose deferred token fetch is pending."""
    from llm_rosetta.gateway.transport.provider_info import (
        TOKEN_PENDING_SENTINEL,
        ProviderInfo,
        openai_auth,
    )

    for name in names:
        config.providers[name] = ProviderInfo(
            name=name,
            api_key=TOKEN_PENDING_SENTINEL,
            base_url=config.providers[name].base_url,
            auth_header_fn=openai_auth,
            url_template="{base_url}/chat/completions",
            token_command=["echo", "x"],
        )
        assert config.providers[name].ready is False


class TestDefaultsUnchanged:
    """With the flag absent or off, nothing about routing changes."""

    def test_flag_defaults_to_off(self):
        config = _make_config({"m": "chat_a"})
        assert config.prefer_same_format is False

    def test_mixed_route_ignores_format_when_off(self):
        """Without the flag, a responses client still lands on WRR's pick."""
        config = _make_config({"m": {"providers": ["chat_a", "resp_a"]}})
        picked = {
            config.resolve("open_responses", "m")[0].provider_name for _ in range(10)
        }
        assert picked == {"chat_a", "resp_a"}

    def test_explicit_false_is_off(self):
        config = _make_config({"m": "chat_a"}, prefer_same_format=False)
        assert config.prefer_same_format is False


class TestFormatAffinity:
    """Provider selection prefers the client's own format."""

    def test_picks_matching_format_over_wrr(self):
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a"]}}, prefer_same_format=True
        )
        for _ in range(10):
            route, _ = config.resolve("open_responses", "m")
            assert route.provider_name == "resp_a"

    def test_affinity_is_per_source_format(self):
        """The same model routes differently for two client formats."""
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a"]}}, prefer_same_format=True
        )
        assert config.resolve("open_responses", "m")[0].provider_name == "resp_a"
        assert config.resolve("openai_chat", "m")[0].provider_name == "chat_a"

    def test_no_match_falls_back_to_full_route(self):
        """An Anthropic client has no affine provider here."""
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a"]}}, prefer_same_format=True
        )
        picked = {config.resolve("messages", "m")[0].provider_name for _ in range(10)}
        assert picked == {"chat_a", "resp_a"}

    def test_homogeneous_route_needs_no_subroute(self):
        """All providers already match — keep the plain route and its WRR state."""
        config = _make_config(
            {"m": {"providers": ["chat_a", "chat_b"]}}, prefer_same_format=True
        )
        picked = [
            config.resolve("openai_chat", "m")[0].provider_name for _ in range(10)
        ]
        assert set(picked) == {"chat_a", "chat_b"}
        groups = config._format_groups("m", "chat_completions")
        assert groups is not None and groups.covers_all

    def test_load_balances_within_matched_subset(self):
        """Two same-format providers share load, excluding the third.

        The anchor consumes the full route's round-robin, so the tier-1
        singletons and tier 3's own counter advance independently: over a short
        run the split is skewed (a ``chat_a`` anchor realigns through tier 3
        while a ``resp_*`` anchor is already in tier 1).  It converges as the
        run lengthens, so assert the invariant plus convergence rather than an
        exact short-run split.
        """
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a", "resp_b"]}},
            prefer_same_format=True,
        )
        picked = [
            config.resolve("open_responses", "m")[0].provider_name for _ in range(100)
        ]
        assert set(picked) == {"resp_a", "resp_b"}
        assert picked.count("resp_a") == picked.count("resp_b") == 50

    def test_tier_balances_exactly_with_pinned_anchor(self):
        """With the anchor held fixed, the tier's own WRR splits the subset evenly."""
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a", "resp_b"]}},
            prefer_same_format=True,
        )
        groups = config._format_groups("m", "open_responses")
        assert groups is not None
        anchor = next(p for p in config.models["m"].providers if p.name == "chat_a")
        picked = [
            (config._select_tiered(groups, anchor, None) or anchor).name
            for _ in range(20)
        ]
        assert picked.count("resp_a") == picked.count("resp_b") == 10

    def test_weights_respected_within_subset(self):
        # 15/5 holds exactly because a ``chat_a`` anchor's tier-3 share (1/5 of
        # the run) re-splits 3:1 inside tier 3 — the same ratio the parent route
        # gives the two resp providers.  A coincidence of the weights, not an
        # invariant; see test_load_balances_within_matched_subset.
        config = _make_config(
            {
                "m": {
                    "providers": [
                        "chat_a",
                        {"name": "resp_a", "weight": 3},
                        {"name": "resp_b", "weight": 1},
                    ]
                }
            },
            prefer_same_format=True,
        )
        picked = [
            config.resolve("open_responses", "m")[0].provider_name for _ in range(20)
        ]
        assert picked.count("resp_a") == 15
        assert picked.count("resp_b") == 5

    def test_format_groups_are_cached(self):
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a"]}}, prefer_same_format=True
        )
        first = config._format_groups("m", "open_responses")
        assert config._format_groups_cache[("m", "open_responses")] is first
        assert config._format_groups("m", "open_responses") is first

    def test_upstream_model_comes_from_matched_entry(self):
        """Per-entry upstream_model follows the affine provider, not WRR."""
        config = _make_config(
            {
                "m": {
                    "providers": [
                        {"name": "chat_a", "upstream_model": "chat-id"},
                        {"name": "resp_a", "upstream_model": "resp-id"},
                    ]
                }
            },
            prefer_same_format=True,
        )
        route, _ = config.resolve("open_responses", "m")
        assert route.upstream_model == "resp-id"


class TestReadinessInteraction:
    """Affinity must not strand a request on an initializing provider."""

    _unready = staticmethod(_make_unready)

    def test_unready_affine_falls_back_to_ready_other_format(self):
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a"]}}, prefer_same_format=True
        )
        self._unready(config, "resp_a")
        route, _ = config.resolve("open_responses", "m")
        assert route.provider_name == "chat_a"

    def test_unready_affine_prefers_ready_affine(self):
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a", "resp_b"]}},
            prefer_same_format=True,
        )
        self._unready(config, "resp_a")
        for _ in range(5):
            route, _ = config.resolve("open_responses", "m")
            assert route.provider_name == "resp_b"

    def test_all_unready_still_raises(self):
        from llm_rosetta.gateway.deferred_startup import ProviderNotReady

        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a"]}}, prefer_same_format=True
        )
        self._unready(config, "chat_a", "resp_a")
        with pytest.raises(ProviderNotReady):
            config.resolve("open_responses", "m")


class TestAccountTiers:
    """Tier 2: same upstream account, different wire format.

    Account identity is derived from the credential source (see
    ``provider_account_tokens``); these fixtures exercise the ways two provider
    entries end up on one account.
    """

    SHARED_KEY_PROVIDERS = {
        "ds_chat": {
            "shim": "deepseek--chat_completions",
            "api_key": "sk-shared",
            "base_url": "https://api.deepseek.com",
        },
        "ds_resp": {
            "shim": "deepseek--open_responses",
            "api_key": "sk-shared",
            "base_url": "https://api.deepseek.com",
        },
        "other_resp": {
            "type": "open_responses",
            "api_key": "sk-other",
            "base_url": "https://other.example.com",
        },
    }

    def _anchor(self, config: GatewayConfig, name: str):
        return next(p for p in config.models["m"].providers if p.name == name)

    def test_shared_key_is_one_account(self):
        config = _make_config(
            {"m": {"providers": ["ds_chat", "ds_resp", "other_resp"]}},
            prefer_same_format=True,
            providers=self.SHARED_KEY_PROVIDERS,
        )
        tokens = config.provider_account_tokens
        assert tokens["ds_chat"] == tokens["ds_resp"]
        assert tokens["other_resp"] != tokens["ds_chat"]

    def test_anchor_realigns_format_within_its_account(self):
        """A chat anchor serving an open_responses client picks its sibling.

        This is the capability tier 2 adds over the old base-only match: the
        request stays on the anchor's account instead of moving to another
        account that happens to speak the client's format.

        ``other_resp`` is listed before ``ds_resp`` on purpose.  That puts the
        other account first in ``matching``, so tier 3 — not just tier 2 — would
        pick ``other_resp``; the assertion then fails if the ``by_account`` tier
        is dropped, which is what makes this a test of tier 2 rather than of the
        subset order.
        """
        config = _make_config(
            {"m": {"providers": ["ds_chat", "other_resp", "ds_resp"]}},
            prefer_same_format=True,
            providers=self.SHARED_KEY_PROVIDERS,
        )
        groups = config._format_groups("m", "open_responses")
        assert groups is not None
        chosen = config._select_tiered(groups, self._anchor(config, "ds_chat"), None)
        assert chosen is not None and chosen.name == "ds_resp"

    def test_tier1_key_separates_endpoints_of_one_format(self):
        """Bare providers in one format on different endpoints are not one group.

        Both halves of the tier-1 key matter, and ``base_url`` is the easy one to
        lose: without it these two — bare, no shim, different endpoints,
        different keys — collapse into a single group and the anchor gets
        swapped to another account and endpoint for nothing.
        """
        providers = {
            "bare_a": {
                "type": "open_responses",
                "api_key": "sk-a",
                "base_url": "https://a.example.com",
            },
            "bare_b": {
                "type": "open_responses",
                "api_key": "sk-b",
                "base_url": "https://b.example.com",
            },
        }
        config = _make_config(
            {"m": {"providers": ["bare_a", "bare_b"]}},
            prefer_same_format=True,
            providers=providers,
        )
        groups = config._format_groups("m", "open_responses")
        assert groups is not None
        chosen = config._select_tiered(groups, self._anchor(config, "bare_b"), None)
        assert chosen is not None and chosen.name == "bare_b"

    def test_flag_off_leaves_the_anchor_alone(self):
        config = _make_config(
            {"m": {"providers": ["ds_chat", "ds_resp", "other_resp"]}},
            providers=self.SHARED_KEY_PROVIDERS,
        )
        picked = {
            config.resolve("open_responses", "m")[0].provider_name for _ in range(30)
        }
        assert picked == {"ds_chat", "ds_resp", "other_resp"}

    def test_env_derived_accounts_group_and_do_not_leak_the_key(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
        providers = {
            "a": {"shim": "deepseek--chat_completions"},
            "b": {"shim": "deepseek--open_responses"},
        }
        config = _make_config(
            {"m": {"providers": ["a", "b"]}},
            prefer_same_format=True,
            providers=providers,
        )
        token = config.provider_account_tokens["a"]
        assert token == config.provider_account_tokens["b"]
        assert "sk-env" not in token

    def test_account_is_the_key_value_not_the_config_spelling(self, monkeypatch):
        """A literal key and an env-resolved one with the same value group."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-same")
        providers = {
            "literal": {"shim": "deepseek--chat_completions", "api_key": "sk-same"},
            "from_env": {"shim": "deepseek--open_responses"},
        }
        config = _make_config(
            {"m": {"providers": ["literal", "from_env"]}},
            prefer_same_format=True,
            providers=providers,
        )
        assert (
            config.provider_account_tokens["literal"]
            == config.provider_account_tokens["from_env"]
        )

    def test_same_key_on_different_base_urls_is_still_one_account(self, monkeypatch):
        """minimax fronts chat and messages from different paths on one key.

        ``base_url`` is deliberately not part of the account token; if it were,
        these two would not group and the client would move accounts for nothing.
        """
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-mm")
        providers = {
            "mm_chat": {"shim": "minimax--chat_completions"},
            "mm_msg": {"shim": "minimax--messages"},
        }
        config = _make_config(
            {"m": {"providers": ["mm_chat", "mm_msg"]}},
            prefer_same_format=True,
            providers=providers,
        )
        assert (
            config.providers["mm_chat"].base_url != config.providers["mm_msg"].base_url
        )
        assert (
            config.provider_account_tokens["mm_chat"]
            == config.provider_account_tokens["mm_msg"]
        )

    def test_multi_key_ring_is_order_insensitive(self):
        providers = {
            "k1": {
                "type": "chat_completions",
                "api_key": "ka,kb",
                "base_url": "https://x.example.com",
            },
            "k2": {
                "type": "open_responses",
                "api_key": "kb,ka",
                "base_url": "https://x.example.com",
            },
            "k3": {
                "type": "open_responses",
                "api_key": "ka,kc",
                "base_url": "https://x.example.com",
            },
        }
        config = _make_config(
            {"m": {"providers": ["k1", "k2", "k3"]}},
            prefer_same_format=True,
            providers=providers,
        )
        tokens = config.provider_account_tokens
        assert tokens["k1"] == tokens["k2"]
        assert tokens["k3"] != tokens["k1"]

    def test_token_command_accounts_group_by_command(self):
        providers = {
            "t1": {
                "type": "chat_completions",
                "base_url": "https://x.example.com",
                "token_command": ["echo", "x"],
            },
            "t2": {
                "type": "open_responses",
                "base_url": "https://x.example.com",
                "token_command": ["echo", "x"],
            },
            "t3": {
                "type": "open_responses",
                "base_url": "https://x.example.com",
                "token_command": ["echo", "y"],
            },
        }
        config = _make_config(
            {"m": {"providers": ["t1", "t2", "t3"]}},
            prefer_same_format=True,
            providers=providers,
        )
        tokens = config.provider_account_tokens
        assert tokens["t1"] == tokens["t2"]
        assert tokens["t3"] != tokens["t1"]
        # Source-based, so a rotated token does not move the identity.
        assert tokens["t1"].startswith("cmd:")

    def test_keyless_providers_never_group(self):
        from llm_rosetta.gateway.transport.provider_info import (
            ProviderInfo,
            openai_auth,
        )

        def make(name: str) -> ProviderInfo:
            return ProviderInfo(
                name=name,
                api_key="",
                base_url="https://same.example.com",
                auth_header_fn=openai_auth,
                url_template="{base_url}/",
                keyless=True,
                credential_token=None,
            )

        assert make("a").account_token("a") != make("b").account_token("b")

    def test_keyless_does_not_group_with_a_keyed_peer(self, monkeypatch):
        providers = {
            "free": {"shim": "kilo--chat_completions"},
            "paid": {"shim": "kilo--chat_completions", "api_key": "sk-paid"},
        }
        config = _make_config(
            {"m": {"providers": ["free", "paid"]}},
            prefer_same_format=True,
            providers=providers,
        )
        assert (
            config.provider_account_tokens["free"]
            != config.provider_account_tokens["paid"]
        )

    def test_unready_account_sibling_falls_through_to_other_account(self):
        config = _make_config(
            {"m": {"providers": ["ds_chat", "ds_resp", "other_resp"]}},
            prefer_same_format=True,
            providers=self.SHARED_KEY_PROVIDERS,
        )
        _make_unready(config, "ds_resp")
        groups = config._format_groups("m", "open_responses")
        assert groups is not None
        chosen = config._select_tiered(groups, self._anchor(config, "ds_chat"), None)
        assert chosen is not None and chosen.name == "other_resp"

    def test_all_in_one_account_falls_back_cross_format(self):
        """No same-format provider is ready — stay on the anchor and convert."""
        config = _make_config(
            {"m": {"providers": ["ds_chat", "ds_resp"]}},
            prefer_same_format=True,
            providers=self.SHARED_KEY_PROVIDERS,
        )
        _make_unready(config, "ds_resp")
        for _ in range(5):
            assert config.resolve("open_responses", "m")[0].provider_name == "ds_chat"
