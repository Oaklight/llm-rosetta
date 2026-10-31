"""Characterization tests for the typed (non-LLM) provider sections.

``embedding``, ``rerank`` and ``decision`` each parse their legacy config
section through their own code path.  These tests write down what each one
does — enabled flag, ``format``/``type`` precedence, shim handling, default
path, model mapping — so a unified parser can be held to all three.

The three sections have drifted apart; where a test pins today's behaviour
rather than the shared one, it says so and the deliberate alignment is
recorded next to it.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta_gateway.config import GatewayConfig


def _auth(pinfo) -> str:
    """The credential header the provider sends (``Authorization`` or Google's)."""
    return next(iter(pinfo.auth_headers()))


class TestRerankSection:
    def test_defaults(self) -> None:
        cfg = GatewayConfig(
            {
                "rerank_providers": {
                    "jina": {"api_key": "k", "base_url": "https://api.jina.ai"}
                },
                "rerank_models": {"jina-reranker-v2": "jina"},
            }
        )
        assert cfg.rerank_providers["jina"] == {
            "format": "jina",
            "rerank_path": "/v1/rerank",
        }
        route = cfg.resolve_rerank("jina-reranker-v2")
        assert route.provider_name == "jina"
        assert route.format == "jina"
        assert route.path == "/v1/rerank"
        assert route.provider_info.name == "rerank:jina"
        assert route.provider_info.upstream_url("") == "https://api.jina.ai/v1/rerank"
        assert _auth(route.provider_info) == "Authorization"

    def test_format_wins_over_type(self) -> None:
        cfg = GatewayConfig(
            {
                "rerank_providers": {
                    "p": {"format": "cohere", "type": "voyage", "base_url": "https://x"}
                }
            }
        )
        assert cfg.rerank_providers["p"]["format"] == "cohere"

    def test_type_is_honoured_without_format(self) -> None:
        cfg = GatewayConfig(
            {"rerank_providers": {"p": {"type": "voyage", "base_url": "https://x"}}}
        )
        assert cfg.rerank_providers["p"]["format"] == "voyage"

    def test_descriptor_path_is_used_by_the_legacy_section(self) -> None:
        """Aligned in #1109: the section uses the descriptor's per-format path.

        A unified ``providers:`` entry with ``rerank_format: cohere`` already
        got ``/v2/rerank``; the legacy section used to fall back to
        ``/v1/rerank`` for every format.
        """
        cfg = GatewayConfig(
            {"rerank_providers": {"p": {"format": "cohere", "base_url": "https://x"}}}
        )
        assert cfg.rerank_providers["p"]["rerank_path"] == "/v2/rerank"

    def test_explicit_path_wins(self) -> None:
        cfg = GatewayConfig(
            {
                "rerank_providers": {
                    "p": {
                        "format": "cohere",
                        "rerank_path": "/custom",
                        "base_url": "https://x",
                    }
                }
            }
        )
        assert cfg.rerank_providers["p"]["rerank_path"] == "/custom"

    def test_disabled_provider_is_skipped(self) -> None:
        cfg = GatewayConfig(
            {
                "rerank_providers": {"p": {"enabled": False, "base_url": "https://x"}},
                "rerank_models": {"m": "p"},
            }
        )
        assert "p" not in cfg.rerank_providers
        assert "m" not in cfg.rerank_models

    def test_model_maps_to_the_first_live_provider(self) -> None:
        cfg = GatewayConfig(
            {
                "rerank_providers": {
                    "off": {"enabled": False, "base_url": "https://x"},
                    "on": {"base_url": "https://x"},
                },
                "rerank_models": {"m": {"providers": ["off", "on"]}},
            }
        )
        assert cfg.rerank_models["m"] == "on"


class TestEmbeddingSection:
    def test_defaults(self) -> None:
        cfg = GatewayConfig(
            {
                "embeddings_providers": {
                    "jina": {"api_key": "k", "base_url": "https://api.jina.ai"}
                },
                "embeddings_models": {"jina-embeddings-v3": "jina"},
            }
        )
        assert cfg.embeddings_providers["jina"] == {
            "format": "openai",
            "embeddings_path": "/v1/embeddings",
        }
        route = cfg.resolve_embedding("jina-embeddings-v3")
        assert route.format == "openai"
        assert route.path == "/v1/embeddings"
        assert route.provider_info.name == "embeddings:jina"
        assert _auth(route.provider_info) == "Authorization"

    def test_cohere_uses_the_descriptor_path(self) -> None:
        cfg = GatewayConfig(
            {
                "embeddings_providers": {
                    "p": {"format": "cohere", "base_url": "https://x"}
                }
            }
        )
        assert cfg.embeddings_providers["p"]["embeddings_path"] == "/v2/embed"

    def test_google_uses_google_auth_and_the_model_in_url_path(self) -> None:
        cfg = GatewayConfig(
            {
                "embeddings_providers": {
                    "p": {
                        "api_key": "k",
                        "format": "google_generate",
                        "base_url": "https://generativelanguage.googleapis.com",
                    }
                }
            }
        )
        info = cfg.embeddings_provider_infos["p"]
        assert _auth(info) == "x-goog-api-key"
        assert info.upstream_url("text-embedding-004") == (
            "https://generativelanguage.googleapis.com"
            "/v1beta/models/text-embedding-004:embedContent"
        )

    def test_format_wins_over_type(self) -> None:
        cfg = GatewayConfig(
            {
                "embeddings_providers": {
                    "p": {"format": "cohere", "type": "jina", "base_url": "https://x"}
                }
            }
        )
        assert cfg.embeddings_providers["p"]["format"] == "cohere"

    def test_disabled_provider_is_skipped(self) -> None:
        cfg = GatewayConfig(
            {
                "embeddings_providers": {
                    "p": {"enabled": False, "base_url": "https://x"}
                },
                "embeddings_models": {"m": "p"},
            }
        )
        assert "p" not in cfg.embeddings_providers
        assert "m" not in cfg.embeddings_models

    def test_shim_is_honoured_by_the_legacy_section(self) -> None:
        """Aligned in #1109: a legacy entry naming ``embeddings_shim`` uses it.

        The unified ``providers:`` entry already honoured the shim; the legacy
        section used to ignore it, so the same shim gave a different format,
        path and connection depending on which section declared it.
        """
        cfg = GatewayConfig(
            {
                "embeddings_providers": {
                    "p": {
                        "embeddings_shim": "cohere--embedding",
                        "api_key": "k",
                        "base_url": "https://x",
                    }
                }
            }
        )
        assert cfg.embeddings_providers["p"] == {
            "format": "cohere",
            "embeddings_path": "/v2/embed",
        }

    def test_model_maps_to_the_first_live_provider(self) -> None:
        cfg = GatewayConfig(
            {
                "embeddings_providers": {
                    "off": {"enabled": False, "base_url": "https://x"},
                    "on": {"base_url": "https://x"},
                },
                "embeddings_models": {"m": {"providers": ["off", "on"]}},
            }
        )
        assert cfg.embeddings_models["m"] == "on"


class TestDecisionSection:
    def test_defaults(self) -> None:
        cfg = GatewayConfig(
            {
                "decision_providers": {
                    "typesafe": {"api_key": "k", "base_url": "https://api.typesafe.ai"}
                },
                "decision_models": {"ts-1": "typesafe"},
            }
        )
        assert cfg.decision_providers["typesafe"] == {
            "format": "systemone",
            "decision_path": "/v1/systemone",
        }
        route = cfg.resolve_decision("ts-1")
        assert route.format == "systemone"
        assert route.path == "/v1/systemone"
        assert route.provider_info.name == "decision:typesafe"
        assert _auth(route.provider_info) == "Authorization"

    def test_format_wins_over_type(self) -> None:
        cfg = GatewayConfig(
            {
                "decision_providers": {
                    "p": {
                        "format": "openai_decisions",
                        "type": "systemone",
                        "base_url": "https://x",
                    }
                }
            }
        )
        assert cfg.decision_providers["p"] == {
            "format": "openai_decisions",
            "decision_path": "/v1/decisions",
        }

    def test_type_is_honoured_without_format(self) -> None:
        cfg = GatewayConfig(
            {
                "decision_providers": {
                    "p": {"type": "openai_decisions", "base_url": "https://x"}
                }
            }
        )
        assert cfg.decision_providers["p"]["format"] == "openai_decisions"

    def test_shim_supplies_format_path_and_connection(self) -> None:
        cfg = GatewayConfig(
            {"decision_providers": {"p": {"decision_shim": "typesafe--systemone"}}}
        )
        assert cfg.decision_providers["p"] == {
            "format": "systemone",
            "decision_path": "/v1/systemone",
        }
        info = cfg.decision_provider_infos["p"]
        assert info.base_url == "https://api.typesafe.ai"

    def test_shim_of_another_type_degrades_that_endpoint_only(self) -> None:
        """A bad endpoint shim is provider-local, not a config-wide failure.

        The provider keeps the endpoints it does parse and the failure is
        recorded on its entry for the admin payload (#1106).
        """
        raw: dict[str, Any] = {
            "decision_providers": {
                "p": {
                    "decision_shim": "deepseek--chat_completions",
                    "base_url": "https://x",
                }
            }
        }
        cfg = GatewayConfig(raw)

        assert "p" not in cfg.decision_providers
        marker: Any = raw["decision_providers"]["p"]["_auto_disabled_endpoints"]
        assert marker["decision"]["category"] == "wrong_model_type"
        assert "serves llm, not decision" in marker["decision"]["reason"]

    def test_a_broken_endpoint_does_not_take_other_providers_down(self) -> None:
        raw: dict[str, Any] = {
            "decision_providers": {
                "bad": {
                    "decision_shim": "nope--decision",
                    "base_url": "https://x",
                },
                "good": {"base_url": "https://y"},
            }
        }
        cfg = GatewayConfig(raw)

        assert "good" in cfg.decision_providers
        assert "good" not in raw["decision_providers"]["good"].get(
            "_auto_disabled_endpoints", {}
        )
        failed: Any = raw["decision_providers"]["bad"]["_auto_disabled_endpoints"]
        assert failed["decision"]["category"] == "unknown_shim"

    def test_disabled_provider_is_skipped(self) -> None:
        cfg = GatewayConfig(
            {
                "decision_providers": {
                    "p": {"enabled": False, "base_url": "https://x"}
                },
                "decision_models": {"m": "p"},
            }
        )
        assert "p" not in cfg.decision_providers
        assert "m" not in cfg.decision_models

    def test_model_maps_to_the_first_live_provider(self) -> None:
        cfg = GatewayConfig(
            {
                "decision_providers": {
                    "off": {"enabled": False, "base_url": "https://x"},
                    "on": {"base_url": "https://x"},
                },
                "decision_models": {"m": {"providers": ["off", "on"]}},
            }
        )
        assert cfg.decision_models["m"] == "on"
