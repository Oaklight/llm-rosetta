"""One provider entry, several faces: the account is shared, the host is not (#1595).

A ``providers:`` entry is an account — one key ring, one mint, one refresh —
and each typed endpoint is a face of it.  Before the face block the endpoint
took the entry's ``base_url`` whatever its shim said, so an account signing one
vendor for chat and another for embeddings sent every embedding to the chat
vendor's host, silently.

These cases pin the block, the precedence it reads with, the mismatch the
runtime now reports, the v4→v5 migration that folds the flat keys into it, and
the admin payload that folds them for the panel.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_rosetta_gateway.config.core import GatewayConfig
from llm_rosetta_gateway.config.typed import face_section, face_value
from llm_rosetta_gateway.maintenance.migrations import CURRENT_VERSION, migrate

# The vendors these tests read moved to contrib (#1660); their frozen copies
# are registered for every test here.
pytestmark = pytest.mark.usefixtures("moved_vendors")


def _config(entry: dict, *, models: dict | None = None) -> GatewayConfig:
    return GatewayConfig({"providers": {"mix": entry}, "models": models or {}})


class TestTheFaceHost:
    def test_an_endpoint_on_another_vendor_uses_its_own_host(self):
        cfg = _config(
            {
                "type": "google_generate",
                "base_url": "https://generativelanguage.googleapis.com",
                "api_key": "k",
                "embeddings": {
                    "shim": "cohere--embedding",
                    "base_url": "https://api.cohere.com/v2",
                },
            }
        )

        assert cfg.embeddings_provider_infos["mix"].base_url == "https://api.cohere.com"
        assert (
            cfg.providers["mix"].base_url == "https://generativelanguage.googleapis.com"
        )

    def test_an_endpoint_without_a_host_inherits_the_entry(self):
        cfg = _config(
            {
                "type": "google_generate",
                "base_url": "https://generativelanguage.googleapis.com",
                "api_key": "k",
                "embeddings": {"shim": "google--embedding"},
            }
        )

        assert (
            cfg.embeddings_provider_infos["mix"].base_url
            == "https://generativelanguage.googleapis.com"
        )

    def test_a_trailing_slash_is_normalised_away(self):
        cfg = _config(
            {
                "type": "chat_completions",
                "base_url": "https://chat.example/v1",
                "api_key": "k",
                "rerank": {"shim": "jina--rerank", "base_url": "https://api.jina.ai/"},
            }
        )

        assert cfg.rerank_provider_infos["mix"].base_url == "https://api.jina.ai"


class TestTheMismatchIsReported:
    def test_an_inherited_host_that_belongs_to_another_vendor_warns(
        self, gateway_warnings
    ):
        _config(
            {
                "type": "chat_completions",
                "base_url": "https://chat.example/v1",
                "api_key": "k",
                "embeddings_shim": "cohere--embedding",
            }
        )

        messages = [record.getMessage() for record in gateway_warnings]
        assert any("inherits base_url" in message for message in messages), messages
        assert any("embeddings: {base_url" in message for message in messages), messages

    def test_the_face_host_silences_the_warning(self, gateway_warnings):
        _config(
            {
                "type": "chat_completions",
                "base_url": "https://chat.example/v1",
                "api_key": "k",
                "embeddings": {
                    "shim": "cohere--embedding",
                    "base_url": "https://api.cohere.com",
                },
            }
        )

        assert not [
            record
            for record in gateway_warnings
            if "inherits base_url" in record.getMessage()
        ]

    def test_a_shim_on_the_same_host_does_not_warn(self, gateway_warnings):
        """The vendor's own two lanes differ by region, not by vendor."""
        _config(
            {
                "type": "chat_completions",
                "base_url": "https://generativelanguage.googleapis.com",
                "api_key": "k",
                "embeddings_shim": "google--embedding",
            }
        )

        assert not [
            record
            for record in gateway_warnings
            if "inherits base_url" in record.getMessage()
        ]


class TestTheGuardrailReachesEverySpelling:
    def test_a_format_only_face_is_checked_against_the_format_s_shim(
        self, gateway_warnings
    ):
        """The docs list `format:` first, and it misroutes like the shim."""
        _config(
            {
                "type": "google_generate",
                "base_url": "https://generativelanguage.googleapis.com",
                "api_key": "k",
                "embeddings": {"format": "cohere"},
            }
        )

        messages = [record.getMessage() for record in gateway_warnings]
        assert any("inherits base_url" in message for message in messages), messages
        assert any("api.cohere.com" in message for message in messages), messages

    @pytest.mark.parametrize(
        "type_name,format_name,host",
        [
            ("embeddings", "openai", "https://api.siliconflow.cn/v1"),
            ("embeddings", "openai", "https://open.bigmodel.cn/api/paas/v4"),
            ("embeddings", "openai", "https://api.z.ai/api/paas/v4"),
            ("embeddings", "openai", "https://ark.cn-beijing.volces.com/api/v3"),
            ("rerank", "cohere", "https://api.siliconflow.cn/v1"),
            ("rerank", "cohere", "https://open.bigmodel.cn/api/paas/v4"),
            ("rerank", "cohere", "https://api.z.ai/api/paas/v4"),
            ("rerank", "cohere", "https://openrouter.ai/api/v1"),
        ],
    )
    def test_a_vendor_serving_its_own_dialect_does_not_warn(
        self, gateway_warnings, type_name, format_name, host
    ):
        """A dialect several vendors serve is not one vendor's host.

        ``embeddings: {format: "openai"}`` on SiliconFlow is served by
        ``siliconflow--embedding``, and the request goes to SiliconFlow's own
        host.  Answering with the *first* registered shim of the pair —
        OpenAI's — warned on all eight shipped vendor entries that spell their
        face this way, while routing correctly (#1595 review).
        """
        _config(
            {
                "type": "chat_completions",
                "base_url": host,
                "api_key": "k",
                type_name: {"format": format_name},
            }
        )

        assert not [
            record
            for record in gateway_warnings
            if "inherits base_url" in record.getMessage()
        ]

    def test_a_migrated_flat_face_on_the_vendor_s_own_host_does_not_warn(
        self, gateway_warnings
    ):
        """A v4 document: the migration must not turn a quiet config loud."""
        raw: dict[str, Any] = {
            "providers": {
                "silicon": {
                    "type": "chat_completions",
                    "base_url": "https://api.siliconflow.cn/v1",
                    "api_key": "k",
                    "embeddings_format": "openai",
                }
            },
            "models": {},
        }

        migrate(raw)
        assert raw["providers"]["silicon"]["embeddings"] == {"format": "openai"}
        GatewayConfig(raw)

        assert not [
            record
            for record in gateway_warnings
            if "inherits base_url" in record.getMessage()
        ]

    def test_a_plaintext_host_against_an_https_shim_is_a_mismatch(
        self, gateway_warnings
    ):
        """A hostname comparison called these the same place."""
        _config(
            {
                "type": "chat_completions",
                "base_url": "http://api.cohere.com",
                "api_key": "k",
                "embeddings_shim": "cohere--embedding",
            }
        )

        assert any(
            "inherits base_url" in record.getMessage() for record in gateway_warnings
        )

    def test_the_guardrail_compares_origins_not_hostnames(self, gateway_warnings):
        """Two local services on one host and two ports are not one service.

        Calls the check directly with a stand-in shim: a real one on the chat
        host would be the same vendor, and the point here is the comparison.
        """
        from types import SimpleNamespace

        from llm_rosetta_gateway.config.typed import _face_base_url

        shim = SimpleNamespace(
            name="stub--embedding",
            format="openai",
            connection=SimpleNamespace(base_url="http://127.0.0.1:8002/v1"),
        )
        _face_base_url(
            "p",
            {},
            "embeddings",
            {"base_url": "http://127.0.0.1:8001/v1"},
            shim,
            "openai",
        )

        assert any(
            "inherits base_url" in record.getMessage() for record in gateway_warnings
        )

    def test_the_guardrail_sees_a_scheme_difference(self, gateway_warnings):
        from types import SimpleNamespace

        from llm_rosetta_gateway.config.typed import _face_base_url

        shim = SimpleNamespace(
            name="stub--embedding",
            format="openai",
            connection=SimpleNamespace(base_url="https://api.cohere.com"),
        )
        _face_base_url(
            "p", {}, "embeddings", {"base_url": "http://api.cohere.com"}, shim, "openai"
        )

        assert any(
            "inherits base_url" in record.getMessage() for record in gateway_warnings
        )

    def test_the_same_origin_does_not_warn(self, gateway_warnings):
        from types import SimpleNamespace

        from llm_rosetta_gateway.config.typed import _face_base_url

        shim = SimpleNamespace(
            name="stub--embedding",
            format="openai",
            connection=SimpleNamespace(base_url="https://api.cohere.com/"),
        )
        _face_base_url(
            "p",
            {},
            "embeddings",
            {"base_url": "https://api.cohere.com:443"},
            shim,
            "openai",
        )

        assert not [
            record
            for record in gateway_warnings
            if "inherits base_url" in record.getMessage()
        ]

    def test_a_trailing_slash_and_a_default_port_do_not_differ(self):
        from llm_rosetta_gateway.config.typed import _origin

        assert _origin("https://api.cohere.com/") == _origin(
            "https://api.cohere.com:443"
        )
        assert _origin("http://x.example") == _origin("http://x.example:80")
        assert _origin("http://x.example") != _origin("https://x.example")
        assert _origin("http://127.0.0.1:8001") != _origin("http://127.0.0.1:8002")


class TestTheFaceReader:
    def test_the_block_wins_over_the_flat_key(self):
        cfg = {"embeddings_shim": "flat--a", "embeddings": {"shim": "block--b"}}

        assert face_value(cfg, "embeddings_shim", "embeddings") == "block--b"

    def test_an_empty_block_value_falls_through(self):
        cfg = {"embeddings_shim": "flat--a", "embeddings": {"shim": ""}}

        assert face_value(cfg, "embeddings_shim", "embeddings") == "flat--a"

    def test_enabled_false_in_the_block_is_a_value(self):
        cfg = {"embeddings_enabled": True, "embeddings": {"enabled": False}}

        assert face_value(cfg, "embeddings_enabled", "embeddings") is False

    def test_a_flat_key_answers_when_the_block_is_absent(self):
        cfg = {"rerank_path": "/v1/rerank"}

        assert face_value(cfg, "rerank_path", "rerank") == "/v1/rerank"

    def test_a_block_that_is_not_a_mapping_is_ignored(self, gateway_warnings):
        assert face_section({"embeddings": "cohere"}, "embeddings") == {}

        assert any(
            "must be a mapping" in record.getMessage() for record in gateway_warnings
        )

    def test_an_endpoint_block_off_disables_only_that_face(self):
        cfg = _config(
            {
                "type": "chat_completions",
                "base_url": "https://chat.example/v1",
                "api_key": "k",
                "embeddings": {"shim": "cohere--embedding", "enabled": False},
            }
        )

        assert "mix" not in cfg.embeddings_provider_infos
        assert "mix" in cfg.providers


class TestTheMigration:
    def test_flat_keys_become_the_face_block(self):
        raw = {
            "providers": {
                "mix": {
                    "type": "google_generate",
                    "base_url": "https://gl.example",
                    "api_key": "k",
                    "embeddings_shim": "cohere--embedding",
                    "embeddings_format": "cohere",
                    "embeddings_path": "/v2/embed",
                    "embeddings_enabled": False,
                }
            },
            "models": {},
        }

        migrate(raw)

        entry = raw["providers"]["mix"]
        assert entry["embeddings"] == {
            "shim": "cohere--embedding",
            "format": "cohere",
            "path": "/v2/embed",
            "enabled": False,
        }
        assert "embeddings_shim" not in entry
        assert raw["config_version"] == CURRENT_VERSION

    def test_an_empty_flat_value_moves_too(self):
        """The form writes ``""`` to clear a setting; both spellings must agree."""
        raw = {
            "providers": {
                "p": {"api_key": "k", "base_url": "https://x", "rerank_path": ""}
            },
            "models": {},
        }

        migrate(raw)

        assert raw["providers"]["p"]["rerank"] == {"path": ""}
        assert "rerank_path" not in raw["providers"]["p"]

    def test_an_existing_block_is_not_overwritten(self):
        raw = {
            "providers": {
                "p": {
                    "base_url": "https://x",
                    "api_key": "k",
                    "embeddings": {"shim": "kept--embedding"},
                    "embeddings_shim": "flat--embedding",
                }
            },
            "models": {},
        }

        migrate(raw)

        entry: dict[str, Any] = raw["providers"]["p"]
        assert entry["embeddings"]["shim"] == "kept--embedding"
        assert "embeddings_shim" not in entry

    def test_an_entry_without_face_keys_is_untouched(self):
        raw = {
            "providers": {"p": {"base_url": "https://x", "api_key": "k"}},
            "models": {},
        }
        before = json.dumps(raw, sort_keys=True)

        migrate(raw)

        assert "embeddings" not in raw["providers"]["p"]
        assert json.dumps(raw["providers"], sort_keys=True) == json.dumps(
            json.loads(before)["providers"], sort_keys=True
        )

    def test_the_legacy_section_is_left_alone(self):
        """``<type>_providers`` is one face already; its keys are its own."""
        raw = {
            "config_version": 0,
            "providers": {},
            "models": {},
            "embeddings_providers": {
                "jina": {
                    "base_url": "https://api.jina.ai",
                    "api_key": "k",
                    "format": "jina",
                }
            },
        }

        migrate(raw)

        section = raw["embeddings_providers"]["jina"]
        assert section["format"] == "jina"
        assert "embeddings" not in section


class TestTheSaveAfterTheFold:
    def test_a_shim_only_entry_is_still_saveable(self, tmp_path):
        """The v5 fold moves the shim; the save check has to find it there.

        `_missing_provider_fields_error` looked for the shim under the flat
        key alone, so an entry the migration had already folded — the shape the
        docs list second — came back 400 asking for a base_url the shim was
        supplying (#1595 review).
        """
        from llm_rosetta_gateway.admin.routes.config.payload import (
            _mask_provider_entry,
        )
        from llm_rosetta_gateway.admin.routes.config.providers import (
            _missing_provider_fields_error,
        )
        from llm_rosetta_gateway.config import load_gateway_config

        path = tmp_path / "config.jsonc"
        path.write_text(
            json.dumps(
                {
                    "providers": {
                        "acme-embed": {
                            "api_key": "k",
                            "embeddings_shim": "cohere--embedding",
                        }
                    },
                    "models": {},
                }
            )
        )
        load_gateway_config(str(path))
        on_disk = json.loads(path.read_text())
        entry = on_disk["providers"]["acme-embed"]
        assert "embeddings_shim" not in entry, "the load folded the key"

        payload: dict[str, Any] = _mask_provider_entry("acme-embed", entry)
        error = _missing_provider_fields_error(
            payload,
            payload.get("api_key", ""),
            payload.get("base_url", ""),
            "acme-embed",
            on_disk["providers"],
        )

        assert error is None


class TestTheSave:
    def test_the_face_block_reaches_the_file(self, tmp_path):
        """The editor's block is stored as sent, not dropped by the whitelist."""
        from llm_rosetta_gateway.admin.routes._shared import (
            _apply_face_blocks,
            _build_provider_entry,
        )

        body = {
            "base_url": "https://generativelanguage.googleapis.com",
            "embeddings": {
                "shim": "cohere--embedding",
                "format": "cohere",
                "path": "/v2/embed",
                "base_url": "https://api.cohere.com/v2",
                "enabled": False,
            },
        }

        built = _build_provider_entry(body, "g-key", body["base_url"], {}, "fresh")
        _apply_face_blocks(built, body)

        assert built["embeddings"] == {
            "shim": "cohere--embedding",
            "format": "cohere",
            "path": "/v2/embed",
            "base_url": "https://api.cohere.com/v2",
            "enabled": False,
        }

    def test_an_empty_block_is_removed(self):
        from llm_rosetta_gateway.admin.routes._shared import _apply_face_blocks

        entry = {"embeddings": {"shim": "old--embedding"}}
        _apply_face_blocks(
            entry, {"embeddings": {"shim": "", "format": "", "path": ""}}
        )

        assert "embeddings" not in entry

    def test_a_capability_the_provider_dropped_leaves_no_block(self):
        from llm_rosetta_gateway.admin.routes._shared import _apply_face_blocks

        entry = {"embeddings": {"shim": "x--embedding"}}
        _apply_face_blocks(entry, {"embeddings": {}})

        assert "embeddings" not in entry

    def test_a_block_that_is_not_a_mapping_is_not_stored(self):
        from llm_rosetta_gateway.admin.routes._shared import _apply_face_blocks

        entry: dict = {}
        _apply_face_blocks(entry, {"embeddings": "cohere--embedding"})

        assert "embeddings" not in entry


class TestTheAdminPayload:
    def test_a_flat_entry_is_folded_into_a_block_for_the_panel(self):
        from llm_rosetta_gateway.admin.routes.config.payload import _mask_provider_entry

        masked: dict[str, Any] = _mask_provider_entry(
            "mix",
            {
                "type": "google_generate",
                "base_url": "https://gl.example",
                "api_key": "secret-key",
                "embeddings_shim": "cohere--embedding",
                "embeddings_enabled": False,
            },
        )

        assert masked["embeddings"] == {
            "shim": "cohere--embedding",
            "enabled": False,
        }
        assert "secret-key" not in json.dumps(masked)

    def test_an_entry_that_already_nests_is_left_as_written(self):
        from llm_rosetta_gateway.admin.routes.config.payload import _mask_provider_entry

        entry = {
            "type": "google_generate",
            "base_url": "https://gl.example",
            "embeddings": {
                "shim": "cohere--embedding",
                "base_url": "https://api.cohere.com",
            },
        }

        masked = _mask_provider_entry("mix", entry)

        assert masked["embeddings"] == entry["embeddings"]


@pytest.mark.parametrize("type_name", ["embeddings", "rerank", "decisions"])
def test_every_typed_endpoint_reads_its_block(type_name):
    """The panel and the parser share one field name per endpoint."""
    from llm_rosetta_gateway.config.typed import _typed_descriptors

    descriptors = {desc.name: desc for desc in _typed_descriptors()}
    desc = descriptors[type_name]
    cfg: dict = {type_name: {"shim": "x--y", "path": "/p"}}

    assert desc.config_shim_key is not None
    assert desc.config_path_key is not None
    assert face_value(cfg, desc.config_shim_key, type_name) == "x--y"
    assert face_value(cfg, desc.config_path_key, type_name) == "/p"


class TestThePresetIsRemembered:
    """The editor's preset is stored beside the entry and read by nothing else."""

    def _build(self, body: dict, existing: dict | None = None) -> dict:
        from llm_rosetta_gateway.admin.routes._shared import _build_provider_entry

        return _build_provider_entry(
            {"api_key": "k", **body}, "k", "https://x.example", existing or {}, "p"
        )

    def test_it_is_written_and_cleared_like_the_other_optional_fields(self):
        entry = self._build({"preset": "openai--chat_completions"})
        assert entry["preset"] == "openai--chat_completions"
        assert "preset" not in self._build({"preset": ""})

    @pytest.mark.parametrize("value", [5, True, ["x"], {"a": 1}, "   ", None])
    def test_only_a_non_blank_string_is_stored(self, value):
        assert "preset" not in self._build({"preset": value})

    def test_surrounding_whitespace_is_trimmed(self):
        assert self._build({"preset": " openai--chat_completions "})["preset"] == (
            "openai--chat_completions"
        )

    def test_an_entry_that_carries_it_still_loads_and_routes_on_its_faces(self):
        cfg = _config(
            {
                "type": "cohere--chat_completions",
                "preset": "openai--chat_completions",
                "base_url": "https://api.cohere.com",
                "api_key": "k",
                "embeddings": {"shim": "openai--embedding"},
            }
        )

        assert "mix" in cfg.providers
        assert "mix" in cfg.embeddings_provider_infos
        # The chat face installed its own shim, not the preset's.
        assert cfg.provider_shim_names["mix"] == "cohere--chat_completions"
