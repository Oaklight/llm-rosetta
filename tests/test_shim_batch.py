"""A shim's ``batch:`` declaration: parsing, validation, and absence.

The contract has three parts:

* an undeclared ``batch:`` block means **none** — every shim written before
  batch existed still loads, and the whole built-in registry is checked here;
* a declared block names a codec its own ``(model_type, format)`` offers (or
  ``emulated``), so a messages lane cannot claim the OpenAI file container;
* a typo is refused at load.  Read as "no batch", it would make a provider
  that does support batching answer ``unsupported`` in silence.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from llm_rosetta.shims import list_shims
from llm_rosetta.contracts.shim import BatchDeclaration, ProviderShim
from llm_rosetta.shims.providers import load_providers_from_dir


def _leaf(tmp_path: Path, body: str) -> Path:
    """Write one leaf under *tmp_path* and return the scan root."""
    leaf = tmp_path / "acme"
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "provider.yaml").write_text(textwrap.dedent(body))
    return tmp_path


def _chat(batch: str = "") -> str:
    return "name: acme\nmodel_type: llm\nformat: chat_completions\n" + batch


def _messages(batch: str = "") -> str:
    return "name: acme\nmodel_type: llm\nformat: messages\n" + batch


class TestUndeclared:
    def test_an_absent_block_means_none(self, tmp_path):
        shim = load_providers_from_dir(_leaf(tmp_path, _chat()))[0]

        assert shim.batch is None

    def test_the_whole_builtin_registry_loads_without_a_declaration(self):
        """Batch is new: a shipped shim that never declares it keeps loading."""
        assert list_shims(), "the built-in registry was not loaded"
        assert all(shim.batch is None for shim in list_shims())

    def test_the_dataclass_default_is_none(self):
        assert ProviderShim(name="acme", format="chat_completions").batch is None


class TestDeclaration:
    def test_a_declared_dialect_is_parsed(self, tmp_path):
        shim = load_providers_from_dir(
            _leaf(tmp_path, _chat("batch:\n  dialect: openai_file\n"))
        )[0]

        assert shim.batch == BatchDeclaration(dialect="openai_file")

    def test_the_files_api_opt_in_is_read(self, tmp_path):
        shim = load_providers_from_dir(
            _leaf(
                tmp_path,
                _chat("batch:\n  dialect: openai_file\n  supports_files: true\n"),
            )
        )[0]

        assert shim.batch is not None
        assert shim.batch.supports_files is True

    def test_path_overrides_are_read(self, tmp_path):
        shim = load_providers_from_dir(
            _leaf(
                tmp_path,
                _chat(
                    "batch:\n"
                    "  dialect: openai_file\n"
                    "  create_path: /v2/batches\n"
                    "  retrieve_path: /v2/batches/{batch_id}\n"
                ),
            )
        )[0]

        assert shim.batch is not None
        assert shim.batch.create_path == "/v2/batches"
        assert shim.batch.retrieve_path == "/v2/batches/{batch_id}"
        assert shim.batch.cancel_path is None

    def test_each_format_accepts_its_own_codec(self, tmp_path):
        layouts = {
            "chat_completions": "openai_file",
            "open_responses": "openai_file",
            "messages": "anthropic_inline",
            "google_generate": "gemini_inline",
        }
        for fmt, dialect in layouts.items():
            leaf = tmp_path / fmt
            leaf.mkdir()
            (leaf / "provider.yaml").write_text(
                f"name: acme_{fmt}\nmodel_type: llm\nformat: {fmt}\n"
                f"batch:\n  dialect: {dialect}\n"
            )

            shim = {s.format: s for s in load_providers_from_dir(tmp_path)}[fmt]

            assert shim.batch is not None
            assert shim.batch.dialect == dialect

    def test_the_gemini_file_variant_is_offered_too(self, tmp_path):
        shim = load_providers_from_dir(
            _leaf(
                tmp_path,
                "name: acme\nmodel_type: llm\nformat: google_generate\n"
                "batch:\n  dialect: gemini_file\n  supports_files: true\n",
            )
        )[0]

        assert shim.batch is not None
        assert shim.batch.dialect == "gemini_file"

    def test_emulated_is_accepted_on_a_measured_format(self, tmp_path):
        shim = load_providers_from_dir(
            _leaf(tmp_path, _messages("batch:\n  dialect: emulated\n"))
        )[0]

        assert shim.batch is not None
        assert shim.batch.dialect == "emulated"

    def test_emulated_is_accepted_on_a_format_without_a_record(self, tmp_path):
        """Interactions has no batch API; emulation is the only route."""
        shim = load_providers_from_dir(
            _leaf(
                tmp_path,
                "name: acme\nmodel_type: llm\nformat: google_interactions\n"
                "batch:\n  dialect: emulated\n",
            )
        )[0]

        assert shim.batch is not None
        assert shim.batch.dialect == "emulated"

    def test_as_dict_omits_unset_paths(self):
        declaration = BatchDeclaration(dialect="anthropic_inline")

        assert declaration.as_dict() == {
            "dialect": "anthropic_inline",
            "supports_files": False,
        }

        declaration = BatchDeclaration(
            dialect="openai_file",
            supports_files=True,
            create_path="/v1/batches",
        )
        assert declaration.as_dict() == {
            "dialect": "openai_file",
            "supports_files": True,
            "create_path": "/v1/batches",
        }


class TestRefusal:
    @pytest.mark.parametrize(
        ("body", "reason"),
        [
            (_chat("batch: openai_file\n"), "batch must be a mapping"),
            (
                _chat("batch:\n  dialect: openai_file\n  endpoint: /x\n"),
                "unknown batch key",
            ),
            (_chat("batch:\n  supports_files: true\n"), "must name a dialect"),
            (_chat("batch:\n  dialect: ''\n"), "must name a dialect"),
            (
                # An OpenAI codec on a Messages lane: different container.
                _messages("batch:\n  dialect: openai_file\n"),
                "not offered by llm/messages",
            ),
            (
                # Interactions has no record, so it offers no codec at all.
                "name: acme\nmodel_type: llm\nformat: google_interactions\n"
                "batch:\n  dialect: gemini_inline\n",
                "not offered by llm/google_interactions",
            ),
            (
                _chat("batch:\n  dialect: openai_file\n  supports_files: yes please\n"),
                "supports_files must be a boolean",
            ),
            (
                _chat("batch:\n  dialect: openai_file\n  create_path: 7\n"),
                "create_path must be a path string",
            ),
            (
                _chat("batch:\n  dialect: openai_file\n  cancel_path: ''\n"),
                "cancel_path must be a path string",
            ),
            (_chat("batch:\n  dialect: nonsense\n"), "not offered"),
        ],
    )
    def test_a_malformed_declaration_is_refused(self, tmp_path, body, reason):
        with pytest.raises(ValueError, match=reason):
            load_providers_from_dir(_leaf(tmp_path, body))

    def test_the_refusal_names_what_the_pair_offers(self, tmp_path):
        with pytest.raises(ValueError, match="offered: anthropic_inline, or emulated"):
            load_providers_from_dir(
                _leaf(tmp_path, _messages("batch:\n  dialect: gemini_inline\n"))
            )
