"""Live coverage for the external shim presets and a source reload (#1106).

The preset path is the documented way to keep ``alcf--*`` / ``argo--*`` /
``asksage--*`` providers working after those shims left core, and every unit
test around it stubs the fetch.  That leaves one gap: nothing proves the
catalogue's repository URLs and ``path`` values are real — a preset that points
at a moved directory still resolves fine against a fake archive.

This module declares the presets for real, over the network, and drives the
same reload the admin panel and the CLI use.  It is skipped when GitHub is
unreachable (see ``tests/integration/conftest.py``), so it never decides a
normal ``make test`` run.

Run it directly (``python tests/integration/test_shim_presets_live.py``) to
check the catalogue after a shim repository moves.
"""

from __future__ import annotations

import asyncio
import socket
import sys

import pytest

from llm_rosetta.shims import get_shim
from llm_rosetta.shims.provider_shim import _reset_registry
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.ops import OpsContext, OpsProviderValidation
from llm_rosetta_gateway.shim_sources import apply_shim_section
from llm_rosetta_gateway.shims import SHIM_SOURCE_PACKAGES

PRESET_KEYS = [package.key for package in SHIM_SOURCE_PACKAGES]


def _github_reachable() -> bool:
    try:
        with socket.create_connection(("github.com", 443), timeout=8):
            return True
    except OSError:
        return False


if not _github_reachable():
    pytest.skip(
        "github.com is unreachable; the shim presets fetch their repositories",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def preset_config(tmp_path_factory) -> str:
    """A config path anchoring the preset cache, warmed once over the network.

    The first apply is the real fetch; every later apply in this module hits
    that cache, which is itself one of the reload behaviours under test.  A
    fetch failure here is a test failure, not a skip: reaching this fixture
    already proved the network is up.
    """
    config_path = str(tmp_path_factory.mktemp("presets") / "config.jsonc")
    report = apply_shim_section({"presets": PRESET_KEYS}, config_path)
    assert report.failures() == [], [(s.location, s.error) for s in report.failures()]
    return config_path


@pytest.fixture(autouse=True)
def presets(preset_config: str):
    """Give every test the same starting registry, then restore the built-ins."""
    from llm_rosetta.shims.providers import load_providers

    _reset_registry()
    report = apply_shim_section({"presets": PRESET_KEYS}, preset_config)
    yield report
    _reset_registry()
    load_providers()


class TestPresetsLoad:
    def test_every_package_loads_over_the_network(self, presets):
        assert len(presets.sources) == len(PRESET_KEYS)
        for status in presets.sources:
            assert status.ok is True, status.error
            assert status.kind == "remote"
            assert status.shim_count > 0, status.location

    def test_each_package_registers_exactly_its_catalogue_shims(self, presets):
        by_url = {status.location: status for status in presets.sources}
        for package in SHIM_SOURCE_PACKAGES:
            status = by_url[package.repo]
            assert set(status.shim_names) == set(package.shims), package.key

    def test_each_registered_shim_resolves_to_its_base_format(self):
        for package in SHIM_SOURCE_PACKAGES:
            for name in package.shims:
                shim = get_shim(name)
                assert shim is not None, name
                assert shim.format == name.partition("--")[2], name

    def test_a_warm_cache_does_not_fetch_again(self, preset_config: str):
        """The second apply is the admin's normal reload: cache, no network."""

        def refuse(url, **kwargs):
            raise AssertionError(f"unexpected fetch of {url}")

        report = apply_shim_section(
            {"presets": PRESET_KEYS}, preset_config, http_get=refuse
        )
        assert report.failures() == []
        assert all(status.from_cache for status in report.sources)


class TestProviderResolvesLive:
    """A provider on a preset shim must build, under both spellings."""

    @pytest.mark.parametrize(
        "shim_name",
        [
            "alcf_sophia--chat_completions",
            "asksage--messages",
            "argo--open_responses",
        ],
    )
    def test_canonical_shim_name_resolves(self, shim_name):
        raw = {
            "providers": {"p": {"shim": shim_name, "api_key": "k"}},
            "models": {"m": "p"},
        }
        config = GatewayConfig(raw)
        assert config.provider_types["p"] == shim_name.partition("--")[2]
        assert config.providers["p"].base_url  # supplied by the shim

    def test_legacy_alcf_spelling_resolves_through_the_preset(self):
        """The migration path this issue asked for, end to end."""
        raw = {
            "providers": {"p": {"type": "alcf--sophia", "api_key": "k"}},
            "models": {"m": "p"},
        }
        config = GatewayConfig(raw)
        assert config.provider_types["p"] == "chat_completions"
        assert config.provider_shim_names["p"] == "alcf_sophia--chat_completions"
        assert config.providers["p"].base_url

    def test_dropping_the_preset_degrades_the_provider(self, preset_config: str):
        """A reload that removes the preset must say why, not fail obscurely."""
        raw = {
            "providers": {
                "p": {"shim": "alcf_sophia--chat_completions", "api_key": "k"}
            },
            "models": {},
        }
        apply_shim_section({"presets": []}, preset_config)
        assert get_shim("alcf_sophia--chat_completions") is None

        result = asyncio.run(OpsProviderValidation(OpsContext(), raw).execute())
        disabled = result["disabled"]
        assert [item["provider"] for item in disabled] == ["p"]
        assert disabled[0]["category"] == "missing_external_shim"
        assert "shims.presets: ['alcf']" in disabled[0]["reason"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "-o", "addopts="]))
