"""The Antigravity lane: what the shim declares and how the gateway reads it.

``antigravity--acp`` is the first provider whose upstream is not HTTP.  Two
facts make it work and both are asserted here: the shim declares its transport
and the agent command, and the gateway refuses a transport name it cannot
serve at config time rather than on the first request.
"""

from __future__ import annotations

import pytest

from llm_rosetta.shims import get_shim
from llm_rosetta.shims.providers import load_providers
from llm_rosetta_gateway.providers import build_provider_info
from llm_rosetta_gateway.transport import TRANSPORT_NAMES

SHIM = "antigravity--acp"


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_providers()


def shim():
    loaded = get_shim(SHIM)
    assert loaded is not None
    return loaded


class TestWhatTheShimDeclares:
    def test_it_is_a_prompt_lane_on_the_acp_format(self):
        profile = shim()

        assert profile.format == "acp"
        assert profile.model_type == "llm"

    def test_the_transport_and_the_agent_command_are_declared(self):
        connection = shim().connection

        assert connection.transport == "acp"
        assert connection.acp_argv == ("agy", "acp")

    def test_it_declares_no_credential_of_its_own(self):
        """``agy`` owns the Google token; the gateway must not hold one."""
        connection = shim().connection

        assert connection.keyless is True
        assert not connection.api_key_env


class TestTheTransportTable:
    def test_acp_is_a_known_transport(self):
        assert "acp" in TRANSPORT_NAMES
        assert "http" in TRANSPORT_NAMES


class TestHowTheGatewayReadsIt:
    def test_the_lane_carries_the_declared_transport_and_command(self):
        info = build_provider_info("acp", {}, shim_name=SHIM)

        assert info.transport == "acp"
        assert info.acp_argv == ("agy", "acp")
        assert info.keyless is True

    def test_a_config_override_wins_over_the_shim(self):
        info = build_provider_info("acp", {"transport": "http"}, shim_name=SHIM)

        assert info.transport == "http"

    def test_an_unknown_transport_is_refused_at_config_time(self):
        with pytest.raises(ValueError, match="unknown transport 'carrier-pigeon'"):
            build_provider_info("acp", {"transport": "carrier-pigeon"}, shim_name=SHIM)
