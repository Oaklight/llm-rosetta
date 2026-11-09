"""The inbound request-body cap: ``server.max_body_size`` (issue #1479).

The cap was hard-coded at 50 MB in the app factory.  It is now a server
setting — default unchanged, floored, and rejected when it is not a number.
``GatewayExtensions.max_body_size`` keeps its precedence, so a caller that
builds the app explicitly still wins over the document.
"""

from __future__ import annotations

import pytest

from llm_rosetta_gateway.app import GatewayExtensions, create_app
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.config.settings import (
    DEFAULT_MAX_BODY_SIZE,
    MIN_MAX_BODY_SIZE,
)


def _raw(**server) -> dict:
    return {"providers": {}, "server": {"open_on_no_keys": True, **server}}


class TestTheConfigKey:
    def test_default_is_today_s_value(self):
        cfg = GatewayConfig(_raw())
        assert cfg.max_body_size == 50_000_000
        assert cfg.max_body_size == DEFAULT_MAX_BODY_SIZE

    def test_a_configured_value_is_taken_in_bytes(self):
        cfg = GatewayConfig(_raw(max_body_size=2_000_000))
        assert cfg.max_body_size == 2_000_000

    def test_a_digit_string_is_accepted(self):
        cfg = GatewayConfig(_raw(max_body_size="8000000"))
        assert cfg.max_body_size == 8_000_000

    def test_a_value_below_the_floor_is_clamped(self):
        cfg = GatewayConfig(_raw(max_body_size=5_000))
        assert cfg.max_body_size == MIN_MAX_BODY_SIZE
        assert cfg.max_body_size == 1_000_000

    @pytest.mark.parametrize("value", ["50MB", "large", [50]])
    def test_a_non_number_is_rejected_at_load(self, value):
        with pytest.raises(ValueError, match="max_body_size"):
            GatewayConfig(_raw(max_body_size=value))

    @pytest.mark.parametrize("value", [1e999, float("nan")])
    def test_an_infinite_body_size_is_rejected_by_name(self, value):
        """``1e999`` parses to ``inf``; ``int(inf)`` raises ``OverflowError``.

        That must reach the same named error as a word does — not escape as an
        uncaught ``OverflowError``, and not fall back to the default.
        """
        with pytest.raises(ValueError, match="server max_body_size"):
            GatewayConfig(_raw(max_body_size=value))

    def test_a_boolean_is_rejected(self):
        """``true`` is not a byte count, and int(True) would make it one."""
        with pytest.raises(ValueError, match="max_body_size"):
            GatewayConfig(_raw(max_body_size=True))


class TestTheAppUsesIt:
    @pytest.mark.asyncio
    async def test_the_document_reaches_the_server(self):
        app = await create_app(GatewayConfig(_raw(max_body_size=2_000_000)))
        assert app.max_body_size == 2_000_000

    @pytest.mark.asyncio
    async def test_an_extension_still_wins(self):
        ext = GatewayExtensions(max_body_size=100 * 1024 * 1024)
        app = await create_app(
            GatewayConfig(_raw(max_body_size=2_000_000)), extensions=ext
        )
        assert app.max_body_size == 100 * 1024 * 1024
