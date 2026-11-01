"""Tests for OpenCode-shaped session and request identifiers.

The gateway mints these for any shim that declares
``session_id_format``/``request_id_format: opencode`` (today the OpenCode
shims); the layout itself is core behaviour, so it is tested here rather than
in a shim package.
"""

from __future__ import annotations

import re
import time

from llm_rosetta_gateway.transport.provider_info import opencode_identifier


class TestOpenCodeIdentifier:
    def test_shape_and_stability(self):
        first = opencode_identifier("ses", "conversation-1")
        assert re.fullmatch(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}", first)
        assert first == opencode_identifier("ses", "conversation-1")
        assert first != opencode_identifier("ses", "conversation-2")

    def test_encoded_timestamp_is_recent(self):
        value = opencode_identifier("ses", "conversation-1")
        head = value[4:16]
        # The low three hex characters are fixed by the bitwise-NOT layout.
        assert head.endswith("ffe")
        payload = int(head, 16)
        ms36 = (~payload & ((1 << 48) - 1)) >> 12
        modulus = 1 << 36
        now_ms = int(time.time() * 1000)
        # The decoder sees 36 bits; the nearest matching instant must be
        # within the measured +/-30 day validity window.
        nearest = ms36 + round((now_ms - ms36) / modulus) * modulus
        assert abs(now_ms - nearest) < 30 * 24 * 3600 * 1000

    def test_request_ids_carry_a_fresh_counter(self):
        first = opencode_identifier("msg", "conversation-1", counter=1)
        second = opencode_identifier("msg", "conversation-1", counter=2)
        assert first != second
        assert re.fullmatch(r"msg_[0-9a-f]{12}[0-9A-Za-z]{14}", first)
