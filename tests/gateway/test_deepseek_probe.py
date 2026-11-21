"""The DeepSeek balance probe: its in-band 402 and its balance reading.

The 402 is classified by the machine field, never by the prose, and the
refusal is marked permanent — a balance does not refill on a timer, so the
gateway parks the provider instead of putting it on the 429 ladder.
"""

from __future__ import annotations

import pytest

from llm_rosetta.shims.builtin.deepseek.quota import DeepSeekBalanceProbe


class TestDeepSeekBalanceProbe:
    """The 402 is classified by its machine field, never by the prose."""

    probe = DeepSeekBalanceProbe()
    BODY = {
        "error": {
            "message": "Insufficient Balance",
            "type": "insufficient_balance_error",
            "param": None,
            "code": "invalid_request_error",
        }
    }

    def test_an_insufficient_balance_402_is_permanent(self):
        signal = self.probe.inband(402, self.BODY)

        assert signal is not None
        assert signal.permanent is True
        assert signal.reason == "balance exhausted"
        assert signal.cooldown_seconds is None, "a balance has no refill timer"

    def test_the_prose_alone_is_not_a_signal(self):
        """Same words, a different type: the ladder stays in charge."""
        body = {"error": {"message": "Insufficient Balance", "type": "other"}}

        assert self.probe.inband(402, body) is None

    def test_any_other_status_is_not_a_signal(self):
        assert self.probe.inband(429, self.BODY) is None
        assert self.probe.inband(500, self.BODY) is None

    def test_only_the_vendor_host_has_a_balance_endpoint(self):
        assert (
            self.probe.match("https://api.deepseek.com/v1")
            == "https://api.deepseek.com/user/balance"
        )
        assert self.probe.match("https://opendesign.example.com/v1") is None

    def test_the_balance_reading_answers_availability(self):
        from llm_rosetta.contracts.quota import QuotaProbeError

        _, windows = self.probe.parse(
            {"is_available": False, "balance_infos": [{"currency": "CNY"}]}
        )
        assert windows[0].status == "balance exhausted"
        assert windows[0].exhausted is True

        _, healthy = self.probe.parse({"is_available": True, "balance_infos": []})
        assert healthy[0].exhausted is False

        with pytest.raises(QuotaProbeError):
            self.probe.parse({"balance_infos": []})
