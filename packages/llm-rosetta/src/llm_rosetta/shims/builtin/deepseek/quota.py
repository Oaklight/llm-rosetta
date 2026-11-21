"""DeepSeek quota probe — the account balance, and its in-band refusal.

DeepSeek has no allowance windows: the account is prepaid, and the only
question is whether it still holds money.  The balance endpoint answers that,
and a 402 from the inference API names it when it does not.

The two halves are used differently:

* ``inband()`` classifies the 402 by its **machine field**
  (``error.type == "insufficient_balance_error"``), never by the prose
  message, and reports the refusal as ``permanent``: a balance does not refill
  on a timer, so the gateway parks the provider instead of putting it on the
  429 ladder.  Any other 402 returns ``None`` and stays the ladder's.
* ``parse()`` reads ``/user/balance``.  A reading that says ``is_available``
  true is what clears the park; the gateway does that, not this module.

Loaded once for the ``deepseek`` group; the chat and open-responses leaves
select it with ``quota.probe: deepseek``.  Module scope imports only the
standard library: shims load before the gateway package, so the types built
here are imported inside the methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from llm_rosetta.contracts.quota import InBandSignal, QuotaWindow

#: The reason the gateway stores with a permanent park.  It is the key the
#: panel's ``BLOCK_REASON_KEYS`` translates, so both sides spell it once.
BALANCE_EXHAUSTED = "balance exhausted"

#: DeepSeek's own error type for an account that cannot pay.  The field, not
#: the message: the prose is translated and reworded, the type is contract.
_INSUFFICIENT_BALANCE_TYPE = "insufficient_balance_error"


class DeepSeekBalanceProbe:
    """The prepaid balance of a DeepSeek account."""

    id = "deepseek"
    #: Only the vendor's own host answers this endpoint; a reseller lane is
    #: read in band, because the reseller key does not authenticate here.
    default_url: str | None = None
    BALANCE_URL = "https://api.deepseek.com/user/balance"

    def match(self, base_url: str) -> str | None:
        host = (urlparse(base_url).hostname or "").lower()
        if host == "api.deepseek.com" or host.endswith(".deepseek.com"):
            return self.BALANCE_URL
        return None

    def headers(self, key: str) -> dict[str, str]:
        return {}

    def parse(self, body: Any) -> tuple[str | None, list[QuotaWindow]]:
        from llm_rosetta.contracts.quota import (
            QuotaProbeError,
            QuotaWindow,
            vendor_message,
        )

        if not isinstance(body, dict):
            raise QuotaProbeError("DeepSeek's balance reply is not an object")
        available = body.get("is_available")
        if not isinstance(available, bool):
            raise QuotaProbeError(
                "DeepSeek's balance reply has no 'is_available' boolean "
                f"(got {vendor_message(body)})"
            )
        # A balance is not a window: there is no share used and no reset time.
        # The status is what the two readers agree on — "ok" or not.
        return None, [
            QuotaWindow(
                name="balance",
                used_percent=None,
                span_seconds=None,
                resets_at=None,
                status="ok" if available else BALANCE_EXHAUSTED,
            )
        ]

    def describe_error(self, status_code: int, body: Any) -> str:
        from llm_rosetta.contracts.quota import vendor_message

        if status_code == 401:
            return "the key does not authenticate (HTTP 401)"
        return f"HTTP {status_code}: {vendor_message(body)}"

    def inband(self, status_code: int, body: Any) -> InBandSignal | None:
        from llm_rosetta.contracts.quota import InBandSignal

        if status_code != 402:
            return None
        error = body.get("error") if isinstance(body, dict) else None
        if not isinstance(error, dict):
            return None
        if error.get("type") != _INSUFFICIENT_BALANCE_TYPE:
            return None
        return InBandSignal(
            cooldown_seconds=None,
            reason=BALANCE_EXHAUSTED,
            permanent=True,
        )


PROBES = (DeepSeekBalanceProbe(),)
