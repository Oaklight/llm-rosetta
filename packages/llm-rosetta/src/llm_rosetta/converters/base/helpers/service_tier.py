"""service_tier across dialects whose value spaces do not line up.

Measured against the live Google routes (2026-11-08): both
``generateContent`` and ``interactions`` accept ``flex``, ``standard`` and
``priority`` and answer ``400`` for anything else -- the Interactions error
names the set -- so those three travel unchanged.  OpenAI's ``auto`` and
``default`` and Anthropic's ``standard_only`` have no Google equivalent;
guessing a mapping would either corrupt the return leg of a round trip or put
a value on the wire the upstream rejects, so they are reported and omitted.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.conversion_warnings import report_loss

#: The values both live Google routes accept, measured 2026-11-08.
GOOGLE_SERVICE_TIERS = frozenset({"flex", "priority", "standard"})


def google_service_tier(value: Any, dialect: str) -> str | None:
    """Return the service tier *dialect* can carry, or ``None`` with a report.

    Args:
        value: The IR ``service_tier``.
        dialect: The Google dialect being written, for the message.

    Returns:
        The value to write, or ``None`` when the dialect cannot express it.
    """
    if isinstance(value, str) and value in GOOGLE_SERVICE_TIERS:
        return value
    report_loss(
        f"{dialect} accepts service_tier flex, priority or standard; "
        f"{value!r} has no equivalent and is omitted",
        subject="service_tier",
    )
    return None
