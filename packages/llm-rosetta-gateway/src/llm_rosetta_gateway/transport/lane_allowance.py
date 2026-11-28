"""Where a lane's reported allowance goes: the sink a provider hands it to.

An agent that serves a lane reports the account's rate-limit windows during a
turn.  The transport does not keep them: it hands each update to the lane's
:class:`LaneAllowanceSink` through ``ProviderInfo.record_lane_reading``, and
reads the last reading back through ``ProviderInfo.lane_reading``.  Whoever owns
the allowance state (the quota layer) provides the sink, so the transport, which
is the foundation layer, needs no import of it and holds no state of its own.

A transport that ships in an extension reaches the same two methods, so they are
the seam an extension's lane uses to report an allowance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class LaneAllowanceSink(Protocol):
    """What keeps the allowance reports of the lanes."""

    def record(self, update: Mapping[str, Any], lane: Any) -> bool:
        """Store the allowance *update* reports for *lane*, if it carries one.

        Args:
            update: A ``session/update`` shaped mapping.
            lane: The lane's ``ProviderInfo``.

        Returns:
            Whether the update carried an allowance.
        """
        ...

    def reading(self, lane: Any) -> Any | None:
        """Return *lane*'s last stored allowance reading, or ``None``."""
        ...
