"""The error a conversion phase raises."""

from __future__ import annotations

from typing import Any


class ConversionError(Exception):
    """Raised when a conversion phase fails.

    Attributes:
        phase: Which pipeline phase failed (``"source_to_ir"``,
            ``"ir_to_target"``, ``"response_to_ir"``,
            ``"ir_to_source"``).
        reason: The terminal reason that could not be expressed, when the
            failure is one (``"error"`` / ``"cancelled"``); else ``""``.
        response_id: The upstream response the failure belongs to, when the
            converter named one.
        provider_metadata: Metadata the converter carried with the failure.
    """

    def __init__(
        self,
        message: str,
        phase: str,
        *,
        reason: str = "",
        response_id: str = "",
        provider_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.phase = phase
        self.reason = reason
        self.response_id = response_id
        self.provider_metadata = dict(provider_metadata or {})
        super().__init__(message)


# ---------------------------------------------------------------------------
# ConversionPipeline
# ---------------------------------------------------------------------------
