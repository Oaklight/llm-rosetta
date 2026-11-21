"""Helpers every part of the quota service uses: the provider's config key and
the timestamp format the readings and the admin payload carry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _iso(value: datetime | None) -> str | None:
    """Render a timestamp as ISO 8601 with a ``Z`` suffix."""
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def provider_config_key(pinfo: Any) -> str:
    """The provider's key in the config, which its routing state is keyed on.

    ``ProviderInfo.name`` is the base format, so two configured providers of
    the same format share it; the config key tells them apart.
    """
    return getattr(pinfo, "config_name", None) or pinfo.name
