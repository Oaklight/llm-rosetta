"""Shared helpers for provider-specific opaque IR items."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...types.ir.passthrough import ProviderPassthroughItem


def restore_provider_passthrough_item(
    item: ProviderPassthroughItem,
    *,
    target_provider: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Restore an opaque item for the matching provider dialect."""
    source_provider = item["provider"]
    if source_provider != target_provider:
        warning = (
            f"Dropped provider passthrough item from {source_provider!r} "
            f"when converting to {target_provider!r}"
        )
        return None, warning
    return dict(item["payload"]), None


def merge_provider_output_items(
    portable_items: Sequence[dict[str, Any]],
    passthrough_items: Sequence[ProviderPassthroughItem],
    *,
    target_provider: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Merge matching provider output items into their recorded positions."""
    merged = [dict(item) for item in portable_items]
    warnings: list[str] = []
    positioned: list[tuple[int, int, dict[str, Any]]] = []
    trailing: list[dict[str, Any]] = []

    for order, item in enumerate(passthrough_items):
        payload, warning = restore_provider_passthrough_item(
            item, target_provider=target_provider
        )
        if warning is not None:
            warnings.append(warning)
            continue
        assert payload is not None
        position = item.get("position")
        if position is None:
            trailing.append(payload)
        else:
            positioned.append((max(position, 0), order, payload))

    inserted = 0
    previous_position: int | None = None
    for position, _order, payload in sorted(positioned):
        if previous_position == position:
            insertion_index = min(position + inserted, len(merged))
            inserted += 1
        else:
            insertion_index = min(position, len(merged))
            previous_position = position
            inserted = 1
        merged.insert(insertion_index, payload)
    merged.extend(trailing)
    return merged, warnings


__all__ = ["merge_provider_output_items", "restore_provider_passthrough_item"]
