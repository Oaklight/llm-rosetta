"""A scaled entry is still the same candidate.

`ModelRoute._candidates` rebuilds entries to apply a request-time weight
factor (an allowance headroom, a pace strategy).  Rebuilding dropped the
entry's declared ``capabilities``, so the entry handed to the strategy — and
carried as ``ctx.chosen`` — no longer described the candidate that was
judged.  Independent fix found while doing #1430.
"""

from __future__ import annotations

import pytest

from llm_rosetta_gateway.routing import ModelRoute, PaceStrategy, ProviderEntry

ROUTE = "scaled"


def test_a_headroom_scaled_entry_keeps_its_capabilities():
    route = ModelRoute(
        [ProviderEntry("p", upstream_model="fast", capabilities=("text",))],
        ROUTE,
    )

    scaled = route._candidates({"p": 0.5})

    assert scaled[0].capabilities == ("text",)
    assert scaled[0].upstream_model == "fast"
    assert scaled[0].weight == pytest.approx(0.5)


def test_the_pace_scaling_keeps_them_too():
    route = ModelRoute(
        [ProviderEntry("p", upstream_model="fast", capabilities=("text", "vision"))],
        ROUTE,
        strategy=PaceStrategy(),
    )

    scaled = route._candidates({"p": 1.0})

    assert scaled[0].capabilities == ("text", "vision")
