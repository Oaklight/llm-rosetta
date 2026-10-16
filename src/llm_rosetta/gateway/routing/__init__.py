"""Gateway routing — how one request picks its upstream.

The package owns *selection*: which provider serves a request, which upstream
credential it uses, and what upstream session identity it carries.  Today every
choice is derived from the request itself — the layer holds no state across
requests — and the strategies below are the terminal stage that picks a
provider out of the candidates the earlier stages left standing.

The public names live here rather than in :mod:`~llm_rosetta.gateway.routing.strategy`
so the internal layout can change without moving the import path::

    from llm_rosetta.gateway.routing import ModelRoute, create_strategy

Anything not listed in ``__all__`` is internal.
"""

from __future__ import annotations

from .context import Identity, SelectionContext
from .select import AnchorStage, select_ready_entry
from .stage import RoutingChain, RoutingStage, StageOrderError
from .strategy import (
    DEFAULT_STRATEGY,
    AffinityRoundRobinStrategy,
    ModelRoute,
    ProviderEntry,
    RoutingStrategy,
    WeightedRoundRobinStrategy,
    create_strategy,
)

__all__ = [
    "DEFAULT_STRATEGY",
    "AffinityRoundRobinStrategy",
    "AnchorStage",
    "Identity",
    "ModelRoute",
    "ProviderEntry",
    "RoutingChain",
    "RoutingStage",
    "RoutingStrategy",
    "SelectionContext",
    "StageOrderError",
    "WeightedRoundRobinStrategy",
    "create_strategy",
    "select_ready_entry",
]
