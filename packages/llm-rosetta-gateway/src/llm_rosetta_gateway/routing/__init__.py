"""Gateway routing — how one request picks its upstream.

The package owns *selection*: which provider serves a request, which upstream
credential it uses, and what upstream session identity it carries.  Today every
choice is derived from the request itself — the layer holds no state across
requests — and the strategies below are the terminal stage that picks a
provider out of the candidates the earlier stages left standing.

The public names live here rather than in :mod:`~llm_rosetta_gateway.routing.strategy`
so the internal layout can change without moving the import path::

    from llm_rosetta_gateway.routing import ModelRoute, create_strategy

Anything not listed in ``__all__`` is internal.
"""

from __future__ import annotations

from .context import Identity, SelectionContext
from .key import ProviderKeyAffinity
from .prefix import compute_affinity_identity, extract_prefix_from_ir
from .quota import QuotaStage
from .same_format import SameFormatStage, shim_group_key
from .session import SessionHeaderAffinity
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
from .unavailable import Unavailable
from .unavailable import classify as classify_unavailable

__all__ = [
    "DEFAULT_STRATEGY",
    "AffinityRoundRobinStrategy",
    "AnchorStage",
    "Identity",
    "ModelRoute",
    "ProviderEntry",
    "ProviderKeyAffinity",
    "QuotaStage",
    "RoutingChain",
    "RoutingStage",
    "RoutingStrategy",
    "SameFormatStage",
    "SelectionContext",
    "SessionHeaderAffinity",
    "StageOrderError",
    "Unavailable",
    "WeightedRoundRobinStrategy",
    "classify_unavailable",
    "compute_affinity_identity",
    "create_strategy",
    "extract_prefix_from_ir",
    "select_ready_entry",
    "shim_group_key",
]
