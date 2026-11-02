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
from .eligibility import (
    BLOCKED,
    CAPABILITY,
    INITIALIZING,
    NO_CREDENTIAL,
    RATE_LIMITED,
    SPENT,
    Eligibility,
    EligibilityStage,
    provider_eligibility,
    quota_weight,
    required_capabilities,
)
from .key import ProviderKeyAffinity
from .model_routing import ModelRouting, parse_model_routing, routing_from_legacy
from .prefix import compute_affinity_identity, extract_prefix_from_ir
from .rules import (
    EFFORT_ORDER,
    Rule,
    RuleFacts,
    RulesStage,
    facts_from_ir,
    matches,
    parse_rules,
)
from .same_format import SameFormatStage, shim_group_key
from .session import SessionHeaderAffinity
from .select import AnchorStage, select_ready_entry
from .stage import RoutingChain, RoutingStage, StageOrderError
from .strategy import (
    DEFAULT_STRATEGY,
    AffinityRoundRobinStrategy,
    LeastUsedStrategy,
    ModelRoute,
    OrderedStrategy,
    PaceStrategy,
    ProviderEntry,
    RotateStrategy,
    RoutingStrategy,
    WeightedRoundRobinStrategy,
    create_strategy,
    strategy_for,
)
from .unavailable import Unavailable
from .unavailable import classify as classify_unavailable

__all__ = [
    "BLOCKED",
    "EFFORT_ORDER",
    "DEFAULT_STRATEGY",
    "INITIALIZING",
    "NO_CREDENTIAL",
    "RATE_LIMITED",
    "SPENT",
    "AffinityRoundRobinStrategy",
    "AnchorStage",
    "Eligibility",
    "EligibilityStage",
    "Identity",
    "LeastUsedStrategy",
    "ModelRoute",
    "ModelRouting",
    "OrderedStrategy",
    "PaceStrategy",
    "ProviderEntry",
    "ProviderKeyAffinity",
    "RotateStrategy",
    "RoutingChain",
    "RoutingStage",
    "RoutingStrategy",
    "Rule",
    "RuleFacts",
    "RulesStage",
    "SameFormatStage",
    "SelectionContext",
    "SessionHeaderAffinity",
    "StageOrderError",
    "Unavailable",
    "WeightedRoundRobinStrategy",
    "CAPABILITY",
    "classify_unavailable",
    "compute_affinity_identity",
    "create_strategy",
    "extract_prefix_from_ir",
    "facts_from_ir",
    "matches",
    "parse_model_routing",
    "parse_rules",
    "provider_eligibility",
    "quota_weight",
    "required_capabilities",
    "routing_from_legacy",
    "select_ready_entry",
    "shim_group_key",
    "strategy_for",
]
