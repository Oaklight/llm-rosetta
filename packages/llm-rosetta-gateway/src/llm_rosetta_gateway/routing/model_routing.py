"""The routing block a model may carry, and what it means.

A model's provider list says *who* may serve it.  This block says *how* the
gateway chooses between them, and it is the only place that says so: before
it, one boolean lived in the config (``prefer_same_format``), one strategy name
sat beside the provider list, and the rest of the behaviour was implicit in
which stages happened to be in the chain.

The block is parsed once, when the config is built, by the same function the
admin save path uses — so a value the panel accepts is a value the router can
honour, and an unknown key fails the load rather than the first request.

A model without a block keeps exactly today's behaviour: weighted round-robin,
no affinity, sink enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: How a model's providers are chosen between when no affinity applies.
STRATEGIES = ("weighted", "ordered", "rotate", "least_used", "pace")

#: Whether one client keeps one provider.
AFFINITIES = ("off", "session")

#: Strategy spellings accepted before this block existed, and the block they
#: stand for.
_LEGACY_STRATEGIES = {
    "weighted_round_robin": ("weighted", "off"),
    "affinity_round_robin": ("weighted", "session"),
}

#: Values the block documents but does not implement yet.  Refusing them by
#: name is better than accepting one and silently ignoring it.
_PENDING_AFFINITIES = {
    "turn": "a conversation-level identity needs the pre-conversion fingerprint",
    "strict": "a conversation that never moves needs a store",
}


@dataclass(frozen=True, slots=True)
class ModelRouting:
    """How one model's providers are chosen between.

    Attributes:
        strategy: One of :data:`STRATEGIES`.
        affinity: One of :data:`AFFINITIES`.  ``session`` keeps one client on
            one provider and takes precedence over the strategy's per-request
            choice, which is why it is only allowed with ``weighted``.
        sink: Whether a recent 429 orders a provider last for this model.
            The provider's *disable* is process-wide; this only says whether
            the model cares about the milder sink.
        first_token_ms: Reserved for the first-token deadline.  Parsed and
            kept, but the selector does not act on it yet.
        rules: Reserved for the per-model rules of the next slice.
    """

    strategy: str = "weighted"
    affinity: str = "off"
    sink: bool = True
    first_token_ms: int | None = None
    rules: tuple[Any, ...] = ()

    def describe(self) -> dict[str, Any]:
        """The block as the admin panel reports it."""
        return {
            "strategy": self.strategy,
            "affinity": self.affinity,
            "sink": self.sink,
            "first_token_ms": self.first_token_ms,
            "rules": list(self.rules),
        }


def parse_model_routing(raw: Any, *, label: str) -> ModelRouting:
    """Parse one model's ``routing`` block.

    Args:
        raw: The block, or ``None`` when the model has none.
        label: The config path, for the error message.

    Returns:
        The parsed block; the defaults when *raw* is ``None``.

    Raises:
        ValueError: If the block is malformed, names an unknown value, or uses
            a value this build does not implement yet.
    """
    if raw is None:
        return ModelRouting()
    if not isinstance(raw, dict):
        raise ValueError(f"config: {label} must be an object, got {raw!r}")

    unknown = set(raw) - {"strategy", "affinity", "sink", "first_token_ms", "rules"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"config: {label} has unknown key(s): {names}")

    strategy, affinity = _parse_strategy_affinity(raw, label)

    sink = raw.get("sink", True)
    if not isinstance(sink, bool):
        raise ValueError(f"config: {label}.sink must be true or false, got {sink!r}")

    first_token_ms = raw.get("first_token_ms")
    if first_token_ms is not None:
        if isinstance(first_token_ms, bool) or not isinstance(first_token_ms, int):
            raise ValueError(
                f"config: {label}.first_token_ms must be an integer, got {first_token_ms!r}"
            )
        if first_token_ms <= 0:
            raise ValueError(
                f"config: {label}.first_token_ms must be greater than 0, "
                f"got {first_token_ms!r}"
            )

    rules = raw.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError(f"config: {label}.rules must be an array, got {rules!r}")
    if rules:
        raise ValueError(
            f"config: {label}.rules is not supported yet; a rule that did "
            "nothing would be worse than no rule"
        )

    return ModelRouting(
        strategy=strategy,
        affinity=affinity,
        sink=sink,
        first_token_ms=first_token_ms,
        rules=(),
    )


def routing_from_legacy(name: str, *, label: str) -> ModelRouting:
    """Translate a model's pre-block ``strategy`` value.

    Args:
        name: The value of ``models.<name>.strategy``.
        label: The config path, for the error message.

    Returns:
        The equivalent block.

    Raises:
        ValueError: If the name is not one this build knows.
    """
    if name in _LEGACY_STRATEGIES:
        strategy, affinity = _LEGACY_STRATEGIES[name]
        return ModelRouting(strategy=strategy, affinity=affinity)
    if name in STRATEGIES:
        return ModelRouting(strategy=name)
    raise ValueError(
        f"config: {label} must be one of {', '.join(STRATEGIES)}; got {name!r}"
    )


def _parse_strategy_affinity(raw: dict[str, Any], label: str) -> tuple[str, str]:
    """Read the strategy and affinity, refusing a combination that cannot work."""
    strategy = raw.get("strategy", "weighted")
    if not isinstance(strategy, str):
        raise ValueError(f"config: {label}.strategy must be a string, got {strategy!r}")
    if strategy in _LEGACY_STRATEGIES:
        legacy_strategy, legacy_affinity = _LEGACY_STRATEGIES[strategy]
        strategy = legacy_strategy
        if "affinity" not in raw:
            return strategy, legacy_affinity
    elif strategy not in STRATEGIES:
        raise ValueError(
            f"config: {label}.strategy must be one of {', '.join(STRATEGIES)}; "
            f"got {strategy!r}"
        )

    affinity = raw.get("affinity", "off")
    if not isinstance(affinity, str):
        raise ValueError(f"config: {label}.affinity must be a string, got {affinity!r}")
    if affinity in _PENDING_AFFINITIES:
        raise ValueError(
            f"config: {label}.affinity {affinity!r} is not implemented yet: "
            f"{_PENDING_AFFINITIES[affinity]}"
        )
    if affinity not in AFFINITIES:
        raise ValueError(
            f"config: {label}.affinity must be one of {', '.join(AFFINITIES)}; "
            f"got {affinity!r}"
        )
    if affinity == "session" and strategy != "weighted":
        raise ValueError(
            f"config: {label}: affinity 'session' keeps one client on one "
            f"provider, so it cannot be combined with strategy {strategy!r}; "
            "use strategy 'weighted'"
        )
    return strategy, affinity
