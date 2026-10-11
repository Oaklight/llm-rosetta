"""Derived / default computations for decision answers.

The IR stores only the *primary* provider output — the probability
distribution plus ``unknown_probability``, or an assertion's scalar
``probability``.  Fields that are mathematically derivable (``choice``,
``score``, ``confidence``, ``abstained``) are exposed here as pure
functions.  Each returns the provider's stored value when present and
otherwise falls back to the documented derivation.

Kept as functions (not ``@property``) because the IR is dict-shaped
(``TypedDict``); see issue #889 comment 6/N.
"""

from __future__ import annotations

from typing import Any

from .schema_ops import compute_confidence


def _stored(answer: dict[str, Any], key: str) -> Any:
    value = answer.get(key)
    return value if value is not None else None


def abstained(answer: dict[str, Any]) -> bool:
    """Whether the model abstained — the unknown mass is the largest option.

    Always derived (``abstained`` is not a stored field).  An answer with no
    ``unknown_probability`` never abstains.  A tie counts as abstained
    (``unknown_probability`` equal to the largest option).
    """
    unknown = answer.get("unknown_probability")
    if unknown is None:
        return False
    probs = answer.get("probabilities")
    if probs:
        return all(unknown >= p for p in probs.values())
    probability = answer.get("probability")
    if probability is None:
        return unknown >= 0.5
    return unknown >= max(probability, 1.0 - probability)


def confidence(answer: dict[str, Any]) -> float:
    """Confidence of the answer.

    Uses the provider's ``confidence`` when present; otherwise derives it from
    the distribution (``1 - normalized Shannon entropy`` of ``probabilities``,
    or the credence ``max(p, 1 - p)`` for an assertion) and scales it by
    ``(1 - unknown_probability)`` so an abstained answer reads as low
    confidence — consistent with :func:`abstained`.
    """
    stored = _stored(answer, "confidence")
    if stored is not None:
        return stored
    probs = answer.get("probabilities")
    if probs:
        base = compute_confidence(probs)
    else:
        probability = answer.get("probability")
        base = max(probability, 1.0 - probability) if probability is not None else 0.0
    unknown = answer.get("unknown_probability")
    if unknown is not None:
        base *= max(0.0, 1.0 - unknown)
    return base


def choice(answer: dict[str, Any]) -> Any:
    """Selected option.

    Uses the provider's ``choice`` when present; otherwise ``argmax`` of
    ``probabilities``.
    """
    stored = _stored(answer, "choice")
    if stored is not None:
        return stored
    probs = answer.get("probabilities")
    if not probs:
        return None
    return max(probs, key=lambda k: probs[k])


def score(answer: dict[str, Any], options: list[dict[str, Any]]) -> float:
    """Weighted mean over ordered options.

    Uses the provider's ``score`` when present; otherwise computes
    ``Σ position · probability`` over ``options`` (the question's ordered
    ``criteria``).  ``probabilities`` is keyed by ``str(label)``.
    """
    stored = _stored(answer, "score")
    if stored is not None:
        return stored
    probs = answer.get("probabilities") or {}
    return sum(
        i * probs.get(str(entry["label"]), 0.0) for i, entry in enumerate(options)
    )


__all__ = ["abstained", "choice", "confidence", "score"]
