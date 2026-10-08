"""Unified retention policy for gateway observability data.

Centralizes retention configuration (caps, age limits) and insert-time
prune scheduling that was previously scattered across PersistenceManager
instance variables, config constants, and ad-hoc insert counters.

The SQL execution stays in :class:`PersistenceManager` — this module
owns the *policy* and *scheduling*, not the queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Prune after this many inserts per table category
_PRUNE_INTERVAL = 100


@dataclass
class RetentionPolicy:
    """Single source of truth for all retention configuration.

    Replaces scattered constants: ``DEFAULT_SUCCESS_MAX``, ``DEFAULT_DUMP_MAX``,
    ``DEFAULT_OPS_INFO_MAX``, ``DEFAULT_OPS_WARN_MAX``, ``DEFAULT_MAX_AGE_DAYS``.
    """

    success_max: int = 50_000
    dump_max: int = 10_000
    ops_info_max: int = 10_000
    ops_warn_max: int = 5_000
    max_age_days: int = 90


@dataclass
class RetentionTracker:
    """Tracks insert counts and decides when to trigger amortized pruning.

    Each ``note_insert()`` call increments the counter for the given
    category.  When the counter reaches the prune interval, ``True`` is
    returned and the counter resets — the caller should run the
    appropriate prune operation.
    """

    _counts: dict[str, int] = field(default_factory=dict)

    def note_insert(self, category: str, n: int = 1) -> bool:
        """Record *n* inserts for *category*; return True when prune is due."""
        self._counts[category] = self._counts.get(category, 0) + n
        if self._counts[category] >= _PRUNE_INTERVAL:
            self._counts[category] = 0
            return True
        return False

    def reset(self, category: str | None = None) -> None:
        """Reset counter(s) — useful after a manual prune."""
        if category is None:
            self._counts.clear()
        else:
            self._counts.pop(category, None)
