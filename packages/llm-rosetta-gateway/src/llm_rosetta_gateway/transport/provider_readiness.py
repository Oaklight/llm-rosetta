"""Provider readiness: the lifecycle state and the refusal that reports it.

A leaf module (standard library only), so routing, config, the pipelines, the
app and the deferred startup can all name these without importing one another.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

__all__ = ["ProviderInitState", "ProviderNotReady"]


class ProviderInitState(Enum):
    """Lifecycle state of a provider's deferred token initialization."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class ProviderNotReady(Exception):
    """Raised when every provider for a model is unavailable right now.

    Five states reach here, and a client should be told which: the providers
    are still initializing after startup, they failed to initialize, they are
    in a quota block with a known end, every one of them is rate limited (a
    run of 429s, or a lane's capacity ladder) with a known or unknown end, or
    every one of them has a reading that says its allowance is spent.  None of
    the three quota-shaped states is "initializing" — saying so sends a client
    that just received an allowance refusal looking for a startup problem.
    """

    def __init__(
        self,
        model: str,
        *,
        failed: bool = False,
        blocked_until: datetime | None = None,
        rate_limited_until: datetime | None = None,
        rate_limited: bool = False,
        exhausted_until: datetime | None = None,
        exhausted: bool = False,
        unsupported: tuple[str, ...] = (),
        now: datetime | None = None,
    ) -> None:
        self.model = model
        #: Capabilities no candidate declared, for the ``unsupported`` state.
        self.unsupported: tuple[str, ...] = tuple(unsupported)
        self.failed = failed
        #: Whether the reason is a spent allowance rather than startup state.
        self.exhausted = False
        #: Seconds the caller should wait before retrying, when the state has
        #: a known end (a quota block or a spent window does; startup does not).
        self.retry_after: int | None = None
        moment = now or datetime.now(timezone.utc)
        if self.unsupported:
            # The route exists, but not for this request: retrying will not
            # help, and the client is told which capability is missing.
            names = ", ".join(self.unsupported)
            super().__init__(
                f"No provider for model '{model}' serves this request; "
                f"every candidate lacks: {names}"
            )
            return
        if exhausted_until is not None:
            self.exhausted = True
            self.retry_after = max(1, int((exhausted_until - moment).total_seconds()))
            rendered = exhausted_until.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            super().__init__(
                f"All providers for model '{model}' have spent their allowance "
                f"until {rendered}; please retry after that"
            )
            return
        if exhausted:
            # Excluded by a reading, but the vendor named no reset: refuse to
            # invent one, and say only what is known.
            self.exhausted = True
            super().__init__(
                f"All providers for model '{model}' have spent their allowance "
                "and no reset time is known; please retry later"
            )
            return
        if blocked_until is not None:
            self.retry_after = max(1, int((blocked_until - moment).total_seconds()))
            rendered = blocked_until.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            super().__init__(
                f"All providers for model '{model}' are in a quota block until "
                f"{rendered}; please retry after that"
            )
            return
        if rate_limited_until is not None:
            self.retry_after = max(
                1, int((rate_limited_until - moment).total_seconds())
            )
            rendered = rate_limited_until.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            super().__init__(
                f"All providers for model '{model}' are rate limited until "
                f"{rendered}; please retry after that"
            )
            return
        if rate_limited:
            # A disable the state cannot date: a run of refusals that left a
            # lane out without a cooldown it could name.  Report the state and
            # say no reset is known, rather than borrowing a deadline.
            super().__init__(
                f"All providers for model '{model}' are rate limited and no "
                "reset time is known; please retry later"
            )
            return
        state = "failed" if failed else "initializing"
        super().__init__(
            f"All providers for model '{model}' are {state}; please retry shortly"
        )
