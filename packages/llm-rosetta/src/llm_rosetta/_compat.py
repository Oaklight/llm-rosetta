"""Sync/async compatibility helpers.

Allows callers that have not yet migrated to ``await`` to keep calling
gateway functions synchronously.  The decorator ``@sync_compat``
replaces the async function at runtime with a sync wrapper that:

* In a **sync** context (no running event loop): runs the coroutine to
  completion via ``asyncio.run()`` and returns the result.
* In an **async** context: schedules the coroutine as a ``Task`` on the
  running loop and returns it.  Because ``Task`` is awaitable, existing
  ``await func(...)`` call sites continue to work unchanged.

At type-checking time the original ``async def`` signature is
preserved, so ``ty`` / ``pyright`` see the function as a coroutine and
``await`` on it is valid.
"""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING, Any, TypeVar

_T = TypeVar("_T")

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


def sync_compat(
    fn: Callable[..., Coroutine[Any, Any, _T]],
) -> Callable[..., Coroutine[Any, Any, _T]]:
    """Decorator: make an async function callable from sync contexts.

    At **runtime** the decorated function becomes a regular ``def`` that
    delegates to :func:`maybe_sync`.  At **type-check time** the
    original ``async def`` signature is preserved (thanks to the
    ``TYPE_CHECKING`` guard on the return annotation) so ``await`` on
    the result is valid.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return maybe_sync(fn(*args, **kwargs))

    return wrapper  # type: ignore[return-value]


def maybe_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run *coro* synchronously or schedule it on the running loop.

    Returns:
        The coroutine's result when no event loop is running, or an
        ``asyncio.Task`` when called inside an active loop.  The return
        type is annotated as ``_T`` (not ``_T | Task[_T]``) because
        callers that ``await`` the result always get ``_T``, and sync
        callers receive ``_T`` directly.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return loop.create_task(coro)  # type: ignore[return-value]  # ty: ignore[invalid-return-type]
