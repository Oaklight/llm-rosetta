"""Unified gateway operations layer.

Every gateway action is an :class:`OpsBase` subclass with automatic
audit recording.  Route handlers construct an Ops object and call
:meth:`~OpsBase.execute` — the base class handles recording.

.. code-block:: python

    result = await OpsClearData(request.app.ops_ctx, table="request_log").execute()
"""

from .base import OpsBase, OpsContext

__all__ = [
    "OpsBase",
    "OpsContext",
]
