"""Give a fake app the ``admin`` context the admin routes reach the config through.

``create_app`` sets ``app.admin`` (an ``AdminContext``); admin routes read the
served config with ``admin_context(request).served_config()``, not with
``request.app.gateway_config``.  A fake app built by hand exposes
``gateway_config`` only, so it needs the same adapter the real app has.  The
adapter reads the app's attributes when it is asked, so a test that assigns
``app.gateway_config`` after the app is built is still seen.

Not a test module: pytest does not collect it (no ``test_`` prefix).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, TypeVar

from llm_rosetta_gateway.app.admin_context import GatewayAdminContext

_App = TypeVar("_App")


def attach_admin(app: _App) -> _App:
    """Set ``app.admin`` to the gateway's own adapter over *app*; return *app*."""
    setattr(app, "admin", GatewayAdminContext(app))
    return app


def fake_app(**attrs: Any) -> SimpleNamespace:
    """A ``SimpleNamespace`` app with *attrs* and an ``admin`` context."""
    return attach_admin(SimpleNamespace(**attrs))
