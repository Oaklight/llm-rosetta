"""Backward-compat shim — real module is at gateway.middleware.auth."""

from .middleware.auth import *  # noqa: F401,F403
from .middleware.auth import api_key_context_var  # noqa: F401 — explicit for type checkers
