"""Backward-compat shim — real module is at gateway.middleware.request_context."""

from .middleware.request_context import *  # noqa: F401,F403
from .middleware.request_context import request_context_var  # noqa: F401 — explicit for type checkers
