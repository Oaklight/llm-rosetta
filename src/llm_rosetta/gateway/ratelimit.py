"""Backward-compat shim — real module is at gateway.middleware.ratelimit."""

from .middleware.ratelimit import *  # noqa: F401,F403
from .middleware.ratelimit import (  # noqa: F401 — underscore names excluded by star import
    _extract_model,
    _rate_limit_response,
    _rate_limit_result_var,
)
