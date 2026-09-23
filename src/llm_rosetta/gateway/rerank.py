"""Backward-compat shim — real module is at gateway.pipelines.rerank."""

from .pipelines.rerank import *  # noqa: F401,F403
from .pipelines.rerank import _detect_source_format  # noqa: F401 — underscore name excluded by star import
