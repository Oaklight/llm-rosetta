"""Conversion pipeline — orchestrates format conversion between LLM APIs.

This module provides two layers of API:

**High-level** — :class:`ConversionPipeline` class that encapsulates the
full conversion lifecycle (Phase 1→2→4).  Use this when you need
request conversion, response conversion, and/or streaming:

    pipeline = ConversionPipeline("chat_completions", "messages", target_shim="anthropic--messages")
    target_body = pipeline.convert_request(body)
    # ... transport sends target_body, receives upstream_response ...
    source_response = pipeline.convert_response(upstream_response)

**Low-level** — :func:`apply_ir_transforms` and the functions in
:mod:`llm_rosetta.ir.passes` for finer control over individual stages.

The pipeline is part of the core library — **no network dependency**.
It produces a target request body and consumes a target response body;
the caller (gateway, argo-proxy, etc.) owns the transport.
"""

from .errors import ConversionError
from .ir_transforms import (
    apply_ir_transforms,
    apply_shim_to_ir,
    configure_context,
    setup_shim_context,
)
from .pipeline import ConversionPipeline
from .streaming import (
    BaselineStreamProcessor,
    StreamProcessor,
    StreamProcessorProtocol,
)

__all__ = [
    "configure_context",
    "apply_ir_transforms",
    "setup_shim_context",
    "apply_shim_to_ir",
    "ConversionError",
    "StreamProcessorProtocol",
    "ConversionPipeline",
    "BaselineStreamProcessor",
    "StreamProcessor",
]
