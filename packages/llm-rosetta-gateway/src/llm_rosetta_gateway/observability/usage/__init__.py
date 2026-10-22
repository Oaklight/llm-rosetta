"""The usage report CLI, which reads the gateway's own database.

The *semantics* of a usage report — what the tokens add up to, and for whom —
live in the core (``llm_rosetta.observability.usage``).  What lives here is the
command that walks a ``gateway.db`` and renders them, because only the gateway
writes that database.

The console script is ``llm-rosetta-usage``; the entry point is
:func:`llm_rosetta_gateway.observability.usage.cli.main`.
"""

from __future__ import annotations
