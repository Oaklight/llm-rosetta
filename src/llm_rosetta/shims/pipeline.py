"""Backward-compatibility shim for ``llm_rosetta.shims.pipeline``.

This module moved to ``llm_rosetta.pipeline`` — the pipeline is an
orchestration layer that *consumes* shim config, not part of the shim
package itself.  Imports from the old path keep working.
"""

from llm_rosetta.pipeline import (  # noqa: F401
    apply_shim_to_ir,
    setup_shim_context,
)

__all__ = ["apply_shim_to_ir", "setup_shim_context"]
