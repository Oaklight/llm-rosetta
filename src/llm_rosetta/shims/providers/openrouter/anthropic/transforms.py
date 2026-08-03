"""OpenRouter Anthropic shim transforms.

OpenRouter's Anthropic-compatible endpoint (`/api`) is a faithful proxy
of the Anthropic Messages API — no request- or response-side transforms
are needed.  The IR-level ``auto_cache_breakpoints`` transform injects
cache breakpoints for cross-format requests.
"""

from llm_rosetta.shims.transforms import auto_cache_breakpoints

post_ir_transforms = ()
pre_ir_transforms = ()
ir_transforms = (auto_cache_breakpoints(),)
