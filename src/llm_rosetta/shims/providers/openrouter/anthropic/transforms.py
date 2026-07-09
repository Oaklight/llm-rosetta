"""OpenRouter Anthropic shim transforms.

OpenRouter's Anthropic-compatible endpoint (`/api`) is a faithful proxy
of the Anthropic Messages API — no request- or response-side transforms
are needed.
"""

post_ir_transforms = ()
pre_ir_transforms = ()
