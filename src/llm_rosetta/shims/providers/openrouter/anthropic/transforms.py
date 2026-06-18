"""OpenRouter Anthropic shim transforms.

OpenRouter's Anthropic-compatible endpoint (`/api`) is a faithful proxy
of the Anthropic Messages API — no request- or response-side transforms
are needed.
"""

to_transforms = ()
from_transforms = ()
