"""Argo OpenAI Responses schema transforms.

The Responses path is much simpler than Chat — most Chat-specific
transforms (role downgrade, content defaulting, system flattening)
do not apply.  Only the shared model-list transform is needed.
"""

from ..model_utils import model_list_transform  # noqa: F401

post_ir_transforms = ()
pre_ir_transforms = ()
ir_transforms = ()
