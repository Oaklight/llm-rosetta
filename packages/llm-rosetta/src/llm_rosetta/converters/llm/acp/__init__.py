"""ACP (Agent Client Protocol) lane.

The lane drives a local agent CLI over ACP instead of an HTTP model API.  See
:mod:`llm_rosetta.converters.llm.acp.converter` for the contract: it is a
target-only format, and its transport lives in the gateway distribution.
"""

from .config_ops import AcpConfigOps
from .content_ops import AcpContentOps
from .converter import AcpConverter, target_finish_reason
from .message_ops import AcpMessageOps
from .tool_ops import AcpToolOps

__all__ = [
    "AcpConverter",
    "AcpContentOps",
    "AcpMessageOps",
    "AcpToolOps",
    "AcpConfigOps",
    "target_finish_reason",
]
