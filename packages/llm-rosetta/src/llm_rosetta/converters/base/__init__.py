"""
LLM-Rosetta - Base Converter Module
转换器基础模块
Base converter module

提供转换器的抽象基类和功能域组织架构：
- BaseConverter: 主转换器抽象基类
- BaseContentOps: 内容转换操作抽象基类
- BaseToolOps: 工具转换操作抽象基类
- BaseMessageOps: 消息转换操作抽象基类
- BaseConfigOps: 配置转换操作抽象基类

Provides abstract base classes for converters and functional domain organization:
- BaseConverter: Main converter abstract base class
- BaseContentOps: Content conversion operations abstract base class
- BaseToolOps: Tool conversion operations abstract base class
- BaseMessageOps: Message conversion operations abstract base class
- BaseConfigOps: Configuration conversion operations abstract base class
"""

from llm_rosetta.conversion_warnings import (  # noqa: F401
    ConversionWarning,
    WarningSink,
    current_sink,
    report_drop,
)

from .configs import BaseConfigOps  # noqa: F401
from .errors import UnrepresentableFinishReason  # noqa: F401
from .content import BaseContentOps  # noqa: F401
from .context import ConversionContext, MetadataMode, StreamContext
from .converter import BaseConverter
from .simple_converter import BaseSimpleConverter
from .embedding_converter import BaseEmbeddingConverter
from .decision_converter import BaseDecisionConverter
from .rerank_converter import BaseRerankConverter
from .messages import BaseMessageOps  # noqa: F401
from .tools import convert_content_blocks_to_ir  # noqa: F401
from .tools import convert_ir_content_blocks_to_p  # noqa: F401
from .tools import sanitize_schema  # noqa: F401
from .tools import BaseToolOps  # noqa: F401
from .tools import IntrinsicLeg, sink_leg  # noqa: F401

__all__ = [
    # 主转换器 Main converter
    "BaseConverter",
    # 非聊天转换器基类 Non-chat converter base
    "BaseSimpleConverter",
    "BaseRerankConverter",
    "BaseEmbeddingConverter",
    "BaseDecisionConverter",
    # 转换上下文 Conversion context
    "ConversionContext",
    "MetadataMode",
    "StreamContext",
    # 转换告警 Conversion warnings
    "ConversionWarning",
    "WarningSink",
    "current_sink",
    "report_drop",
]
