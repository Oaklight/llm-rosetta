"""
LLMIR - Base Converter Module
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

from .configs import BaseConfigOps
from .content import BaseContentOps
from .converter import BaseConverter
from .messages import BaseMessageOps
from .stream_context import StreamContext
from .tools import BaseToolOps

__all__ = [
    # 主转换器 Main converter
    "BaseConverter",
    # 流式上下文 Stream context
    "StreamContext",
    # 功能域操作类 Functional domain operation classes
    "BaseContentOps",
    "BaseToolOps",
    "BaseMessageOps",
    "BaseConfigOps",
]
