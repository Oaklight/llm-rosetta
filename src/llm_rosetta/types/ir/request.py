"""
LLM-Rosetta - IR Request Types

基于 sdk_body_structures.md 设计的统一请求参数类型
Unified request parameter types based on sdk_body_structures.md

设计原则：
- 核心参数（90%场景）：必需且普遍支持
- 可选参数（少见）：通过 provider_extensions 支持
- 渐进式复杂度：简单场景不需要了解所有参数
"""

import sys
from typing import Any, TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required

from .configs import (
    CacheConfig,
    GenerationConfig,
    ReasoningConfig,
    ResponseFormatConfig,
    StreamConfig,
)
from .messages import Message
from .tools import ToolCallConfig, ToolChoice, ToolDefinition

# ============================================================================
# 主请求类型 Main Request Type
# ============================================================================


class IRRequest(TypedDict):
    """
    统一的IR请求类型
    Unified IR request type

    这个类型整合了所有provider的核心请求参数，提供统一的接口。

    必需字段 Required fields:
    - model: 模型ID
    - messages: 消息列表

    可选字段按功能分组:
    - 系统指令: system_instruction
    - 工具相关: tools, tool_choice, tool_config
    - 生成控制: generation, response_format
    - 流式输出: stream
    - 推理配置: reasoning
    - 缓存配置: cache
    """

    # ========== 必需字段 Required Fields ==========
    model: Required[str]
    messages: Required[list[Message]]

    # ========== 系统指令 System Instruction ==========
    # 映射关系:
    # - OpenAI Chat: messages[0] with role="system"
    # - OpenAI Responses: instructions
    # - Anthropic: system
    # - Google: config.system_instruction
    system_instruction: NotRequired[str]

    # ========== 工具相关 Tool Related ==========
    tools: NotRequired[list[ToolDefinition]]
    tool_choice: NotRequired[ToolChoice]
    tool_config: NotRequired[ToolCallConfig]

    # ========== 生成控制 Generation Control ==========
    generation: NotRequired[GenerationConfig]
    response_format: NotRequired[ResponseFormatConfig]

    # ========== 流式输出 Streaming ==========
    stream: NotRequired[StreamConfig]

    # ========== 推理配置 Reasoning ==========
    reasoning: NotRequired[ReasoningConfig]

    # ========== 缓存配置 Cache ==========
    cache: NotRequired[CacheConfig]

    # ========== Provider特定扩展 Provider-specific Extensions ==========
    # 用于存储少见或provider特定的参数:
    # - 元数据相关: metadata, user (少见)
    # - 音频配置: audio (少见)
    # - 服务配置: service_tier, store, background (少见)
    # - 安全配置: safety_identifier, safety_settings (少见)
    # - 会话配置: conversation, previous_response_id (少见，OpenAI Responses)
    # - 其他: truncation已移到generation，verbosity等
    provider_extensions: NotRequired[dict[str, Any]]


# ============================================================================
# 导出的主要类型 Main Exported Types
# ============================================================================

__all__ = [
    # 主请求类型
    "IRRequest",
]
