"""
LLM-Rosetta - Base Tool Operations
工具转换操作的抽象基类
Abstract base class for tool conversion operations

处理所有工具相关的转换：
- 工具定义：函数签名、参数schema
- 工具调用：调用请求、参数传递
- 工具结果：执行结果、错误处理
- 工具配置：选择策略、调用配置
Handles all tool-related conversions:
- Tool definitions: function signatures, parameter schemas
- Tool calls: call requests, parameter passing
- Tool results: execution results, error handling
- Tool configurations: choice strategies, call configurations
"""

from abc import ABC, abstractmethod
from typing import Any

from ...types.ir import (
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)
from ...types.ir.tools import ToolCallConfig


class BaseToolOps(ABC):
    """工具转换操作的抽象基类
    Abstract base class for tool conversion operations

    统一处理工具生命周期的所有阶段：定义 → 选择 → 调用 → 结果。
    Uniformly handles all stages of the tool lifecycle: definition → choice → call → result.
    """

    # ==================== 工具定义转换 Tool definition conversion ====================

    @staticmethod
    @abstractmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> Any:
        """IR ToolDefinition → Provider Tool Definition
        将IR工具定义转换为Provider工具定义

        处理工具的基本信息：名称、描述、参数schema等。
        Handles basic tool information: name, description, parameter schema, etc.

        Args:
            ir_tool: IR格式的工具定义
            **kwargs: 额外参数

        Returns:
            Provider格式的工具定义
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_definition_to_ir(provider_tool: Any, **kwargs: Any) -> ToolDefinition:
        """Provider Tool Definition → IR ToolDefinition
        将Provider工具定义转换为IR工具定义

        Args:
            provider_tool: Provider格式的工具定义
            **kwargs: 额外参数

        Returns:
            IR格式的工具定义
        """
        pass

    # ==================== 工具选择转换 Tool choice conversion ====================

    @staticmethod
    @abstractmethod
    def ir_tool_choice_to_p(ir_tool_choice: ToolChoice, **kwargs: Any) -> Any:
        """IR ToolChoice → Provider Tool Choice Config
        将IR工具选择转换为Provider工具选择配置

        处理工具选择策略：none、auto、any、specific tool等。
        Handles tool choice strategies: none, auto, any, specific tool, etc.

        Args:
            ir_tool_choice: IR格式的工具选择
            **kwargs: 额外参数

        Returns:
            Provider格式的工具选择配置
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        """Provider Tool Choice Config → IR ToolChoice
        将Provider工具选择配置转换为IR工具选择

        Args:
            provider_tool_choice: Provider格式的工具选择配置
            **kwargs: 额外参数

        Returns:
            IR格式的工具选择
        """
        pass

    # ==================== 工具调用转换 Tool call conversion ====================

    @staticmethod
    @abstractmethod
    def ir_tool_call_to_p(ir_tool_call: ToolCallPart, **kwargs: Any) -> Any:
        """IR ToolCallPart → Provider Tool Call
        将IR工具调用部分转换为Provider工具调用

        处理工具调用请求：调用ID、工具名称、输入参数等。
        Handles tool call requests: call ID, tool name, input parameters, etc.

        Args:
            ir_tool_call: IR格式的工具调用部分
            **kwargs: 额外参数

        Returns:
            Provider格式的工具调用
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        """Provider Tool Call → IR ToolCallPart
        将Provider工具调用转换为IR工具调用部分

        Args:
            provider_tool_call: Provider格式的工具调用
            **kwargs: 额外参数

        Returns:
            IR格式的工具调用部分
        """
        pass

    # ==================== 工具结果转换 Tool result conversion ====================

    @staticmethod
    @abstractmethod
    def ir_tool_result_to_p(ir_tool_result: ToolResultPart, **kwargs: Any) -> Any:
        """IR ToolResultPart → Provider Tool Result
        将IR工具结果部分转换为Provider工具结果

        处理工具执行结果：结果数据、错误信息、状态等。
        Handles tool execution results: result data, error information, status, etc.

        Args:
            ir_tool_result: IR格式的工具结果部分
            **kwargs: 额外参数

        Returns:
            Provider格式的工具结果
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        """Provider Tool Result → IR ToolResultPart
        将Provider工具结果转换为IR工具结果部分

        Args:
            provider_tool_result: Provider格式的工具结果
            **kwargs: 额外参数

        Returns:
            IR格式的工具结果部分
        """
        pass

    # ==================== 工具配置转换 Tool configuration conversion ====================

    @staticmethod
    @abstractmethod
    def ir_tool_config_to_p(ir_tool_config: ToolCallConfig, **kwargs: Any) -> Any:
        """IR ToolCallConfig → Provider Tool Call Config
        将IR工具调用配置转换为Provider工具调用配置

        处理工具调用的控制参数：并行调用、最大调用数等。
        Handles tool call control parameters: parallel calls, max call count, etc.

        Args:
            ir_tool_config: IR格式的工具调用配置
            **kwargs: 额外参数

        Returns:
            Provider格式的工具调用配置
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        """Provider Tool Call Config → IR ToolCallConfig
        将Provider工具调用配置转换为IR工具调用配置

        Args:
            provider_tool_config: Provider格式的工具调用配置
            **kwargs: 额外参数

        Returns:
            IR格式的工具调用配置
        """
        pass
