"""
LLM-Rosetta - Base Tool Operations

Abstract base class for tool conversion operations.

Handles the full tool lifecycle: definition → choice → call → result → config.
All methods are ``@staticmethod @abstractmethod``.

Utility functions (orphan fixing, schema sanitization) live in the
``helpers`` subpackage.
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
    """Abstract base class for tool conversion operations.

    Uniformly handles all stages of the tool lifecycle:
    definition → choice → call → result → config.
    """

    # ==================== Tool Definition ====================

    @staticmethod
    @abstractmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> Any:
        """IR ToolDefinition → Provider Tool Definition.

        Args:
            ir_tool: IR tool definition.
            **kwargs: Extra arguments.

        Returns:
            Provider tool definition.
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_definition_to_ir(
        provider_tool: Any, **kwargs: Any
    ) -> ToolDefinition | list[ToolDefinition] | None:
        """Provider Tool Definition → IR ToolDefinition.

        Args:
            provider_tool: Provider tool definition.
            **kwargs: Extra arguments.

        Returns:
            IR tool definition(s), or None if the entry cannot be converted.
        """
        pass

    # ==================== Tool Choice ====================

    @staticmethod
    @abstractmethod
    def ir_tool_choice_to_p(ir_tool_choice: ToolChoice, **kwargs: Any) -> Any:
        """IR ToolChoice → Provider Tool Choice Config.

        Args:
            ir_tool_choice: IR tool choice.
            **kwargs: Extra arguments.

        Returns:
            Provider tool choice config.
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        """Provider Tool Choice Config → IR ToolChoice.

        Args:
            provider_tool_choice: Provider tool choice config.
            **kwargs: Extra arguments.

        Returns:
            IR tool choice.
        """
        pass

    # ==================== Tool Call ====================

    @staticmethod
    @abstractmethod
    def ir_tool_call_to_p(ir_tool_call: ToolCallPart, **kwargs: Any) -> Any:
        """IR ToolCallPart → Provider Tool Call.

        Args:
            ir_tool_call: IR tool call part.
            **kwargs: Extra arguments.

        Returns:
            Provider tool call.
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        """Provider Tool Call → IR ToolCallPart.

        Args:
            provider_tool_call: Provider tool call.
            **kwargs: Extra arguments.

        Returns:
            IR tool call part.
        """
        pass

    # ==================== Tool Result ====================

    @staticmethod
    @abstractmethod
    def ir_tool_result_to_p(ir_tool_result: ToolResultPart, **kwargs: Any) -> Any:
        """IR ToolResultPart → Provider Tool Result.

        Args:
            ir_tool_result: IR tool result part.
            **kwargs: Extra arguments.

        Returns:
            Provider tool result.
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        """Provider Tool Result → IR ToolResultPart.

        Args:
            provider_tool_result: Provider tool result.
            **kwargs: Extra arguments.

        Returns:
            IR tool result part.
        """
        pass

    # ==================== Tool Config ====================

    @staticmethod
    @abstractmethod
    def ir_tool_config_to_p(ir_tool_config: ToolCallConfig, **kwargs: Any) -> Any:
        """IR ToolCallConfig → Provider Tool Call Config.

        Args:
            ir_tool_config: IR tool call config.
            **kwargs: Extra arguments.

        Returns:
            Provider tool call config.
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        """Provider Tool Call Config → IR ToolCallConfig.

        Args:
            provider_tool_config: Provider tool call config.
            **kwargs: Extra arguments.

        Returns:
            IR tool call config.
        """
        pass
