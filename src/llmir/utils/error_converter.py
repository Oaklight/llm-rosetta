"""
LLMIR - 错误转换工具类
Error conversion utilities
"""

import json
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

from ..exceptions import (
    WarningInfo,
)
from ..types.ir import ToolResultPart


class ErrorConverter:
    """错误转换工具类
    Error conversion utility class

    负责处理工具执行结果中的错误信息，统一错误格式，
    并提供错误信息在不同provider之间的转换。
    """

    @staticmethod
    def normalize_tool_error(
        result: Any,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
    ) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
        """
        标准化工具执行错误
        Normalize tool execution error

        Args:
            result: 工具执行结果
            tool_name: 工具名称
            tool_call_id: 工具调用ID

        Returns:
            Tuple[错误消息, 是否为错误, 错误元数据]
        """
        if isinstance(result, dict):
            # 检查常见的错误字段
            if "error" in result:
                error_msg = str(result["error"])
                metadata = {
                    "error_type": result.get("error_type", "unknown"),
                    "error_code": result.get("error_code"),
                    "details": result.get("details"),
                }
                return error_msg, True, metadata

            elif "exception" in result:
                error_msg = str(result["exception"])
                metadata = {
                    "error_type": result.get("exception_type", "exception"),
                    "traceback": result.get("traceback"),
                }
                return error_msg, True, metadata

            elif result.get("success") is False:
                error_msg = result.get("message", "Tool execution failed")
                metadata = {
                    "error_type": "execution_failure",
                    "status": result.get("status"),
                }
                return error_msg, True, metadata

            elif "status" in result and result["status"] in [
                "error",
                "failed",
                "failure",
            ]:
                error_msg = result.get(
                    "message", f"Tool execution status: {result['status']}"
                )
                metadata = {
                    "error_type": "status_error",
                    "status": result["status"],
                }
                return error_msg, True, metadata

        elif isinstance(result, Exception):
            error_msg = str(result)
            metadata = {
                "error_type": result.__class__.__name__,
                "traceback": traceback.format_exception(
                    type(result), result, result.__traceback__
                ),
            }
            return error_msg, True, metadata

        elif isinstance(result, str):
            # 检查字符串是否包含错误关键词
            error_keywords = [
                "error:",
                "exception:",
                "failed:",
                "failure:",
                "traceback",
            ]
            lower_result = result.lower()

            if any(keyword in lower_result for keyword in error_keywords):
                return result, True, {"error_type": "string_error"}

        # 不是错误，返回原始结果
        return str(result), False, None

    @staticmethod
    def create_tool_result_part(
        tool_call_id: str,
        result: Any,
        tool_name: Optional[str] = None,
        auto_detect_error: bool = True,
    ) -> ToolResultPart:
        """
        创建工具结果部分
        Create tool result part

        Args:
            tool_call_id: 工具调用ID
            result: 工具执行结果
            tool_name: 工具名称
            auto_detect_error: 是否自动检测错误

        Returns:
            标准化的工具结果部分
        """
        if auto_detect_error:
            normalized_result, is_error, error_metadata = (
                ErrorConverter.normalize_tool_error(result, tool_name, tool_call_id)
            )
        else:
            normalized_result = str(result)
            is_error = False
            error_metadata = None

        tool_result = ToolResultPart(
            type="tool_result",
            tool_call_id=tool_call_id,
            result=normalized_result,
            is_error=is_error,
        )

        # 添加错误元数据到provider_metadata中
        if error_metadata:
            tool_result["provider_metadata"] = {"error_info": error_metadata}

        return tool_result

    @staticmethod
    def convert_provider_error_to_ir(
        provider_error: Dict[str, Any],
        provider_name: str,
    ) -> ToolResultPart:
        """
        将provider特定的错误格式转换为IR格式
        Convert provider-specific error format to IR format

        Args:
            provider_error: Provider特定的错误数据
            provider_name: Provider名称 (anthropic, openai, google)

        Returns:
            IR格式的工具结果部分
        """
        if provider_name.lower() == "anthropic":
            return ErrorConverter._convert_anthropic_error(provider_error)
        elif provider_name.lower() in ["openai", "openai_chat", "openai_responses"]:
            return ErrorConverter._convert_openai_error(provider_error)
        elif provider_name.lower() == "google":
            return ErrorConverter._convert_google_error(provider_error)
        else:
            # 通用转换
            return ErrorConverter._convert_generic_error(provider_error, provider_name)

    @staticmethod
    def _convert_anthropic_error(error_data: Dict[str, Any]) -> ToolResultPart:
        """转换Anthropic错误格式"""
        tool_call_id = error_data.get("tool_use_id", "unknown")
        error_content = error_data.get("content", "Unknown error")
        is_error = error_data.get("is_error", True)

        return ToolResultPart(
            type="tool_result",
            tool_call_id=tool_call_id,
            result=str(error_content),
            is_error=is_error,
        )

    @staticmethod
    def _convert_openai_error(error_data: Dict[str, Any]) -> ToolResultPart:
        """转换OpenAI错误格式"""
        tool_call_id = error_data.get("tool_call_id", "unknown")
        error_content = error_data.get("content", "Unknown error")

        # OpenAI通常通过content内容判断是否为错误
        is_error = True
        if isinstance(error_content, str):
            is_error = any(
                keyword in error_content.lower()
                for keyword in ["error", "failed", "exception"]
            )

        return ToolResultPart(
            type="tool_result",
            tool_call_id=tool_call_id,
            result=str(error_content),
            is_error=is_error,
        )

    @staticmethod
    def _convert_google_error(error_data: Dict[str, Any]) -> ToolResultPart:
        """转换Google错误格式"""
        # Google使用function_response格式
        func_response = error_data.get("function_response", {})
        tool_call_id = func_response.get("name", "unknown")
        response_data = func_response.get("response", {})

        # Google明确区分error和output
        if "error" in response_data:
            result = str(response_data["error"])
            is_error = True
        else:
            result = str(response_data.get("output", "Unknown error"))
            is_error = False

        return ToolResultPart(
            type="tool_result",
            tool_call_id=tool_call_id,
            result=result,
            is_error=is_error,
        )

    @staticmethod
    def _convert_generic_error(
        error_data: Dict[str, Any], provider_name: str
    ) -> ToolResultPart:
        """转换通用错误格式"""
        tool_call_id = error_data.get("id", error_data.get("tool_call_id", "unknown"))
        result = error_data.get("result", error_data.get("content", "Unknown error"))
        is_error = error_data.get("is_error", True)

        return ToolResultPart(
            type="tool_result",
            tool_call_id=tool_call_id,
            result=str(result),
            is_error=is_error,
        )

    @staticmethod
    def convert_ir_error_to_provider(
        tool_result: ToolResultPart,
        provider_name: str,
    ) -> Dict[str, Any]:
        """
        将IR格式的错误转换为provider特定格式
        Convert IR format error to provider-specific format

        Args:
            tool_result: IR格式的工具结果
            provider_name: 目标provider名称

        Returns:
            Provider特定格式的错误数据
        """
        if provider_name.lower() == "anthropic":
            return ErrorConverter._convert_ir_to_anthropic_error(tool_result)
        elif provider_name.lower() in ["openai", "openai_chat", "openai_responses"]:
            return ErrorConverter._convert_ir_to_openai_error(tool_result)
        elif provider_name.lower() == "google":
            return ErrorConverter._convert_ir_to_google_error(tool_result)
        else:
            return ErrorConverter._convert_ir_to_generic_error(
                tool_result, provider_name
            )

    @staticmethod
    def _convert_ir_to_anthropic_error(tool_result: ToolResultPart) -> Dict[str, Any]:
        """将IR错误转换为Anthropic格式"""
        return {
            "type": "tool_result",
            "tool_use_id": tool_result["tool_call_id"],
            "content": tool_result["result"],
            "is_error": tool_result.get("is_error", False),
        }

    @staticmethod
    def _convert_ir_to_openai_error(tool_result: ToolResultPart) -> Dict[str, Any]:
        """将IR错误转换为OpenAI格式"""
        content = tool_result["result"]

        # 如果是错误，可以在内容前添加错误标识
        if tool_result.get("is_error", False):
            if not str(content).lower().startswith("error"):
                content = f"Error: {content}"

        return {
            "role": "tool",
            "tool_call_id": tool_result["tool_call_id"],
            "content": str(content),
        }

    @staticmethod
    def _convert_ir_to_google_error(tool_result: ToolResultPart) -> Dict[str, Any]:
        """将IR错误转换为Google格式"""
        is_error = tool_result.get("is_error", False)
        result_content = tool_result["result"]

        if is_error:
            response_data = {"error": str(result_content)}
        else:
            response_data = {"output": str(result_content)}

        return {
            "function_response": {
                "name": tool_result[
                    "tool_call_id"
                ],  # Google需要函数名，这里使用tool_call_id
                "response": response_data,
            }
        }

    @staticmethod
    def _convert_ir_to_generic_error(
        tool_result: ToolResultPart, provider_name: str
    ) -> Dict[str, Any]:
        """将IR错误转换为通用格式"""
        return {
            "tool_call_id": tool_result["tool_call_id"],
            "result": tool_result["result"],
            "is_error": tool_result.get("is_error", False),
            "provider": provider_name,
        }

    @staticmethod
    def create_error_summary(
        errors: List[Union[Exception, ToolResultPart]],
        warnings: List[Union[str, WarningInfo]],
    ) -> Dict[str, Any]:
        """
        创建错误和警告的汇总信息
        Create summary of errors and warnings

        Args:
            errors: 错误列表
            warnings: 警告列表

        Returns:
            错误汇总信息
        """
        error_summary = {
            "total_errors": len(errors),
            "total_warnings": len(warnings),
            "errors": [],
            "warnings": [],
            "severity_counts": {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0,
            },
            "category_counts": {},
        }

        # 处理错误
        for error in errors:
            if isinstance(error, Exception):
                if hasattr(error, "to_dict"):
                    error_dict = error.to_dict()
                else:
                    error_dict = {
                        "error_type": error.__class__.__name__,
                        "message": str(error),
                        "category": "unknown",
                        "severity": "medium",
                    }
            elif isinstance(error, dict) and error.get("type") == "tool_result":
                # 检查是否是ToolResultPart字典
                error_dict = {
                    "error_type": "ToolExecutionError",
                    "message": error.get("result", "Unknown tool error"),
                    "category": "tool_execution",
                    "severity": "medium",
                    "context": {
                        "tool_call_id": error.get("tool_call_id", "unknown"),
                        "is_error": error.get("is_error", False),
                    },
                }
            else:
                error_dict = {
                    "error_type": "UnknownError",
                    "message": str(error),
                    "category": "unknown",
                    "severity": "medium",
                }

            error_summary["errors"].append(error_dict)

            # 统计严重程度
            severity = error_dict.get("severity", "medium")
            error_summary["severity_counts"][severity] += 1

            # 统计分类
            category = error_dict.get("category", "unknown")
            error_summary["category_counts"][category] = (
                error_summary["category_counts"].get(category, 0) + 1
            )

        # 处理警告
        for warning in warnings:
            if isinstance(warning, WarningInfo):
                warning_dict = warning.to_dict()
            elif isinstance(warning, str):
                warning_dict = {
                    "message": warning,
                    "category": "unknown",
                    "severity": "low",
                }
            else:
                warning_dict = {
                    "message": str(warning),
                    "category": "unknown",
                    "severity": "low",
                }

            error_summary["warnings"].append(warning_dict)

            # 统计严重程度
            severity = warning_dict.get("severity", "low")
            error_summary["severity_counts"][severity] += 1

            # 统计分类
            category = warning_dict.get("category", "unknown")
            error_summary["category_counts"][category] = (
                error_summary["category_counts"].get(category, 0) + 1
            )

        return error_summary

    @staticmethod
    def format_error_for_display(
        error: Union[Exception, Dict[str, Any]],
        include_context: bool = True,
        include_suggestions: bool = True,
    ) -> str:
        """
        格式化错误信息用于显示
        Format error information for display

        Args:
            error: 错误对象或字典
            include_context: 是否包含上下文信息
            include_suggestions: 是否包含建议

        Returns:
            格式化的错误信息字符串
        """
        if isinstance(error, Exception):
            if hasattr(error, "to_dict"):
                error_dict = error.to_dict()
            else:
                error_dict = {
                    "error_type": error.__class__.__name__,
                    "message": str(error),
                }
        else:
            error_dict = error

        parts = []

        # 错误类型和消息
        error_type = error_dict.get("error_type", "Error")
        message = error_dict.get("message", "Unknown error")
        parts.append(f"[{error_type}] {message}")

        # 上下文信息
        if include_context and "context" in error_dict:
            context = error_dict["context"]
            if context:
                context_str = ", ".join(f"{k}={v}" for k, v in context.items())
                parts.append(f"Context: {context_str}")

        # 建议
        if include_suggestions and "suggestions" in error_dict:
            suggestions = error_dict["suggestions"]
            if suggestions:
                parts.append(f"Suggestions: {'; '.join(suggestions)}")

        return " | ".join(parts)
