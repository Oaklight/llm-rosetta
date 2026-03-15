"""
LLM-Rosetta - IR Type Guards

IR类型守卫系统，提供类似isinstance的智能类型检查
IR type guard system providing isinstance-like intelligent type checking
"""

from typing import Any, Dict, Type, TypeVar, Union

from typing_extensions import TypedDict, get_args, get_origin

from .parts import (
    AudioPart,
    CitationPart,
    ContentPart,
    FilePart,
    ImagePart,
    ReasoningPart,
    RefusalPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)

# TypeVar for generic type checking
T = TypeVar("T", bound=TypedDict)


def is_part_type(part: Any, part_class: Type[T]) -> bool:
    """
    通用的类型检查函数，类似isinstance但针对TypedDict优化
    Generic type checking function, similar to isinstance but optimized for TypedDict

    Args:
        part: 要检查的内容部分 Content part to check
        part_class: 目标类型类 Target type class

    Returns:
        是否匹配指定类型 Whether it matches the specified type

    Examples:
        >>> part = {"type": "text", "text": "hello"}
        >>> is_part_type(part, TextPart)  # True
        >>> is_part_type(part, ToolCallPart)  # False

        >>> tool_call = {"type": "tool_call", "tool_call_id": "1", "tool_name": "func", "tool_input": {}}
        >>> is_part_type(tool_call, ToolCallPart)  # True
    """
    if not isinstance(part, dict):
        return False

    # 简化的类型检查：直接使用 type 字段和类型映射
    part_type = part.get("type")
    if not part_type:
        return False

    # 获取期望的类型值
    expected_type = None
    if part_class == TextPart:
        expected_type = "text"
    elif part_class == ImagePart:
        expected_type = "image"
    elif part_class == FilePart:
        expected_type = "file"
    elif part_class == ToolCallPart:
        expected_type = "tool_call"
    elif part_class == ToolResultPart:
        expected_type = "tool_result"
    elif part_class == ReasoningPart:
        expected_type = "reasoning"
    elif part_class == RefusalPart:
        expected_type = "refusal"
    elif part_class == CitationPart:
        expected_type = "citation"
    elif part_class == AudioPart:
        expected_type = "audio"
    else:
        return False

    # 检查类型是否匹配
    if part_type != expected_type:
        return False

    # 基本的必需字段检查
    if part_class == TextPart:
        return "text" in part
    elif part_class == ToolCallPart:
        return all(
            field in part for field in ["tool_call_id", "tool_name", "tool_input"]
        )
    elif part_class == ToolResultPart:
        return all(field in part for field in ["tool_call_id", "result"])
    elif part_class == ImagePart:
        return "image_url" in part or "image_data" in part
    elif part_class == FilePart:
        return "file_url" in part or "file_data" in part
    elif part_class == AudioPart:
        return "audio_id" in part
    elif part_class == RefusalPart:
        return "refusal" in part

    return True


def _extract_literal_value(annotation: Any) -> Union[str, None]:
    """
    从类型注解中提取Literal的值
    Extract Literal value from type annotation
    """
    origin = get_origin(annotation)
    if origin is Union:
        # 处理Required[Literal["text"]]这种情况
        args = get_args(annotation)
        for arg in args:
            if hasattr(arg, "__origin__") and str(arg.__origin__) == "typing.Literal":
                literal_args = get_args(arg)
                if literal_args:
                    return literal_args[0]
    elif (
        hasattr(annotation, "__origin__")
        and str(annotation.__origin__) == "typing.Literal"
    ):
        # 直接的Literal类型
        literal_args = get_args(annotation)
        if literal_args:
            return literal_args[0]

    return None


def _is_required_field(annotation: Any) -> bool:
    """
    检查字段是否是必需的（Required）
    Check if field is required (Required)
    """
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        # 检查是否有Required标记
        for arg in args:
            if hasattr(arg, "__origin__") and "Required" in str(arg.__origin__):
                return True
    return False


# ============================================================================
# 预定义的类型检查函数已移除，请使用通用的 is_part_type 函数
# Predefined type checking functions removed, please use the generic is_part_type function
# ============================================================================

# 示例用法 Example usage:
# is_part_type(part, TextPart)      # 替代 is_text_part(part)
# is_part_type(part, ToolCallPart)  # 替代 is_tool_call_part(part)
# is_part_type(part, ImagePart)     # 替代 is_image_part(part)


# ============================================================================
# 类型映射表 Type mapping table
# ============================================================================

# 类型字符串到类型类的映射
TYPE_CLASS_MAP: Dict[str, Type[ContentPart]] = {
    "text": TextPart,
    "image": ImagePart,
    "file": FilePart,
    "tool_call": ToolCallPart,
    "tool_result": ToolResultPart,
    "reasoning": ReasoningPart,
    "refusal": RefusalPart,
    "citation": CitationPart,
    "audio": AudioPart,
}

# 类型类到检查函数的映射已移除，请直接使用 is_part_type
# Type class to check function mapping removed, please use is_part_type directly


def get_part_type(part: Any) -> Union[Type[ContentPart], None]:
    """
    获取内容部分的具体类型
    Get the specific type of content part

    Args:
        part: 内容部分 Content part

    Returns:
        对应的类型类，如果无法确定则返回None
        Corresponding type class, None if cannot be determined

    Examples:
        >>> part = {"type": "text", "text": "hello"}
        >>> get_part_type(part)  # TextPart

        >>> tool_call = {"type": "tool_call", "tool_call_id": "1", "tool_name": "func", "tool_input": {}}
        >>> get_part_type(tool_call)  # ToolCallPart
    """
    if not isinstance(part, dict):
        return None

    part_type = part.get("type")
    if part_type in TYPE_CLASS_MAP:
        type_class = TYPE_CLASS_MAP[part_type]
        # 验证是否真的匹配这个类型
        if is_part_type(part, type_class):
            return type_class

    return None


def isinstance_part(part: Any, *part_types: Type[ContentPart]) -> bool:
    """
    类似isinstance的函数，支持多个类型检查
    isinstance-like function supporting multiple type checking

    Args:
        part: 要检查的内容部分 Content part to check
        *part_types: 一个或多个类型类 One or more type classes

    Returns:
        是否匹配任一指定类型 Whether it matches any of the specified types

    Examples:
        >>> part = {"type": "text", "text": "hello"}
        >>> isinstance_part(part, TextPart)  # True
        >>> isinstance_part(part, TextPart, ImagePart)  # True
        >>> isinstance_part(part, ToolCallPart)  # False
    """
    for part_type in part_types:
        if is_part_type(part, part_type):
            return True
    return False


# ============================================================================
# 导出的主要函数 Main Exported Functions
# ============================================================================

__all__ = [
    # 核心函数 Core functions
    "is_part_type",
    "isinstance_part",
    "get_part_type",
    # 映射表 Mapping tables
    "TYPE_CLASS_MAP",
]
