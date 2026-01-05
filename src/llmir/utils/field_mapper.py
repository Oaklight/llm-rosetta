"""
字段映射工具
Field mapping utility

处理不同provider之间的字段名差异，提供统一的字段访问接口
Handles field name differences between providers, provides a unified field access interface
"""

from typing import Any, Dict, List, Optional, Union


class FieldMapper:
    """字段映射工具，处理不同provider的字段名差异
    Field mapping utility, handles field name differences between providers
    """

    # 定义常见的字段映射规则 Define common field mapping rules
    TOOL_NAME_FIELDS = ["tool_name", "name", "function.name"]
    TOOL_INPUT_FIELDS = ["tool_input", "arguments", "args", "input", "parameters"]
    TOOL_ID_FIELDS = ["tool_call_id", "id", "call_id"]
    RESULT_FIELDS = ["result", "content", "output", "response"]
    IMAGE_URL_FIELDS = ["image_url", "url", "source.url"]
    IMAGE_DATA_FIELDS = ["image_data", "data", "inline_data", "source.data"]

    @staticmethod
    def get_field(
        data: Dict[str, Any], field_names: Union[List[str], str], default: Any = None
    ) -> Any:
        """
        从多个可能的字段名中获取值

        Args:
            data: 要查询的字典
            field_names: 字段名或字段名列表，支持点号表示嵌套
            default: 默认值

        Returns:
            找到的值或默认值

        Examples:
            >>> data = {"name": "test", "function": {"name": "func"}}
            >>> FieldMapper.get_field(data, ["tool_name", "name"])
            'test'
            >>> FieldMapper.get_field(data, "function.name")
            'func'
        """
        if isinstance(field_names, str):
            field_names = [field_names]

        for field_name in field_names:
            if "." in field_name:
                # 处理嵌套字段
                value = FieldMapper._get_nested_field(data, field_name)
                if value is not None:
                    return value
            else:
                value = data.get(field_name)
                if value is not None:
                    return value
        return default

    @staticmethod
    def _get_nested_field(data: Dict[str, Any], field_path: str) -> Any:
        """获取嵌套字段的值
        Get value of nested field
        """
        parts = field_path.split(".")
        value = data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    @classmethod
    def get_tool_name(cls, data: Dict[str, Any], default: str = "") -> str:
        """获取工具名称
        Get tool name
        """
        return cls.get_field(data, cls.TOOL_NAME_FIELDS, default)

    @classmethod
    def get_tool_input(
        cls, data: Dict[str, Any], default: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """获取工具输入参数
        Get tool input parameters
        """
        result = cls.get_field(data, cls.TOOL_INPUT_FIELDS, default)
        return result if result is not None else {}

    @classmethod
    def get_tool_id(cls, data: Dict[str, Any], default: str = "") -> str:
        """获取工具调用ID
        Get tool call ID
        """
        return cls.get_field(data, cls.TOOL_ID_FIELDS, default)

    @classmethod
    def get_result_content(cls, data: Dict[str, Any], default: Any = None) -> Any:
        """获取结果内容
        Get result content
        """
        return cls.get_field(data, cls.RESULT_FIELDS, default)

    @classmethod
    def get_image_url(
        cls, data: Dict[str, Any], default: Optional[str] = None
    ) -> Optional[str]:
        """获取图像URL
        Get image URL
        """
        return cls.get_field(data, cls.IMAGE_URL_FIELDS, default)

    @classmethod
    def get_image_data(
        cls, data: Dict[str, Any], default: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """获取图像数据
        Get image data
        """
        return cls.get_field(data, cls.IMAGE_DATA_FIELDS, default)

    @staticmethod
    def normalize_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将各种格式的工具调用标准化为IR格式

        Args:
            data: 原始工具调用数据

        Returns:
            标准化的IR格式工具调用
        """
        return {
            "type": "tool_call",
            "tool_call_id": FieldMapper.get_tool_id(data),
            "tool_name": FieldMapper.get_tool_name(data),
            "tool_input": FieldMapper.get_tool_input(data),
            "tool_type": data.get("tool_type", "function"),
        }

    @staticmethod
    def normalize_tool_result(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将各种格式的工具结果标准化为IR格式

        Args:
            data: 原始工具结果数据

        Returns:
            标准化的IR格式工具结果
        """
        return {
            "type": "tool_result",
            "tool_call_id": FieldMapper.get_tool_id(data),
            "result": FieldMapper.get_result_content(data),
            "is_error": data.get("is_error", False),
        }
