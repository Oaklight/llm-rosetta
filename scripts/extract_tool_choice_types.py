"""
提取各个LLM提供商的工具选择类型定义

此脚本用于提取OpenAI、Anthropic和Google GenAI的工具选择相关类型定义，
包括ChatCompletionToolChoiceOptionParam、ToolChoiceParam和ToolConfig。
"""

import inspect
import json
import os
import sys
from typing import Any, get_type_hints

# 添加conda环境路径
sys.path.append(
    os.path.expanduser("~/miniforge3/envs/l_t_c/lib/python3.10/site-packages")
)

import anthropic
from google import genai


def get_class_info(cls: type) -> dict[str, Any]:
    """获取类的详细信息，包括字段、类型注解和文档字符串"""
    result = {
        "name": getattr(cls, "__name__", str(cls)),
        "module": getattr(cls, "__module__", "unknown"),
        "doc": inspect.getdoc(cls),
        "annotations": {},
    }

    # 尝试获取基类
    try:
        if hasattr(cls, "__bases__"):
            result["bases"] = [
                base.__name__ for base in cls.__bases__ if base.__name__ != "object"
            ]
        elif hasattr(cls, "__origin__") and hasattr(cls, "__args__"):
            result["origin"] = str(cls.__origin__)
            result["args"] = [str(arg) for arg in cls.__args__]
    except Exception as e:
        result["bases_error"] = str(e)

    # 获取类型注解
    try:
        type_hints = get_type_hints(cls)
        for name, type_hint in type_hints.items():
            result["annotations"][name] = str(type_hint)
    except Exception as e:
        result["annotations_error"] = str(e)

    # 尝试获取__dict__属性
    try:
        if hasattr(cls, "__dict__"):
            attrs = {}
            for key, value in cls.__dict__.items():
                if not key.startswith("_"):
                    attrs[key] = str(type(value))
            if attrs:
                result["attributes"] = attrs
    except Exception as e:
        result["attributes_error"] = str(e)

    return result


def find_classes_by_name(module: Any, name_patterns: list[str]) -> list[type]:
    """在模块中查找名称匹配指定模式的类"""
    classes = []
    visited = set()

    # 递归查找模块中的所有类
    def search_module(obj, path=""):
        # 避免循环引用
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if inspect.ismodule(obj):
            # 使用list复制字典的键值对，避免在迭代过程中修改字典
            try:
                for key, value in list(obj.__dict__.items()):
                    if not key.startswith("_"):  # 跳过私有属性
                        try:
                            search_module(value, f"{path}.{key}" if path else key)
                        except Exception:
                            # 忽略访问某些属性时可能出现的错误
                            pass
            except Exception:
                pass
        elif inspect.isclass(obj):
            for pattern in name_patterns:
                if pattern in obj.__name__:
                    classes.append(obj)
                    break

    try:
        search_module(module)
    except Exception as e:
        print(f"警告：搜索模块时出错: {e}")

    return classes


def extract_openai_tool_choice_types() -> list[dict[str, Any]]:
    """提取OpenAI的工具选择相关类型"""
    # 只提取特定的类型
    target_classes = [
        "ChatCompletionToolChoiceOptionParam",
        "ChatCompletionToolChoiceParam",
        "ChatCompletionNamedToolChoiceParam",
        "ChatCompletionToolParam",
    ]
    classes = []

    # 直接在openai.types模块中查找
    try:
        import openai.types.chat

        for name in target_classes:
            if hasattr(openai.types.chat, name):
                classes.append(getattr(openai.types.chat, name))
    except (ImportError, AttributeError) as e:
        print(f"警告：无法从openai.types.chat导入类: {e}")

    # 如果没有找到，则在整个openai包中搜索
    if not classes:
        patterns = ["ToolChoice", "Tool"]
        classes = find_classes_by_name(openai, patterns)

    return [get_class_info(cls) for cls in classes]


def extract_anthropic_tool_choice_types() -> list[dict[str, Any]]:
    """提取Anthropic的工具选择相关类型"""
    # 只提取特定的类型
    target_classes = ["ToolChoiceParam", "ToolParam"]
    classes = []

    # 直接在anthropic.types模块中查找
    try:
        for name in target_classes:
            if hasattr(anthropic, name):
                classes.append(getattr(anthropic, name))
    except AttributeError as e:
        print(f"警告：无法从anthropic导入类: {e}")

    # 如果没有找到，则在整个anthropic包中搜索
    if not classes:
        patterns = ["ToolChoice", "Tool"]
        classes = find_classes_by_name(anthropic, patterns)

    return [get_class_info(cls) for cls in classes]


def extract_google_tool_config_types() -> list[dict[str, Any]]:
    """提取Google GenAI的工具配置相关类型"""
    # 只提取特定的类型
    target_classes = ["ToolConfig", "Tool", "FunctionDeclaration"]
    classes = []

    # 直接在genai.types模块中查找
    try:
        for name in target_classes:
            if hasattr(genai.types, name):
                classes.append(getattr(genai.types, name))
    except AttributeError as e:
        print(f"警告：无法从genai.types导入类: {e}")

    # 如果没有找到，则在整个genai包中搜索
    if not classes:
        patterns = ["ToolConfig", "Tool"]
        classes = find_classes_by_name(genai, patterns)

    return [get_class_info(cls) for cls in classes]


def main():
    """主函数"""
    # 提取各提供商的工具选择类型
    print("正在提取OpenAI工具选择类型...")
    openai_types = extract_openai_tool_choice_types()
    print(f"找到 {len(openai_types)} 个OpenAI相关类")

    print("正在提取Anthropic工具选择类型...")
    anthropic_types = extract_anthropic_tool_choice_types()
    print(f"找到 {len(anthropic_types)} 个Anthropic相关类")

    print("正在提取Google工具选择类型...")
    google_types = extract_google_tool_config_types()
    print(f"找到 {len(google_types)} 个Google相关类")

    # 合并结果
    result = {
        "openai": openai_types,
        "anthropic": anthropic_types,
        "google": google_types,
    }

    # 保存为JSON文件
    output_path = "docs/provider_messages_typing_schemas/tool_choice_types_info.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"已将工具选择类型定义保存到 {output_path}")


if __name__ == "__main__":
    main()
