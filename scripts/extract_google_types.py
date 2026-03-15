#!/usr/bin/env python3
"""
提取Google GenAI类型定义的脚本。
使用inspect模块动态分析google.genai.types模块，提取关键类型定义。
"""

import inspect
import json
import os
from typing import Any, get_type_hints

# 确保使用conda环境中的Python
# 如果需要，可以取消下面的注释并修改路径
# sys.path.insert(0, '/data/pding/miniforge3/envs/l_t_c/lib/python3.10/site-packages')
import google.genai.types as types


def get_class_info(cls) -> dict[str, Any]:
    """提取类的详细信息，包括字段、类型注解和文档字符串"""
    result = {
        "name": cls.__name__,
        "module": cls.__module__,
        "doc": inspect.getdoc(cls),
        "fields": {},
        "bases": [base.__name__ for base in cls.__bases__ if base.__name__ != "object"],
        "annotations": {},
    }

    # 获取类型注解
    try:
        type_hints = get_type_hints(cls)
        for name, hint in type_hints.items():
            result["annotations"][name] = str(hint)
    except (TypeError, ValueError) as e:
        result["annotations_error"] = str(e)

    # 获取字段和默认值
    if hasattr(cls, "__annotations__"):
        for name, annotation in cls.__annotations__.items():
            result["fields"][name] = {
                "annotation": str(annotation),
                "doc": None,  # 我们将在下面尝试提取字段文档
            }

    # 尝试从文档字符串中提取字段文档
    if result["doc"]:
        lines = result["doc"].split("\n")
        current_field = None
        field_doc = []

        for line in lines:
            if ":" in line and not line.startswith(" "):
                # 可能是字段定义
                parts = line.split(":", 1)
                field_name = parts[0].strip()
                if field_name in result["fields"]:
                    # 如果之前有字段，保存它的文档
                    if current_field and field_doc:
                        result["fields"][current_field]["doc"] = "\n".join(field_doc)

                    # 开始新字段的文档
                    current_field = field_name
                    field_doc = [parts[1].strip()]
                else:
                    # 不是字段，继续添加到当前字段的文档
                    if current_field:
                        field_doc.append(line)
            elif current_field:
                # 继续添加到当前字段的文档
                field_doc.append(line)

        # 保存最后一个字段的文档
        if current_field and field_doc:
            result["fields"][current_field]["doc"] = "\n".join(field_doc)

    return result


def get_union_type_info(type_obj) -> dict[str, Any]:
    """提取Union类型的信息"""
    result = {
        "name": str(type_obj),
        "origin": str(getattr(type_obj, "__origin__", None)),
        "args": [str(arg) for arg in getattr(type_obj, "__args__", [])],
    }
    return result


def extract_types_info() -> dict[str, Any]:
    """提取所有相关类型的信息"""
    result = {"classes": {}, "type_aliases": {}, "enums": {}}

    # 提取我们感兴趣的类
    target_classes = ["Content", "ContentDict", "Part", "PartDict", "File", "FileDict"]

    # 提取我们感兴趣的类型别名
    target_type_aliases = ["ContentListUnionDict", "ContentUnionDict", "PartUnionDict"]

    # 提取类
    for name, obj in inspect.getmembers(types):
        if inspect.isclass(obj) and name in target_classes:
            result["classes"][name] = get_class_info(obj)

    # 提取类型别名
    for name in target_type_aliases:
        if hasattr(types, name):
            type_obj = getattr(types, name)
            result["type_aliases"][name] = get_union_type_info(type_obj)

    return result


def main():
    """主函数"""
    # 提取类型信息
    types_info = extract_types_info()

    # 创建输出目录
    os.makedirs("docs/provider_messages_typing_schemas", exist_ok=True)

    # 保存为JSON文件，方便查看
    with open("docs/provider_messages_typing_schemas/google_types_info.json", "w") as f:
        json.dump(types_info, f, indent=2)

    # 打印摘要
    print("提取的类:")
    for name in types_info["classes"]:
        print(f"  - {name}")

    print("\n提取的类型别名:")
    for name in types_info["type_aliases"]:
        print(f"  - {name}")

    # 生成Markdown文档
    generate_markdown(types_info)


def generate_markdown(types_info: dict[str, Any]):
    """生成Markdown文档"""
    with open("docs/provider_messages_typing_schemas/google.md", "w") as f:
        f.write("# Google GenAI ContentListUnionDict 类型定义\n\n")

        f.write("## 概述\n\n")
        f.write(
            "Google GenAI的消息类型系统基于`ContentListUnionDict`，这是一个非常灵活的Union类型，支持多种不同的内容表示方式。\n\n"
        )

        f.write("## 类型层次结构\n\n")
        f.write("```mermaid\n")
        f.write("graph TD\n")
        f.write("    A[ContentListUnionDict] --> B[ContentUnionDict]\n")
        f.write("    A --> C[list[ContentUnionDict]]\n")
        f.write("    B --> D[Content]\n")
        f.write("    B --> E[ContentDict]\n")
        f.write("    B --> F[PartUnionDict]\n")
        f.write("    B --> G[list[PartUnionDict]]\n")
        f.write("    F --> H[str]\n")
        f.write("    F --> I[PIL_Image]\n")
        f.write("    F --> J[File]\n")
        f.write("    F --> K[FileDict]\n")
        f.write("    F --> L[Part]\n")
        f.write("    F --> M[PartDict]\n")
        f.write("```\n\n")

        # 类型别名
        f.write("## 主要类型别名\n\n")
        for name, info in types_info["type_aliases"].items():
            f.write(f"### {name}\n\n")
            f.write(f"**定义**: `{name} = {info['name']}`\n\n")
            f.write("**组成**:\n")
            for arg in info["args"]:
                f.write(f"- `{arg}`\n")
            f.write("\n")

        # 类定义
        f.write("## 主要类定义\n\n")
        for name, info in types_info["classes"].items():
            f.write(f"### {name}\n\n")
            if info["doc"]:
                f.write(f"{info['doc']}\n\n")

            if info["bases"]:
                f.write(f"**继承**: {', '.join(info['bases'])}\n\n")

            f.write("**字段**:\n\n")
            if info["fields"]:
                f.write("| 字段 | 类型 | 说明 |\n")
                f.write("|------|------|------|\n")
                for field_name, field_info in info["fields"].items():
                    annotation = (
                        field_info["annotation"]
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    doc = field_info["doc"] or ""
                    doc = doc.replace("\n", " ")
                    f.write(f"| `{field_name}` | `{annotation}` | {doc} |\n")
            else:
                f.write("*无字段或无法提取字段信息*\n")
            f.write("\n")

        # 使用示例
        f.write("## 使用示例\n\n")

        f.write("### 简单文本消息\n")
        f.write("```python\n")
        f.write("# 使用字符串\n")
        f.write('content = "Hello, how are you?"\n\n')
        f.write("# 使用Content对象\n")
        f.write(
            'content = types.Content(parts=[types.Part(text="Hello, how are you?")])\n\n'
        )
        f.write("# 使用字典\n")
        f.write('content = {"parts": [{"text": "Hello, how are you?"}]}\n')
        f.write("```\n\n")

        f.write("### 多模态消息\n")
        f.write("```python\n")
        f.write("# 使用Content对象\n")
        f.write("content = types.Content(\n")
        f.write("    parts=[\n")
        f.write('        types.Part(text="What\'s in this image?"),\n')
        f.write("        types.Part(inline_data=types.Blob(\n")
        f.write('            mime_type="image/jpeg",\n')
        f.write("            data=base64.b64encode(image_bytes).decode()\n")
        f.write("        ))\n")
        f.write("    ]\n")
        f.write(")\n\n")
        f.write("# 使用字典\n")
        f.write("content = {\n")
        f.write('    "parts": [\n')
        f.write('        {"text": "What\'s in this image?"},\n')
        f.write('        {"inline_data": {\n')
        f.write('            "mime_type": "image/jpeg",\n')
        f.write('            "data": base64.b64encode(image_bytes).decode()\n')
        f.write("        }}\n")
        f.write("    ]\n")
        f.write("}\n")
        f.write("```\n\n")

        f.write("### 对话历史\n")
        f.write("```python\n")
        f.write("# 使用Content对象列表\n")
        f.write("contents = [\n")
        f.write(
            '    types.Content(role="user", parts=[types.Part(text="Hello, how are you?")]),\n'
        )
        f.write(
            '    types.Content(role="model", parts=[types.Part(text="I\'m doing well, thank you!")]),\n'
        )
        f.write(
            '    types.Content(role="user", parts=[types.Part(text="Tell me about yourself.")])\n'
        )
        f.write("]\n\n")
        f.write("# 使用字典列表\n")
        f.write("contents = [\n")
        f.write('    {"role": "user", "parts": [{"text": "Hello, how are you?"}]},\n')
        f.write(
            '    {"role": "model", "parts": [{"text": "I\'m doing well, thank you!"}]},\n'
        )
        f.write(
            '    {"role": "user", "parts": [{"text": "Tell me about yourself."}]}\n'
        )
        f.write("]\n")
        f.write("```\n\n")

        # 关键特性总结
        f.write("## 关键特性总结\n\n")
        f.write("### 1. 灵活的类型系统\n")
        f.write("- **多种表示方式**: 同一内容可以用字符串、对象或字典表示\n")
        f.write("- **嵌套结构**: 支持复杂的嵌套内容结构\n")
        f.write("- **类型转换**: 自动在不同表示之间转换\n\n")

        f.write("### 2. 角色系统\n")
        f.write("- **用户和模型**: 主要使用`user`和`model`角色\n")
        f.write("- **系统消息**: 可以使用`system`角色设置上下文\n\n")

        f.write("### 3. 多模态支持\n")
        f.write("- **文本**: 通过`text`字段\n")
        f.write("- **图片**: 通过`inline_data`字段\n")
        f.write("- **混合内容**: 一条消息可以包含多种媒体类型\n\n")

        f.write("### 4. 与其他提供商的主要差异\n")
        f.write("| 特性 | Google GenAI | OpenAI | Anthropic |\n")
        f.write("|------|-------------|--------|----------|\n")
        f.write(
            "| 类型灵活性 | 高（多种表示方式） | 中（固定结构） | 中（固定结构） |\n"
        )
        f.write("| 角色数量 | 3种（user, model, system） | 6种 | 2种 |\n")
        f.write("| 多模态支持 | 内联数据 | 内容部分 | 内容块 |\n")
        f.write("| 工具调用 | 函数调用 | 工具调用 | 工具使用块 |\n\n")

        # 注意事项
        f.write("## 注意事项\n\n")
        f.write(
            "1. **类型灵活性**: Google GenAI的类型系统非常灵活，同一内容可以有多种表示方式\n"
        )
        f.write(
            "2. **自动转换**: API会自动在不同表示之间转换，但显式使用正确的类型可以避免潜在问题\n"
        )
        f.write("3. **字典表示**: 在大多数情况下，使用字典表示是最简单的方式\n")
        f.write("4. **对象表示**: 使用对象表示可以获得更好的类型检查和IDE支持\n")
        f.write(
            "5. **字符串限制**: 简单字符串只适用于纯文本内容，不支持角色或多模态\n\n"
        )

        # 版本信息
        f.write("## 版本信息\n\n")
        f.write("- **来源**: Google GenAI Python SDK\n")
        f.write("- **包路径**: `google.genai.types`\n")


if __name__ == "__main__":
    main()
