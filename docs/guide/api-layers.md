---
title: API 分层与导入指南
---

# API 分层与导入指南

LLM-Rosetta 将公共 API 分为三个稳定性层级。本指南帮助你选择正确的导入路径，并了解每个层级的稳定性承诺。

## 稳定性层级

### 稳定 API（Stable）

可以在次版本间安全依赖的符号，推荐大多数用户使用。

| 类别 | 符号 | 导入路径 |
|------|------|----------|
| 转换器 | `OpenAIChatConverter`、`AnthropicConverter`、`GoogleGenAIConverter`、`GoogleConverter`、`OpenAIResponsesConverter`、`BaseConverter` | `llm_rosetta` |
| 上下文 | `ConversionContext`、`StreamContext` | `llm_rosetta` |
| 便捷接口 | `detect_provider`、`get_converter_for_provider`、`convert`、`ProviderType` | `llm_rosetta` |
| 核心 IR 类型 | `Message`、`IRRequest`、`IRResponse`、`ContentPart`、`TextPart`、`ToolCallPart`、流式事件等 | `llm_rosetta.types.ir` |

```python
# 推荐：稳定导入
from llm_rosetta import OpenAIChatConverter, AnthropicConverter
from llm_rosetta import ConversionContext
from llm_rosetta.types.ir import Message, IRRequest, IRResponse, TextPart
```

### 高级 API（Advanced）

面向需要精细控制的高级用户。可能在次版本中变更，但会在更新日志中说明。

| 类别 | 示例 | 导入路径 |
|------|------|----------|
| 类型守卫 | `is_text_part`、`is_tool_call_part`、`is_stream_start_event`、`is_message` 等 | `llm_rosetta.types.ir.type_guards`、`llm_rosetta.types.ir.messages` |
| 消息工厂函数 | `create_system_message`、`create_user_message`、`create_assistant_message`、`create_tool_message` | `llm_rosetta.types.ir.messages` |
| 内容辅助函数 | `extract_text_content`、`extract_all_text`、`extract_tool_calls`、`create_tool_result_message` | `llm_rosetta.types.ir.helpers` |
| 扩展项守卫 | `is_extension_item` | `llm_rosetta.types.ir.extensions` |

```python
# 高级：从子模块导入
from llm_rosetta.types.ir.type_guards import is_text_part, is_tool_call_part
from llm_rosetta.types.ir.messages import create_user_message
from llm_rosetta.types.ir.helpers import extract_text_content
```

!!! note
    这些符号也可以从 `llm_rosetta.types.ir` 直接导入，但它们不在稳定的 `__all__` 暴露面中。

### 内部 API（Internal）

库内部使用的实现细节。使用需自行承担风险——可能在不通知的情况下变更。

| 类别 | 示例 | 导入路径 |
|------|------|----------|
| 类型映射 | `TYPE_CLASS_MAP`、`get_part_type`、`isinstance_part` | `llm_rosetta.types.ir.type_guards` |
| 验证工具 | `ValidationError`、`validate_ir_request`、`validate_ir_response` | `llm_rosetta.types.ir.validation` |
| Ops 基类 | `BaseContentOps`、`BaseToolOps`、`BaseMessageOps`、`BaseConfigOps` | `llm_rosetta.converters.base.content` 等 |
| Schema 工具 | `sanitize_schema` | `llm_rosetta.converters.base.tools` |
| 内容转换辅助 | `convert_content_blocks_to_ir`、`convert_ir_content_blocks_to_p` | `llm_rosetta.converters.base.tool_content` |

```python
# 内部：使用深层子模块导入
from llm_rosetta.types.ir.validation import validate_ir_request
from llm_rosetta.converters.base.tools import sanitize_schema
```

## 快速参考

```text
llm_rosetta                          # 稳定：转换器、上下文、便捷接口
llm_rosetta.types.ir                 # 稳定：核心 IR 数据类型
llm_rosetta.types.ir.type_guards     # 高级：is_*_part、is_*_event 守卫
llm_rosetta.types.ir.messages        # 高级：create_*、is_*_message
llm_rosetta.types.ir.helpers         # 高级：extract_*、create_tool_result_message
llm_rosetta.types.ir.validation      # 内部：验证工具
llm_rosetta.converters.base          # 稳定：BaseConverter、上下文类
llm_rosetta.converters.base.content  # 内部：BaseContentOps
llm_rosetta.converters.base.tools    # 内部：BaseToolOps、sanitize_schema
```

## CI 强制检查

导出面由 `tests/test_public_api.py` 强制执行。如果向任何 `__all__` 添加新符号，测试将失败，除非同步更新预期导出列表。这可以防止公共 API 的意外膨胀。
