---
title: 命名约定
---

# 命名约定

本页面记录了 LLM-Rosetta 转换器层使用的函数命名约定。遵循这些约定可以确保
在任何位置读到函数名时，转换方向都是明确无歧义的。

## 域

LLM-Rosetta 在两个域之间进行转换：

| 缩写 | 含义 |
|---|---|
| `ir` | 中间表示（Intermediate Representation）— 中枢格式 |
| `p` | 提供方（Provider）— 任意提供方特定格式（OpenAI、Anthropic 等） |

## 规则 1：内部函数使用 `{source}_X_to_{target}`

所有内部/私有的跨域转换函数，名称中同时编码**来源**和**目标**：

```python
# IR → Provider：前缀 ir_，后缀 _to_p
ir_text_to_p(part: IRTextPart) -> dict
ir_tool_definition_to_p(tool: IRTool) -> dict

# Provider → IR：前缀 p_，后缀 _to_ir
p_text_to_ir(part: dict) -> IRTextPart
p_tool_definition_to_ir(tool: dict) -> IRTool
```

介词 `_to_` 始终指向**目标**，前缀始终标识**来源**。内部函数名中不使用
`_from_`。

此规则统一适用于：

- **Ops 文件**（`content_ops.py`、`config_ops.py`、`tool_ops.py`、`message_ops.py`）
- `converter.py` 中的**流式处理 handler**
- 转换器层中任何位置的**构建/转换辅助函数**

### 流式处理 handler

```python
# 提供方事件 → IR 事件 (P→IR)
_handle_p_choice_to_ir(event)
_handle_p_content_block_delta_to_ir(event)

# IR 事件 → 提供方事件 (IR→P)
_handle_ir_text_delta_to_p(event)
_handle_ir_tool_call_start_to_p(event)
```

### 构建辅助函数

```python
# 从提供方 usage 数据构建 IR usage (P→IR)
_build_p_usage_to_ir(p_usage: dict) -> IRUsage

# 从 IR usage 数据构建提供方 usage (IR→P)
_build_ir_usage_to_p(ir_usage: IRUsage) -> dict
```

## 规则 2：公开 API 使用自然英语

转换器类的公开 API 方法使用 `_to_provider` / `_from_provider`，
以提高可读性：

```python
# 公开方法 — 自然英语风格，通过 to/from 表达方向
request_to_provider(ir_request)      # IR → Provider
request_from_provider(p_request)     # Provider → IR
response_to_provider(ir_response)    # IR → Provider
response_from_provider(p_response)   # Provider → IR

stream_response_to_provider(...)     # IR → Provider（流式）
stream_response_from_provider(...)   # Provider → IR（流式）
```

只有这些公开方法允许使用 `_from_`，且必须使用完整单词 `provider`
（不允许缩写为 `_from_p`）。

## 规则 3：域缩写

- 内部函数始终使用 `ir` 和 `p` — 不使用 `provider`、`responses`、
  `anthropic` 等长格式。
- 公开 API 方法使用完整单词 `provider`。

## 总结

| 范围 | 模式 | 示例 |
|---|---|---|
| Ops 文件（公开） | `ir_X_to_p` / `p_X_to_ir` | `ir_text_to_p()`、`p_text_to_ir()` |
| 流式 handler（私有） | `_handle_ir_X_to_p` / `_handle_p_X_to_ir` | `_handle_ir_text_delta_to_p()` |
| 构建/转换辅助函数（私有） | `_{verb}_{source}_X_to_{target}` | `_build_p_usage_to_ir()` |
| 公开转换器 API | `X_to_provider` / `X_from_provider` | `request_to_provider()` |
