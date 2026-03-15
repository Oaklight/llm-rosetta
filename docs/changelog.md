---
title: 更新日志
---

# 更新日志

## 未发布

### 新特性

- 流式支持：通过 `StreamContext` 实现有状态的数据块处理
- `stream_response_from_provider()` 和 `stream_response_to_provider()` 方法
- `accumulate_stream_to_assistant_message()` 辅助函数
- 跨提供商流式示例（SDK 和 REST 版本）

### 改进

- 转换器架构重构为基于组合的 Ops 模式
- 为 Google GenAI REST 流式/响应字段添加 camelCase 回退

## [0.0.1] - 2024-12-01

初始发布。

### 新增

- 基于中枢辐射的转换器架构，以中央 IR 格式为核心
- `OpenAIChatConverter`：OpenAI Chat Completions API 转换器
- `OpenAIResponsesConverter`：OpenAI Responses API 转换器
- `AnthropicConverter`：Anthropic Messages API 转换器
- `GoogleGenAIConverter`：Google GenAI API 转换器
- 双向转换：`request_to/from_provider`、`response_to/from_provider`、`messages_to/from_provider`
- IR 类型系统：消息、内容部分、工具、配置、请求/响应
- 通过 `detect_provider()` 自动检测提供商格式
- `convert()` 便捷函数实现一步格式转换
- 辅助函数：`extract_text_content`、`extract_tool_calls`、`create_tool_result_message`
- 完整的 TypedDict 注解确保类型安全
- 24 个跨提供商示例脚本（12 个基于 SDK，12 个基于 REST）
