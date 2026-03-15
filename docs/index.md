---
title: 首页
author: Oaklight
hide:
  - navigation
---

# LLMIR

[![PyPI version](https://badge.fury.io/py/llmir.svg?icon=si%3Apython)](https://badge.fury.io/py/llmir)
[![GitHub version](https://badge.fury.io/gh/oaklight%2Fllmir.svg?icon=si%3Agithub)](https://badge.fury.io/gh/oaklight%2Fllmir)

**大语言模型中间表示（Large Language Model Intermediate Representation）** — 用于 LLM 提供商 API 之间的统一消息格式转换库。

## 概述

不同的 LLM 提供商（OpenAI、Anthropic、Google）使用互不兼容的 API 格式。LLMIR 通过中枢辐射（Hub-and-Spoke）架构解决了这一问题：每个提供商只需与中央中间表示（IR）进行转换，仅需 N 个转换器，而非 N²。

## 快速开始

```bash
pip install llmir
```

```python
from llmir import OpenAIChatConverter, AnthropicConverter

openai_conv = OpenAIChatConverter()
anthropic_conv = AnthropicConverter()

# OpenAI 格式 → IR → Anthropic 格式
ir_request = openai_conv.request_from_provider(openai_request)
anthropic_request, warnings = anthropic_conv.request_to_provider(ir_request)
```

## 支持的提供商

| 提供商 | API | 转换器 |
|--------|-----|--------|
| OpenAI | Chat Completions | `OpenAIChatConverter` |
| OpenAI | Responses | `OpenAIResponsesConverter` |
| Anthropic | Messages | `AnthropicConverter` |
| Google | GenAI | `GoogleGenAIConverter` |

## 核心特性

- **中枢辐射架构** — 中央 IR 格式消除 N² 转换问题
- **双向转换** — 请求、响应和消息均支持双向转换
- **流式支持** — 通过有状态上下文管理转换流式数据块
- **工具调用** — 跨提供商的统一工具定义和调用处理
- **自动检测** — 从请求结构自动识别提供商格式
- **类型安全** — 所有类型均有完整的 TypedDict 注解
- **零运行时开销** — 纯字典转换，无验证成本

## 架构

```text
OpenAI Chat ──────┐
                   │
OpenAI Responses ──┤
                   ├──── IR（中间表示）
Anthropic ─────────┤
                   │
Google GenAI ──────┘
```

## 文档目录

- **[快速上手](getting-started/installation.md)** — 安装和入门
- **[指南](guide/concepts.md)** — 核心概念、转换器、IR 类型、流式处理
- **[示例](examples/)** — 跨提供商对话、工具调用
- **[API 参考](api/)** — 完整 API 文档
- **[更新日志](changelog.md)** — 版本历史

## 许可证

MIT 许可证
