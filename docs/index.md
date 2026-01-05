---
title: 首页
summary: LLMIR - 大语言模型中间表示
description: LLMIR 是一个 Python 库，提供统一的中间表示来处理不同 LLM 提供商的消息格式，支持 OpenAI、Anthropic、Google 等主流提供商。
keywords: python, llm, 大语言模型, 中间表示, openai, anthropic, google
author: LLMIR Team
hide:
  - navigation
---

# LLMIR

[![PyPI version](https://badge.fury.io/py/llmir.svg?icon=si%3Apython)](https://badge.fury.io/py/llmir)
[![GitHub version](https://badge.fury.io/gh/Oaklight%2Fllmir.svg?icon=si%3Agithub)](https://badge.fury.io/gh/Oaklight%2Fllmir)

**大语言模型中间表示 (Large Language Model Intermediate Representation)** - 为不同 LLM 提供商提供统一的消息格式转换。

## 🚀 快速开始

```bash
pip install llmir
```

```python
from llmir import auto_detect, convert_to_openai

# 自动检测并转换消息格式
messages = [{"role": "user", "content": "Hello, world!"}]
provider = auto_detect(messages)
openai_format = convert_to_openai(messages, provider)
```

## 🎯 核心特性

- **🔄 统一转换**: 支持 OpenAI、Anthropic、Google 等主流 LLM 提供商
- **🤖 自动检测**: 智能识别消息格式来源
- **⚡ 高性能**: 优化的转换算法，最小化性能开销
- **🛡️ 类型安全**: 完整的 TypeScript 风格类型注解
- **📚 易于使用**: 简洁的 API 设计，快速上手
- **🔧 可扩展**: 支持自定义转换器和格式

## 🛠️ 支持的提供商

LLMIR 目前支持以下 LLM 提供商的消息格式转换：

- **OpenAI** - Chat Completions 和 Responses API
- **Anthropic** - Claude 消息格式
- **Google** - Gemini/PaLM 消息格式
- **通用格式** - 标准化的中间表示

## 📖 文档导航

- **[快速开始](getting-started/)** - 安装和基本使用
- **[用户指南](guide/)** - 详细的使用指南和最佳实践
- **[API 参考](api/)** - 完整的 API 文档
- **[示例](examples/)** - 实际使用案例和代码示例

## 🌟 为什么选择 LLMIR？

- **🎯 专注**: 专门为 LLM 消息格式转换而设计
- **🔧 灵活**: 支持多种转换模式和自定义配置
- **📈 可靠**: 经过充分测试，适用于生产环境
- **🌐 开源**: MIT 许可证，社区驱动开发
- **📚 文档完善**: 详细的文档和示例

## 🤝 参与贡献

- **[GitHub 仓库](https://github.com/Oaklight/llmir)** - 源代码和问题反馈
- **[English Documentation](../en/)** - 英文文档
- **[开发指南](development/)** - 贡献代码和开发指南

---

_LLMIR: 让 LLM 消息格式转换变得简单可靠。_