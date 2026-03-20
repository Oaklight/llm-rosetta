# LLM-Rosetta

[![PyPI version](https://img.shields.io/pypi/v/llm-rosetta?color=green)](https://pypi.org/project/llm-rosetta/)
[![CI](https://github.com/Oaklight/llm-rosetta/actions/workflows/ci.yml/badge.svg)](https://github.com/Oaklight/llm-rosetta/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Oaklight/llm-rosetta)

[English Version](README_en.md) | [中文版](README_zh.md)

**Large Language Model Rosetta Stone** — 一个通过中心化中间表示（IR）的轴辐式架构，在不同 LLM 提供商 API 格式之间进行转换的 Python 库。

## 完整文档

完整文档请访问：

- **English**: [https://llm-rosetta.readthedocs.io/en/latest/](https://llm-rosetta.readthedocs.io/en/latest/)
- **中文**: [https://llm-rosetta.readthedocs.io/zh-cn/latest/](https://llm-rosetta.readthedocs.io/zh-cn/latest/)

## 解决的问题

当构建需要对接多个 LLM 提供商的应用时，你会面临 N² 转换问题——每对提供商之间都需要专门的转换逻辑。LLM-Rosetta 通过轴辐式（hub-and-spoke）方案解决这一问题：每个提供商只需要一个与共享 IR 格式之间的转换器。

```
Provider A ──→ IR ──→ Provider B
Provider C ──→ IR ──→ Provider D
         ... 以此类推
```

## 支持的提供商

| 提供商 | API 标准 | 请求 | 响应 | 流式 |
|--------|---------|:----:|:----:|:----:|
| OpenAI | Chat Completions | ✅ | ✅ | ✅ |
| OpenAI | Responses API | ✅ | ✅ | ✅ |
| Anthropic | Messages API | ✅ | ✅ | ✅ |
| Google | GenAI API | ✅ | ✅ | ✅ |

### Ollama 及其他 OpenAI 兼容服务

LLM-Rosetta 可直接与任何提供 OpenAI 兼容接口的服务配合使用。[Ollama](https://ollama.com/)（v0.13+）是一个典型示例——它支持 LLM-Rosetta 所转换的四种 API 格式中的三种：

| Ollama 端点 | LLM-Rosetta 转换器 | 起始版本 |
|---|---|---|
| `/v1/chat/completions` | `openai_chat` | 早期版本 |
| `/v1/responses` | `openai_responses` | v0.13.3 |
| `/v1/messages` | `anthropic` | v0.14.0 |

其他兼容服务包括 [HuggingFace TGI](https://github.com/huggingface/text-generation-inference)、[vLLM](https://github.com/vllm-project/vllm) 和 [LM Studio](https://lmstudio.ai/)。

## 功能特性

- 统一的 IR 格式，支持消息、工具调用和内容块
- 双向转换：请求转为提供商格式，响应从提供商格式转出
- 流式传输支持，带类型化的流事件
- 自动检测请求/响应对象的提供商类型
- 支持文本、图片、工具调用和工具结果
- 零必需依赖（仅需 `typing_extensions`）；提供商 SDK 为可选依赖

## 安装

### 基本安装

安装核心包（需要 **Python >= 3.8**）：

```bash
pip install llm-rosetta
```

### 安装提供商 SDK

```bash
# 单个提供商
pip install llm-rosetta[openai]
pip install llm-rosetta[anthropic]
pip install llm-rosetta[google]

# 所有提供商
pip install llm-rosetta[openai,anthropic,google]
```

### 可选依赖

| 附加项 | 包 | 说明 |
|--------|---|------|
| `openai` | `openai` | OpenAI Chat Completions 和 Responses API |
| `anthropic` | `anthropic` | Anthropic Messages API |
| `google` | `google-genai` | Google GenAI API |

## 快速开始

```python
from llm_rosetta import OpenAIChatConverter, AnthropicConverter

# 创建转换器
openai_conv = OpenAIChatConverter()
anthropic_conv = AnthropicConverter()

# 将 OpenAI 响应转换为 IR，再转换为 Anthropic 格式
ir_messages = openai_conv.response_from_provider(openai_response)
anthropic_request = anthropic_conv.request_to_provider(ir_messages)
```

### 自动检测

```python
from llm_rosetta import convert, detect_provider

# 自动检测提供商并转换
provider = detect_provider(some_response)
ir_messages = convert(some_response, direction="from_provider")
```

### 跨提供商对话

```python
from llm_rosetta import OpenAIChatConverter, GoogleGenAIConverter
from llm_rosetta.types.ir import Message, ContentPart

# 共享的 IR 消息历史
ir_messages = []

# 第 1 轮：向 OpenAI 提问
ir_messages.append(Message(role="user", content=[ContentPart(type="text", text="你好！")]))
openai_request = openai_conv.request_to_provider({"messages": ir_messages})
openai_response = openai_client.chat.completions.create(**openai_request)
ir_messages.extend(openai_conv.response_from_provider(openai_response))

# 第 2 轮：继续使用 Google —— 完整上下文保持
google_request = google_conv.request_to_provider({"messages": ir_messages})
```

## 相关项目

LLM-Rosetta 是 ToolRegistry 生态系统的一部分：

- **[ToolRegistry](https://github.com/Oaklight/ToolRegistry)** — LLM 函数调用的工具注册与管理
- **[ToolRegistry Hub](https://github.com/Oaklight/toolregistry-hub)** — 开箱即用的工具集合
- **[ToolRegistry Server](https://github.com/Oaklight/toolregistry-server)** — OpenAPI 和 MCP 服务端适配器

## 贡献

欢迎贡献！请访问 [GitHub 仓库](https://github.com/Oaklight/llm-rosetta) 开始参与。

## 许可证

本项目采用 MIT 许可证——详见 [LICENSE](LICENSE) 文件。
