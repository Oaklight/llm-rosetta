# LLMIR 跨 Provider 示例

## 概述

这些示例展示了 **LLMIR** 在 **4 种不同 LLM API 标准**之间进行跨 provider 多轮对话的能力。通过 IR（中间表示）消息格式，对话上下文可以在不同 provider 之间无缝传递。

支持的 4 种 API 标准：

- **OpenAI Chat Completions** (`oc`)
- **OpenAI Responses** (`or`)
- **Anthropic Messages** (`an`)
- **Google GenAI** (`gg`)

每对 provider 组合都有 **SDK** 和 **REST** 两种实现，共计 12 个示例脚本，覆盖所有 6 种唯一 provider 组合 × 2 种传输方式。

## 对话场景

所有示例运行相同的 **8 轮对话**，两个 provider 交替处理请求。场景是一个**旅行规划**讨论，比较旧金山和东京：

| 轮次 | Provider | 类型 | 描述 |
|------|----------|------|------|
| 1 | A | 纯文本 | 询问旧金山作为旅行目的地的信息 |
| 2 | B | 图片 + 文本 | 从照片中识别金门大桥 |
| 3 | A | 工具调用 | `get_current_weather` 查询旧金山天气 |
| 4 | B | 工具调用 | `get_flight_info` 查询纽约到旧金山的航班 |
| 5 | A | 纯文本 | 总结目前收集到的信息 |
| 6 | B | 图片 + 文本 | 从照片中识别东京塔，比较两座城市 |
| 7 | A | 工具调用 | `get_current_weather` 查询东京天气 |
| 8 | B | 纯文本 | 基于所有信息给出最终推荐 |

### 覆盖的功能

- **纯文本**生成与多轮上下文
- **图像分析**（视觉），使用 URL 引用的图片
- **工具调用**（函数调用 + 结果返回循环）
- **跨 provider 上下文共享**，通过 IR 消息实现

### 工具

提供两个模拟工具：

- `get_current_weather` — 返回指定位置的模拟天气数据
- `get_flight_info` — 返回两个城市之间的模拟航班信息

## 文件结构

```
examples/
├── README.md              # 英文版说明
├── README_zh.md           # 本文件（中文版）
├── common.py              # 公共资源：工具定义、对话轮次、辅助函数、
│                          # provider 配置加载器、图片 URL 转 base64
├── tools.py               # 旧版工具定义（供早期示例使用）
├── sdk_based/             # SDK 版本示例（使用 provider SDK）
│   ├── cross_oc_an.py     # OpenAI Chat ↔ Anthropic
│   ├── cross_oc_or.py     # OpenAI Chat ↔ OpenAI Responses
│   ├── cross_oc_gg.py     # OpenAI Chat ↔ Google GenAI
│   ├── cross_an_or.py     # Anthropic ↔ OpenAI Responses
│   ├── cross_an_gg.py     # Anthropic ↔ Google GenAI
│   └── cross_gg_or.py     # Google GenAI ↔ OpenAI Responses
└── rest_based/            # REST 版本示例（使用 httpx）
    ├── cross_oc_an_rest.py    # OpenAI Chat ↔ Anthropic
    ├── cross_oc_or_rest.py    # OpenAI Chat ↔ OpenAI Responses
    ├── cross_oc_gg_rest.py    # OpenAI Chat ↔ Google GenAI
    ├── cross_an_or_rest.py    # Anthropic ↔ OpenAI Responses
    ├── cross_an_gg_rest.py    # Anthropic ↔ Google GenAI
    └── cross_gg_or_rest.py    # Google GenAI ↔ OpenAI Responses
```

### Provider 缩写说明

| 缩写 | Provider |
|------|----------|
| `oc` | OpenAI Chat Completions |
| `or` | OpenAI Responses |
| `an` | Anthropic Messages |
| `gg` | Google GenAI |

## 环境设置

### API 密钥

每个 provider 需要各自的 API 密钥。可以设置为环境变量，或在项目根目录使用 `.env` 文件（通过 `python-dotenv` 自动加载）。

#### 必需的环境变量

| 变量 | Provider | 何时需要 |
|------|----------|----------|
| `OPENAI_API_KEY` | OpenAI（Chat 和 Responses） | 运行 OpenAI 相关示例时 |
| `ANTHROPIC_API_KEY` | Anthropic | 运行 Anthropic 相关示例时 |
| `GOOGLE_API_KEY` | Google GenAI | 运行 Google 相关示例时 |

#### 可选的环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API 基础 URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI 模型名称 |
| `OPENAI_RESPONSES_API_KEY` | 回退到 `OPENAI_API_KEY` | Responses API 的独立密钥 |
| `OPENAI_RESPONSES_BASE_URL` | 回退到 `OPENAI_BASE_URL` | Responses API 的独立基础 URL |
| `OPENAI_RESPONSES_MODEL` | 回退到 `OPENAI_MODEL` | Responses API 的独立模型 |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Anthropic API 基础 URL |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic 模型名称 |
| `GOOGLE_MODEL` | `gemini-2.0-flash` | Google GenAI 模型名称 |

#### `.env` 文件示例

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
GOOGLE_API_KEY=AIza...
GOOGLE_MODEL=gemini-2.0-flash
```

### Conda 环境

```bash
conda activate llmir
```

### 依赖安装

SDK 版本示例需要安装对应的 provider SDK：

```bash
pip install openai anthropic google-genai python-dotenv httpx
```

## 运行示例

### 基本用法

```bash
# 从项目根目录运行
python examples/sdk_based/cross_oc_an.py
python examples/rest_based/cross_oc_an_rest.py
```

### 网络代理

在受限网络环境中，OpenAI 和 Google API 可能需要代理。使用 `proxychains -q` 将流量路由到代理：

```bash
# OpenAI + Anthropic（OpenAI 需要代理）
proxychains -q python examples/sdk_based/cross_oc_an.py

# OpenAI + Google（两者都需要代理）
proxychains -q python examples/sdk_based/cross_oc_gg.py

# Anthropic + Google（Google 需要代理）
proxychains -q python examples/sdk_based/cross_an_gg.py

# Anthropic + OpenAI Responses（OpenAI 需要代理）
proxychains -q python examples/sdk_based/cross_an_or.py
```

Anthropic 通常可以直接访问，无需代理。

## 架构说明

### IR 消息作为共享状态

核心思想是所有 provider 共享一个 **IR（中间表示）消息列表**。每个 provider 的 converter 负责在 IR 格式和 provider 原生格式之间进行转换：

```
                    ┌─────────────────────┐
                    │   IR 消息列表       │
                    │  （共享状态）        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌────────────────┐ ┌───────────┐ ┌────────────────┐
     │ OpenAI Chat    │ │ Anthropic │ │ Google GenAI   │
     │ Converter      │ │ Converter │ │ Converter      │
     └────────────────┘ └───────────┘ └────────────────┘
```

### 转换流程

每一轮对话的流程如下：

1. 构建 IR `UserMessage` 并追加到共享的 `ir_messages` 列表
2. 调用 `converter.request_to_provider(ir_request)` 将 IR 转换为 provider 原生格式
3. 将请求发送到 provider（通过 SDK 或 REST）
4. 调用 `converter.response_from_provider(response)` 将 provider 响应转换为 IR 格式
5. 提取 assistant 消息并追加到 `ir_messages`

### 工具调用循环

当 assistant 响应包含工具调用时：

1. 从 IR assistant 消息中提取工具调用
2. 执行每个工具（模拟实现）并创建 IR 工具结果消息
3. 将工具结果追加到 `ir_messages`
4. 将更新后的对话发送回**同一个 provider** 获取后续响应

### 图片处理差异

不同 provider 处理图片的方式不同：

- **OpenAI Chat / Responses**：直接支持图片 URL，但可能无法下载对话历史中的某些 URL。示例中在发送给 OpenAI 时会剥离历史中的图片部分（`_strip_images()`）。
- **Anthropic**：原生支持图片 URL 和 base64 内联数据。
- **Google GenAI**：**不**直接支持图片 URL。示例中在发送给 Google 之前会将图片 URL 转换为内联 base64 数据（`convert_image_urls_to_inline()`）。