---
title: 合规性测试
---

# 使用 llm-comply 进行合规性测试

[llm-comply](https://github.com/Oaklight/llm-comply) 是一个配套工具，用于验证 LLM API 端点是否符合官方规范。它通过内置的 OpenAPI 规范进行 schema 校验，并执行语义检查（生命周期顺序、流式事件、字段存在性）。

**[在线体验 →](https://llm-comply.service.oaklight.top)**

## 支持的格式

| 格式 | 标志 | 测试数 |
|------|------|:-----:|
| OpenAI Chat Completions | `openai-chat` | 8 |
| Open Responses | `open-responses` | 12 |
| Anthropic Messages | `anthropic` | 8 |
| Google GenAI | `google-genai` | 9 |

## 安装

```bash
pip install llm-comply

# 彩色终端输出：
pip install llm-comply[rich]
```

## CLI 使用

### 基本用法

```bash
# 使用 Open Responses 格式测试（默认）
llm-comply -u https://your-gateway/v1 -k $API_KEY -m your-model

# 测试特定格式
llm-comply -u https://your-gateway/v1 -k $API_KEY -m your-model --format openai-chat
```

### 通过 LLM-Rosetta 网关测试

当网关运行时（通过 `llm-rosetta-gateway` 或 [argo-proxy](https://github.com/Oaklight/argo-proxy)）：

```bash
# OpenAI Chat 格式
llm-comply --format openai-chat \
  -u http://localhost:44497/v1 \
  -k $API_KEY -m gpt-4o-mini

# Anthropic 格式（需要自定义认证头）
llm-comply --format anthropic \
  -u http://localhost:44497/v1 \
  -k $API_KEY -m claude-haiku-4-5 \
  --auth-header x-api-key --no-bearer \
  -H anthropic-version:2023-06-01

# Google GenAI 格式
llm-comply --format google-genai \
  -u http://localhost:44497 \
  -k $API_KEY -m gemini-2.5-flash \
  --auth-header x-goog-api-key --no-bearer
```

### 常用选项

```
-u, --base-url URL     API 基础 URL（必填）
-k, --api-key KEY      API 密钥
-m, --model MODEL      模型名称（默认：gpt-4o-mini）
--format FORMAT        API 格式：open-responses、openai-chat、anthropic、google-genai
-f, --filter IDS       以逗号分隔的测试 ID，仅运行指定测试
-i, --ignore PATTERNS  忽略匹配指定子串的错误
-H, --header K:V       额外请求头
--auth-header NAME     认证头名称（默认：Authorization）
--no-bearer            不在 API 密钥前添加 "Bearer " 前缀
-v, --verbose          失败时显示请求/响应详情
--json                 以 JSON 格式输出结果
--list                 列出可用测试但不运行
```

## Web UI

llm-comply 提供了基于浏览器的交互式测试界面：

```bash
llm-comply run --web --host 0.0.0.0 --port 8080
```

在线托管实例：[llm-comply.service.oaklight.top](https://llm-comply.service.oaklight.top)。

## CI 集成

LLM-Rosetta 仓库包含一个按需触发的 [Compliance 工作流](https://github.com/Oaklight/llm-rosetta/actions/workflows/compliance.yml)，用于对网关端点运行 llm-comply 测试。

### 通过 CLI 触发

```bash
gh workflow run Compliance \
  -f base_url=https://rosetta-dev.service.oaklight.top/v1 \
  -f api_key=your-key \
  -f model=deepseek-v4-flash \
  -f formats="open-responses openai-chat" \
  -f timeout=60
```

### 通过 GitHub UI 触发

进入 **Actions → Compliance → Run workflow**，填写输入参数后点击 **Run workflow**。结果会以汇总表格形式显示在工作流运行页面，JSON 产物保留 30 天。

### 工作流输入参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_url` | `https://rosetta-dev.service.oaklight.top/v1` | 待测试的 API 基础 URL |
| `api_key` | — | API 密钥（必填） |
| `model` | `deepseek-v4-flash` | 测试使用的模型 |
| `formats` | `open-responses openai-chat` | 以空格分隔的待测试格式 |
| `timeout` | `60` | 请求超时时间（秒） |
