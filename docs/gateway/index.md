---
title: 网关
---

# 网关（Gateway）

LLM-Rosetta 网关是一个 HTTP 代理服务，可以实时在 LLM 提供商 API 格式之间进行转换。发送任意支持格式的请求，网关会自动转换并转发到配置的上游提供商。

```text
客户端 (OpenAI 格式) ──→ 网关 ──→ Anthropic API
客户端 (Anthropic 格式) ──→ 网关 ──→ OpenAI API
客户端 (Google 格式) ──→ 网关 ──→ 任意提供商
```

## 安装

```bash
pip install "llm-rosetta[gateway]"
```

这会安装网关依赖：[Starlette](https://www.starlette.io/)、[uvicorn](https://www.uvicorn.org/) 和 [httpx](https://www.python-httpx.org/)。

## 快速开始

### 1. 创建配置文件

创建 `config.jsonc`（支持注释的 JSON）：

```jsonc
{
  "providers": {
    "openai_chat":      { "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "openai_responses": { "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "anthropic":        { "api_key": "${ANTHROPIC_API_KEY}",  "base_url": "https://api.anthropic.com" },
    "google":           { "api_key": "${GOOGLE_API_KEY}",     "base_url": "https://generativelanguage.googleapis.com" }
  },
  "models": {
    "gpt-4o": "openai_chat",
    "gpt-4o-mini": "openai_chat",
    "claude-sonnet-4-20250514": "anthropic",
    "claude-3-5-haiku-20241022": "anthropic",
    "gemini-2.0-flash": "google",
    "gemini-2.5-pro": "google"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8765
  }
}
```

API 密钥支持 `${ENV_VAR}` 语法 — 启动时从环境变量读取。

### 2. 启动网关

```bash
# CLI 命令（pip install 后可用）
llm-rosetta-gateway --config config.jsonc

# 或作为 Python 模块运行
python -m llm_rosetta.gateway --config config.jsonc
```

### 3. 发送请求

使用任意提供商格式 — 网关根据模型名称自动路由：

```bash
# 发送 OpenAI 格式请求，路由到 Anthropic
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# 发送 Anthropic 格式请求，路由到 OpenAI
curl http://localhost:8765/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## 端点

| 路径 | 来源格式 | 说明 |
|------|---------|------|
| `POST /v1/chat/completions` | OpenAI Chat | 兼容 OpenAI SDK |
| `POST /v1/messages` | Anthropic | 兼容 Anthropic SDK |
| `POST /v1/responses` | OpenAI Responses | 兼容 OpenAI Responses SDK |
| `POST /v1beta/models/{model}:generateContent` | Google GenAI | 兼容 Google REST API |
| `POST /v1beta/models/{model}:streamGenerateContent` | Google GenAI（流式） | 兼容 Google 流式 API |
| `GET /health` | — | 健康检查 |

端点路径决定了来源格式 — 无需自动检测。

## 流式传输

所有提供商组合均支持流式传输。请求方式与原生 API 相同：

```bash
# OpenAI 格式流式请求，路由到任意提供商
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

网关实时在提供商格式之间转换 SSE 数据块。

## 配置说明

### 提供商配置

每个提供商需要 `api_key` 和 `base_url`：

```jsonc
"providers": {
  "openai_chat":      { "api_key": "sk-...",     "base_url": "https://api.openai.com/v1" },
  "anthropic":        { "api_key": "sk-ant-...", "base_url": "https://api.anthropic.com" },
  "google":           { "api_key": "AIza...",    "base_url": "https://generativelanguage.googleapis.com" }
}
```

提供商名称必须是以下之一：`openai_chat`、`openai_responses`、`anthropic`、`google`。

### 模型路由

`models` 部分将模型名称映射到提供商：

```jsonc
"models": {
  "gpt-4o": "openai_chat",
  "claude-sonnet-4-20250514": "anthropic",
  "gemini-2.0-flash": "google"
}
```

当请求包含 `"model": "claude-sonnet-4-20250514"` 时，网关查找到 `anthropic` 并相应转发。

### CLI 选项

```
llm-rosetta-gateway [OPTIONS]

  --config PATH      配置文件路径（默认：config.jsonc）
  --host HOST        覆盖服务器主机
  --port PORT        覆盖服务器端口
  --log-level LEVEL  日志级别：debug, info, warning, error（默认：info）
```

## 编程方式使用

网关也可以作为库使用：

```python
from llm_rosetta.gateway import create_app, GatewayConfig, load_config

# 加载配置并创建 ASGI 应用
raw = load_config("config.jsonc")
config = GatewayConfig(raw)
app = create_app(config)

# 挂载到你自己的 ASGI 应用中，或使用任意 ASGI 服务器运行
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8765)
```

## 工作原理

网关使用 LLM-Rosetta 的转换器管道：

```text
1. 接收请求（来源格式）
2. source_converter.request_from_provider() → IR 请求
3. 查找模型 → 目标提供商
4. target_converter.request_to_provider() → 目标格式
5. 转发到上游 API
6. target_converter.response_from_provider() → IR 响应
7. source_converter.response_to_provider() → 来源格式
8. 返回客户端
```

对于流式传输，同样的管道在 SSE 数据块级别运行，使用 `stream_response_from_provider()` 和 `stream_response_to_provider()` 配合 `StreamContext` 进行有状态转换。
