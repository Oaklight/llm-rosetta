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
llm-rosetta-gateway

# 或显式指定配置文件
llm-rosetta-gateway --config /path/to/config.jsonc

# 或作为 Python 模块运行
python -m llm_rosetta.gateway
```

网关会按以下顺序自动搜索配置文件（首个匹配生效）：

1. `./config.jsonc`（当前目录）
2. `~/.config/llm-rosetta-gateway/config.jsonc`
3. `~/.llm-rosetta-gateway/config.jsonc`

也可以使用 `init` 或 `add` 子命令快速创建配置文件：

```bash
# 在 ~/.config/llm-rosetta-gateway/config.jsonc 创建模板配置
llm-rosetta-gateway init

# 或通过 add 子命令逐步构建配置
llm-rosetta-gateway add provider openai_chat
llm-rosetta-gateway add provider anthropic
llm-rosetta-gateway add provider google

# 添加模型路由条目
llm-rosetta-gateway add model gpt-4o --provider openai_chat
llm-rosetta-gateway add model claude-sonnet-4-20250514 --provider anthropic
llm-rosetta-gateway add model gemini-2.0-flash --provider google
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
| `GET /v1/models` | OpenAI / Anthropic | 列出已配置模型（兼容两种 SDK） |
| `GET /v1beta/models` | Google GenAI | 列出已配置模型（Google SDK 格式） |
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

#### API 密钥轮转

每个提供商支持通过逗号分隔配置多个 API 密钥，网关以轮询方式依次使用：

```jsonc
"openai_chat": { "api_key": "sk-key1,sk-key2,sk-key3", "base_url": "https://api.openai.com/v1" }
```

#### 逐提供商代理

可为单个提供商指定代理：

```jsonc
"anthropic": { "api_key": "sk-ant-...", "base_url": "https://api.anthropic.com", "proxy": "http://proxy:8080" }
```

### 代理配置

可在 `server` 部分设置全局代理，适用于所有提供商（除非逐提供商覆盖）：

```jsonc
{
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "proxy": "http://proxy.example.com:8080"
  }
}
```

CLI `--proxy` 参数会覆盖配置文件中的全局代理设置。

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
llm-rosetta-gateway [选项] [命令]

选项:
  --config, -c PATH    配置文件路径（未指定时自动搜索）
  --version, -V        显示版本并退出
  --no-banner          抑制启动横幅显示
  --edit, -e           在 $EDITOR 中打开配置文件进行编辑
  --host HOST          覆盖服务器主机
  --port PORT          覆盖服务器端口
  --proxy URL          所有上游请求的 HTTP/SOCKS 代理 URL
  --log-level LEVEL    日志级别：debug, info, warning, error（默认：info）

命令:
  init                 在 ~/.config/llm-rosetta-gateway/ 创建模板 config.jsonc

  add provider <name>  添加提供商条目到配置
    --api-key KEY        API 密钥或 ${ENV_VAR} 占位符
    --base-url URL       提供商基础 URL（已知提供商自动填充）

  add model <name>     添加模型路由条目到配置
    --provider NAME      目标提供商名称
```

#### 配置文件自动发现

未指定 `--config` 时，网关按以下顺序搜索配置文件：

1. `./config.jsonc` — 当前工作目录
2. `~/.config/llm-rosetta-gateway/config.jsonc` — XDG 标准位置
3. `~/.llm-rosetta-gateway/config.jsonc` — dotfile 约定

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

## CLI 工具集成

网关可以作为主流 AI 编程 CLI 工具的后端。每个工具使用不同的 API 格式——网关会自动处理格式转换。

### Claude Code

Claude Code 使用 Anthropic Messages API (`/v1/messages`)。

```bash
export ANTHROPIC_BASE_URL=http://localhost:8765
export ANTHROPIC_API_KEY=your-key  # 或任意占位符
export CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1
claude --model claude-sonnet-4-20250514
```

或在 `~/.claude/settings.json` 中配置：

```json
{
  "env": {
    "ANTHROPIC_MODEL": "claude-sonnet-4-20250514",
    "ANTHROPIC_BASE_URL": "http://localhost:8765",
    "CLAUDE_CODE_SKIP_ANTHROPIC_AUTH": "1"
  }
}
```

**支持功能**：对话、多轮对话、图片、工具调用、流式传输 ✅

### Kilo Code

Kilo Code 使用 OpenAI Chat Completions API (`/v1/chat/completions`)。

在 `~/.config/kilo/kilo.jsonc` 中添加自定义提供商：

```jsonc
{
  "provider": {
    "rosetta": {
      "api": "openai",
      "name": "Rosetta Gateway",
      "models": {
        "claude-sonnet-4-20250514": {
          "name": "Claude Sonnet 4",
          "attachment": true,
          "tool_call": true,
          "cost": { "input": 0, "output": 0 },
          "limit": { "context": 200000, "output": 8192 }
        }
        // 根据需要添加更多模型
      },
      "options": {
        "apiKey": "your-key",
        "baseURL": "http://localhost:8765/v1"
      }
    }
  }
}
```

然后使用：`kilo --model rosetta/claude-sonnet-4-20250514`

**支持功能**：对话、多轮对话、工具调用、流式传输 ✅

### OpenAI Codex CLI

Codex CLI 使用 OpenAI Responses API (`/v1/responses`)。

创建 `~/.codex/config.toml`：

```toml
model = "gpt-4o"
model_provider = "rosetta"

[model_providers.rosetta]
name = "Rosetta Gateway"
base_url = "http://localhost:8765/v1"
env_key = "ROSETTA_API_KEY"
wire_api = "responses"
```

然后：

```bash
export ROSETTA_API_KEY=your-key
codex "your prompt here"
```

**支持功能**：对话、多轮对话、工具调用、流式传输 ✓

### Ollama

[Ollama](https://ollama.com/)（v0.13+）在本地提供 OpenAI 兼容接口，非常适合作为网关的上游提供商或客户端。

#### 将 Ollama 作为上游提供商

将网关提供商指向本地 Ollama 实例：

```jsonc
"providers": {
  "openai_chat": { "api_key": "ollama", "base_url": "http://localhost:11434/v1" }
},
"models": {
  "llama3.2": "openai_chat",
  "qwen3:8b": "openai_chat"
}
```

这样任何客户端（Anthropic SDK、Google SDK 等）都可以通过网关查询本地 Ollama 模型，格式自动转换。

#### 将 Ollama 作为客户端

Ollama v0.13+ 支持网关可以提供的三种 API 格式：

| Ollama 端点 | 网关路由 | 转换器 |
|---|---|---|
| `/v1/chat/completions` | 相同 | `openai_chat` |
| `/v1/responses` | 相同 | `openai_responses`（v0.13.3+） |
| `/v1/messages` | 相同 | `anthropic`（v0.14.0+） |

这意味着基于 Ollama OpenAI 兼容层构建的工具可以通过网关访问云提供商（Anthropic、Google 等），无需更改代码——只需将 base URL 指向网关即可。

### Gemini CLI

Gemini CLI 使用 Google GenAI API (`/v1beta/models/...`)。

```bash
export GOOGLE_GEMINI_BASE_URL=http://localhost:8765
export GEMINI_API_KEY=your-key
gemini -m gemini-2.5-pro -p "your prompt here"
```

!!! note "TTY 要求"
    Gemini CLI 即使在无头模式（`-p`）下也需要 TTY。在脚本或非交互式 shell 中运行时，请使用 `script` 包装：

    ```bash
    script -qec 'gemini -m gemini-2.5-pro -p "your prompt"' /dev/null
    ```

!!! note "网络依赖"
    Gemini CLI 在启动时会连接 `github.com` 和 `play.googleapis.com`。这些地址必须可达（直连或通过代理）。

**支持功能**：对话、流式传输 ✅

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
