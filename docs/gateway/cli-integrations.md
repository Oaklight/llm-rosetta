---
title: CLI 工具集成
---

# CLI 工具集成

网关可以作为主流 AI 编程 CLI 工具的后端。每个工具使用不同的 API 格式——网关会自动处理格式转换。

## Claude Code

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

## Kilo Code

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

## OpenAI Codex CLI

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

## Ollama

[Ollama](https://ollama.com/)（v0.13+）在本地提供 OpenAI 兼容接口，非常适合作为网关的上游提供商或客户端。

### 将 Ollama 作为上游提供商

将网关提供商指向本地 Ollama 实例：

```jsonc
"providers": {
  "local-ollama": { "type": "openai_chat", "api_key": "ollama", "base_url": "http://localhost:11434/v1" }
},
"models": {
  "llama3.2": "local-ollama",
  "qwen3:8b": "local-ollama"
}
```

这样任何客户端（Anthropic SDK、Google SDK 等）都可以通过网关查询本地 Ollama 模型，格式自动转换。

### 将 Ollama 作为客户端

Ollama v0.13+ 支持网关可以提供的三种 API 格式：

| Ollama 端点 | 网关路由 | 转换器 |
|---|---|---|
| `/v1/chat/completions` | 相同 | `openai_chat` |
| `/v1/responses` | 相同 | `openai_responses`（v0.13.3+） |
| `/v1/messages` | 相同 | `anthropic`（v0.14.0+） |

这意味着基于 Ollama OpenAI 兼容层构建的工具可以通过网关访问云提供商（Anthropic、Google 等），无需更改代码——只需将 base URL 指向网关即可。

## Gemini CLI

Gemini CLI 使用 Google GenAI API (`/v1beta/models/...`)。

=== "配置文件（推荐）"

    **`~/.gemini/.env`** — Gemini CLI 启动时自动读取此文件：

    ```bash
    GEMINI_API_KEY=your-key
    GOOGLE_GEMINI_BASE_URL=http://localhost:8765
    ```

    **`~/.gemini/settings.json`** — 设置认证模式和默认模型：

    ```json
    {
        "model": {
            "name": "gemini-2.5-pro"
        },
        "security": {
            "auth": {
                "selectedType": "gemini-api-key"
            }
        }
    }
    ```

    两个文件配置好后，直接运行 `gemini` 即可，无需额外参数。

=== "环境变量"

    ```bash
    export GOOGLE_GEMINI_BASE_URL=http://localhost:8765
    export GEMINI_API_KEY=your-key
    gemini -m gemini-2.5-pro -p "your prompt here"
    ```

!!! tip "Bearer token 认证"
    如果上游代理要求 Bearer token 认证（如 OneAPI），在 `~/.gemini/.env` 中添加：

    ```bash
    GEMINI_API_KEY_AUTH_MECHANISM=bearer
    ```

    这会将 API key 作为 `Bearer` token 放在 `Authorization` 请求头中发送，而非作为查询参数。

!!! note "TTY 要求"
    Gemini CLI 即使在无头模式（`-p`）下也需要 TTY。在脚本或非交互式 shell 中运行时，请使用 `script` 包装：

    ```bash
    script -qec 'gemini -m gemini-2.5-pro -p "your prompt"' /dev/null
    ```

!!! note "网络依赖"
    Gemini CLI 在启动时会连接 `github.com` 和 `play.googleapis.com`。这些地址必须可达（直连或通过代理）。

**支持功能**：对话、流式传输 ✅
