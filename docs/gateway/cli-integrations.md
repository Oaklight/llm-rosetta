---
title: CLI 工具集成
---

# CLI 工具集成

网关可以作为主流 AI 编程 CLI 工具的后端。每个工具使用不同的 API 格式——网关会自动处理格式转换。

!!! note "前提条件"
    以下所有示例假设网关运行在 `http://localhost:8765`，请替换为你的实际地址和端口。

## Claude Code

[Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) 通过 **Anthropic Messages API** (`/v1/messages`) 连接。

=== "配置文件（推荐）"

    ```bash
    claude config set -g apiKeyHelper "echo 'your-api-key'"
    claude config set -g env.ANTHROPIC_BASE_URL "http://localhost:8765"
    claude config set -g env.CLAUDE_CODE_SKIP_ANTHROPIC_AUTH "1"
    ```

    这会写入 `~/.claude/settings.json`：

    ```json
    {
        "apiKeyHelper": "echo 'your-api-key'",
        "env": {
            "ANTHROPIC_BASE_URL": "http://localhost:8765",
            "CLAUDE_CODE_SKIP_ANTHROPIC_AUTH": "1"
        }
    }
    ```

=== "环境变量"

    ```bash
    export ANTHROPIC_BASE_URL="http://localhost:8765"
    export ANTHROPIC_API_KEY="your-api-key"
    export CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1
    claude
    ```

!!! important
    - `CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1` **必须设置** — 跳过 Anthropic 默认的认证流程
    - `ANTHROPIC_BASE_URL` 设置为代理根路径（如 `http://localhost:8765`），**不要**加 `/v1/messages` — Claude Code 会自动拼接

**支持功能**：对话、多轮对话、图片、工具调用、流式传输 ✅

---

## Codex CLI (OpenAI)

[Codex CLI](https://github.com/openai/codex) 通过 **OpenAI Responses API** (`/v1/responses`) 连接。

=== "配置文件（推荐）"

    创建 `~/.codex/config.toml`：

    ```toml
    model = "gpt-5-nano"
    model_provider = "rosetta"

    [model_providers.rosetta]
    name = "Rosetta Gateway"
    base_url = "http://localhost:8765/v1"
    env_key = "ROSETTA_API_KEY"
    wire_api = "responses"
    ```

    然后在 shell 配置中设置 API key：

    ```bash
    export ROSETTA_API_KEY="your-api-key"
    ```

=== "环境变量"

    ```bash
    export OPENAI_BASE_URL="http://localhost:8765/v1"
    export OPENAI_API_KEY="your-api-key"
    codex "your prompt here"
    ```

!!! note
    Codex CLI 默认使用 **Responses API** 格式。配置文件中 `wire_api = "responses"` 使其显式声明。

**支持功能**：对话、多轮对话、工具调用、流式传输 ✅

---

## Aider

[Aider](https://aider.chat/) 支持 OpenAI 和 Anthropic 两种后端。

=== "OpenAI 模式"

    ```bash
    export OPENAI_API_BASE="http://localhost:8765/v1"
    export OPENAI_API_KEY="your-api-key"
    aider --model gpt-5-nano
    ```

=== "Anthropic 模式"

    ```bash
    export ANTHROPIC_BASE_URL="http://localhost:8765"
    export ANTHROPIC_API_KEY="your-api-key"
    aider --model claude-sonnet-4-6
    ```

!!! tip
    可以将这些设置写入 `.aider.conf.yml` 或 shell 配置文件中持久化。

**支持功能**：对话、多轮对话、工具调用、流式传输 ✅

---

## Antigravity CLI (agy)

[Antigravity CLI](https://antigravity.google/docs/cli/install/) 是 Google 的智能编程 CLI（Gemini CLI 的后继者），使用 **Google GenAI API** (`/v1beta/models/...`)。

=== "配置文件（推荐）"

    **1. `~/.gemini/antigravity-cli/settings.json`**：

    ```json
    {
        "modelProvider": "gemini",
        "model": {
            "name": "gemini-3.5-flash"
        }
    }
    ```

    **2. 环境变量**（添加到 shell 配置文件）：

    ```bash
    export GEMINI_API_KEY="your-api-key"
    export GOOGLE_GEMINI_BASE_URL="http://localhost:8765"
    ```

    然后直接运行 `agy`。

=== "环境变量"

    ```bash
    GOOGLE_GEMINI_BASE_URL=http://localhost:8765 \
    GEMINI_API_KEY=your-api-key \
    agy
    ```

模型名称可以是网关上配置的**任何模型** — 包括非 Google 模型如 `claude-sonnet-4-6`、`deepseek-v4-flash` 或 `gpt-5-nano`。网关会自动处理跨格式转换。

!!! warning "需要 Planner 模型"
    agy 内部使用硬编码的 planner 模型（`gemini-3.1-pro-preview`），**无法覆盖** — 你的网关必须配置此模型。

!!! tip "使用非 Google 模型"
    这是核心用途：通过网关让 agy 使用 Claude、DeepSeek、GPT 或任何其他 provider。网关自动处理 Google GenAI 格式与目标 provider 格式之间的转换。

**支持功能**：对话、流式传输、视觉、工具调用 ✅

---

## OpenCode

[OpenCode](https://github.com/opencode-ai/opencode) 支持 OpenAI 兼容端点。

=== "配置文件（推荐）"

    在 `~/.config/opencode/opencode.json` 中添加自定义 provider：

    ```json
    {
        "provider": {
            "rosetta": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Rosetta Gateway",
                "options": {
                    "baseURL": "http://localhost:8765/v1",
                    "apiKey": "your-api-key"
                },
                "models": {
                    "gpt-5-nano": { "name": "GPT-5 Nano" },
                    "claude-sonnet-4-6": { "name": "Claude Sonnet 4.6" }
                }
            }
        }
    }
    ```

=== "环境变量"

    ```bash
    export OPENAI_BASE_URL="http://localhost:8765/v1"
    export OPENAI_API_KEY="your-api-key"
    opencode
    ```

**支持功能**：对话、多轮对话、工具调用、流式传输 ✅

---

## Kilo Code

[Kilo Code](https://kilocode.ai/)（VS Code 扩展）使用 OpenAI Chat Completions API (`/v1/chat/completions`)。

在 `~/.config/kilo/kilo.jsonc` 中添加自定义 provider：

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
      },
      "options": {
        "apiKey": "your-api-key",
        "baseURL": "http://localhost:8765/v1"
      }
    }
  }
}
```

然后使用：`kilo --model rosetta/claude-sonnet-4-20250514`

**支持功能**：对话、多轮对话、工具调用、流式传输 ✅

---

## Ollama

[Ollama](https://ollama.com/) (v0.13+) 提供 OpenAI 兼容端点，既可以作为上游 provider，也可以作为客户端。

### 作为上游 Provider

将网关 provider 指向本地 Ollama 实例：

```jsonc
"providers": {
  "local-ollama": { "type": "openai_chat", "api_key": "ollama", "base_url": "http://localhost:11434/v1" }
},
"models": {
  "llama3.2": "local-ollama",
  "qwen3:8b": "local-ollama"
}
```

### 作为客户端

Ollama v0.13+ 支持网关提供的三种 API 格式：

| Ollama 端点 | 网关路由 | 转换器 |
|---|---|---|
| `/v1/chat/completions` | 相同 | `openai_chat` |
| `/v1/responses` | 相同 | `openai_responses` (v0.13.3+) |
| `/v1/messages` | 相同 | `anthropic` (v0.14.0+) |

---

## 通用 SDK

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8765/v1",
    api_key="your-api-key",
)

response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Anthropic Python SDK

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8765",
    api_key="your-api-key",
)

message = client.messages.create(
    model="gpt-5-nano",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)
```

---

## Gemini CLI（已停止维护）

!!! warning "已停止维护"
    Gemini CLI 于 2026 年 6 月 18 日停止服务，已被 [Antigravity CLI (agy)](#antigravity-cli-agy) 取代。以下配置仅供参考。

??? note "旧版 Gemini CLI 配置"

    **`~/.gemini/.env`**：

    ```bash
    GEMINI_API_KEY=your-api-key
    GOOGLE_GEMINI_BASE_URL=http://localhost:8765
    ```

    **`~/.gemini/settings.json`**：

    ```json
    {
        "model": { "name": "gemini-2.5-pro" },
        "security": { "auth": { "selectedType": "gemini-api-key" } }
    }
    ```

---

## 总结

| 工具 | API 格式 | Base URL 环境变量 | 值 |
|------|---------|-----------------|------|
| Claude Code | Anthropic | `ANTHROPIC_BASE_URL` | `http://localhost:8765` |
| Codex CLI | OpenAI Responses | `OPENAI_BASE_URL` | `http://localhost:8765/v1` |
| Aider (OpenAI) | OpenAI | `OPENAI_API_BASE` | `http://localhost:8765/v1` |
| Aider (Anthropic) | Anthropic | `ANTHROPIC_BASE_URL` | `http://localhost:8765` |
| agy | Google GenAI | `GOOGLE_GEMINI_BASE_URL` | `http://localhost:8765` |
| OpenCode | OpenAI | `OPENAI_BASE_URL` | `http://localhost:8765/v1` |
| Kilo Code | OpenAI | —（配置文件）| `http://localhost:8765/v1` |
| OpenAI SDK | OpenAI | `OPENAI_BASE_URL` | `http://localhost:8765/v1` |
| Anthropic SDK | Anthropic | `ANTHROPIC_BASE_URL` | `http://localhost:8765` |
