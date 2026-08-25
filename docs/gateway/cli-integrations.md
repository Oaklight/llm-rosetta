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

## Antigravity CLI (agy)

[Antigravity CLI](https://antigravity.google/docs/cli/install/) 是 Google 的 AI 编程 CLI 工具（Gemini CLI 的继任者），使用 Google GenAI API (`/v1beta/models/...`)。

**安装：**

```bash
curl -fsSL https://antigravity.google/cli/install.sh | bash
```

**配置** `~/.gemini/antigravity-cli/settings.json`：

```json
{
    "modelProvider": "gemini",
    "model": {
        "name": "gemini-3.5-flash"
    }
}
```

模型名称可以是网关上配置的**任意模型**，包括非 Google 模型如 `claude-sonnet-4-6`、`deepseek-v4-flash`、`gpt-5-nano`。网关会自动处理跨格式转换。

**设置环境变量：**

```bash
export GEMINI_API_KEY=your-gateway-api-key
export GOOGLE_GEMINI_BASE_URL=http://localhost:8765
```

**运行：**

```bash
agy
```

或无头模式：

```bash
agy -p "your prompt here"
```

!!! warning "需要配置 planner 模型"
    agy 内部使用一个硬编码的 planner 模型（`gemini-3.1-pro-preview`），会与用户选择的主模型一起发送请求。网关上**必须**配置此模型，否则 planner 会在主模型执行前失败。

    在网关配置中添加一个别名，指向任意可用的 provider：

    ```json
    "gemini-3.1-pro-preview": {
        "provider": "YourProvider",
        "capabilities": ["text", "vision", "tools", "reasoning"],
        "upstream_model": "any-capable-model"
    }
    ```

!!! tip "使用非 Google 模型"
    这是 llm-rosetta 配合 agy 的核心用例：你可以通过 agy 使用 Claude、DeepSeek、GPT 等任意 provider。网关会自动在 Google GenAI 格式和目标 provider 格式之间进行转换。

    已验证的组合：

    | 模型 | Provider | 文本 | 图像 | 工具 |
    |------|----------|:----:|:----:|:----:|
    | `claude-sonnet-4-6` | Anthropic | ✅ | ✅ | ✅ |
    | `deepseek-v4-flash` | DeepSeek | ✅ | ✅ | ✅ |
    | `gemini-3.5-flash` | Google | ✅ | ✅ | ✅ |

**支持功能**：对话、流式传输、图像识别、工具调用 ✅

---

## Gemini CLI（已停止服务）

!!! warning "已停止服务"
    Gemini CLI 已于 2026 年 6 月 18 日停止服务，由 [Antigravity CLI (agy)](#antigravity-cli-agy) 取代。以下配置仅供参考。

??? note "旧版 Gemini CLI 配置"

    Gemini CLI 使用 Google GenAI API (`/v1beta/models/...`)。

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
