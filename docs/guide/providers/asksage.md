# AskSage

[AskSage](https://asksage.ai) 是一个面向政府的 AI 平台，提供 OpenAI、Anthropic 和 Google Gemini 模型的标准格式 API 端点。LLM-Rosetta 通过四个 shim 支持 AskSage——每种 API 格式一个——使用单一 API key。

!!! warning "以独立包形式发布"
    自 0.14.0 起，四个 AskSage shim 不再内置在 LLM-Rosetta 中，而是由
    [llm-rosetta-shim-asksage](https://github.com/Oaklight/llm-rosetta-shim-asksage)
    包提供。将它声明在 `shims.remotes` 下，下面的名称即可继续使用。
    **Settings → Shim Sources** 提供同一条目的一键预设。

    ```jsonc
    "shims": {
      "remotes": [{
        "url": "https://github.com/Oaklight/llm-rosetta-shim-asksage",
        "path": "src/llm_rosetta_shim_asksage/providers",
        "token_env": "GITHUB_TOKEN"      // 该仓库是私有的
      }]
    }
    ```

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `asksage--chat_completions` | `chat_completions` | `https://api.asksage.ai/server/openai/v1` | `ASKSAGE_API_KEY` |
| `asksage--open_responses` | `open_responses` | `https://api.asksage.ai/server/openai/v1` | `ASKSAGE_API_KEY` |
| `asksage--messages` | `messages` | `https://api.asksage.ai/server/anthropic` | `ASKSAGE_API_KEY` |
| `asksage--google_generate` | `google_generate` | `https://api.asksage.ai/server/google/v1beta` | `ASKSAGE_API_KEY` |

## 认证

AskSage 的 provider-compatible 端点直接接受 API key——不需要 token 交换。

- **OpenAI 端点**：标准 `Authorization: Bearer <key>` header
- **Anthropic 端点**：标准 `x-api-key: <key>` header
- **Gemini 端点**：使用 `x-access-tokens: <key>` header（非标准——通过 `connection.auth_header` 覆盖处理）

## 转换

| Shim | 类型 | 转换 | 用途 |
|:---|:---|:---|:---|
| `asksage--chat_completions` | Post-IR | `rename_field("max_tokens", "max_completion_tokens")` | AskSage 在 OpenAI 端点拒绝 `max_tokens` |
| `asksage--open_responses` | Post-IR | `rename_field("max_tokens", "max_completion_tokens")` | 同上 |

Anthropic 和 Gemini shim 不需要转换——AskSage 对这些使用标准格式。

## 可用模型

每个端点暴露不同的模型子集：

**OpenAI 端点**：GPT 模型（`gpt-4.1-*`、`gpt-5-*`、`gpt-6-*`、`gpt-o3`、`gpt-o4-mini`）、Bedrock 模型（`aws-bedrock-*`）

**Anthropic 端点**：Claude 模型（`claude-haiku-4-5`、`claude-sonnet-4-5`、`claude-opus-*`）

**Gemini 端点**：Gemini 模型（`gemini-2.5-flash`、`gemini-2.5-pro`、`gemini-3-flash`、`gemini-3-pro`）

!!! note "Gemini 没有模型列表端点"
    Gemini 端点不提供 `/models` 列表 API。需要在网关配置中直接指定模型名称。

## 实例特定的 Base URL

AskSage 部署使用实例特定的主机名。默认的 `api.asksage.ai` 是商业实例。对于其他实例（如 Argonne 国家实验室），在网关配置中覆盖 base URL：

```jsonc
{
  "providers": {
    "asksage--chat_completions": {
      "api_key": "${ASKSAGE_API_KEY}",
      "base_url": "https://api.asksage.anl.gov/server/openai/v1"
    }
  }
}
```

## 网关配置

```jsonc
{
  "providers": {
    "asksage--chat_completions": {
      "api_key": "${ASKSAGE_API_KEY}"
    },
    "asksage--open_responses": {
      "api_key": "${ASKSAGE_API_KEY}"
    },
    "asksage--messages": {
      "api_key": "${ASKSAGE_API_KEY}"
    },
    "asksage--google_generate": {
      "api_key": "${ASKSAGE_API_KEY}"
    }
  },
  "models": {
    "gpt-4.1-nano": "asksage--chat_completions",
    "gpt-4.1-mini": "asksage--chat_completions",
    "claude-haiku-4-5": "asksage--messages",
    "claude-sonnet-4-5": "asksage--messages",
    "gemini-2.5-flash": "asksage--google_generate",
    "gemini-3-flash": "asksage--google_generate"
  }
}
```
