---
title: 配置
---

# 配置

本页详细介绍网关的配置文件格式。

## 提供商

每个提供商条目需要 `api_key`、`base_url`，以及可选的 `type` 字段指定 API 标准：

```jsonc
"providers": {
  "my-openai":   { "type": "openai_chat",      "api_key": "sk-...",     "base_url": "https://api.openai.com/v1" },
  "my-anthropic": { "type": "anthropic",        "api_key": "sk-ant-...", "base_url": "https://api.anthropic.com" },
  "my-google":   { "type": "google",            "api_key": "AIza...",    "base_url": "https://generativelanguage.googleapis.com" }
}
```

提供商名称是用户自定义的字符串（如 `"my-openai"`、`"prod-claude"`）。`type` 字段指定使用哪种 API 标准。

可用类型：`openai_chat`、`openai_responses`、`anthropic`、`google`。

### 使用 Shim

除了 `type`，还可以使用 `shim` 字段引用已注册的提供商 shim。Shim 是一个轻量级身份卡片，声明了提供商使用的基础 API 标准、连接默认值和字段级转换规则。

```jsonc
"providers": {
  "my-deepseek":   { "shim": "deepseek",   "api_key": "${DEEPSEEK_API_KEY}" },
  "my-volcengine": { "shim": "volcengine",  "api_key": "${VOLCENGINE_API_KEY}", "base_url": "https://ark.cn-beijing.volces.com/api/v3" }
}
```

指定 `shim` 时：

- **基础类型**会自动解析（如 `deepseek` → `openai_chat`）
- 如果配置中未设置，会从 shim 填充默认的 **`base_url`** 和 **`api_key` 环境变量**
- 请求/响应转换时会应用**字段级转换**（如 Volcengine 的 shim 会剥离其 API 不支持的 `logprobs` 和 `top_logprobs` 字段）

内置 shim：`openai`、`openai_responses`、`anthropic`、`google`、`deepseek`、`volcengine`。

也可以通过 `register_shim()` 以编程方式注册自定义 shim。

!!! tip "解析优先级"
    提供商类型解析顺序为：`shim` → `type` → 提供商名称（兜底）。

!!! note "向后兼容"
    如果 `shim` 和 `type` 都省略，提供商名称本身将用作类型。这意味着使用旧格式（提供商名称为 `openai_chat`、`anthropic` 等）的配置无需修改即可继续使用。

### 启用 / 禁用提供商

每个提供商支持 `enabled` 字段（默认 `true`）。禁用的提供商及其关联模型将从路由中静默排除：

```jsonc
"my-openai": { "type": "openai_chat", "api_key": "sk-...", "base_url": "https://api.openai.com/v1", "enabled": false }
```

这在需要临时下线提供商但不删除配置时很有用。[管理面板](admin-panel.md)提供了切换开关来操作此功能。

### API 密钥轮转

每个提供商支持通过逗号分隔配置多个 API 密钥，网关以轮询方式依次使用：

```jsonc
"my-openai": { "type": "openai_chat", "api_key": "sk-key1,sk-key2,sk-key3", "base_url": "https://api.openai.com/v1" }
```

### 环境变量替换

API 密钥支持 `${ENV_VAR}` 语法 — 启动时从环境变量读取：

```jsonc
"my-openai": { "type": "openai_chat", "api_key": "${OPENAI_API_KEY}", "base_url": "https://api.openai.com/v1" }
```

### 逐提供商代理

可为单个提供商指定代理：

```jsonc
"my-anthropic": { "type": "anthropic", "api_key": "sk-ant-...", "base_url": "https://api.anthropic.com", "proxy": "http://proxy:8080" }
```

## 代理配置

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

同时支持 HTTP 和 SOCKS5 代理：

```jsonc
// HTTP 代理
"proxy": "http://proxy.example.com:8080"

// SOCKS5 代理（无认证）
"proxy": "socks5://proxy.example.com:1080"

// SOCKS5 代理（用户名/密码认证）
"proxy": "socks5://username:password@proxy.example.com:1080"
```

CLI `--proxy` 参数会覆盖配置文件中的全局代理设置。

## 模型路由

`models` 部分将模型名称映射到提供商：

```jsonc
"models": {
  "gpt-4o": "my-openai",
  "claude-sonnet-4-20250514": "my-anthropic",
  "gemini-2.0-flash": "my-google"
}
```

当请求包含 `"model": "claude-sonnet-4-20250514"` 时，网关查找到 `my-anthropic` 并相应转发。

### 模型能力声明

模型可以使用字典格式声明能力：

```jsonc
"models": {
  "gpt-4o": { "provider": "my-openai", "capabilities": ["text", "vision", "tools"] },
  "gemini-2.0-flash": { "provider": "my-google", "capabilities": ["text", "tools"] }
}
```

可用能力：`text`、`vision`、`tools`、`embedding`、`reasoning`。如未指定，默认为 `["text"]`。注意 `embedding` 与 `vision`/`tools` 互斥，`reasoning` 与 `embedding` 互斥。

能力信息显示在[管理面板](admin-panel.md)中，也可在面板中编辑。

## 网关 API Key

通过网关级 API Key 保护 AI 请求端点：

```jsonc
{
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "api_key": "my-secret-gateway-key"
  }
}
```

配置后，所有 `/v1/*` 端点需要使用对应 API 标准的原生格式进行认证：

| API 标准 | 凭证格式 |
|---------|---------|
| OpenAI Chat / Responses | `Authorization: Bearer <key>` |
| Anthropic | `x-api-key: <key>` |
| Google GenAI | `x-goog-api-key: <key>` 或 `?key=<key>` 查询参数 |

API Key 也支持 `${ENV_VAR}` 替换：

```jsonc
"api_key": "${GATEWAY_API_KEY}"
```

!!! note "管理面板"
    管理面板（`/admin/*`）**不需要**网关 API Key。如需保护管理面板，请使用反向代理（如 Caddy 的 `basicauth`、Nginx 的 `auth_basic`）。

未配置 `api_key` 时，所有请求无需认证直接通过（向后兼容）。

## 调试选项

```jsonc
{
  "debug": {
    "verbose": true,       // 启用 DEBUG 级别日志
    "log_bodies": true     // 记录完整的请求/响应体
  }
}
```

也可以通过环境变量设置：`LLM_ROSETTA_VERBOSE=1`、`LLM_ROSETTA_LOG_BODIES=1`。

## 完整示例

```jsonc
{
  "providers": {
    "openai-prod":    { "type": "openai_chat",      "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "openai-resp":    { "type": "openai_responses",  "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "anthropic-prod": { "type": "anthropic",         "api_key": "${ANTHROPIC_API_KEY}",  "base_url": "https://api.anthropic.com" },
    "google-prod":    { "type": "google",            "api_key": "${GOOGLE_API_KEY}",     "base_url": "https://generativelanguage.googleapis.com" },
    // 基于 Shim 的提供商 — base_url 和 transforms 自动解析
    "deepseek":       { "shim": "deepseek",          "api_key": "${DEEPSEEK_API_KEY}" },
    "volcengine":     { "shim": "volcengine",         "api_key": "${VOLCENGINE_API_KEY}", "base_url": "https://ark.cn-beijing.volces.com/api/v3" }
  },
  "models": {
    "gpt-4o":                     { "provider": "openai-prod",    "capabilities": ["text", "vision", "tools"] },
    "claude-sonnet-4-20250514":   { "provider": "anthropic-prod", "capabilities": ["text", "vision", "tools"] },
    "gemini-2.0-flash":           { "provider": "google-prod",    "capabilities": ["text", "tools"] },
    "deepseek-r1":                { "provider": "deepseek",       "capabilities": ["text", "tools"] }
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "api_key": "${GATEWAY_API_KEY}"
  }
}
```
