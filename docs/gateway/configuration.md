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

!!! note "向后兼容"
    如果省略 `type`，提供商名称本身将用作类型。这意味着使用旧格式（提供商名称为 `openai_chat`、`anthropic` 等）的配置无需修改即可继续使用。

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

可用能力：`text`、`vision`、`tools`。如未指定，默认为 `["text"]`。

能力信息显示在[管理面板](admin-panel.md)中，也可在面板中编辑。

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
    "google-prod":    { "type": "google",            "api_key": "${GOOGLE_API_KEY}",     "base_url": "https://generativelanguage.googleapis.com" }
  },
  "models": {
    "gpt-4o":                     { "provider": "openai-prod",    "capabilities": ["text", "vision", "tools"] },
    "claude-sonnet-4-20250514":   { "provider": "anthropic-prod", "capabilities": ["text", "vision", "tools"] },
    "gemini-2.0-flash":           { "provider": "google-prod",    "capabilities": ["text", "tools"] }
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8765
  }
}
```
