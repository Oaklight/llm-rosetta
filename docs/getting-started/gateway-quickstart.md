---
title: 网关快速开始
---

# 网关快速开始

5 分钟内启动一个格式转换 HTTP 代理。

## 1. 安装

```bash
pip install "llm-rosetta[gateway]"
```

或使用 Docker（Alpine 二进制版，~21 MB）：

```bash
docker pull oaklight/llm-rosetta-gateway:latest
```

也可以从 [GitHub Releases](https://github.com/Oaklight/llm-rosetta/releases) 下载独立二进制文件 — 无需 Python。

## 2. 创建配置文件

创建 `config.jsonc`（支持注释的 JSON）：

```jsonc
{
  "providers": {
    "my-openai":    { "type": "openai_chat",      "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "my-anthropic": { "type": "anthropic",         "api_key": "${ANTHROPIC_API_KEY}",  "base_url": "https://api.anthropic.com" },
    "my-google":    { "type": "google",            "api_key": "${GOOGLE_API_KEY}",     "base_url": "https://generativelanguage.googleapis.com" }
  },
  "models": {
    "gpt-4o": "my-openai",
    "gpt-4o-mini": "my-openai",
    "claude-sonnet-4-20250514": "my-anthropic",
    "gemini-2.0-flash": "my-google"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8765
  }
}
```

提供方名称是用户自定义的字符串。`type` 字段指定 API 标准（`openai_chat`、`openai_responses`、`anthropic`、`google`）。

## 3. 启动网关

```bash
llm-rosetta-gateway
```

网关会按以下顺序自动搜索配置文件（首个匹配生效）：

1. `./config.jsonc`（当前目录）
2. `~/.config/llm-rosetta-gateway/config.jsonc`
3. `~/.llm-rosetta-gateway/config.jsonc`

也可以显式指定配置文件：

```bash
llm-rosetta-gateway --config /path/to/config.jsonc
```

## 4. 发送请求

使用任意提供方格式 — 网关根据模型名称自动路由：

```bash
# 发送 OpenAI 格式请求，路由到 Anthropic
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

流式传输也支持 — 只需添加 `"stream": true`。

## 下一步

- [配置](../gateway/configuration.md) — 完整配置参考
- [CLI 参考](../gateway/cli.md) — 所有 CLI 选项和子命令
- [CLI 工具集成](../gateway/cli-integrations.md) — 配合 Claude Code、Codex、Gemini CLI 使用
- [管理面板](../gateway/admin-panel.md) — Web UI 配置与监控
