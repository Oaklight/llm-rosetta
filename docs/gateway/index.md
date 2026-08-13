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

初次使用？从[网关快速开始](../getting-started/gateway-quickstart.md)入手。

## 端点

| 路径 | 来源格式 | 说明 |
|------|---------|------|
| `POST /v1/chat/completions` | OpenAI Chat | 兼容 OpenAI SDK |
| `POST /v1/messages` | Anthropic | 兼容 Anthropic SDK |
| `POST /v1/responses` | OpenAI Responses | 兼容 OpenAI Responses SDK |
| `POST /v1/embeddings` | OpenAI Embeddings | 通过 IR 跨格式转换（OpenAI、Cohere、Jina、Voyage） |
| `POST /v1/rerank` | Rerank（默认 Jina） | 通过 IR 跨格式转换（Jina、Cohere、Voyage） |
| `POST /v2/rerank` | Rerank（Cohere） | 自动检测 Cohere 格式；与 `/v1/rerank` 共用处理器 |
| `POST /v1beta/models/{model}:generateContent` | Google GenAI | 兼容 Google REST API |
| `POST /v1beta/models/{model}:streamGenerateContent` | Google GenAI（流式） | 兼容 Google 流式 API |
| `GET /v1/models` | OpenAI / Anthropic | 列出已配置模型，含 `api_standard` 和 `capabilities` |
| `GET /v1beta/models` | Google GenAI | 列出已配置模型（Google SDK 格式） |
| `GET /health` | — | 健康检查 |
| `GET /admin/` | — | [管理面板](admin-panel.md)（Web UI） |

端点路径决定了来源格式 — 无需自动检测。

## 流式传输

所有提供商组合均支持流式传输。请求方式与原生 API 相同：

```bash
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

网关实时在提供商格式之间转换 SSE 数据块。

## 认证

通过 `server` 配置中的网关 API Key 保护 AI 端点：

```jsonc
"server": { "api_key": "my-secret-key" }
```

请求必须以对应 API 标准的原生格式提供 Key（Bearer token、`x-api-key` 头部等）。详见[配置 — 网关 API Key](configuration.md#网关-api-key)。

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
