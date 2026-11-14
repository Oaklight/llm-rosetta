# Cohere

Cohere 同时提供原生 API 与 OpenAI 兼容 API。embedding 与 rerank 使用原生 v2 结构；
chat 走兼容端点，使用 OpenAI Chat Completions 结构。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `cohere--chat_completions` | `chat_completions` | `https://api.cohere.com/compatibility/v1` | `COHERE_API_KEY` |
| `cohere--embedding` | `embeddings`（`cohere`） | `https://api.cohere.com` | `COHERE_API_KEY` |
| `cohere--rerank` | `rerank`（`cohere`） | `https://api.cohere.com` | `COHERE_API_KEY` |

chat 叶子覆盖厂商清单里的原生 `base_url`；凭据只在清单上声明一次，其余线路继承。

## 说明

- chat 与 tool call 在 `api.cohere.com` 与 `api.cohere.ai` 上均可用；
  shim 选用 `.com`，使全部线路共用一个主机。
- `/compatibility/v1/models` 返回与原生 API 相同的目录。模型列表钩子读取 `name`
  而不是 `id`，因为 Cohere 自己的列表使用该字段。
- embedding 必须提供 `input_type`；shim 默认填 `retrieval_document`，
  客户端自己的 `task_type` 仍然优先。

## 网关配置

```jsonc
{
  "providers": {
    "cohere": {
      "type": "cohere--chat_completions",
      "api_key": "${COHERE_API_KEY}"
    },
    "cohere-embed": {
      "type": "cohere--embedding",
      "api_key": "${COHERE_API_KEY}"
    }
  }
}
```
