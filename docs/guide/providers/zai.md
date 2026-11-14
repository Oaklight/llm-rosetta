# Z.ai

Z.ai 是智谱 AI 的国际站。它与国内 [Zhipu](zhipu.md) 线路提供同一 GLM 模型族，
但主机与凭据不同，中国大陆以外应使用这一线路。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `zai--chat_completions` | `chat_completions` | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` |
| `zai--messages` | `messages` | `https://api.z.ai/api/anthropic` | `ZAI_API_KEY` |
| `zai--embedding` | `embeddings`（`openai`） | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` |
| `zai--rerank` | `rerank`（`cohere`） | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` |

一个 Key 覆盖全部线路。

## 各线路的声明

- **Chat Completions** 将 `temperature` 钳制到 `[0, 1]`：上游对 `2` 返回 400，
  转换时钳制并在 IR 路径上报告改动。`top_p` 不做限制——与国内线路不同，`0` 可用。
  `n`、`seed`、`logprobs` 均被接受，因此不声明字段剥离。
- **Messages** 兼容 Anthropic 协议：返回带 `signature` 的 `thinking` 块，
  接受 `thinking` 与 `output_config.effort`。该线路没有 adaptive 模式，故不映射 `auto`。
- **Embeddings** 使用 OpenAI 请求与响应结构（`embedding-3` 返回浮点数组）。
- **Rerank** 使用 Cohere 结构，返回 `results[].index/relevance_score`，不含 `document`。

## 网关配置

```jsonc
{
  "providers": {
    "zai": {
      "type": "zai--chat_completions",
      "api_key": "${ZAI_API_KEY}"
    },
    "zai-claude": {
      "type": "zai--messages",
      "api_key": "${ZAI_API_KEY}"
    }
  }
}
```

## 免费额度

Z.ai 上部分 GLM 模型定价为零（`glm-4.7-flash`、`glm-4.6v-flash`）。
服务过载时免费线路可能返回 `429`，应重试，而不是当作额度耗尽。
