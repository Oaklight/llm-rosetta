# SiliconFlow

硅基流动在同一凭据下托管开源权重模型，chat、embedding、rerank 都在同一主机上。
目录中有一部分模型永久免费。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `siliconflow--chat_completions` | `chat_completions` | `https://api.siliconflow.cn/v1` | `SILICONFLOW_API_KEY` |
| `siliconflow--embedding` | `embeddings`（`openai`） | `https://api.siliconflow.cn/v1` | `SILICONFLOW_API_KEY` |
| `siliconflow--rerank` | `rerank`（`cohere`） | `https://api.siliconflow.cn/v1` | `SILICONFLOW_API_KEY` |

## 带类型的模型目录

普通的 `GET /v1/models` 不含类型字段，选择器无法区分 chat 模型与图像模型。
SiliconFlow 在另一台主机上提供带类型的目录：

```
https://busy-bear.siliconflow.cn/api/v1/playground/comprehensive/all
```

每条记录带 `type`、`subType`，以及 `normal` 或 `disable` 的 `status`。
撰写时实测：201 条记录，91 条启用，其中 `subType: chat` 61 条、
`embedding` 8 条、`reranker` 6 条。

shim 在每个叶子上以绝对 URL 形式声明该 `models_path`，每个叶子各带一个
`model_list_transform`，只保留自己子类型中启用的条目。模型 id 为 `modelName`，
即端点接受的字符串。

## 已实测的端点

- chat 返回 OpenAI 结构。
- embedding（`BAAI/bge-m3`）返回 `data[].embedding` 浮点数组。
- rerank（`BAAI/bge-reranker-v2-m3`）接受 Cohere 请求结构，
  返回 `results[].index/relevance_score`。

## 网关配置

```jsonc
{
  "providers": {
    "sf": {
      "type": "siliconflow--chat_completions",
      "api_key": "${SILICONFLOW_API_KEY}"
    },
    "sf-rerank": {
      "type": "siliconflow--rerank",
      "api_key": "${SILICONFLOW_API_KEY}"
    }
  }
}
```

## 开通条件

免费模型不需要信用卡，但需要实名认证（支持中国大陆证件）。国际用户需联系客服。
