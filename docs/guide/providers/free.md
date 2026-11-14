# 免密钥免费端点

有四家厂商提供无需账号的免费池，且都没有合同约束：模型清单、速率限制甚至端点本身
都可能随时变更或关闭。因此它们作为独立包发布：
[llm-rosetta-shim-free](https://github.com/Oaklight/llm-rosetta-shim-free)——
某个池消失时可以在不发版的情况下撤下。

| Shim 名称 | 厂商 | 默认 Base URL | 免费范围 |
|:---|:---|:---|:---|
| `kilo_free--chat_completions` | Kilo Code | `https://api.kilo.ai/api/gateway` | 标记 `isFree` 的模型 |
| `llm7--chat_completions` | LLM7.io | `https://api.llm7.io/v1` | `turbo` 档 |
| `opencode_zen_free--chat_completions` | OpenCode (Zen) | `https://opencode.ai/zen/v1` | 零成本模型 |
| `ovhcloud--chat_completions` | OVHcloud AI Endpoints | `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1` | 整个目录 |

四个都是 OpenAI Chat Completions 线路，都声明 `keyless`，都带
`free_source: true`，因此在管理台中归入 **Free Resource** 分组。

!!! warning "免费不等于私密，也不等于无限"
    这些上游都声明免费额度的 prompt 可能被记录。切勿通过免费源发送密钥、凭据或敏感数据。

## 声明该包

```jsonc
{
  "shims": { "presets": ["free"] },
  "providers": {
    "kilo-free": { "type": "kilo_free--chat_completions" },
    "llm7-free": { "type": "llm7--chat_completions" },
    "ovh-free": { "type": "ovhcloud--chat_completions" }
  }
}
```

`free` 取代了此前的 `opencode_zen_free` 预设。OpenCode 的 shim 名称未变，因此针对旧包
写的配置只需改动预设声明。

## 速率限制

| 厂商 | 匿名限制 | 提供 key 后 |
|:---|:---|:---|
| Kilo | 每 IP 每小时 200 次请求 | key 会把同一 shim 切换到 Bearer 认证 |
| LLM7 | `turbo` 档 | 免费 token 把上限提高到 60 RPM / 每小时 250 次请求 |
| OpenCode Zen | 受限于 OpenCode 客户端的请求形态 | — |
| OVHcloud | 每 IP、每模型每分钟 2 次请求 | Public Cloud key 提高到每项目每模型 400 RPM |

## 各 shim 修正了什么

- **推理字段。** LLM7 与 OVHcloud 把推理文本放在 `message.reasoning` /
  `delta.reasoning`，而转换器读取 `reasoning_content`。两者各带一个改名转换，
  在响应不含推理时是空操作。Kilo 的免费线路不带改名：它的推理字段尚未实测。
- **模型清单。** Kilo 与 OpenCode 的目录把付费模型与免费模型列在一起，因此它们的
  `model_list_transform` 只保留免费子集。LLM7 的目录带类型与档位。OVHcloud 的目录
  没有类型字段，整份提供。
- **请求形态。** OpenCode 的线路会拒绝未声明名为 `read` 工具的请求；shim 注入一个
  占位声明，并从响应中移除对它的调用。

## Kilo 的付费线路

Kilo 是唯一同时出现在两侧的厂商。付费网关是内置 shim `kilo--chat_completions`，
始终发送 `KILO_API_KEY` 并开放完整目录；参见 [Kilo](kilo.md)。
