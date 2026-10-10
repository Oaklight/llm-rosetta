---
title: Kilo
---

# Kilo

Kilo 的公共网关提供一个免费模型池。LLM-Rosetta 将其作为一个无需凭据的普通 provider
暴露，在管理台中显示为 **Kilo (Free)**。

!!! warning "免费不等于私密，也不等于无限"
    免费池声明 prompt 可能被上游服务方记录。切勿通过免费源发送密钥、凭据或敏感数据。
    模型清单与配额政策归上游所有，随时可能变更或消失。

## Shim

| Shim 名称 | 基础类型 | 默认 Base URL | 凭据 |
|:---|:---|:---|:---|
| `kilo--openai_chat` | `openai_chat` | `https://api.kilo.ai/api/gateway` | 无（keyless） |

该 shim 声明了 `display_name: Kilo`、`free_source: true`、一个 logo，以及
`connection.keyless: true`：

- **Keyless** —— 未配置 `api_key` 时，请求不携带任何鉴权头，该条目显示为
  **Kilo (Free)**，归入 *Free Resource* 分区。
- **Key 可选** —— 提供 `api_key`（或 `api_key_env`）后，同一 shim 切换为
  `Authorization: Bearer <key>`。此时它是一个普通 provider：以你起的名字显示，
  归入 *Providers* 分区，而不是 *Free Resource*。

## 模型清单

上游 `GET /models` 会列出全部模型，并用 `isFree: true` 标记免费切片。shim 的
`model_list_transform` **仅**暴露该切片。

清单是发现出来的，而非硬编码。在管理台中，**Fetch Models** 拉取清单，provider 卡片上的
**Refresh** 按钮会重新拉取、报告差异（`{total}` 个免费模型、`{new}` 个新增），并把新增项
提供给一键应用。Refresh 不会静默地增删模型。

服务端**不会**重新校验某个已路由模型是否仍然免费 —— 路由表由运维控制。若上游撤下了某个
模型，对它的请求会在上游失败；请将其从路由表中移除。

## 网关配置

```jsonc
{
  "providers": {
    "kilo-free": {
      "type": "kilo--openai_chat"
      // 无需 api_key —— 免费池不需要凭据。
      // 填入 "api_key": "${KILO_API_KEY}" 可改用你自己的 Kilo 账号。
    }
  }
}
```

也可以使用管理台中的 **+ Kilo (Free)** 预设，它会创建该条目，并把可选的 API Key 字段
折叠到 "Use my own key" 链接之后。

## 隐私

免费模型流量会发往 Kilo。管理台会显示一条中性披露 ——
*"Served by an external provider; prompts may be logged by that provider."*

## 聚合多个免费源

每个免费源都是独立的 provider 条目（`kilo-free`，以后还有 OpenCode 等），它们统一归入
**Free Resource** 分区 —— 只有至少配置了一个这样的 provider 时该分区才会出现。当同一个
模型在两个免费源上都可用时，把它配置为一条带多个 provider 的路由：

```jsonc
{
  "models": {
    "some-free-model": {
      "providers": ["kilo-free", "opencode-free"],
      "strategy": "affinity_round_robin"
    }
  }
}
```

网关的多 provider 路由支持 `weighted_round_robin` 和 `affinity_round_robin` 两种策略。
