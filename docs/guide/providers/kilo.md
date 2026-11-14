---
title: Kilo
---

# Kilo

Kilo Code 运营一个公共模型网关。我们用两个 shim 分别对接，这个拆分是有意的：
付费线路发送凭据并开放完整目录，免费池不发送凭据、只开放上游标记为免费的模型。

| Shim 名称 | 基础类型 | 默认 Base URL | 凭据 | 所在位置 |
|:---|:---|:---|:---|:---|
| `kilo--chat_completions` | `chat_completions` | `https://api.kilo.ai/api/gateway` | `KILO_API_KEY` | 核心包 |
| `kilo_free--chat_completions` | `chat_completions` | `https://api.kilo.ai/api/gateway` | 无（keyless） | [llm-rosetta-shim-free](https://github.com/Oaklight/llm-rosetta-shim-free) |

## 付费线路

`kilo--chat_completions` 始终携带凭据。未配置 key 的 provider 会在启动时被拒绝，
而不是无凭据发往上游；它的模型列表是账号可访问的完整目录。两个 shim 都声明
`group_label: Kilo`，因此在选择器中归到同一个厂商下。

```jsonc
{
  "providers": {
    "kilo": {
      "type": "kilo--chat_completions",
      "api_key": "${KILO_API_KEY}"
    }
  }
}
```

## 免费池

`kilo_free--chat_completions` 是 keyless 的，带 `free_source: true`，发布在核心包之外
——免费池归上游所有，可能随时变更或关闭。使用前先声明包：

```jsonc
{
  "shims": { "presets": ["free"] },
  "providers": {
    "kilo-free": { "type": "kilo_free--chat_completions" }
  }
}
```

!!! warning "免费不等于私密，也不等于无限"
    免费池声明 prompt 可能被上游服务方记录。切勿通过免费源发送密钥、凭据或敏感数据。
    模型清单与配额政策归上游所有，随时可能变更或消失。

### 模型清单

上游 `GET /models` 返回完整目录，并用 `isFree: true` 标记免费子集。免费 shim 的
`model_list_transform` 只暴露该子集；付费 shim 不做过滤。

在管理台中，**Fetch Models** 拉取清单，provider 卡片上的 **Refresh** 重新拉取、
报告差异并为新增项提供一键应用。Refresh 从不静默增删模型。

系统**不会**在服务端复查被路由的模型是否仍然免费——路由表由运维掌控。若上游下线
某模型，针对它的请求会在上游失败，需要从路由表中移除。

### 速率限制

免费池每 IP 每小时 200 次请求。给免费 shim 配置 `api_key` 会切换为
`Authorization: Bearer <key>`，这是在免密钥线路上使用自己 Kilo 账号的受支持方式；
但要开放完整目录，应使用付费 shim。

## 聚合多个免费源

每个免费源都是独立的 provider 条目（`kilo-free`，以后也可以有 OVHcloud 或 LLM7），
统一归在 **Free Resource** 分组下——该分组只在至少配置了一个免费源后出现。同一个模型
若有两个免费源，可配置为一个模型加两个 provider 和一种策略：

```jsonc
{
  "models": {
    "some-free-model": {
      "providers": ["kilo-free", "ovh-free"],
      "strategy": "affinity_round_robin"
    }
  }
}
```

网关的多 provider 路由支持 `weighted_round_robin` 与 `affinity_round_robin` 两种策略。
