---
title: OpenCode
---

# OpenCode

OpenCode 的付费服务是同一个凭据下的两条通道，只有 base URL 不同：

| 通道 | Base URL | 计费 | `x-opencode-session` |
|:---|:---|:---|:---|
| **Go** | `https://opencode.ai/zen/go/v1` | 月度订阅 | 必需 |
| **Zen** | `https://opencode.ai/zen/v1` | 账户余额 | 不要求 |

两条通道在同一个 base 下都提供三种 API 标准，因此 LLM-Rosetta 暴露三个 shim，
默认走 Go 通道。要用 Zen，在 provider 条目上覆盖 `base_url` 即可。

## Shims

| Shim 名称 | Base 类型 | 默认 Base URL | 凭据 |
|:---|:---|:---|:---|
| `opencode--chat_completions` | `chat_completions` | `https://opencode.ai/zen/go/v1` | `OPENCODE_API_KEY` |
| `opencode--messages` | `messages` | `https://opencode.ai/zen/go/v1` | `OPENCODE_API_KEY` |
| `opencode--open_responses` | `open_responses` | `https://opencode.ai/zen/go/v1` | `OPENCODE_API_KEY` |

三者分别解析到 base 下各自格式的路径：`/chat/completions`、`/messages`
（key 通过 `x-api-key` 发送）和 `/responses`。并非每个模型都在每个端点上——
具体哪个模型走哪个端点见 [Go 文档](https://opencode.ai/docs/go/)。

## Session 头

Go 通道会以 `400 MissingSessionID` 拒绝任何不带 `x-opencode-session` 的推理请求；
上游用它来路由，并为该会话绑定一个对 prompt cache 友好的后端。这些 shim 声明
`connection.session_header: x-opencode-session`，由网关填充：

- 默认填充一个**按会话派生**的值——对 key 选择所用的同一个 affinity identity
  （客户端 key 加消息前缀）做哈希，因此同一会话的每一轮都复用同一个 session，
  不同会话则得到不同的值；
- 若请求本身已带该头，则原样透传客户端的值——对会发送原生 session id 的客户端
  （如 Codex）很有用。

Zen 会忽略该头。

## 网关配置

Go 通道（默认）：

```jsonc
{
  "providers": {
    "opencode": { "shim": "opencode--chat_completions" }
  },
  "models": {
    "glm-5.3": "opencode"
  }
}
```

Zen 通道——同一个 shim 覆盖 `base_url`：

```jsonc
{
  "providers": {
    "opencode-zen": {
      "shim": "opencode--chat_completions",
      "base_url": "https://opencode.ai/zen/v1"
    }
  },
  "models": {
    "glm-5.3": "opencode-zen"
  }
}
```

在网关进程的环境里设置 `OPENCODE_API_KEY`——两条通道用的是同一个凭据。

## 额度

Go 订阅的窗口（5 小时 / 每周 / 每月）可以按 key 读取，显示在 provider 卡片上，
并用于跳过已耗尽的凭据。见[提供方额度](../quota.md)。

## 公共免费通道 {#public-free-lane}

keyless 的 `/zen/v1` 免费池是一个独立 shim：
`opencode_zen_free--chat_completions`。上游会拒绝任何不像它自家桌面客户端的请求，
所以该 shim 在 `connection` 块和 `transforms.py` 里复现这个形状：

- `opencode/<version>` 的 `User-Agent`，以及 `x-opencode-client` /
  `x-opencode-project` 标记；
- `x-opencode-session` 的值可以解码回最近的时间戳（`session_id_format: opencode`），
  外加每个请求独立的 `x-opencode-request`；
- 恰好一个名为 `read` 的工具声明——仅在客户端没有声明时注入。该声明明确写着自己只是
  传输占位符、模型不应调用，客户端自己的工具才是可用的；对它的调用会被移除而不是
  转发，若一轮里所有调用都被移除，`finish_reason` 会改为 `stop`，客户端不会一直等待
  一个永远不会到达的结果；
- `stream: true`（`stream_required`），非流式回复由网关重新拼装。

它的模型列表会过滤为 `-free` 子集：lane 的 `/models` 返回整个 Zen 目录，但它只服务
id 以 `-free` 结尾的模型，因此面板只提供这些。

这是反滥用网关，而不是公共池：它可能随时变化或关闭，额度按出网地址块计（IPv6 一个
`/64`，IPv4 一个 `/32`），每天重置，prompt 也可能被上游记录。该 provider 会出现在
**Free Resource** 分组下，并带上面板的第三方声明。测量过程见
[issue #884](https://github.com/Oaklight/llm-rosetta/issues/884)；
[opencode-unlimited-proxy](https://github.com/ebouther/opencode-unlimited-proxy)
另有一份关于各道门与额度维度的独立记录。

```jsonc
{
  "providers": {
    "zen-free": { "type": "opencode_zen_free--chat_completions" }
  },
  "models": {
    "beam-free": { "provider": "zen-free" }
  }
}
```

不需要任何 key。Go 订阅本身也通过付费通道提供两个 *Preview Free* 模型。
