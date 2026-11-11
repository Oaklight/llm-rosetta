---
title: Open Design
---

# Open Design

Open Design Cloud 在一个 HTTPS 网关上提供其模型。三个 shim 分别暴露该网关接受的三种
聊天 API 标准；它们共用一个 base URL，只有请求体和凭据头不同。

## Shims

| Shim 名称 | Base 类型 | 默认 Base URL | 凭据 |
|:---|:---|:---|:---|
| `opendesign--chat_completions` | `chat_completions` | `https://amr-link.open-design.ai/v1` | `OPENDESIGN_API_KEY` |
| `opendesign--open_responses` | `open_responses` | `https://amr-link.open-design.ai/v1` | `OPENDESIGN_API_KEY` |
| `opendesign--messages` | `messages` | `https://amr-link.open-design.ai/v1` | `OPENDESIGN_API_KEY` |

模型列表地址为 `https://amr-link.open-design.ai/v1/models`。

Open Design 在三种 API 上都提供每个模型，因此按客户端使用的格式选择即可。shim 会按端点要求
签署请求：

- `/chat/completions` 与 `/responses` 发送 `Authorization: Bearer sk-amr-…`。
- `/messages` 发送 `x-api-key: sk-amr-…` 和 `anthropic-version: 2023-06-01`。

## 独立包

这些 shim 位于
[`llm-rosetta-shim-opendesign`](https://github.com/Oaklight/llm-rosetta-shim-opendesign)，
不在核心包内。使用这些名称前需先声明预设：

```jsonc
{
  "shims": { "presets": ["opendesign"] }
}
```

## 网关配置

```jsonc
{
  "shims": { "presets": ["opendesign"] },
  "providers": {
    "opendesign": {
      "type": "opendesign--chat_completions",
      "api_key": "${OPENDESIGN_API_KEY}"
    }
  },
  "models": {
    "deepseek-v4.1-flash": { "provider": "opendesign" }
  }
}
```

厂商清单提供 base URL、`OPENDESIGN_API_KEY` 环境变量默认值和模型列表地址；提供方条目可以
覆盖其中任何一项。

## 额度

此 shim 不读取 Coding Plan 额度。Open Design 在单独的控制面上报告该额度，而控制面不接受
运行时 API Key；控制 Key 由设备授权流程签发，网页控制台并不显示。网关会将该提供方的额度
报告为不支持。跟踪议题：[#1499](https://github.com/Oaklight/llm-rosetta/issues/1499)。

## 状态

端点、按格式区分的请求头和模型列表路径遵循 Open Design Cloud 的公开接口。本项目的测试尚未
对真实账户进行现场验证。
