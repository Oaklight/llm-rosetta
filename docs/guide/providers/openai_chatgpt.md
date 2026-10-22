---
title: OpenAI（ChatGPT 套餐）
---

# OpenAI（ChatGPT 套餐）

`openai_chatgpt--open_responses` shim 对接的是 ChatGPT 的**套餐**通道：消耗
Go/Plus/Pro 套餐额度，而不是 Platform API 计费。它使用与 `openai--open_responses`
相同的 [Open Responses](../formats.md) 传输格式，因此请求的转换与其他 Responses
请求完全一致；不同的只是凭据从哪来、能看到哪些模型，以及这条通道拒绝什么。

!!! warning "它不是 API Key 的替代品"
    套餐通道是 OpenAI 面向本地与开源应用提供的集成方式。商业用途或远程托管的部署需要
    先加入 OpenAI 的等待名单。本页不改变 OpenAI 对你的计费方式。

## Shim

| Shim 名称 | 基础类型 | 默认 Base URL | 凭据 |
|:---|:---|:---|:---|
| `openai_chatgpt--open_responses` | `open_responses` | `https://api.openai.com/v1` | 无 —— 通过登录取得 |

这里刻意没有 `api_key_env`：凭据由 shim 自带的 `auth.py` 生成。若在声明了交换却
`consumes_key=false` 的提供方旁配置 `api_key`，加载配置时会被拒绝 —— 两者是同一个凭据的
两种来源。

## 前置条件

- 登录所用账号所在的套餐支持这条通道（Go、Plus 或 Pro）。
- 一台运行网关的主机，以及**首次登录时的一个浏览器**。登录不必发生在网关主机上 ——
  见[登录](#_3)。
- 一个持久的数据目录：凭据记录保存在 `<data_dir>/shim-auth/<provider>`，必须能跨重启保留。

## 配置

1. 添加一个类型为 `openai_chatgpt--open_responses` 的提供方，不需要 API Key。
2. 完成登录（见下）。在拿到记录之前，该提供方会显示“需要交互式登录”，并且不会在每次
   请求时重试。
3. 拉取模型列表 —— 通道返回的是 `{models: [{slug, …}]}` 而不是 `{data: [...]}`，shim 会
   保留其中标记为 `visibility: "list"` 的条目。
4. 发送流式请求。

```jsonc
{
  "providers": {
    "chatgpt": {
      "type": "openai_chatgpt--open_responses",
      "base_url": "https://api.openai.com/v1"
    }
  }
}
```

## 登录

登录需要一个浏览器和一个 loopback 监听，因此它在有浏览器的一侧完成 —— 不一定是网关主机：

```bash
llm-rosetta-gateway credential chatgpt --out record.json
```

该命令运行 shim 的交互式登录，并把凭据记录以 `0600` 权限写入文件。请尽快把记录交给网关
—— ID token 在收到时即刻验签，放了一夜的记录会被拒绝，只能重新登录：

- **管理台** → Providers → 该提供方 → **凭据** → **安装记录**。管理台同时显示已登录的
  账号、访问令牌的到期时间、**管理用量**（shim 声明了页面时）以及 **退出登录**。
- **API**：`PUT /admin/api/config/providers/<name>/credential`，请求体即记录本身。

安装成功后会立即重新填充运行中的提供方，无需重启。重复登录是安全的：轮换的 refresh token
会被替换，而账号身份（验签后的 `sub`）不变。

## 网关实际发送的内容

这条通道接受的 Responses 字段远少于格式本身允许的范围。shim 的请求变换会去掉它拒绝的
字段，而不是让请求失败，同时强制它要求的字段：

| 字段 | 处理方式 |
|:---|:---|
| `store` | 强制为 `false`（通道要求如此，无论是否发送） |
| `previous_response_id` | 去掉 —— 通道不保留服务端状态，历史必须放在 `input` 里 |
| `temperature`、`top_p`、`max_output_tokens`、`top_logprobs` | 去掉 |
| `background`、`conversation`、`max_tool_calls`、`metadata`、`moderation`、`multi_agent`、`prompt`、`prompt_cache_retention`、`safety_identifier`、`truncation`、`user` | 去掉 |

被去掉的字段会出现在请求日志的已应用变换列表里（`drop_refused_fields(...)`），请求本身
仍然成功。`instructions` 承载系统提示，`input` 承载完整历史，函数工具按通道期望的命名空间
发送。

**必须流式。** 通道只返回 SSE，因此即使客户端只要一次完整响应，网关也会强制上游
`stream: true`。非流式客户端会在任何上游调用之前收到 `400` 并指明需要 `stream=true`：
把 Responses 流重新组装成一个 JSON 响应体尚未实现
（[#1061](https://github.com/Oaklight/llm-rosetta/issues/1061)）。多数 agent 客户端都是
流式的，因此这只影响简单的调用方。

## 限制

- **除网页搜索外没有托管工具。** 代码解释器、文件搜索、图像生成、计算机操作、托管 MCP 与
  `tool_search` 都不在声明之列，因为通道会拒绝它们；声明它们等于宣称提供不了的能力。
  `web_search` 工具可用。
- **无法控制输出长度** —— 被去掉的 `max_output_tokens` 网关无法代为执行。
- **套餐窗口**：Plus 的 5 小时窗口在多个应用间共享，每个应用另有周上限。上限在你的
  ChatGPT 设置里、不在 API 后面，所以管理台只能给出链接。
- **非流式客户端**会被拒绝（见上）。

## 错误

流开始**之后**才到达的套餐类失败会保留其机器可读的错误码 —— 网关用它终止客户端的流，
而不是笼统的“stream failed”，因为恢复方式不同：

| 错误码 | 状态 | 处理 |
|:---|:---|:---|
| `subscription_sharing_usage_limit_exceeded` | 429 | 暂停套餐请求；查看用量页 |
| `subscription_sharing_user_not_eligible` | 403 | 账号不在该通道 —— 不要重试，也不要重跑登录 |
| `subscription_sharing_usage_unavailable` | 503 | 有上限的退避；保留凭据 |
| `subscription_sharing_unsupported_capability` | 400 | 查看 `error.param`，去掉该字段或工具 |
| `subscription_sharing_route_not_supported` | 403 | 端点或方法不对 |
| `subscription_sharing_invalid_user` | 401 | 凭据失效 —— 重新登录 |
| `subscription_sharing_user_unavailable` | 503 | 有上限的退避 |
| `chatpass_v2_scope_not_authorized`、`chatpass_v2_invalid_authorization_context` | 403 | 授权或配置问题 —— 不要重试 |

在 Responses 层**之前**就被拒绝的请求会返回普通的 `{"detail": "…"}` 响应体；网关保留
该状态码、响应体与 request id，不做改写。

## 安全

- 记录以 `0600` 写在 `0700` 目录下，原子写入，并持有排他锁。每次读写都取锁，因为 refresh
  token 用一次就作废：两次并发刷新会把它轮换两次并丢掉账号。
- 令牌不会出现在管理台、请求日志或凭据列表里；管理台只显示交换摘要允许的账号与到期时间。
- **退出登录**会在厂商允许时于上游撤销授权，并立刻把令牌从网关的 key ring 中移除。手工
  删除记录目录会让网关以为自己仍然连着，直到下一次刷新失败 —— 请使用管理台或
  `DELETE …/credential`。

## 故障排查

| 现象 | 原因 | 处理 |
|:---|:---|:---|
| 启动时提示“需要交互式登录” | `<data_dir>/shim-auth/<provider>` 下没有记录 | 完成登录并安装记录 |
| 粘贴时记录被拒绝 | ID token 已过期，或属于另一个客户端 | 重新登录并尽快粘贴 |
| `subscription_sharing_user_not_eligible` | 账号不在该通道 | 换用有资格的账号；这不是瞬时错误 |
| 几次请求后出现 `usage_limit_exceeded` | 套餐窗口或应用的周上限 | 暂停；打开用量页 |
| 登录后请求仍然 401 | 记录装在了别的提供方名下 | 两个提供方就是两个账号，需分别安装 |
