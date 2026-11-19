---
title: Google（Antigravity）
---

# Google（Antigravity）

`antigravity--acp` shim 通过厂商自带的 `agy` CLI 使用 Google **Antigravity**
套餐。它不调用 Google 的 Code Assist API：网关在主机上启动 `agy acp`，通过
stdio 与之讲 [Agent Client Protocol](https://agentclientprotocol.com/)，因此
Google 登录、访问 token 与 project 都由 `agy` 保管，与你亲自运行 `agy` 时完全一样。

!!! warning "在把账号指过来之前先读这一段"
    Google 的条款决定谁可以使用该套餐，并且 Google 可能封禁被判定在 Antigravity
    自家客户端之外使用的账号。驱动 `agy` 是从网关使用该套餐最不绕路的方式 —— 它
    本身就是厂商的客户端 —— 但这仍然是一个可能失去的套餐。请使用可以承受损失的账号，
    也不要把这条通道开放给你无法支持的最终用户。

## 为什么用 `agy` 而不是直接调 API

用 Antigravity 自己的 OAuth client 直接调 Code Assist，是 magpie 早先的做法，
而它已经撤回了这条路：Google 会封禁那些被发现在 Antigravity 客户端之外调用
Code Assist 的账号（magpie 源码 `internal/plugin/withdrawn.go`）。magpie 推荐的
替代方案 —— 也是这里采用的方案 —— 是驱动厂商 CLI。因此凭据归 `agy` 所有，本
shim 不声明任何凭据。

| 路线 | 谁持有登录 | 状态 |
|:---|:---|:---|
| 用 Antigravity 的 OAuth client 调 Code Assist API | 网关 | 已撤回 —— 账号会被封 |
| `agy acp`（本 shim） | `agy` | 推荐路线 |

## Shim

| Shim 名称 | 基础类型 | 传输 | 凭据 |
|:---|:---|:---|:---|
| `antigravity--acp` | `acp` | `acp`（本机进程，stdio） | 无 —— token 在 `agy` 手里 |

`acp` 是一个**仅作上游**的格式：没有客户端会用它请求网关。该类型的提供方是你
已在提供的各格式的**目标**，所以 Chat Completions、Messages、Responses 或
generateContent 请求照常转换后就到达它。

## 前置条件

- **在网关主机上**安装 `agy`，且在 `PATH` 中（或用 `connection.acp_argv` 指定
  其他命令），并已登录：

  ```bash
  agy            # 首次运行：完成 Google 登录
  agy acp        # 网关启动的就是这个命令；Ctrl-D 退出
  ```

- 一个已经拥有你要使用的 Antigravity 套餐的 Google 账号。
- 网关进程有启动子进程的权限。容器部署需要在镜像内带 `agy`，或挂载一份。

## 配置

1. 添加一个类型为 `antigravity--acp` 的提供方，不需要 API Key。
2. 在某个模型的 `providers` 列表里引用它，请求一个 `agy` 提供的模型（见
   [模型](#_2)）：

```jsonc
{
  "providers": {
    "antigravity": {
      "type": "antigravity--acp"
      // 连接参数来自 shim：transport 为 "acp"，argv 为 ["agy", "acp"]
      // 可在提供方上覆盖：
      // "transport": "acp",
      // "acp_argv": ["/usr/local/bin/agy", "acp"]
    }
  },
  "models": {
    "gemini-3.1-pro": {
      "providers": [{"name": "antigravity"}]
    }
  }
}
```

3. 发送请求。流式与非流式客户端都能被服务：网关边接收 agent 的更新边按客户端
   自己的方言写出。

## 模型

请求里的模型名会作为该会话的 `model` 选项交给 agent。模型名是 agent 自己的，
不是本仓库的 —— 从 agent 取（`agy` 会列出模型；ACP 会话也会以 `model` 选项
公布可选值）。agent 拒绝某个名字时只记录一条警告，该轮改用会话自己的默认模型，
而不是让请求失败。

## 传递了什么

| 请求侧 | 处理 |
|:---|:---|
| System instruction | agent 接受时作为单个字符串发送 |
| 对话 | 按顺序渲染成带标签的文本块（`[User]`、`[Assistant]`、`[Tool result]`） |
| 图片 | 作为内联图片块，跟随所属消息的文本 |
| 工具声明、`tool_choice`、`tool_config` | **丢弃并报告** —— 目前没有工具桥 |
| `generation`、`response_format`、`reasoning`、缓存设置 | 丢弃并报告 |
| 文件与音频部分 | 丢弃并报告 |

| 响应侧 | 处理 |
|:---|:---|
| 消息块 | 作为 text delta 流出 |
| 思考块 | 作为 reasoning delta 流出 |
| usage 更新 | 记为上下文占用，而不是单轮 token 账单 |
| 停止原因 | 映射为 IR 的 finish reason（`end_turn` → `stop`、`max_tokens` → `length` 等） |
| 工具调用 | 暂不呈现（第二期） |

每个被丢弃的字段都会产生一条转换警告，所以请求日志会说清这条通道没能带走什么。

## 限制

- **目前没有工具调用。** 声明了工具的请求仍会成功，但模型无法调用它们。这是
  下一期第一件要做的事，通过通往 agent 的 MCP 桥实现。
- **每个请求一个会话。** 每个请求开一个会话、发送完整对话、然后关闭：重放历史
  不会静默丢掉某一轮；而要跨请求复用 agent 的会话，需要一个目前还不存在的台账。
  预期 agent 每轮都会重读历史。
- **agent 自己的工具由它自己负责。** `agy` 可能在它运行的主机上动作。请给它一个
  你愿意让模型写入的工作目录 —— 除非请求自带 `cwd`，网关会指着系统临时目录下的
  一个草稿目录（`llm-rosetta-acp/<provider>`）。
- **套餐窗口与限额由 Google 决定**，这里看不到。
- **`base_url` 只是占位。** `http://agy.local` 存在是因为传输契约与 HTTP 通道共用；
  没有任何代码会去连接它。

## 错误

| 现象 | 含义 | 解决 |
|:---|:---|:---|
| `503` “could not open a session … Sign the agent CLI in” | agent 拒绝 `session/new`，通常是没人登录 | 在网关主机上用 `agy` 登录后重试 |
| `503` “could not start the ACP agent ['agy', 'acp']” | 命令不在 `PATH`，或启动即退出 | 为网关进程的用户安装 `agy`，或设置 `connection.acp_argv` |
| `502`，正文是 agent 自己的话 | 非登录类拒绝（agent 不认的模型、上游错误） | 读这条消息；它是 agent 给的 |
| 流中途出现错误事件（`finish_reason: error`） | 轮次开始后 agent 失败 | 请求日志里有原因 |
| 每轮都很慢 | 每个请求都在重放完整历史 | 本期的预期行为（见「限制」） |

## 安全

- 网关从不持有、读取或记录 Google token：`agy` 把它存在自己使用的位置。
- agent 每个提供方启动一次并跨请求复用；网关退出时停止它。一个提供方是一个进程，
  而不是每个请求一个进程。
- 子进程继承网关的环境变量。如果 `agy` 会从中读取敏感内容，用 `acp_argv` 包一层
  来清理环境。
