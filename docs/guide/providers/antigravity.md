---
title: Google（Antigravity）
---

# Google（Antigravity）

`antigravity--acp` shim 通过服务方自带的 `agy` CLI 使用 Google **Antigravity**
套餐。它不调用 Google 的 Code Assist API：网关在主机上启动 `agy acp`，通过
stdio 与之讲 [Agent Client Protocol](https://agentclientprotocol.com/)，因此
Google 登录、访问 token 与 project 都由 `agy` 保管，与你亲自运行 `agy` 时完全一样。

!!! warning "在把账号指过来之前先读这一段"
    Google 的条款决定谁可以使用该套餐，并且 Google 可能封禁被判定在 Antigravity
    自家客户端之外使用的账号。驱动 `agy` 是从网关使用该套餐最不绕路的方式 —— 它
    本身就是服务方的客户端 —— 但这仍然是一个可能失去的套餐。请使用可以承受损失的账号，
    也不要把这条通道开放给你无法支持的最终用户。

## 为什么用 `agy` 而不是直接调 API

用 Antigravity 自己的 OAuth client 直接调 Code Assist，正是 Google 会封号的做法：
它看到这个请求并非来自 Antigravity 自己的客户端。驱动服务方 CLI 则让每个请求都留在
Google 认可的客户端里。因此凭据归 `agy` 所有，本 shim 不声明任何凭据。

| 路线 | 谁持有登录 | 状态 |
|:---|:---|:---|
| 用 Antigravity 的 OAuth client 调 Code Assist API | 网关 | 不提供 —— 账号会被封 |
| `agy acp`（本 shim） | `agy` | 本仓库支持的路线 |

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

## 登录与持久配置

网关给每个提供方一个位于数据目录下的独立目录，默认是
`<data_dir>/acp/<provider>`，并用 `--gemini-dir` 指向它来启动 ACP agent 和管理面板的
登录。桌面 Linux 上 `agy` 把 token 放在整机共享的 keyring 里；无头主机或容器里没有
keyring，`agy` 会把文件形式的 token 写在该目录下。把网关的数据目录挂载到卷上，
重启网关或部署新镜像后都能复用这次登录。在提供方条目里设置 `acp.home` 可以切换到
另一个已保存的目录：

```jsonc
"antigravity": {
  "type": "antigravity--acp",
  "acp": {"home": "saved/antigravity-work"}
}
```

认证卡片提供 **Start browser sign-in**：它运行
`agy acp --login --gemini-dir <home>`，打开 Google 登录页，接收粘贴的授权码，
并报告账号。无头主机无法反查账号，因此登录成功后还会在同一目录写一个不含机密的
标记文件（`.llm-rosetta-login.json`）；重启后面板仍显示 **已连接** 就是靠它。
标记里只有时间戳，没有任何 token。

## 切换 agent 版本

agent 命令指定的是**安装**，`acp.home` 指定的是**配置**。要运行另一个 `agy`
版本，把 `acp.argv` 指向它，配置目录保持不变：

```jsonc
"antigravity": {
  "type": "antigravity--acp",
  "acp": {
    "home": "saved/antigravity-work",
    "argv": ["/opt/agy-1.7/bin/agy", "acp"]
  }
}
```

旧的顶层 `acp_argv` 仍可作为别名使用。默认按可执行文件名决定是否加固：命令的
可执行文件名为 `agy`（`agy`、`agy.real`、`agy.exe`）时，网关会套用自己的配置：
`--gemini-dir`、无工具 agent、拒绝一切的 `PreToolUse` 钩子和会话闸门都保留。
因此切换版本既不会重新放开主机工具，也不会丢失登录。

改名后的下载（如 `agy-1.7`）或包装脚本无法按名字识别。用 `"harden": true`
强制对该命令套用配置——包装脚本必须把 `agy` 的参数继续传下去。设为
`"harden": false` 时命令逐字执行；如果它启动的是 `agy`，还必须显式指定
`--gemini-dir`，否则 agent 与登录无法共用配置，该条目会被拒绝。

如果命令本身已经带 `--gemini-dir`，该目录就是登录写入的配置目录。不要设置
`acp.home`，或把它设为同一路径：两个不同目录会被拒绝，而不是让 agent 与登录
各写一处。

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
| 工具声明 | 作为回环 MCP 服务器交给 agent —— 模型可以调用它们 |
| `tool_choice`、`tool_config` | 丢弃并报告 |
| `generation`、`response_format`、`reasoning`、缓存设置 | 丢弃并报告 |
| 文件与音频部分 | 丢弃并报告 |

| 响应侧 | 处理 |
|:---|:---|
| 消息块 | 作为 text delta 流出 |
| 思考块 | 作为 reasoning delta 流出 |
| usage 更新 | 记为上下文占用，而不是单轮 token 账单 |
| 停止原因 | 映射为 IR 的 finish reason（`end_turn` → `stop`、`max_tokens` → `length` 等） |
| 工具调用 | 作为工具调用事件呈现；该轮以 `tool_calls` 停止，直到下一个请求带回结果 |

每个被丢弃的字段都会产生一条转换警告，所以请求日志会说清这条通道没能带走什么。

客户端工具把客户端留在回路里。请求声明一个会话 id 与一个或多个工具；agent 通过
网关的回环 MCP 端点调用其中一个，网关以该调用和 `stop_reason: "tool_calls"`
结束这次响应，客户端在下一个请求里带回结果。网关回答挂起的 MCP 调用，agent 继续
同一轮。声明了工具但没有会话 id 的请求会收到 `400`，因为结果无法找回对应的轮次。
已在 `agy` 1.5.14 上实测；1.5.13 通过 `call_mcp_tool` 元工具走同一条调用。

工具集合是会话的属性，不是单轮的设置。agent 在会话打开时固定它的 MCP 服务器
列表，所以只要请求声明的工具与该会话不同 —— 新增或移除工具、禁用工具、刷新
schema —— 网关就关闭会话，把整段对话重放进一个列出新工具集合的新会话。仅有顺序
不同不算变化：会话的工具摘要是规范化的，同一组定义换一个顺序仍复用会话。带工具
结果的请求如果同时也改变了工具集合，同样如此：网关放弃挂起的调用并重放历史，
而不是继续使用旧的桥接。

## 限制

- **客户端工具需要会话 id。** 客户端的结果按会话匹配回挂起的那一轮，所以声明了
  工具的一次性请求会被拒绝（`400`），而不是让调用丢失。请求带工具时请声明会话
  id（见下一条）。
- **改变工具集合要付一次重放。** agent 的 MCP 列表在一个会话内固定，所以第一个
  新增、移除或刷新工具的请求会打开新会话并重发整段对话。这是正确的，但不是免费
  的；对延迟敏感时，尽量保持一个会话的工具集合稳定。
- **agent 自带的工具始终关闭。** `agy` 在网关自有的配置中运行：不带工具的自定义
  agent、只允许网关回环 MCP 服务器的 `PreToolUse` hook、
  `toolPermission: strict`，以及拒绝每一个内置工具的 ACP 客户端。一旦内置工具
  启动，该轮返回 `502`，而不是在主机上执行。
- **客户端声明了会话，就一个会话对应一个对话。** 声明了会话 id 的请求
  （`x-claude-code-session-id`、`x-opencode-session`、任何值形如 id 的
  `x-<vendor>-session-id`，或请求体的 `metadata.user_id.session_id` /
  `prompt_cache_key`）会跨轮保留它的 agent 会话：台账记着 agent 已经看过哪些
  prompt 块，所以下一轮只发送新增的尾部。历史不再是对该记录的延展时 —— 某一轮被
  改写、上下文被压缩 —— 网关关闭会话，把整段对话重放进一个新会话。没有声明会话的
  请求仍然一个请求一个会话，因为猜出来的身份可能把一个对话的记录交给另一个对话。
- **重启后每个对话各重放一次。** 台账是易失的：它描述的是活着的 agent 会话，
  所以重启后从空开始，每个对话的第一轮都会重新发送完整历史。
- **进程与会话都有上限。** `acp.processes`（默认 1）决定该提供方运行几个 agent
  进程 —— 一个进程一次只处理它的会话中的一轮，所以这项才决定两个对话能否并行；
  `acp.max_sessions`（默认 32）与 `acp.session_ttl`（秒，默认 1800）约束这些进程
  持有多少，空闲会话由周期性清扫交还，而不是等到下一个请求到来。
- **工作目录仍是主机的。** agent 的内置工具无法运行，但会话的 `cwd` 是真实的。
  除非请求自带 `cwd`，网关会指着系统临时目录下的一个草稿目录
  （`llm-rosetta-acp/<provider>`）。
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
| 每轮都很慢 | 请求没有声明会话 id，所以每轮都开会话并重放历史 | 声明会话 id（见「限制」） |

## 安全

- 网关从不持有、读取或记录 Google token：`agy` 把它存在 keyring 里，或在无头主机上
  存在该通道的持久目录里。网关只在那个目录写一个不含机密的时间戳标记。
- agent 在网关自有、且每次会话前都会被重写的配置中运行：它的内置工具无法在主机上
  执行，客户端工具只能通过网关的回环 MCP 端点到达它。
- agent 每个提供方启动一次并跨请求复用；网关退出时停止它。一个提供方是一个进程，
  而不是每个请求一个进程。
- 子进程继承网关的环境变量。shim 的 `connection.acp_env` 或提供方的 `acp_env`
  可以新增或覆盖指定的变量，同时保留 `PATH` 与登录存储；如果 `agy` 会从环境中读取
  敏感内容，用 `acp_argv` 包一层来清理环境。
