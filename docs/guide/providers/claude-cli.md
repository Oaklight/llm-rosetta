---
title: Claude（官方 CLI）
---

# Claude（官方 CLI）

`claude--messages` shim 把官方 **Claude Code CLI** 当作模型客户端，用来服务
Claude 订阅或 Console 账号。请求由 CLI 自己构建并签名，网关不持有 Claude
凭据：账号就是 CLI 自己的登录存储。

该 shim 位于 contrib 源，不在核心仓库。安装 `claude` 厂商并选择 Messages
线路：

```jsonc
{
  "shims": { "sources": ["contrib"], "install": ["claude"] },
  "providers": {
    "claude-cli": {
      "type": "claude--messages",
      "models": ["claude-haiku-4-5", "claude-sonnet-5"]
    }
  }
}
```

该厂商同时安装 `claude--acp`（ACP 线路）；由 `type` 决定 provider 使用哪一条。

!!! warning "把账号接进来之前请先读这段"
    Anthropic 的条款决定订阅可以怎样被驱动，在 Anthropic 自家客户端之外使用
    的账号可能被停用。这与 ACP 线路的风险相同。请使用一个可以承受损失的账号。

## 一个回合如何工作

1. 网关把客户端请求转换成 Messages，并把对话作为 NDJSON 重放帧写给 CLI。
2. 启动 `claude -p --input-format stream-json --output-format stream-json`，
   把 `ANTHROPIC_BASE_URL` 指向本回合专用的回环中继。
3. CLI 发出它唯一的一次上游请求；中继用 CLI 自己的请求头转发，并为网关捕获
   流式响应。
4. 网关把 CLI 的流事件映射回 Messages 事件。要求非流式回复的客户端会拿到
   同样事件折叠成的一个响应体。

中继同时是单请求闸门：CLI 自己的恢复重试会在本地被拒绝；唯一的例外是上游
返回 401 后放行一次，让 CLI 刷新登录。

## 前置条件

- 官方 `claude` CLI 位于网关主机的 `PATH`，或在 `cli_argv` 中给出绝对路径。
  本线路在 2.1.299 上测量。
- CLI 已登录：

  ```bash
  claude auth login --claudeai   # Pro / Max 订阅
  claude auth login --console    # Console 账号，按 token 计费
  ```

- 网关进程有权创建子进程。容器镜像里需要包含 CLI，或挂载一份。

## 配置

厂商事实由 shim 声明，provider 条目可以覆盖：

| 键 | 含义 |
|:---|:---|
| `cli_argv` | 要运行的命令。替换 shim 的 `["claude"]`。 |
| `cli_args` | 额外参数，追加在最后。替换 shim 的列表。 |
| `cli_env` | 子进程的环境覆盖；逐键以 provider 为准。 |
| `cli_env_unset` | 在应用 `cli_env` 前移除的变量名。shim 声明的名字保持移除。 |
| `cli_upstream` | 中继转发的目标。默认 Anthropic API；仅支持 `https://`，回环主机除外。 |
| `cli_processes` | 该 provider 同时运行的 CLI 进程数上限。默认 `1`；超过上限的请求排队。 |

shim 会设置 `--setting-sources user`，因为主机可能通过 `apiKeyHelper`
提供凭据；没有它时 CLI 启动时没有凭据，上游返回 401。shim 还会移除
`ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`、模型覆盖变量以及
`CLAUDE_CODE_USE_*` 后端选择器，避免导出的 API key 压过已保存的登录。中继
通过私有的 `--settings` 文件钉住子进程的 base URL，因为用户设置文件的优先级
高于进程环境。

```jsonc
"claude-cli": {
  "type": "claude--messages",
  "cli_argv": ["/opt/claude/bin/claude"],
  "cli_env": {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000"},
  "cli_processes": 2
}
```

`max_tokens` 保持 CLI 自己的值：它请求 32,000 个输出 token，因为自适应思考
与输出共享这份预算。需要硬上限时在 `cli_env` 中设置
`CLAUDE_CODE_MAX_OUTPUT_TOKENS`。

## 能力

`text`、`vision` 与 `reasoning` 默认开启。图片原样通过重放，thinking 块从
CLI 的流事件返回。

`tools` **关闭**。上游请求体由 CLI 构建，传输层不会把主机的工具定义注入
其中：一次声明了工具的实测请求退化成文本，而不是 `tool_use` 块。工具*历史*
（`tool_use` / `tool_result`、id 与嵌套内容）可以原样重放，因此缺口只在
出站定义。在该路径修复并实测之前，不要为这条线路开启 `tools`。

## 运维

- 每个请求一个进程，约 250 MB。超过 `cli_processes` 上限的回合排队，而不是
  失败。
- CLI 超过上游超时仍无输出时回合失败；关闭流会终止整棵进程树（POSIX 用
  `killpg`，Windows 用 `taskkill /T`）。
- 请求日志的 profile 记录该线路独有的事实：CLI 原生 `usage`、
  `total_cost_usd`（按标价的等价成本）、`modelUsage[*]`（含
  `contextWindow`），以及中继的 `restored` 标志、上游状态码和
  `anthropic-ratelimit-*` 响应头。
- 模型 id 原样传给 `--model`。`[1m]` 后缀仍是 CLI 自己的 1M 选项，未知 id
  回落到 CLI 的 200K 默认值。
- CLI 会在每条重放的 user 消息后追加环境/日期注释。中继丢弃这些注释，并把
  缓存断点移到本回合最后一个块，前缀因此保持可缓存。

## 限制

- **实验性且与 CLI 版本耦合。** 重放帧、确认、设置注入和会话 id 均在
  CLI 2.1.299 上测量。升级版本后需要重跑实测检查。
- **工具定义不会发往上游。** 见[能力](#能力)。
- **错误映射不完整。** 被拒绝的重试会复现真实上游状态；把
  `anthropic-ratelimit-unified-reset` 映射为 `resets_at` 的套餐/会话 429，
  以及 401/402/529/503 的映射尚未完成。
- **暂不支持代理。** 中继尚未接入 `https_proxy` / `HTTP_PROXY`。

## 另见

- [Provider Shims](../shims.md)
- [ACP 线路（Antigravity）](antigravity.md) —— 另一种服务厂商 CLI 的方式，
  由 agent 自己掌握循环。
