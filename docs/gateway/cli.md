---
title: CLI 参考
---

# CLI 参考

## 命令面 { #the-command-surface }

`llm-rosetta` 命令是一个分发器。core 拥有两个名字：`ext`（管理扩展）和 `help`；其他命令都由已安装的发行版贡献。网关发行版添加 `gateway` 和 `serve`。不带参数运行 `llm-rosetta` 会打印命令列表，不会启动服务器。

| 命令 | 作用 |
|---------|------|
| `llm-rosetta gateway <verb>` | 网关的动词：`serve`、`init`、`add`、`credential`、`password`、`db`、`usage`、`migrate` |
| `llm-rosetta serve` | `llm-rosetta gateway serve` 的快捷方式 |
| `llm-rosetta ext <verb>` | 扩展：`list`、`show`、`install`、`remove`、`update`、`doctor`、`new`、`check`、`migrate`；见[扩展](../guide/extensions.md) |
| `llm-rosetta <id> <verb>` | 已安装扩展新增的命令 |
| `llm-rosetta-gateway [COMMAND]` | 网关自己的入口；动词相同，不带命令时启动服务器 |

`llm-rosetta-gateway` 是网关发行版唯一的控制台脚本。早先 0.14 开发版附带的控制台脚本 `llm-rosetta-usage` 和 `llm-rosetta-migrate` 已移除，`python -m llm_rosetta_gateway` 也已移除：

| 早先的 0.14 开发版 | 现在 |
|--------|-----|
| `llm-rosetta-usage ...` | `llm-rosetta gateway usage ...` |
| `llm-rosetta-migrate ...` | `llm-rosetta gateway migrate ...` |
| `llm-rosetta-gateway shims new / check / migrate` | `llm-rosetta ext new / check / migrate`（见[添加扩展](../contributing/adding-an-extension.md)） |
| `llm-rosetta-gateway set-password`（0.13.0） | `llm-rosetta gateway password`（仍接受 `set-password` 这个别名） |
| `python -m llm_rosetta_gateway` | `llm-rosetta serve` 或 `llm-rosetta-gateway` |

每个动词也可以写成 `llm-rosetta-gateway <verb>`，所以下面的示例两种写法都适用。`llm-rosetta-gateway` 还会把 `ext`、`help` 和已安装扩展的命令转发给分发器：编译后的二进制和 Docker 镜像旁边没有单独的 `llm-rosetta` 脚本，没有 pip 的运维人员就是这样安装扩展的（`llm-rosetta-gateway ext install <id>`）。放在这类命令之前的选项会被拒绝。

退出码：`0` 成功，`1` 命令失败，`2` 用法错误；文档中另有说明的命令保持自己的退出码（`gateway migrate --check` 在有待处理工作时返回 `3`）。未预期的异常会打印一行 `llm-rosetta <command>: <message>` 并以 `1` 退出；把 `LLM_ROSETTA_DEBUG` 设为任意值可看到 traceback。

## 用法

```
llm-rosetta serve [选项]
llm-rosetta-gateway [选项] [命令]
```

## 选项

| 参数 | 说明 |
|------|------|
| `--config`, `-c` PATH | 配置文件路径（未指定时自动搜索） |
| `--version`, `-V` | 显示版本并退出 |
| `--no-banner` | 抑制启动横幅显示 |
| `--edit`, `-e` | 在 `$EDITOR` 中打开配置文件进行编辑 |
| `--host` HOST | 覆盖服务器主机 |
| `--port` PORT | 覆盖服务器端口 |
| `--proxy` URL | 所有上游请求的 HTTP/SOCKS 代理 URL |
| `--verbose`, `-v` | 启用详细（DEBUG）日志；覆盖配置文件和 `--log-level` |
| `--log-level` LEVEL | 日志级别：`debug`、`info`、`warning`、`error`（默认：`info`） |
| `--socket`, `-S` PATH | 监听 Unix 域套接字而不是 TCP |
| `--tls-cert` PEM、`--tls-key` PEM | 提供 TLS；二者需同时给出 |
| `--data-dir` DIR | 存放 `gateway.db` 等持久化数据的目录（覆盖配置） |

## 命令

### `init`

在 `~/.config/llm-rosetta-gateway/` 创建模板 `config.jsonc`：

```bash
llm-rosetta gateway init
```

### `add provider <name>`

添加提供方条目到配置文件：

```bash
llm-rosetta-gateway add provider chat_completions
llm-rosetta-gateway add provider anthropic --api-key "${ANTHROPIC_API_KEY}"
```

| 参数 | 说明 |
|------|------|
| `--api-key` KEY | API 密钥或 `${ENV_VAR}` 占位符 |
| `--base-url` URL | 提供方基础 URL（已知提供方自动填充） |

### `add model <name>`

添加模型路由条目到配置文件：

```bash
llm-rosetta-gateway add model gpt-4o --provider my-openai
llm-rosetta-gateway add model claude-sonnet-4-20250514 --provider my-anthropic
llm-rosetta-gateway add model gemini-2.0-flash --provider my-google
```

| 参数 | 说明 |
|------|------|
| `--provider` NAME | 目标提供方名称 |

### `password [config]`

设置或修改管理员密码（忘记密码时的恢复手段）：

```bash
llm-rosetta gateway password --config config.jsonc
```

### `db <action>`

数据库维护。用 `--data-dir`（或 `--config`）指定存放 `gateway.db` 的目录。

| 动作 | 作用 |
|------|------|
| `cleanup [--max-age-days N]` | 删除早于 `N` 天（默认 90）的全部记录并执行 vacuum |
| `cleanup-logs [--max-age-days N]` | 删除早于 `N` 天的请求日志 |
| `cleanup-errors [--max-age-days N]` | 删除早于 `N` 天的错误转储 |
| `export-errors [--start D] [--end D] [-o FILE]` | 把错误转储导出为 `tar.gz`（默认 `error-dumps.tar.gz`） |

### `usage`

见[从 shell 读取用量](#reading-usage-from-the-shell)。

### `migrate`

重写 0.14.0 的移除所破坏的内容：配置文件、网关数据库中持久化的标签，以及源代码。默认是试运行。

```bash
llm-rosetta gateway migrate --config config.jsonc            # 试运行：差异与报告
llm-rosetta gateway migrate --config config.jsonc --write    # 先备份 .bak，再重写
llm-rosetta gateway migrate --db /var/lib/llm-rosetta --write
llm-rosetta gateway migrate --scan src/ --json
```

见[迁移到 0.14](../guide/migrating-to-0.14.md#llm-rosetta-migrate)。

### `ext`

`llm-rosetta ext` 不是网关的动词，但网关二进制会转发它。见[扩展](../guide/extensions.md#the-ext-verbs)。

### `credential <provider>`

在本机运行 shim 的交互式登录，输出它生成的凭据记录：

```bash
llm-rosetta gateway credential chatgpt --out record.json
```

shim 随包提供 `auth.py` 的提供方没有可填的密钥 —— 交换通过需要浏览器的 OAuth 登录
生成凭据。网关跑在别处时，登录要在有浏览器的一侧完成，本命令就是那一半：它运行交换的
`login()`，返回的内容正是管理 API `PUT
/admin/api/config/providers/<name>/credential` 接收的记录。命令不联系网关 —— 记录在管理台
里交过去。

| 参数 | 说明 |
|------|------|
| `--out` FILE | 把记录写入 `FILE`（权限 `0600`），而不是输出到标准输出 |

记录里是有效令牌（access、refresh 与 ID token）。请尽快粘贴到管理台：ID token 在收到时
即刻验签，放置过久的记录会被拒绝，只能重新登录。登录还会在本机的记录目录
（`<data_dir>/shim-auth/<provider>`）里存一份副本；网关在别处时，交付完成后请把它删掉。
`--out` 通过同目录下随机命名的临时文件写入：以 `0600` 独占创建、改名之前 `fsync`，因此在
共享目录里按可预测的文件名预先放置的文件或符号链接都截不走记录。命令会拒绝两种提供方：shim 没有
`auth.py` 的（`mints no credential of its own`），以及交换没有浏览器流程的（`does not
implement login()`）。

## 配置文件自动发现

未指定 `--config` 时，网关按以下顺序搜索配置文件：

1. `./config.jsonc` — 当前工作目录
2. `~/.config/llm-rosetta-gateway/config.jsonc` — XDG 标准位置
3. `~/.llm-rosetta-gateway/config.jsonc` — dotfile 约定

## 编程方式使用

网关也可以作为库使用：

```python
import asyncio

import uvicorn

from llm_rosetta_gateway import create_app, GatewayConfig, load_config

# 加载配置并创建 ASGI 应用。create_app 自 0.14.0 起是 async 函数。
raw = load_config("config.jsonc")
config = GatewayConfig(raw)
app = asyncio.run(create_app(config))

# 挂载到你自己的 ASGI 应用中，或使用任意 ASGI 服务器运行
uvicorn.run(app, host="0.0.0.0", port=8765)
```

## 从 shell 读取用量 { #reading-usage-from-the-shell }

`llm-rosetta gateway usage` 报告网关保存的 token 台账，即管理面板仪表盘和 API Keys 标签页展示的同一组数字，无需启动任何服务：

```console
$ llm-rosetta gateway usage --db /var/lib/llm-rosetta --window 24h --by subject,model
```

它读取存储，并在 keys 数据库就在旁边时一并读取，因此可以直接对运行中的网关文件出报告。参见[可观测性：usage 命令](../api/observability.md#the-llm-rosetta-usage-command)。
