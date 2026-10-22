---
title: CLI 参考
---

# CLI 参考

## 用法

```
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

## 命令

### `init`

在 `~/.config/llm-rosetta-gateway/` 创建模板 `config.jsonc`：

```bash
llm-rosetta-gateway init
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

### `credential <provider>`

在本机运行 shim 的交互式登录，输出它生成的凭据记录：

```bash
llm-rosetta-gateway credential chatgpt --out record.json
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
from llm_rosetta_gateway import create_app, GatewayConfig, load_config

# 加载配置并创建 ASGI 应用
raw = load_config("config.jsonc")
config = GatewayConfig(raw)
app = create_app(config)

# 挂载到你自己的 ASGI 应用中，或使用任意 ASGI 服务器运行
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8765)
```

## 从 shell 读取用量

`llm-rosetta-usage` 是另一个命令，用于读取网关保存的 token 台账——与管理面板
仪表盘和 API Keys 标签页展示的同一组数字，无需启动任何服务：

```console
$ llm-rosetta-usage --db /var/lib/llm-rosetta --window 24h --by subject,model
```

它读取存储，并在 keys 数据库就在旁边时一并读取，因此可以直接对运行中的网关
文件出报告。参见
[可观测性——`llm-rosetta-usage` 命令](../api/observability.md#the-llm-rosetta-usage-command)。
