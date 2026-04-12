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

添加提供商条目到配置文件：

```bash
llm-rosetta-gateway add provider openai_chat
llm-rosetta-gateway add provider anthropic --api-key "${ANTHROPIC_API_KEY}"
```

| 参数 | 说明 |
|------|------|
| `--api-key` KEY | API 密钥或 `${ENV_VAR}` 占位符 |
| `--base-url` URL | 提供商基础 URL（已知提供商自动填充） |

### `add model <name>`

添加模型路由条目到配置文件：

```bash
llm-rosetta-gateway add model gpt-4o --provider my-openai
llm-rosetta-gateway add model claude-sonnet-4-20250514 --provider my-anthropic
llm-rosetta-gateway add model gemini-2.0-flash --provider my-google
```

| 参数 | 说明 |
|------|------|
| `--provider` NAME | 目标提供商名称 |

## 配置文件自动发现

未指定 `--config` 时，网关按以下顺序搜索配置文件：

1. `./config.jsonc` — 当前工作目录
2. `~/.config/llm-rosetta-gateway/config.jsonc` — XDG 标准位置
3. `~/.llm-rosetta-gateway/config.jsonc` — dotfile 约定

## 编程方式使用

网关也可以作为库使用：

```python
from llm_rosetta.gateway import create_app, GatewayConfig, load_config

# 加载配置并创建 ASGI 应用
raw = load_config("config.jsonc")
config = GatewayConfig(raw)
app = create_app(config)

# 挂载到你自己的 ASGI 应用中，或使用任意 ASGI 服务器运行
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8765)
```
