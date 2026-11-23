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

### `shims new [vendor]`

创建一个新的 provider shim 所需的文件，然后用网关所用的同一个 loader 装载一遍：

```bash
llm-rosetta-gateway shims new                  # 逐项询问
llm-rosetta-gateway shims new acme --no-input \
  --base-url https://api.acme.example/v1 --check
```

在终端上不带参数运行时它是一份问卷 —— 先问扫描根，再问服务方、模型类型、线上格式、lane、base URL、
凭据变量、可选的展示名 / logo / 分组标题、以及要写出哪些可选模块 —— 回车即采用中括号里的值。
扫描根排在第一位，因为它之后的每一项都跟随包已有的做法（见下文）。**默认是交互模式**；已经用
flag 给出的值不会再问。`--no-input`（stdin 不是终端时自动生效）从不提问，所以同一条命令可以
直接放进脚本。

| 参数 | 说明 |
|------|------|
| `--root DIR` | 网关扫描的目录；默认 `providers` |
| `--model-type` | `llm`（默认）、`embeddings`、`rerank`、`decisions` |
| `--format` | 线上格式；问卷与 `--format` 的帮助都会列出该模型类型注册的全部格式，未注册的格式在写任何文件之前就被拒绝 |
| `--lane` | 服务方的具名实例，写在服务方与类型目录之间（`<vendor>/<lane>/<type>/`）；ALCF 的集群 |
| `--base-url` | 上游 base URL；省略时写入占位值 |
| `--probe` | 检查 `--base-url` 是否可达，并打印它返回了什么 |
| `--models-path` | 模型列表 URL 或路径（非默认时） |
| `--api-key-env` | 凭据变量；默认 `<VENDOR>_API_KEY` |
| `--display-name` | 服务方分组内的行标签 |
| `--logo`、`--group`、`--group-label` | 服务方身份，写入服务方清单 |
| `--auto-logo` | 按服务方名到 jsDelivr 上的 lobehub 图标集里查找标志图；查不到就失败 |
| `--localize-logo` | 把 `--logo` 存进包的 `assets/`，并引用它的 jsDelivr URL |
| `--transform` | 同时生成一个 `transforms.py` 桩 |
| `--auth`、`--quota` | 同时生成服务方级的 `auth.py` 或 `quota.py` 桩 |
| `--package DISTRIBUTION` | 同时写出外部包外壳（`pyproject.toml`、入口点、CI、README），目录树落在 `src/<package>/providers` |
| `--no-registry-manifest` | 不改动包的 `shims.yaml` |
| `--check` | 装载新叶子并报告解析结果 |

服务方已有 `provider.yaml` 时，凭据和服务方身份留在那里，叶子只写自己独有的字段。扫描根提供
`shims.yaml` 时，新叶子会被追加进去 —— 那份清单对网关是权威的，没列出的叶子永远不会装载。

**扫描根只被读取，不被照抄。** 一切都按约定形状写出 ——
`<vendor>[/<lane>]/<type>[/<format>]/provider.yaml` —— 扫描根只决定是否需要 `<format>` 层：
该类型下已有第二条 lane 的服务方需要，第一条 lane 则直接落在类型目录里。服务方仍把格式目录直接放在
其下的根（`kilo_free/chat_completions/`）属于约定之前的形状，因此新叶子仍按规范写出，只对该服务方
给出提示，并附上报告其余问题的 `shims check` 命令。

以下三种目标在写任何文件之前就被拒绝：

- 服务方自己的 `provider.yaml` 是一个 shim —— loader 会把该目录当成那一个 shim，从不向下看，
  因此它下面的第二个 lane 根本不会被加载；
- 另一个文件已经声明的 shim 名，并指出那个文件；
- loader 会跳过或误读的服务方名 —— `_`/`.` 开头，或 `llm`、`embeddings`、`rerank`、`decisions`。

服务方仍把格式目录直接放在其下时只提示、不阻断：loader 同样能读那些目录，`shims migrate`
负责把它们移过去。

粘贴进来的 base URL 在写入前会先检查。结尾的 `/models`（服务方控制台常见）会被去掉，然后不带
凭据请求 `<base>/models`：`200`、`401`、`403` 都证明主机在该路径上有应答，其余结果会连同
base URL 自身的状态一起报告，于是后缀写错现在就暴露，而不是以后表现为空模型列表。交互模式
总是检查；flag 模式需加 `--probe`。检查不通过只是警告，不是拒绝 —— 本机未必能到达该主机。

按一次 <kbd>Ctrl</kbd>+<kbd>C</kbd> 会重新询问当前问题；再按一次则以退出码 130 取消本次运行，
并说明是否已经写入了文件。按 <kbd>Ctrl</kbd>+<kbd>D</kbd> 立即取消 —— 输入流已结束，没有值可
取，而后续每个提问都会同样读到结束。三种情况都不会出现 traceback。

标志图属于包本身，而不是别人的主机：给出 logo URL 时，交互模式会问是否把它下载到包的
`assets/`，并把 `logo:` 指向该文件的 jsDelivr 副本。包通过扫描根的 git remote 识别，因此
写下的就是本仓库实际会提供的 URL；扫描根不在 git 仓库里时保留原 URL 并说明原因。
`--localize-logo` 与提问等价，只是不问。

提问之前，会先按服务方名到 lobehub 图标集（jsDelivr 上的 `@lobehub/icons-static-svg`）里查找，
命中就作为默认答案 —— 回车即采用。先试完整名字，再试首段以及去掉池标记后的名字，所以
`opencode_zen_free` 仍能找到 `opencode`；每次尝试是一个 `HEAD`，CDN 不可达时立即结束查找，
而不是每个候选各等一次超时。已经在 jsDelivr 上的 URL 原样保留，因为没有什么可复制的 ——
两种情况下标志图都由 CDN 提供。`--auto-logo` 不做问卷、直接查表，查不到就失败。

### `shims check [--root DIR]`

报告 loader 会读错的 shim 目录树，只要发现问题就以非零退出：

```bash
llm-rosetta-gateway shims check --root src/llm_rosetta_shim_acme/providers
# ok      .../providers: no deviation from the shim contract
# problem acme/chat_completions: <type>/ is missing above this leaf
```

它不装载、不写文件、也不需要配置，因此可以在还没配好网关的包仓库里直接运行。会报告的情形：
叶子不在规范路径上、把服务方目录当叶子用、类型目录同时混有叶子层和格式层、类型目录之上超过一个
lane 目录、服务方级 `transforms.py`、服务方与叶子两级都声明了 `auth.py`、文件不在允许清单内、缺少
`provider.yaml`。约定本身见 `llm_rosetta/devtools/_shim_template/README_zh.md`。

### `shims migrate [--root DIR] [--write]`

把一棵树移到约定形状上。不加 `--write` 时只打印移动计划，什么都不动：

```bash
llm-rosetta-gateway shims migrate --root src/llm_rosetta_shim_acme/providers
llm-rosetta-gateway shims migrate --root src/llm_rosetta_shim_acme/providers --write
```

它移动目录，并改写每个被移动叶子的 `shims.yaml` 登记，包括第二条 lane 的情况 —— 已有叶子要下沉
到 `<type>/<它自己的格式>/`。它**不**重写相对导入：原先向上导入 `model_utils` 的 `transforms.py`
需要多一个点，靠包自己的测试兜住。加 `--write` 之后请跑一遍包的测试。

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

## 从 shell 读取用量

`llm-rosetta-usage` 是另一个命令，用于读取网关保存的 token 台账——与管理面板
仪表盘和 API Keys 标签页展示的同一组数字，无需启动任何服务：

```console
$ llm-rosetta-usage --db /var/lib/llm-rosetta --window 24h --by subject,model
```

它读取存储，并在 keys 数据库就在旁边时一并读取，因此可以直接对运行中的网关
文件出报告。参见
[可观测性——`llm-rosetta-usage` 命令](../api/observability.md#the-llm-rosetta-usage-command)。
