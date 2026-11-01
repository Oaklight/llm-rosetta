<p align="center">
  <img src="https://raw.githubusercontent.com/Oaklight/llm-rosetta/master/design/logo/out/rosetta-icon-github.svg" alt="LLM-Rosetta" width="96">
</p>

<h1 align="center">llm-rosetta-gateway</h1>

<p align="center">
  <a href="https://pypi.org/project/llm-rosetta-gateway/"><img src="https://img.shields.io/pypi/v/llm-rosetta-gateway?color=green" alt="PyPI"></a>
  <a href="https://pypi.org/project/llm-rosetta-gateway/"><img src="https://img.shields.io/pypi/pyversions/llm-rosetta-gateway" alt="Python"></a>
  <a href="https://hub.docker.com/r/oaklight/llm-rosetta-gateway"><img src="https://img.shields.io/docker/v/oaklight/llm-rosetta-gateway?label=Docker&color=blue" alt="Docker"></a>
</p>

<p align="center">
  <a href="https://llm-rosetta.readthedocs.io/en/latest/gateway/"><img src="https://img.shields.io/badge/docs-English-2f6fdd" alt="Docs (English)"></a>
  <a href="https://llm-rosetta.readthedocs.io/zh-cn/latest/gateway/"><img src="https://img.shields.io/badge/docs-%E4%B8%AD%E6%96%87-2f6fdd" alt="Docs (中文)"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT"></a>
</p>

<p align="center">
  <a href="https://github.com/Oaklight/llm-rosetta/blob/master/packages/llm-rosetta-gateway/README_en.md">English</a> | 中文
</p>

自托管的 HTTP 代理：接受任意受支持格式的 LLM API 请求，转发给任意已配置的上游
提供方，并实时完成转换（chat 格式也支持流式）。网关还提供密钥轮换、路由、故障
转移和管理面板。

这是**网关发行包**。它依赖 `llm-rosetta` 提供转换核心，本身不增加其他第三方
运行时依赖。在 Python 3.10 上，核心包还会安装 `typing_extensions`。内置的 HTTP
服务器和客户端模块仅使用标准库。

重点在于掌控：每个上游的配置、凭据和路由都由你自己控制。LLM-Rosetta 本身不会在
链路中增加托管聚合服务。

## 最上层

核心库提供工具箱、原始转换器和管线。网关再增加 HTTP 代理这一层：负责调用提供方、
路由、密钥和管理面板。底层仍是同一套 IR 和转换器，向上迁移时翻译逻辑不变。

## 安装

```bash
pip install llm-rosetta-gateway
```

`pip install "llm-rosetta[gateway]"` 是一个过渡别名，会同时安装两个发行包。

## 运行

```bash
llm-rosetta-gateway init     # 生成模板配置
llm-rosetta-gateway          # 在 http://localhost:8765 启动
```

配置会自动在 `./config.jsonc`、`~/.config/llm-rosetta-gateway/config.jsonc` 或
`~/.llm-rosetta-gateway/config.jsonc` 中查找；也可以用 `--config` 指定。管理面板
位于 `http://localhost:8765/admin/`。

## 配置

在 `providers` 中列出上游，在 `models` 中把客户端发送的模型名映射到某个提供方。
网关按模型名路由，并在来源格式与上游格式之间转换：

```jsonc
{
  "providers": {
    "my-openai":    { "type": "chat_completions", "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "my-anthropic": { "type": "messages",         "api_key": "${ANTHROPIC_API_KEY}", "base_url": "https://api.anthropic.com" }
  },
  "models": {
    "gpt-4o": "my-openai",
    "claude-sonnet-4-5": "my-anthropic"
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    // 仅用于本地演示。共享部署请绑定 0.0.0.0 并置于自己的网络控制之后，
    // 在 /admin 生成网关密钥，再按各格式的原生鉴权头发送。
    "open_on_no_keys": true
  }
}
```

## 发送请求

向 Anthropic 模型发送 OpenAI 格式的请求——网关会双向转换，也支持流式响应：

```bash
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hello!"}]}'
```

配置了网关密钥后，按各格式的原生方式发送——OpenAI 风格路由用
`Authorization: Bearer <key>`，Messages 用 `x-api-key: <key>`，Google 用
`x-goog-api-key: <key>`（或 `?key=<key>`）。裸 `Authorization: Bearer` 在任何
路由上也可用；反过来，OpenAI 风格路由会拒绝 `x-api-key`。

网关支持 OpenAI Chat Completions、Open Responses、Anthropic Messages、
Google generateContent 和 Google Interactions 路由，以及 embedding、rerank 和
decision 端点。

## 管理面板

内置管理面板位于 `http://localhost:8765/admin/`，可管理提供方、模型、API 密钥和
可观测性数据：

![管理面板](https://llm-rosetta.readthedocs.io/zh-cn/latest/gateway/images/admin-dashboard.png)

## 为什么不用 LiteLLM 或 OpenRouter？

**OpenRouter 主要是一项托管服务。** 在标准使用方式下，OpenRouter 管理提供方
接入和路由，你的流量会经过其服务；它也提供自带密钥选项。这正是它的长处：不用在每家提供方分别注册就能用到模型。LLM-Rosetta 的前提相反——每个上游凭据都归你所有，网关由你
自己运行。

**LiteLLM 可自托管，并提供广泛的提供方支持。** 它还提供团队密钥、预算和花费追踪。
它的核心接口基本沿用 OpenAI 格式，其他格式都映射到这个中心上。LLM-Rosetta 的 IR
不属于任何提供方：每种支持的格式都能放在任意一侧，客户端方言和模型来源因此互相
独立，应用也可以把 payload 保存在 IR 里。

两边的覆盖范围都变化很快；请以它们当前的文档为准。

- **选 OpenRouter**：你完全不想自己持有提供方账户。
- **选 LiteLLM**：你需要广泛的提供方支持或团队级成本控制。
- **选 LLM-Rosetta**：模型来源归你所有，而你需要用不同格式的客户端去访问它们。

## Docker

```bash
docker run -p 8765:8765 -v /path/to/config:/config oaklight/llm-rosetta-gateway
```

镜像读取 `/config/config.jsonc`。

## CLI 集成

可将网关作为 Claude Code、Codex CLI、Aider、Antigravity CLI、OpenCode 和
Kilo Code 的后端，也可将其作为 Ollama 的前置网关——见
[CLI 集成](https://llm-rosetta.readthedocs.io/zh-cn/latest/gateway/cli-integrations/)。

## 文档

- [完整 README](https://github.com/Oaklight/llm-rosetta/blob/master/README_zh.md)
- [网关快速开始](https://llm-rosetta.readthedocs.io/zh-cn/latest/getting-started/gateway-quickstart/)
- [配置](https://llm-rosetta.readthedocs.io/zh-cn/latest/gateway/configuration/)
- [管理面板](https://llm-rosetta.readthedocs.io/zh-cn/latest/gateway/admin-panel/)
- [核心发行包](https://pypi.org/project/llm-rosetta/)

## 许可证

本项目使用 MIT 许可证——详见 [LICENSE](https://github.com/Oaklight/llm-rosetta/blob/master/LICENSE)。
