<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="design/logo/out/rosetta-icon-github.svg">
    <source media="(prefers-color-scheme: light)" srcset="design/logo/out/rosetta-icon-docs.svg">
    <img src="design/logo/out/rosetta-icon-github.svg" alt="LLM-Rosetta" width="128">
  </picture>
</p>

<h1 align="center">LLM-Rosetta</h1>

<p align="center">
  <a href="https://pypi.org/project/llm-rosetta/"><img src="https://img.shields.io/pypi/v/llm-rosetta?color=green" alt="PyPI"></a>
  <a href="https://test.pypi.org/project/llm-rosetta/"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Ftest.pypi.org%2Fpypi%2Fllm-rosetta%2Fjson&query=%24.info.version&label=TestPyPI&color=blue" alt="TestPyPI"></a>
  <a href="https://github.com/Oaklight/llm-rosetta/releases/latest"><img src="https://img.shields.io/github/v/release/Oaklight/llm-rosetta?color=green" alt="Release"></a>
  <a href="https://hub.docker.com/r/oaklight/llm-rosetta-gateway"><img src="https://img.shields.io/docker/v/oaklight/llm-rosetta-gateway?label=Docker&color=blue" alt="Docker"></a>
  <a href="https://github.com/Oaklight/llm-rosetta/actions/workflows/ci.yml"><img src="https://github.com/Oaklight/llm-rosetta/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <a href="https://llm-rosetta.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/docs-English-2f6fdd" alt="Docs (English)"></a>
  <a href="https://llm-rosetta.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/docs-%E4%B8%AD%E6%96%87-2f6fdd" alt="Docs (中文)"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT"></a>
  <a href="https://arxiv.org/abs/2604.09360"><img src="https://img.shields.io/badge/arXiv-2604.09360-b31b1b.svg" alt="arXiv"></a>
</p>

<p align="center">
  <a href="README_en.md">English</a> | 中文
</p>

**任意受支持的 LLM API 格式进，任意受支持的格式出——模型来源由你自己掌控。**
LLM-Rosetta 通过中心化中间表示（IR），在 OpenAI、Anthropic、Google 等提供方的
API 格式之间双向转换。可以把它当作 Python 库嵌入应用，也可以运行自托管网关，
作为 Claude Code、Codex CLI、Aider 以及各类 OpenAI 兼容客户端的后端。网关把上游
配置、凭据和路由都留在你自己的部署里；LLM-Rosetta 本身不会在链路中增加托管聚合
服务。Python 3.11+ 上没有第三方运行时依赖；3.10 上也只多一个 `typing_extensions`，它是标准库 `typing` 模块的向后移植包
（backport）。

## 为什么需要它

如果要两两支持 N 种格式，就需要 O(N²) 个适配器——每对格式之间都要写一套转换逻辑。
把 IR 放在中间后，每种格式只需要一个与 IR 互转的双向转换器，任意格式之间的转换
即可实现：

```text
提供方 A ──→ IR ──→ 提供方 B
提供方 C ──→ IR ──→ 提供方 D
```

请求、响应、流式 chunk 和工具调用都走同一条路径。新增一种格式需要一个转换器；
在已有格式上新增一个提供方，需要一个 shim，而不是为每个已有提供方各写一个适配器。

### 为什么不直接用支持多种格式的提供方？

如果客户端和提供方原生使用同一种方言，你不需要转换层。LLM-Rosetta 处理的是这种
配对覆盖不到的情况。

- **兼容层只覆盖部分字段。** 提供方的 `responses` 或 `messages` 端点只实现它选择的字段。
  其余字段——`store`、thinking 签名、缓存控制——可能被拒绝、以不同方式承载，或者
  直接丢弃。shim 记录了每个上游接受什么，然后适配请求或报告损失。
- **会话要在提供方之间迁移。** 中途换提供方，需要转换的是已有对话记录：工具调用、
  推理块、图片和缓存标记。签名、提供方生成的 id 这类不透明产物，无法变成对另一家
  有效的值；它们会被保留、剥离或替换，并报告损失。
- **有些配对没有原生路径。** Codex CLI 使用 Responses 方言，不支持该方言的提供方
  接不了；Claude Code 使用 Messages 方言，已经加了兼容端点的提供方能覆盖常见情况，
  但字段覆盖仍然因提供方而异。

如果一组稳定的客户端—提供方配对已经可用，就不要加转换层。需要让客户端和提供方各自
可以独立替换时再用——并接受转换有时会丢信息。

### 为什么不用 LiteLLM 或 OpenRouter？

它们解决的是相邻的问题，但中心不同。

**OpenRouter 主要是一项托管服务。** 在标准使用方式下，OpenRouter 管理提供方
接入和路由，你的流量会经过其服务；它也提供自带密钥选项。这正是它的长处：不用在每家提供方分别注册就能用到模型。LLM-Rosetta 的前提相反——每个上游凭据都归你所有，网关由你
自己运行。

**LiteLLM 可自托管，并提供广泛的提供方支持。** 它还提供团队密钥、预算和花费追踪。
它的核心接口基本沿用 OpenAI 格式，其他格式都映射到这个中心上。LLM-Rosetta 的 IR
不属于任何提供方：每种支持的格式都能放在任意一侧，客户端方言和模型来源因此互相
独立，应用也可以把 payload 保存在 IR 里。转换器检测到的损失会以 warning 报告，
不会被静默丢弃。

两边的覆盖范围都变化很快；请以它们当前的文档为准。

- **选 OpenRouter**：你完全不想自己持有提供方账户。
- **选 LiteLLM**：你需要广泛的提供方支持或团队级成本控制。
- **选 LLM-Rosetta**：模型来源归你所有，而你需要用不同格式的客户端去访问它们，
  并且希望转换损失可检查。

## 三种控制粒度与一个工具箱

| | 工具箱 | 原始转换器 | 管线 | 网关 |
|---|---|---|---|---|
| **适用场景** | 转换单个 IR 值或工具 | 在自己的循环里转换单条链路 | 一次调用完成 source → IR → target | 面向任意客户端的 HTTP 代理 |
| **入口** | `tool_ops`、`transforms`、`capabilities` | `ChatCompletionsConverter()` | `convert()` / `ConversionPipeline` | `llm-rosetta-gateway` |
| **调用提供方** | 你 | 你 | 你 | 网关 |
| **路由、密钥、管理面板** | — | — | — | 内置 |

三种粒度共用同一套 IR 和转换器，工具箱也基于同一套 IR。从需要的层级开始，向上迁移时
不用重写翻译逻辑。`tool_ops` 在单个工具定义、选择、调用或结果与 IR 之间转换；
`transforms` 和 `capabilities` 只应用一个字段变换或能力 pass，不需要创建转换器。

Python 3.11+ 没有第三方运行时依赖；3.10 上也只需要 `typing_extensions`，它是标准库 `typing` 模块的向后移植包（backport）。对多提供方网关来说这并不常见，也意味着同一条管线可以在不同应用和
部署之间搬动，不需要解析依赖图，只要 Python 版本兼容即可。

### 原始转换器

```python
from llm_rosetta import ChatCompletionsConverter, MessagesConverter

openai_request = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
}

# OpenAI Chat Completions → IR → Anthropic Messages
ir = ChatCompletionsConverter().request_from_provider(openai_request)
anthropic_request, warnings = MessagesConverter().request_to_provider(ir)
```

用你自己的客户端调用提供方 API，再把响应转成 IR：

```python
ir_response = MessagesConverter().response_from_provider(response)
```

流式转换是有状态的：转换器用一个 `StreamContext` 跨 chunk 保存状态。详见
[流式转换](https://llm-rosetta.readthedocs.io/zh-cn/latest/guide/streaming/)。

### 管线

`convert()` 会解析来源与目标的 shim，一次调用走完 source → IR → target 的完整路径。
需要保留管线对象、传入 shim 或读取 `pipeline.warnings` 时，用 `ConversionPipeline`
的 `convert_request()` 和 `convert_response()`。

```python
from llm_rosetta import convert, detect_provider

detect_provider(openai_request)                          # 'chat_completions'
anthropic_request = convert(openai_request, target_provider="messages")
```

管线会解析两侧的 shim，因此目标 shim 的默认值可能让输出与原始转换器不同——Anthropic
默认的 `cache_control` 断点就是一例。两者共用 IR 和转换器，差别来自 shim 层。

IR 是类型化的、提供方中立的 payload。你可以把对话历史、路由状态或评测数据都保存在
IR 里，只在调用边界把它渲染成目标提供方的方言。换提供方时改变的是渲染器，不是应用
的数据模型。核心库在 Python 3.11+ 上没有第三方依赖（3.10 仅需 `typing_extensions`），
而且 payload 始终保存在 IR 里，因此支持另一种 API 格式不会给应用增加依赖或格式专用
分支。

### 网关

```bash
pip install llm-rosetta-gateway
llm-rosetta-gateway init     # 生成模板配置
llm-rosetta-gateway          # 在 http://localhost:8765 启动
```

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

向 Anthropic 模型发送 OpenAI 格式的请求——网关会双向转换，也支持流式响应：

```bash
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hello!"}]}'
```

配置了网关密钥的共享部署接受各格式的原生鉴权头——OpenAI 风格路由用
`Authorization: Bearer <key>`，Messages 用 `x-api-key: <key>`，Google 用
`x-goog-api-key: <key>`（或 `?key=<key>`）——并在任何路由上也接受裸
`Authorization: Bearer`，多数 CLI 客户端发送的就是这种形式。反过来不成立：
OpenAI 风格路由会拒绝 `x-api-key`。

内置管理面板位于 `http://localhost:8765/admin/`，可管理提供方、模型、API 密钥和
可观测性数据：

![管理面板](https://llm-rosetta.readthedocs.io/zh-cn/latest/gateway/images/admin-dashboard.png)

网关可作为 Claude Code、Codex CLI、Aider、Antigravity CLI、OpenCode 和
Kilo Code 的后端，也可以作为 Ollama 前面的转换代理——见
[CLI 集成](https://llm-rosetta.readthedocs.io/zh-cn/latest/gateway/cli-integrations/)。

## 支持的格式

| 模型类型 | 格式 | 网关路由 |
|----------|------|----------|
| LLM（chat） | `chat_completions`、`open_responses`、`messages`、`google_generate`、`google_interactions` | `/v1/chat/completions`、`/v1/responses`、`/v1/messages`、`/v1beta/models/…`、`/v1beta/interactions` |
| Embeddings | `openai`、`cohere`、`jina`、`voyage`、`google_generate`、`minimax` | `/v1/embeddings` |
| Rerank | `jina`、`cohere`、`voyage` | `/v1/rerank`、`/v2/rerank` |
| Decisions | `systemone`、`openai_decisions` | `/v1/decisions`、`/v1/systemone` |

每种格式都能与 IR 互转（请求和响应）；chat 格式还支持流式 chunk 转换。
`open_responses` 是提供方中立的 Open Responses 规范；OpenAI profile
（`openai--open_responses` shim）在此基础上增加 `resp_` ID、`store: true` 和
回显字段。

### Provider Shim

提供方与基础格式的差异体现在字段名、默认值、鉴权和 URL 路径上。shim 声明这些差异，
转换器因此保持通用。内置 39 个 shim，覆盖 19 个提供方前缀，包括 Anthropic、Cohere、
DeepSeek、Google、Jina、MiniMax、Moonshot、OpenAI、OpenRouter、Qwen、Volcengine、
Voyage、xAI 和 Zhipu。

大多数 shim 名采用 `<vendor>--<format>` 形式，例如 `openrouter--open_responses`、
`deepseek--chat_completions`、`anthropic--messages`。少数单 lane shim 保留裸提供方名——
`moonshot`、`qwen`、`xai` 和 `zhipu`——它们的 embedding 与 rerank 兄弟节点仍带后缀。
使用裸格式类型的提供方（例如
`type: open_responses`）不带提供方默认值，必须显式设置 `base_url`；提供方 shim 会提供
默认 URL 和凭据环境变量。shim 还可以带凭据交换（`auth.py`）和额度探测
（`quota.py`）。详见
[Provider Shim](https://llm-rosetta.readthedocs.io/zh-cn/latest/guide/shims/)。

### 兼容服务

符合上述任一格式规范的服务可以配置为上游。[Ollama](https://ollama.com/)（v0.14.0+）提供
其中三种格式，因此网关可以用任意一种方言代理它：

| Ollama 端点 | 格式 | 起始版本 |
|-------------|------|----------|
| `/v1/chat/completions` | `chat_completions` | 早期版本 |
| `/v1/responses` | `open_responses` | v0.13.3 |
| `/v1/messages` | `messages` | v0.14.0 |

其他兼容服务——包括 vLLM、Hugging Face TGI 和 LM Studio——可尝试通过它们各自
暴露的兼容格式接入。

## 功能

- **双向转换**——覆盖请求、响应和流式 chunk，并提供有状态的流式上下文和类型化
  流事件。
- **丰富内容**——IR 可以表示文本、图片、工具调用与结果、推理/思考、引用和
  token 用量。
- **自动识别**——从请求体判断来源格式，或用 `convert()` 一步完成转换。
- **IR 作为应用 payload**——用提供方中立的形式保存历史和编排状态，只在调用时渲染成
  目标方言。
- **Provider shim**——按提供方声明字段变换、URL 默认值、凭据交换和额度探测，无需
  写提供方专用转换代码。
- **网关**——加权轮询与亲和路由、密钥轮换、故障转移、限流、请求日志、指标、
  错误转储和用量报告。
- **管理面板**——管理提供方、模型、密钥、日志和设置，无需手改配置文件。
- **Python ≥ 3.11 无第三方依赖**——3.10 上也只需 `typing_extensions`，它是标准库 `typing` 模块的向后移植包（backport）；网关内置 HTTP 栈。

## 安装

### 库

```bash
pip install llm-rosetta
```

需要 **Python ≥ 3.10**。

### 安装网关

```bash
pip install llm-rosetta-gateway
# 过渡别名（保留一个版本），会同时安装两个发行包：
pip install "llm-rosetta[gateway]"
```

Docker（Alpine，约 21 MB）：

```bash
docker run -p 8765:8765 -v /path/to/config:/config oaklight/llm-rosetta-gateway
```

## 合规测试

[llm-comply](https://github.com/Oaklight/llm-comply) 是一个配套工具，按官方规范校验
LLM API 端点。可以用它验证网关或代理是否正确实现了 OpenAI Chat、Open Responses、
Anthropic Messages 和 Google GenAI 格式。

```bash
pip install llm-comply

# 测试你的网关端点
llm-comply -u https://your-gateway/v1 -k $API_KEY -m your-model --format openai-chat
```

托管版本位于
[llm-comply.service.oaklight.top](https://llm-comply.service.oaklight.top)。仓库还
提供一个按需触发的
[Compliance workflow](https://github.com/Oaklight/llm-rosetta/actions/workflows/compliance.yml)。

## 文档

- **English**: [https://llm-rosetta.readthedocs.io/en/latest/](https://llm-rosetta.readthedocs.io/en/latest/)
- **中文**: [https://llm-rosetta.readthedocs.io/zh-cn/latest/](https://llm-rosetta.readthedocs.io/zh-cn/latest/)

## 引用

如果你在研究中使用了 LLM-Rosetta，请引用：

```bibtex
@article{ding2026llm,
  title={LLM-Rosetta: A Hub-and-Spoke Intermediate Representation for Cross-Provider LLM API Translation},
  author={Ding, Peng},
  journal={arXiv preprint arXiv:2604.09360},
  year={2026}
}
```

## 贡献

欢迎贡献！提交 PR 前请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 和
[贡献指南](https://llm-rosetta.readthedocs.io/zh-cn/latest/contributing/guide/)。

## 社区

欢迎通过 [GitHub Issues](https://github.com/Oaklight/llm-rosetta/issues) 或
[LINUX DO](https://linux.do/) 社区讨论本项目。

## 许可证

本项目使用 MIT 许可证——详见 [LICENSE](LICENSE)。
