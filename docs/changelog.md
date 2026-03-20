---
title: 更新日志
---

# 更新日志

LLM-Rosetta 的所有重要变更均记录于此。本项目遵循 [Keep a Changelog](https://keepachangelog.com/) 规范。

## 未发布

### 新增

- **网关请求/响应体日志**：可配置的调试日志，支持彩色输出、请求体脱敏和截断 — 通过配置（`"debug": {"verbose": true, "log_bodies": true}`）、环境变量（`LLM_ROSETTA_VERBOSE`、`LLM_ROSETTA_LOG_BODIES`）或 `--verbose` CLI 参数启用
- **Google `request_to_provider()` 支持 `output_format="rest"`**：传入 `output_format="rest"` 可直接获得 REST API 格式的请求体，`tools`/`tool_config` 提升至顶层，生成参数包装在 `generationConfig` 中 — 无需再手动进行 SDK→REST 格式转换

### 变更

- **网关模块化重构**：将 `app.py`（1057 行）拆分为 `proxy.py`（代理引擎、SSE 处理、上游请求）、`cli.py`（CLI 入口、argparse、子命令）和精简后的 `app.py`（路由处理、应用工厂，约 210 行）
- **Google REST 请求体转换迁移至核心包**：`_fixup_google_body()` 逻辑从 `gateway/proxy.py` 迁移至 `GoogleGenAIConverter._to_rest_body()`，消除了网关和全部 6 个 REST 示例中的重复 SDK→REST 转换代码

### 修复

- OpenAI Responses 流式传输：为 `response.completed` 添加缺失的 `id`/`object`/`model` 字段，为文本增量事件添加 `output_index`/`content_index`，并补充完整的生命周期事件（`output_item.added`、`content_part.added`、`content_part.done`、`output_item.done`）（#56）
- OpenAI Chat 流式传输：`tool_calls` 条目现在始终包含必需的 `index` 字段，当上游 IR 事件未明确提供时默认为 `0`（#57）
- OpenAI Chat 流式传输：usage-only 数据块现在包含 `"choices": []`，以满足要求每个 `chat.completion.chunk` 必须包含 `choices` 数组的客户端验证（#55）
- `stream_options`（Chat Completions 专用字段）不再泄漏到 OpenAI Responses API 请求中 — Responses 转换器的 `ir_stream_config_to_p()` 之前错误地输出了 `stream_options`，导致 Chat 格式客户端（Kilo、OpenCode）通过网关代理到 Responses API 时被上游拒绝（#58）
- Google GenAI 转换器现在可以处理 REST 格式请求中顶层的 tools 和 tool_config（除了 SDK 格式的 `config.tools`）— 之前只识别 SDK 格式，导致网关代理请求中的工具定义被静默丢弃（#59）

## v0.2.0 — 2026-03-18

### 新增

- **独立 API 测试脚本** (`llm_api_simple_tests/`)：20 个测试脚本（每个提供商 5 个），直接使用官方 SDK，覆盖简单查询、多轮对话、图片、函数调用和综合场景 — 作为 git 子模块从 [Oaklight/llm_api_simple_tests](https://github.com/Oaklight/llm_api_simple_tests) 引入
- **LLM-Rosetta Gateway**：跨提供商 HTTP 代理的 REST 网关应用
- CLI 入口 (`llm-rosetta-gateway`) 及网关包结构
- 网关配置文件自动发现：依次搜索 `./config.jsonc`、`~/.config/llm-rosetta-gateway/config.jsonc`、`~/.llm-rosetta-gateway/config.jsonc`
- `--edit` / `-e` 标志：在 `$EDITOR` 中打开配置文件（回退到 nano/vi/vim）
- `--version` / `-V` 标志：显示当前版本
- ASCII 艺术启动横幅，支持 `--no-banner` 选项抑制显示
- `add provider <name>` 子命令：添加提供商条目到配置（支持 `--api-key`、`--base-url` 参数或交互式提示；已知提供商自动填充默认值）
- `add model <name>` 子命令：添加模型路由条目（支持 `--provider` 参数或交互式提示）
- **网关提供商模块** (`providers.py`)：集中管理提供商定义，包括认证头构建器、URL 模板、默认基础 URL 和 API 密钥环境变量名
- **API 密钥轮转**：每个提供商支持逗号分隔的多 API 密钥，通过 `KeyRing` 轮询使用
- **代理支持**：全局 `server.proxy` 和逐提供商 `proxy` 配置，支持 HTTP/SOCKS 代理；CLI `--proxy` 参数覆盖配置
- Makefile 新增 `test-integration` 目标，使用 `proxychains`（如已安装）运行集成测试
- `init` 子命令：在 XDG 默认位置 (`~/.config/llm-rosetta-gateway/`) 创建模板 `config.jsonc` 文件
- **模型列表端点**：`GET /v1/models`（兼容 OpenAI 和 Anthropic SDK）和 `GET /v1beta/models`（Google GenAI SDK 格式）— 使三种 SDK 的 `client.models.list()` 均可正常使用（#54）

### 变更

- 最低 Python 版本提升至 3.10+；迁移至标准库 `typing`（移除 `typing_extensions`）
- 使用 `ruff` 格式化整个代码库
- 更新 Makefile，增加 `lint`、`test` 和 `build` 目标
- 新增 `ty`（类型检查器）配置
- 在 `pyproject.toml` 中配置 `ruff` lint 规则（`E`、`F`、`UP`）；忽略 `UP007`（Union 语法）和 `E501`（行长度）
- 现代化 `src/`、`tests/`、`examples/` 和 `scripts/` 中的 typing 导入 — 将 `typing.Dict`、`List`、`Tuple`、`Optional`、`Type` 替换为标准库内建类型

### 修复

- 修复 Anthropic 提供商流式传输中 usage tokens 为 `null` 时的崩溃 — 所有转换器中 `TypeError: NoneType + int`（将 `.get("*_tokens", 0)` 替换为 `.get("*_tokens") or 0`）
- 网关提供商 `base_url` 验证 — 配置错误（如 `https:example.com` 缺少 `//`）时提前报错并给出清晰提示
- 网关依赖新增 `socksio` 以支持 SOCKS 代理（`httpx[socks]`）
- 补充 `types` 包缺失的 `__init__.py`
- 更新文档中 `git clone` URL，从 `llm-rosetta` 改为 `llm-rosetta`
- 解决 `src/` 中所有 `ty` 类型检查器诊断（31 → 0）：
    - 修复 `is_part_type()` TypeGuard 类型窄化 — 替换为特定类型守卫函数（`is_text_part` 等）
    - 补充缺失的 TypedDict 字段：`TextPart`/`ReasoningPart` 上的 `provider_metadata`，`ImagePart`/`FilePart` 上的 `file_id`
    - 修复 `IRRequest.messages` 类型，从 `Required[Message]` 改为 `Required[Iterable[Message]]`
    - 使用 `cast()` 桥接 `dict[str, Any]` 中间值到 TypedDict 返回类型
    - 修复转换器响应构建器中的 dict 字面量类型推断冲突
- 解决 `tests/` 中所有 `ty` 类型检查器诊断（1506 → 0）：
    - 为传递给期望 TypedDict 参数的函数的 dict 字面量添加 `cast()` 包装（`GenerationConfig`、`IRRequest`、`IRResponse`、`ToolDefinition`、`ToolChoice` 等）
    - 使用 `cast(list[Any], ...)` 或 `cast(Message, ...)` 窄化 `Message | ExtensionItem` 联合类型结果
    - 将 `Iterable` 内容字段转换为 `list` 以支持下标和 `len()` 访问
    - 在对可选返回类型进行下标访问前添加 `assert ... is not None` 守卫
    - 修复 `FinishReason`，从裸字符串改为 TypedDict 形式 `{"reason": "stop"}`
    - 修复 `IRResponse.object` 字面量，从 `"chat.completion"` 改为 `"response"`
- 解决 `src/` 和 `tests/` 中所有 `ruff` lint 违规（UP035 弃用导入、F401 未使用导入）
- Google `thought_signature` 在网关往返中的保留 — 新版 Google 模型要求在函数调用部分中回传 `thoughtSignature`；网关现在按 `tool_call_id` 缓存 `provider_metadata`（含 `thought_signature`），并在后续请求中重新注入，支持流式和非流式模式（#51）
- OpenAI Responses 转换器现在支持全部 3 种 `input` 格式：裸字符串（`"input": "hello"`）、简写列表（`[{"role": "user", "content": "hi"}]`）和结构化列表 — 此前仅支持结构化格式，导致 OpenAI Python SDK 发送的简写项被静默丢弃，跨提供商转换到 Anthropic 或 Google 时生成空 IR 消息

---

## 2026-03-15 — 品牌重塑为 LLM-Rosetta

### 变更

- **项目从 LLM-Rosetta 重命名为 LLM-Rosetta**，涵盖所有代码、文档及配置
- 包名从 `llm-rosetta` 改为 `llm_rosetta`；`pyproject.toml` 相应更新
- 使用 Zensical 全面重写英文 (`docs_en`) 和中文 (`docs_zh`) 文档
- README（中/英）更新品牌标识、徽章及 `pyproject.toml` 元数据

---

## 2026-03-06 — 流式传输与 StreamContext

### 新增

- **`StreamContext`**：为所有 4 个提供商提供有状态的流式数据块处理
- 所有转换器新增 `stream_response_from_provider()` 和 `stream_response_to_provider()` 方法
- `accumulate_stream_to_assistant_message()` 辅助函数
- `BaseConverter` 新增流式抽象方法（`stream_response_to_provider`、`stream_response_from_provider`）
- 4 种新 IR 流式事件类型：`StreamStart`、`StreamEnd`、`ContentBlockStart`、`ContentBlockEnd`
- `ReasoningDeltaEvent` 及 IR 流式类型新增 `tool_call_index` 字段
- 所有提供商组合的跨提供商流式示例（SDK 和 REST 版本）
- 示例中图片下载新增本地文件缓存和重试逻辑

### 变更

- 流式方法签名更新，增加可选 `context` 参数
- 移除已弃用的 `from_provider` 方法；`auto_detect` 更新为新 API
- 移除过时的单提供商示例脚本（已被跨提供商示例替代）
- `_normalize()` 提取至 `BaseConverter` 作为共享工具方法

### 修复

- Google GenAI REST 流式/响应字段的 camelCase 回退处理
- Anthropic 流式转换器：`thinking_delta`、`signature_delta`、`tool_call_id` 处理
- OpenAI Chat 流式转换器：`reasoning_content`、空字符串、`tool_call_index` 处理
- 补充测试包发现所需的 `__init__.py`
- `google_genai_rest_e2e` 集成测试中的 `from_provider` 调用

---

## 2026-02-14 — 跨提供商示例与流式转换器

### 新增

- 所有 4 个提供商的**流式转换器**：OpenAI Chat、Anthropic、Google GenAI、OpenAI Responses
- 所有提供商的流式转换器单元测试
- **6 个跨提供商对话示例**（基于 SDK）：OpenAI Chat ↔ Anthropic、OpenAI Chat ↔ Google GenAI、OpenAI Chat ↔ OpenAI Responses、Anthropic ↔ Google GenAI、Anthropic ↔ OpenAI Responses、Google GenAI ↔ OpenAI Responses
- 跨提供商对话示例的公共资源模块
- Google GenAI 兼容性的图片 URL 转内联 base64 辅助工具
- OpenAI Responses E2E 集成测试（REST + SDK）
- OpenAI Responses Ops 类及转换器的单元测试
- 示例 README（中英文）

### 变更

- **OpenAI Responses 转换器**重构为 Bottom-Up Ops 模式
- 重构后清理：移除弃用工具和空目录

### 修复

- 为 Google GenAI 提供商兼容性将图片 URL 转换为内联 base64

---

## 2026-02-13 — Bottom-Up Ops 架构

### 新增

- **Google GenAI 转换器**使用 Bottom-Up Ops 模式重建
- **OpenAI Responses API** 类型的 TypedDict 副本
- **Google GenAI SDK** 类型的 TypedDict 副本
- Google GenAI REST 和 SDK E2E 集成测试
- `google_genai` 转换器 Ops 类的单元测试
- Anthropic SDK 和 REST E2E 集成测试
- OpenAI Chat E2E 测试拆分为 SDK 和 REST 版本
- **GitHub Actions** CI/CD 工作流及 Dependabot 配置

### 变更

- **Anthropic 转换器**重新设计为 Bottom-Up Ops 架构
- 导入更新为使用新的 `google_genai` 转换器模块
- 移除旧的 `google/` 转换器及遗留测试

---

## 2026-02-12 — 转换器重新设计

### 新增

- **Anthropic SDK** 类型的 TypedDict 副本
- **OpenAI Chat** 类型的 TypedDict 副本，包含向后兼容性和测试
- 保留遗留 body 转换器设计作为历史参考

### 变更

- **OpenAI Chat 转换器**使用 Bottom-Up Ops 架构重新设计
- 修复整个代码库的 ruff lint 错误

---

## 2026-01-06 — 分层架构与文档

### 新增

- 初始化英文和中文文档结构（`docs_en`、`docs_zh`）
- 完整的错误处理文档
- OpenAI Chat Converter 集成测试
- `BaseConverter` 测试类的完整 mock 实现
- 基础转换器的文件处理功能
- 提供商到 IR 的映射文档

### 变更

- 转换器基类完善为分层抽象模板
- 所有 4 个转换器重构为分层架构（Anthropic、OpenAI Chat、OpenAI Responses、Google GenAI）
- IR 内容/部件转换方法的类型注解更新
- IR 类型系统重组和增强
- 代码注释和文档字符串添加英文翻译

### 修复

- 修正推理内容字段断言
- 修复 OpenAI Chat Completions 转换器的文件内容处理

---

## 2026-01-05 — 自动检测与包成熟化

### 新增

- **`detect_provider()`**：自动检测提供商格式
- **`convert()`**：一步格式转换便捷函数
- 消息验证中支持 `developer` 角色
- `BaseConverter`、Anthropic、Google GenAI 和 OpenAI 转换器的综合验证测试
- 工具调用和工具定义转换测试
- pytest 配置及 `pytest-cov` 依赖
- 竞品分析文档

### 变更

- **包重命名**：从 `llm-provider-converter` 改为 `llm-rosetta`
- 所有提供商标准化 IR 格式用法
- 示例中使用 `Message` 类标准化消息创建
- 测试套件从 unittest 迁移至 pytest
- 提取公共逻辑至共享工具模块

### 修复

- OpenAI Responses 转换器中无当前消息上下文时的独立工具调用处理
- Google GenAI Pydantic 模型处理重新排序以兼容元组
- OpenAI 单文本部件的内容处理逻辑简化

---

## 2026-01-04 — 示例与打包

### 新增

- `pyproject.toml` 包配置
- 带工具集成的多轮对话示例
- 多轮对话示例中的 Anthropic 切换
- 多轮对话示例中的 Google GenAI 函数调用

### 变更

- 工具函数从转换器移至 IR 类型模块
- OpenAI Chat 转换器代码格式化改进
- 移除弃用的多提供商查询和天气工具模块

---

## 2025-12-24 — 初始实现

### 新增

- **IR 类型系统**：消息、内容部分、工具、配置、请求/响应的中间表示类型
- **`BaseConverter`** 抽象类：LLM 提供商转换基类
- **`AnthropicConverter`**：Anthropic Messages API 双向转换
- **`OpenAIChatConverter`**：OpenAI Chat Completions API 双向转换
- **`OpenAIResponsesConverter`**：OpenAI Responses API 双向转换
- **`GoogleGenAIConverter`**：Google GenAI SDK 格式双向转换
- 所有 4 个转换器的综合测试套件
- 包初始化和导出
- 带模拟数据的天气工具示例

---

## 2025-12-09 — 研究与设计

### 新增

- 初始项目结构
- LLM 提供商消息类型 schema 文档及比较
- 提供商消息 IR 设计文档
- 跨提供商 MCP 支持对比（OpenAI、Anthropic、Google）
- Google GenAI Interactions API 类型分析
- 多提供商查询示例函数
- 查询示例中增加 OpenAI Responses API 支持
