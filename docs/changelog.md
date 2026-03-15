---
title: 更新日志
---

# 更新日志

LLM-Rosetta 的所有重要变更均记录于此。本项目遵循 [Keep a Changelog](https://keepachangelog.com/) 规范。

## 未发布

### 新增

- **LLM-Rosetta Gateway**：跨提供商 HTTP 代理的 REST 网关应用
- CLI 入口 (`llm-rosetta-gateway`) 及网关包结构

### 变更

- 最低 Python 版本提升至 3.10+；迁移至标准库 `typing`（移除 `typing_extensions`）
- 使用 `ruff` 格式化整个代码库
- 更新 Makefile，增加 `lint`、`test` 和 `build` 目标
- 新增 `ty`（类型检查器）配置

### 修复

- 补充 `types` 包缺失的 `__init__.py`
- 更新文档中 `git clone` URL，从 `llmir` 改为 `llm-rosetta`

---

## 2026-03-15 — 品牌重塑为 LLM-Rosetta

### 变更

- **项目从 LLMIR 重命名为 LLM-Rosetta**，涵盖所有代码、文档及配置
- 包名从 `llmir` 改为 `llm_rosetta`；`pyproject.toml` 相应更新
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

- **包重命名**：从 `llm-provider-converter` 改为 `llmir`
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
