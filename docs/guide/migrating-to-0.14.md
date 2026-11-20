---
title: 迁移到 0.14
---

# 迁移到 0.14

0.14.0 移除了 0.13.0 为过渡保留一个版本的已弃用导入路径与类别名（epic [#919](https://github.com/Oaklight/llm-rosetta/issues/919)）。被移除的名字会以普通的 `ImportError`、`ModuleNotFoundError` 或 `AttributeError` 失败；[retired-names](#retired-names) 诊断只覆盖提供方、shim 与模型类型名称，不覆盖 Python 导入路径。本页列出已移除的内容、替代写法，以及仍然可用的部分。要改写网关配置和数据库，请使用 [`llm-rosetta-migrate`](#llm-rosetta-migrate)；[升级流程](#upgrade-recipe)给出了步骤顺序。

截至 [已移除的 helper 模块、网关存根与 `tool_ops` 别名](#removed-shims) 的各行已针对 `origin/master`（commit `756f084169`）核对。从 [已移除的 facade](#removed-facades) 起的各行已针对 PR [#1656](https://github.com/Oaklight/llm-rosetta/pull/1656) 的 head（commit `47fe04b2ea`）核对：每个被移除的名字导入失败或抛出异常，每个替代写法都能导入。

## 已移除的转换器包 {#removed-converter-packages}

| 已移除的导入路径 | 改用 |
| --- | --- |
| `llm_rosetta.converters.openai_chat` | `llm_rosetta.converters.llm.chat_completions` |
| `llm_rosetta.converters.openai_responses` | `llm_rosetta.converters.llm.open_responses` |
| `llm_rosetta.converters.anthropic` | `llm_rosetta.converters.llm.messages` |
| `llm_rosetta.converters.google_genai` | `llm_rosetta.converters.llm.google_generate` |
| `llm_rosetta.converters.google_generate` | `llm_rosetta.converters.llm.google_generate` |
| `llm_rosetta.converters.google_interactions` | `llm_rosetta.converters.llm.google_interactions` |
| `llm_rosetta.converters.embedding` | `llm_rosetta.converters.embeddings` |

`google_generate` 与 `google_interactions` 两行是旧的顶层位置，同名模块现在位于 `converters.llm` 之下。子模块路径跟随其包：`converters.openai_chat.converter` 变为 `converters.llm.chat_completions.converter`。

## 已移除的类名 {#removed-class-names}

| 已移除的名字 | 改用 |
| --- | --- |
| `OpenAIChatConverter` | `ChatCompletionsConverter` |
| `AnthropicConverter` | `MessagesConverter` |
| `GoogleGenAIConverter` | `GoogleGenerateConverter` |
| `GoogleConverter` | `GoogleGenerateConverter` |

这四个名字已从 `llm_rosetta`、`llm_rosetta.converters`、`llm_rosetta.converters.llm` 以及规范包（`converters.llm.chat_completions`、`converters.llm.messages`、`converters.llm.google_generate`）中移除。替代名从相同位置导入。`ChatCompletionsConverter` 与 `MessagesConverter` 也从 `llm_rosetta.converters.llm.<format>` 导出。

## 已移除的 ops 与 stream-context 别名 {#removed-ops-aliases}

| 已移除的名字 | 改用 |
| --- | --- |
| `OpenAIChatConfigOps`、`OpenAIChatContentOps`、`OpenAIChatMessageOps`、`OpenAIChatToolOps` | `converters.llm.chat_completions` 中的 `ChatCompletions*Ops` |
| `AnthropicConfigOps`、`AnthropicContentOps`、`AnthropicMessageOps`、`AnthropicToolOps` | `converters.llm.messages` 中的 `Messages*Ops` |
| `OpenAIResponsesConfigOps` | `OpenResponsesConfigOps` |
| `OpenAIResponsesContentOps` | `OpenResponsesContentOps` |
| `OpenAIResponsesMessageOps` | `OpenResponsesMessageOps` |
| `OpenAIResponsesToolOps` | `OpenResponsesToolOps` |
| `OpenAIResponsesStreamContext` | `OpenResponsesStreamContext` |

`OpenResponses*` 位于 `llm_rosetta.converters.llm.open_responses`。`OpenAIResponsesConverter` 是真实的类（Open Responses 转换器的 OpenAI profile），**没有**被移除。

## 已移除的 helper 模块、网关存根与 `tool_ops` 别名 {#removed-shims}

在 [#1647](https://github.com/Oaklight/llm-rosetta/pull/1647) 中移除。这些 shim 曾声明保留到 0.16.0，现在 0.14.0 即移除。导入已移除的路径会抛 `ModuleNotFoundError`。

| 已移除的导入路径 | 改用 |
| --- | --- |
| `llm_rosetta.converters.base.helpers.cache_breakpoints` | `llm_rosetta.ir.primitives.cache_breakpoints` |
| `llm_rosetta.converters.base.helpers.image_limit` | `llm_rosetta.ir.primitives.image_limit` |
| `llm_rosetta.converters.base.helpers.system_message_hoist` | `llm_rosetta.ir.primitives.system_message_hoist` |
| `llm_rosetta_gateway.auth` | `llm_rosetta_gateway.middleware.auth` |
| `llm_rosetta_gateway.headers` | `llm_rosetta_gateway.middleware.headers` |
| `llm_rosetta_gateway.embeddings` | `llm_rosetta_gateway.pipelines.embeddings` |
| `llm_rosetta_gateway.migrations` | `llm_rosetta_gateway.maintenance.migrations` |

| 已移除的 `llm_rosetta.tool_ops` 函数 | 改用 |
| --- | --- |
| `to_openai_chat` | `to_chat_completions` |
| `from_openai_chat` | `from_chat_completions` |
| `to_openai_responses` | `to_open_responses` |
| `from_openai_responses` | `from_open_responses` |
| `to_anthropic` | `to_messages` |
| `from_anthropic` | `from_messages` |

`tool_ops.to_provider` 及其他以 provider 为键的函数不再接受旧的 provider 拼写（`openai_chat`、`anthropic`、`google-genai`，以及[提供方类型](#provider-types)表中的其余名字）。它们抛 `ValueError`，消息会写出替代名，例如 `'openai_chat' was renamed to 'chat_completions' in 0.14 and the old spelling is no longer accepted (use 'chat_completions')`。

## 已移除的 facade（P4） {#removed-facades}

在 [#1648](https://github.com/Oaklight/llm-rosetta/pull/1648) 中移除。每个模块都只是右侧规范模块的星号再导出。它们在 0.13.0 中都不存在，因此没有已发布的导入路径被破坏；导入其中任何一个都会抛出 `ModuleNotFoundError`（`ImportError` 的子类）。

| 已移除的导入路径 | 改用 |
| --- | --- |
| `llm_rosetta.converters.base.helpers.tool_group` | `llm_rosetta.ir.primitives.tool_group` |
| `llm_rosetta.converters.base.helpers.truncate` | `llm_rosetta.ir.primitives.truncate` |
| `llm_rosetta.converters.base.tools.call_unwind` | `llm_rosetta.ir.primitives.call_unwind` |
| `llm_rosetta.converters.base.tools.allowed_tools` | `llm_rosetta.ir.tools.allowed_tools` |
| `llm_rosetta.converters.base.tools.batch` | `llm_rosetta.ir.tools.batch` |
| `llm_rosetta.converters.base.tools.call_id` | `llm_rosetta.ir.tools.call_id` |
| `llm_rosetta.converters.base.tools.error_marker` | `llm_rosetta.ir.tools.error_marker` |
| `llm_rosetta.converters.base.tools.intrinsic` | `llm_rosetta.ir.tools.intrinsic` |
| `llm_rosetta.converters.base.tools.orphan_fix` | `llm_rosetta.ir.tools.orphan_fix` |
| `llm_rosetta.converters.base.tools.strict_mode` | `llm_rosetta.ir.tools.strict_mode` |
| `llm_rosetta_gateway.middleware.sanitize` | `llm_rosetta_gateway.observability.sanitize` |

`from llm_rosetta.converters.base.tools import batch`、`intrinsic` 以及 `helpers.tool_group` 现在会抛出 `ImportError`：这三者是模块属性，不是包 `__init__` 导出的名字。`converters.base.helpers` 与 `converters.base.tools` 自己导出的名字不变。星号再导出过去连带泄露、但包 `__init__` 从未导出的名字（`ToolGroup`、`group_tools`、`tool_call_group`、`report_loss`、`report_drop`、`ToolChoice`、`ToolRef`、`truncate_with_digest`、`history_native_calls`、`UNANSWERED_TOOL_CALL`）现在只能从 `llm_rosetta.ir.*` 下导入。

## 已移除的名称别名（P5） {#removed-name-aliases}

在 PR [#1656](https://github.com/Oaklight/llm-rosetta/pull/1656) 中移除。别名表已删除；已退役的提供方类型或 shim 名称不会再被翻译，而是经由 [retired-names](#retired-names) 表抛出错误，写出替代名称，并以迁移命令结尾。

| 已移除的名字 | 改用 |
| --- | --- |
| `llm_rosetta.provider_names`（模块） | 直接传规范名称；不再有需要规范化的步骤 |
| `canonical_provider_name`、`canonical_shim_name`、`normalize_provider_name` | 从 [retired-names](#retired-names) 表中查规范名称 |
| `warn_legacy_provider_names`、`warn_legacy_shim_names` | 无；退役名称会直接抛错 |
| `LEGACY_PROVIDER_ALIASES`、`LEGACY_SHIM_ALIASES` | `llm_rosetta.retired_names.RETIRED_NAMES` |
| `legacy_names`（含 `llm_rosetta.api` 中的导出） | `llm_rosetta.retired_names.RETIRED_NAMES` |
| `ProviderType` 中的旧成员 | 规范格式名 |

这些名字已从 `llm_rosetta`、`llm_rosetta.api` 与 `llm_rosetta.fidelity` 中移除。会抛错的边界有：`get_converter_for_provider`、`ConversionPipeline`、`convert`、`convert_response`、embedding 转换器、`ProviderShim`、`provider.yaml` 的 format，以及网关配置。

网关在加载配置时不再改写退役名称，因此仍写着退役名称的配置无法启动。错误信息写出文件与键名，并以迁移命令结尾，例如：

```text
llm-rosetta-gateway: error: config.jsonc: providers.p.type: 'openai_chat' was renamed to 'chat_completions' in 0.14 and the old spelling is no longer accepted (use 'chat_completions'); run 'llm-rosetta-migrate --config <path>' (add --write to apply), see the migration guide
```

进程以退出码 1 结束，不打印 traceback。在代码里，该异常是 `RetiredConfigName`（`ValueError` 的子类），消息格式为 `<配置文件>: <json 路径>: <retired-names 消息>`。

## 已移除的 shim 声明（P6） {#removed-shim-declarations}

在 PR [#1656](https://github.com/Oaklight/llm-rosetta/pull/1656) 中移除。`ProviderShim(...)` 会抛出写明替代写法的 `TypeError`，loader 会拒绝仍使用旧键的 `provider.yaml`。

| 已移除的 `ProviderShim` 关键字 | 改用 |
| --- | --- |
| `base` | `format` |
| `from_transforms` | `pre_ir_transforms` |
| `to_transforms` | `post_ir_transforms` |
| `default_base_url` | `connection=ConnectionConfig(base_url=...)` |
| `default_api_key_env` | `connection=ConnectionConfig(api_key_env=...)` |
| `models_path`、`default_path`、`model_id_field` | `connection=ConnectionConfig(<同名>=...)` |
| `supports_custom_tools` | `tools=ToolsConfig(custom_tools=...)` |
| `max_tool_description_length` | `tools=ToolsConfig(max_description_length=...)` |
| `tool_search_mode` | `tools=ToolsConfig(search_mode=...)` |
| `multimodal_tool_result` | `tools=ToolsConfig(multimodal_result=...)` |

| 已移除的 `provider.yaml` 键 | 改用 |
| --- | --- |
| `base:` | `format:` 与 `model_type:` |
| `default_base_url:` | `connection.base_url` |
| `default_api_key_env:` | `connection.api_key_env` |
| 写在顶层的连接键（`base_url`、`models_path`、`default_path`、`model_id_field`、`auth_header`、`keyless`、`token_command`，以及 `ConnectionConfig` 的其他字段） | 放到 `connection:` 下的同名键 |
| `supports_custom_tools`、`max_tool_description_length`、`tool_search_mode`、`multimodal_tool_result`、`tool_groups`、`intrinsic_tools` | `tools.custom_tools`、`tools.max_description_length`、`tools.search_mode`、`tools.multimodal_result`、`tools.tool_groups`、`tools.intrinsic_tools` |
| 导出 `from_transforms` / `to_transforms` 的 `transforms.py` | `pre_ir_transforms` / `post_ir_transforms` |

```yaml
# 之前
name: deepseek
format: chat_completions
default_base_url: https://api.deepseek.com
default_api_key_env: DEEPSEEK_API_KEY

# 之后
name: deepseek
model_type: llm
format: chat_completions
connection:
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
```

同一步骤中移除的其他内容：

| 已移除的名字 | 改用 |
| --- | --- |
| `llm_rosetta.shims.provider_shim.resolve_base`（以及 `llm_rosetta.resolve_base`） | `resolve_format` |
| `llm_rosetta.shims.transforms`（模块） | `llm_rosetta.transforms` |
| `llm_rosetta.ir.tools.fix_orphaned_tool_calls_ir` | `normalize_tool_pairing` |
| `GoogleInteractionsToolOps.ir_function_call_to_p`、`ir_function_result_to_p` | `ir_tool_call_to_p`、`ir_tool_result_to_p` |

`ProviderShim.base`、`.default_path` 与 `.max_tool_description_length` 作为只读属性保留，因为网关或 shim 包里仍有读取方。管理面板的 shim 编辑器现在用 loader 自己的检查来校验保存的 `provider.yaml`，因此对 `base:` 给出与 loader 相同的拒绝信息。

## 已移除的模型类型写法（P7） {#removed-model-type-spellings}

在 PR [#1656](https://github.com/Oaklight/llm-rosetta/pull/1656) 中移除。`retired_names.ALIASES_ENABLED` 与模型类型别名辅助项（`LEGACY_MODEL_TYPES`、`canonical_model_type`、`legacy_model_type`）已删除：retired-names 表的每一行都会抛错，没有开关。模型类型名称是 `llm`、`embeddings`、`rerank` 与 `decisions`。

已退役的模型类型会在以下位置失败，并写出替代名称与迁移命令：类型化转换器注册表与管道、`ProviderShim(model_type=...)`、`provider.yaml` 的 `model_type` 及其 `embedding:` 块、用量读取器、保真度检查器、配置中的 `type:` 与 `capabilities:`、管理面板的模型编辑器，以及 `/models` 上的 `?type=`（HTTP 400）。

网关配置加载器不再读取下列键；每一项都会在加载时失败，并写出替代它的键名。

| 已移除的配置键 | 改用 |
| --- | --- |
| `embedding_providers` | `embeddings_providers` |
| `embedding_models` | `embeddings_models` |
| `default_embedding_format` | `default_embeddings_format` |
| 提供方的 `embedding_format`、`embedding_path`、`embedding_shim`、`embedding_enabled`，或其 `embedding:` 块 | `embeddings_format`、`embeddings_path`、`embeddings_shim`、`embeddings_enabled`，或 `embeddings:` 块 |
| `decision_*` 与 `default_decision_*` 对应项 | `decisions_*` 与 `default_decisions_*` |
| 模型的 `"type": "embedding"` / `"decision"`，或 `capabilities` 中的这种写法 | `"embeddings"` / `"decisions"` |

带版本号的 `config_version` 迁移仍会升级比当前版本更旧的文档，因为它们通过 retired-names 表读取历史写法。只有已经处于当前版本的文档才会因旧写法被拒绝。以当前名称书写的段（`embeddings_providers`、`rerank_providers`）仍然可读。

## 用量账本中的退役标签（P8） {#ledger-tags}

在 PR [#1656](https://github.com/Oaklight/llm-rosetta/pull/1656) 中移除。`openai_chat`、`openai_responses`、`anthropic` 与 `google` 不再是 `STANDARD_SHAPES` 或 `ADDITIVE_CACHE_STANDARDS`（`llm_rosetta.observability.usage`）的行。用量词汇表只列当前标准：`chat_completions`、`open_responses`、`messages`、`google_generate`。

以退役标签写入的数据库行不会被误读，也不会让网关无法启动。对这样的行，`shape_for` 返回 `None`，`cache_is_additive` 返回 `False`，因此启动时的总量修复会保持该行总量为旧写入方存下的值。`backfill_total_tokens` 记录一条警告，写出这类行的数量、退役标签和命令 `llm-rosetta-migrate --db`。运行该命令即可改写标签，见[下文](#llm-rosetta-migrate)。

## 在代码中查找旧名字 {#find-old-names}

```bash
grep -rnE "converters\.(openai_chat|openai_responses|anthropic|google_genai|google_generate|google_interactions|embedding)\b|\b(OpenAIChat|Anthropic)(Converter|(Config|Content|Message|Tool)Ops)\b|\bGoogle(GenAI)?Converter\b|\bOpenAIResponses(Config|Content|Message|Tool)Ops\b|\bOpenAIResponsesStreamContext\b" .
```

## 仍然可用的部分 {#still-works}

- **`llm_rosetta.gateway`** 仍可导入。它会告警并转发到 `llm_rosetta_gateway`；其文字只说明它已弃用、将在后续版本移除，不写具体版本。请现在就直接导入 `llm_rosetta_gateway`。`llm_rosetta.observability` finder 与 `llm-rosetta[gateway]` extra 条件相同。
- **持久化的内部键**不会改名：`provider_metadata` 标签 `openai_chat`、`google`、`anthropic`，转换器标签，原生工具定义，以及已提交的基准基线，有意保留旧写法。
- **已退役的提供方、shim 与模型类型名称**（见下节）不再解析。它们会抛错，错误信息写出替代名称。

## 已退役的名字 {#retired-names}

`llm_rosetta.retired_names.RETIRED_NAMES` 列出 31 个早先版本改过名的名字，每一行说明该名字变成了什么。这张表是错误表，不是别名表：每一行都会抛 `ValueError`，不做任何翻译，例如：

```text
'openai_chat' was renamed to 'chat_completions' in 0.14 and the old spelling is no longer accepted (use 'chat_completions'); run 'llm-rosetta-migrate --config <path>' (add --write to apply), see the migration guide
```

对于已迁入独立包的 shim，消息还会写出需要声明的 `shims.presets` 键。每条退役名称错误都以迁移命令结尾（`retired_names.MIGRATE_HINT`），网关的配置检查也一样。

### 提供方类型与格式 {#provider-types}

提供方条目的 `type:` 值，以及 `convert()`、`ConversionPipeline`、`get_converter_for_provider` 的 `format` 参数。

| 旧名字 | 替代 |
| --- | --- |
| `google` | `google_generate` |
| `google-genai` | `google_generate` |
| `openai_chat` | `chat_completions` |
| `openai_responses` | `openai--open_responses` |
| `anthropic` | `messages` |
| `openai-chat` | `chat_completions` |
| `openai-responses` | `openai--open_responses` |
| `open-responses` | `open_responses` |
| `google-interactions` | `google_interactions` |

### Shim 名称 {#shim-names}

提供方条目的 `shim:` 值。`shims.presets` 一列是该 shim 已作为独立包发布时需要声明的预设。

| 旧名字 | 替代 | `shims.presets` |
| --- | --- | --- |
| `openai` | `openai--chat_completions` |  |
| `google` | `google--google_generate` |  |
| `google_generate` | `google--google_generate` |  |
| `google_interactions` | `google--google_interactions` |  |
| `anthropic` | `anthropic--messages` |  |
| `argo--openai_chat` | `argo--chat_completions` | `argo` |
| `argo--openai_responses` | `argo--open_responses` | `argo` |
| `minimax--openai_chat` | `minimax--chat_completions` |  |
| `openrouter--openai_chat` | `openrouter--chat_completions` |  |
| `volcengine--openai_chat` | `volcengine--chat_completions` |  |
| `volcengine--openai_responses` | `volcengine--open_responses` |  |
| `argo--anthropic` | `argo--messages` | `argo` |
| `minimax--anthropic` | `minimax--messages` |  |
| `openrouter--anthropic` | `openrouter--messages` |  |
| `alcf--metis` | `alcf_metis--chat_completions` | `alcf` |
| `alcf--minerva` | `alcf_minerva--chat_completions` | `alcf` |
| `alcf--sophia` | `alcf_sophia--chat_completions` | `alcf` |
| `asksage--anthropic` | `asksage--messages` | `asksage` |
| `asksage--openai_chat` | `asksage--chat_completions` | `asksage` |
| `asksage--openai_responses` | `asksage--open_responses` | `asksage` |

### 模型类型 {#model-types}

模型类型的写法跟随其端点。

| 旧名字 | 替代 |
| --- | --- |
| `embedding` | `embeddings` |
| `decision` | `decisions` |

## `llm-rosetta-migrate` {#llm-rosetta-migrate}

`llm-rosetta-migrate` 是 `llm-rosetta-gateway` 发行版的控制台脚本。它改写网关配置中的退役名称，改写 `gateway.db` 中持久化的标签，并扫描源码中的退役名称。它只读取 `llm_rosetta.retired_names`，因此别名表删除后仍可使用；它从不猜测：无法映射的名称会被报告，并原样保留。

```bash
llm-rosetta-migrate --config config.jsonc                  # 试运行：输出 diff 与报告
llm-rosetta-migrate --config config.jsonc --write          # 先做 .bak 副本，再改写
llm-rosetta-migrate --db /var/lib/llm-rosetta --write      # gateway.db，或其所在目录
llm-rosetta-migrate --config config.jsonc --check          # 有待改写项时退出码为 3
llm-rosetta-migrate --scan src/                            # 仅提示，不修改任何文件
```

`--config`、`--db` 与 `--scan` 可重复使用。三者都不给时，命令会查找网关的配置文件。

**配置。** 编辑在文本层面进行：被改写名称之外的注释、键顺序与排版保持不变。它改写已退役的 `type:` 与 `shim:` 值、模型的 `type:` 与 `capabilities:` 项、`embedding_*` 与 `decision_*` 的键和段（改为复数名），并在 shim 已迁入独立包时添加 `shims.presets` 项。例如 `"type": "openai_chat"` 变为 `"type": "chat_completions"`，`"shim": "argo--openai_chat"` 变为 `"argo--chat_completions"` 并添加 `"shims": {"presets": ["argo"]}`，`embedding_providers` 变为 `embeddings_providers`。

**数据库。** 它改写 `request_log.source_provider`、`request_log.target_provider`、`error_dumps.source_provider` 与 `error_dumps.target_provider` 中的退役标签：退役的格式名，以及值 `embedding`（改为 `embeddings`）。它还把 `metrics.counters.by_source_provider["embedding"]` 合并进 `embeddings`。`by_target_provider` 中的 `embedding` 键只报告、不改写，`request_log.model_type` 不变。写入在备份之后的一个事务中完成，另一个进程占用该文件时命令拒绝执行（`--force` 可覆盖）。改写是幂等的。在一份生产规模数据库的副本上（268275 行 `request_log`，775 行 `error_dumps`），334 行标记为 `embedding` 的数据在两列中都变成了 `embeddings`，行数、token 总和、未触及列的校验和以及 `integrity_check` 均未变化，第二次运行没有写入任何内容。

持久化的内部键有意保持原样：`provider_metadata` 标签、转换器标签、原生工具定义与已提交的基线。你没有迁移的数据库行不是错误；网关照常启动并记录一条警告（见 [P8](#ledger-tags)）。

**模式。** 默认是试运行。`--write` 应用改写，并先做副本：`<file>.bak`，该名称已被占用时为 `<file>.bak-<时间戳>`。`--check` 是试运行，有待改写项时以退出码 `3` 结束，供 CI 使用。`--json` 输出机器可读的报告。

**`--scan`。** 扫描只做提示。它报告路径下 `.py`、`.json`、`.jsonc`、`.yaml` 与 `.yml` 文件中的退役名称，不修改任何文件。它只保留在任何位置都已退役的字符串：当前格式 `google_generate` 与 `google_interactions`，以及作为 shim 的裸厂商词 `openai`、`google`、`anthropic`，都不会被报告。带引号的字符串是匹配项，不是证明；`"anthropic"` 可能是你自己的变量。

**退出码。**

| 退出码 | 含义 |
| --- | --- |
| `0` | 无事可做，或全部已映射。 |
| `1` | 有项目需要人工处理：存在无法映射的项，或任何 `--scan` 发现。 |
| `2` | 命令无法运行：文件不可读、解析错误，或数据库被运行中的进程占用。 |
| `3` | `--check` 发现有待改写项。 |

## 升级流程 {#upgrade-recipe}

1. 停止网关。`llm-rosetta-migrate --db` 会拒绝被运行中的进程占用的数据库。
2. 备份配置文件与数据目录（`gateway.db`）。`--write` 也会保留 `.bak` 副本，但自己做的备份不依赖该命令。
3. 试运行：`llm-rosetta-migrate --config <config> --db <数据目录>`。阅读 diff 与报告。退出码 `1` 或 `2` 需要先人工处理再继续。
4. 写入：`llm-rosetta-migrate --config <config> --db <数据目录> --write`。
5. 启动网关。仍含退役名称的配置会在启动时以一行 `llm-rosetta-gateway: error: ...` 终止，其中写出文件与键名。
6. 可选：运行 `llm-rosetta-migrate --scan <你的代码>` 并阅读结果。库调用方、shim 包与脚本不会被该命令改写。

## core 接口 {#core-surface}

0.14.0 同时也是网关严格 core 接口的发布版本：网关只通过 `llm_rosetta.api` 与四个契约模块导入 core，`llm_rosetta.api.SURFACE_VERSION`（现为 `1`）只在移除时递增。导入其他 `llm_rosetta` 模块的宿主或 shim 作者不在该契约之内。参见 [core 接口](../api/index.md#the-core-surface)。
