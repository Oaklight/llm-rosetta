---
title: API 参考
---

# API 参考

## 库 API

- [转换器](converters.md) — 各提供方的转换器类
- [IR 类型](types.md) — 中间表示类型定义
- [Tool Ops](tool-ops.md) — IR 与各提供方格式之间的工具定义转换
- [Shims](shims.md) — 提供方和模型 shim 类与注册表
- [工具函数](utils.md) — 辅助函数和自动检测

## 网关 API

- [Admin API](admin.md) — 网关管理面板 REST API 端点
- [可观测性](observability.md) — 网关的遥测栈（指标、请求日志、持久化、性能分析）与核心的用量计量

## core 接口 {#the-core-surface}

core 在 `llm_rosetta/_tiers.py` 中一次性写明每个公开模块的层级。该文件没有列出的模块都是内部实现，任何版本都可能移动。

| 层级 | 模块 | 承诺 |
| --- | --- | --- |
| 用户 API | `llm_rosetta`、`auto_detect`、`converters`（含 `converters.base`、`converters.registry` 与各格式包）、`fidelity`、`pipeline`、`retired_names`、`shims`、`tool_ops`、`types`、`types.ir` | 库 API：转换请求与响应、转换器类、IR 类型、管道。破坏性变更遵循弃用策略。 |
| 扩展接口 | `llm_rosetta.api`、`llm_rosetta.contracts` 及其子模块 | 宿主或 shim 作者构建所依赖的接口。在一个次版本内只增加名字。 |
| 兼容 | `llm_rosetta.gateway`、`llm_rosetta.observability`、`llm_rosetta.shims.provider_shim`（其中对数据类的再导出）、`llm_rosetta.routing`、`llm_rosetta.capabilities`、`llm_rosetta.types.ir.extensions` | 为曾经发布过它们的版本保留的已弃用路径。每个都写明替代路径，已定移除版本的也写明版本。 |
| 内部 | 其余所有模块 | 任何版本都可能移动。 |

`tests/test_tiers.py` 检查前三个层级的每个模块都能导入。`scripts/dev/public_api_diff.py` 从同一个文件读取要比较的 core 模块列表，并读取每棵被比较的树自己的那份：base 正式版按它自己声明的模块测量，因此从层级表删掉的模块会被报告为已移除。没有该文件的树（v0.13.0）用脚本自带的列表测量。

### 谁导入什么 {#who-imports-what}

- 宿主（例如网关，或构建在 core 之上的路由器）导入 `llm_rosetta.api` 与 `llm_rosetta.contracts`。
- shim 只导入 `llm_rosetta.contracts`。这包括它的旁挂模块（`transforms.py`、`auth.py`、`quota.py`）和 shim 包的 entry point。

shim 不得导入 `llm_rosetta.api`。插件 entry point 与旁挂模块在 `import llm_rosetta` 尚未完成时就会运行，而 `api` 需要完整加载的包。见 [shim 可以导入什么](../guide/shims.md#what-a-shim-may-import)。

`llm_rosetta.api` 有一份字面量、分组的 `__all__`，共 63 个名字：聊天转换、类型化管道、格式目录与模型类型词汇表、提供方与 shim 名称、shim 注册表（读侧与写侧）与加载器、清单词汇表与叶子合并、脚手架入口、保真度与性能分析，以及宿主要检查的 IR 类型。core 自己从不导入 `llm_rosetta.api`：依赖方向是从宿主指向 core。

### 契约模块 {#contract-modules}

`llm_rosetta.contracts` 每个契约一个模块。包本身只含常量，因此导入它不会加载任何转换器，也不会加载 shim 注册表。

| 模块 | 内容 | 原路径（0.14 开发版） |
| --- | --- | --- |
| `llm_rosetta.contracts` | `SURFACE_VERSION`、`IR_VERSION`、旁挂模块的文件名（`TRANSFORMS_MODULE`、`AUTH_MODULE`、`QUOTA_MODULE`），以及 shim 加载器从中读取的名字（`PRE_IR_TRANSFORMS`、`POST_IR_TRANSFORMS`、`IR_TRANSFORMS`、`RESPONSE_BODY_TRANSFORMS`、`MODEL_LIST_TRANSFORM`、`QUOTA_PROBES` = `"PROBES"`、`AUTH_EXCHANGE` = `"EXCHANGE"`） | `SURFACE_VERSION` 原定义在 `llm_rosetta.api`，`api` 现在再导出同一个对象 |
| `llm_rosetta.contracts.shim` | `ProviderShim`、`ConnectionConfig`、`ToolsConfig`、`ReasoningCapability`、`VendorManifest`、`QuotaDeclaration`、`BatchDeclaration`、`SoftErrorPattern`、`TYPE_PARAM_KEYS`、字面量类型 `ToolSearchMode`、`ReplayMissing`、`UnsignedBlocks`，以及 `load_providers_from_dir` | `llm_rosetta.shims.provider_shim`（已弃用的再导出，0.15.0 移除） |
| `llm_rosetta.contracts.transform` | 钩子类型 `Transform`、`IRTransform`、`TransformContext`、`NamedTransform`、`NamedIRTransform`；`apply_transforms`、`apply_ir_transforms`；body 原语（`strip_fields`、`rename_field`、`set_defaults`、`force_fields` 等）与 IR 原语（`auto_cache_breakpoints`、`hoist_late_system_messages`、`truncate_images`、`unwind_parallel_tool_calls`） | `llm_rosetta.transforms`（已移除） |
| `llm_rosetta.contracts.auth` | shim 在 `auth.py` 中提供的凭据交换：`AuthExchange`、`BaseAuthExchange`、`AuthContext`、`AuthExchangeError`、`InteractiveSetupRequired`、`credential_for` 与字段限制常量 | `llm_rosetta.shims.auth_exchange`（已移除） |
| `llm_rosetta.contracts.quota` | shim 在 `quota.py` 中提供的额度探测：`QuotaProbe`、`QuotaWindow`、`InBandSignal`、`ProbeTarget`、`ProbeMatch`、`resolve_probe`、`probe_fetch` 及辅助函数 | `llm_rosetta.shims.quota_probe`（已移除） |
| `llm_rosetta.contracts.routing` | `ProviderType`、`ResolvedRoute`、`Router` 协议、`SoftErrorPattern` | `llm_rosetta.routing`（现为 0.13.0 兼容转发模块，0.15.0 移除） |
| `llm_rosetta.contracts.usage` | 用量计量语义与报告：`UsageLedger`、`UsageReport`、`token_total`、`usage_columns`、`STANDARD_SHAPES` 等。只导入这个包，不导入它的子模块。 | `llm_rosetta.observability.usage`（已移除） |

### 网关规则

网关只通过两个前缀访问 core：`llm_rosetta.api` 与 `llm_rosetta.contracts`（`contracts.usage` 只导入包本身）。规则是严格的，没有白名单：网关源码中导入任何其他 core 模块或私有（以 `_` 开头）core 名称，都会使 `tests/gateway/test_core_surface_boundary.py` 失败。兼容转发模块 `llm_rosetta.routing` 与 `llm_rosetta.capabilities` 同样被拒绝。

shim 遵循同一规则，只有一个前缀。`shims/providers/<vendor>/` 下的 `.py` 文件、脚手架模板或脚手架生成的文件，若不经 `llm_rosetta.contracts` 导入 core，或导入其他厂商的目录，`tests/test_shim_sidecar_boundary.py` 就会失败。

### 版本常量 {#version-constants}

| 常量 | 定义位置 | 值 | 何时递增 | 由谁读取 |
| --- | --- | --- | --- | --- |
| `SURFACE_VERSION` | `llm_rosetta.contracts`；`llm_rosetta.api` 再导出同一个对象 | `1` | 扩展接口中的名字被移除或含义改变。新增名字不会使它递增。 | 网关，启动时；`scripts/dev/public_api_diff.py` |
| `IR_VERSION` | `llm_rosetta.types.ir.version`；由 `llm_rosetta.types.ir` 与 `llm_rosetta.contracts` 再导出 | `1` | IR 类型的变更使旧级别写出的 IR 值在新级别下无法读取。新增可选字段不会使它递增。 | 目前没有代码读取它 |

由于 `SURFACE_VERSION` 只在移除或含义改变时递增，接口在一个次版本内只增不减，宿主断言的是最低级别。0.14.0 是该接口的发布版本。网关声明自己能运行的最低级别（`llm_rosetta_gateway.surface.REQUIRED_SURFACE_VERSION`），`create_app` 在启动时断言它：core 过旧时网关以 `RuntimeError` 停止，信息写出两个级别，而不是在第一次请求时报缺少属性。早于该常量的 core 视为级别 0。`scripts/dev/public_api_diff.py` 强制执行递增：相对上一个正式版，扩展层模块中的破坏性变更（即使已在 allowlist 中确认）会使检查失败，除非 head 的 `SURFACE_VERSION` 高于该正式版（没有该常量的正式版视为 0）。网关对 core 的版本约束保持为 `llm-rosetta>=X.Y.0.dev0,<X.(Y+1)`。

### 随严格规则新增的名字

| 名字 | 用途 |
| --- | --- |
| `plugin_source`、`builtin_providers_dir` | 管理面板的 shim 文件路由 |
| `add_scaffold_arguments`、`run_scaffold`、`run_scaffold_check`、`run_scaffold_migrate` | 网关的 `shims new`、`shims check` 与 `shims migrate` 命令 |
| `leaf_connection`、`manifest_vocabulary`、`merge_connection`、`ManifestVocabulary` | 厂商清单词汇表及其类型，以及把清单的连接配置合并进叶子 |
| `SURFACE_VERSION` | 上文所述的接口级别 |

`manifest_vocabulary()` 返回 `ManifestVocabulary`，`api` 导出该类型，宿主可以用它做标注。`merge_connection(manifest_connection, leaf)` 先接收清单，参数顺序与加载器的私有合并函数相反，并且只返回合并后的映射：相对 `token_command` 据以解析的目录不返回，因为传入的清单没有对应文件。已移除的名字列在 [迁移到 0.14](../guide/migrating-to-0.14.md) 中。
