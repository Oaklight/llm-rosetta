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

网关与外部 shim 作者通过一个扁平的再导出模块 `llm_rosetta.api` 和四个契约模块来构建在 core 之上。`llm_rosetta` 下的其余内容都是内部实现，任何版本都可能移动。允许的五条导入路径是：

- `llm_rosetta.api`
- `llm_rosetta.routing`
- `llm_rosetta.observability.usage`（只导入这个包，不导入它的子模块）
- `llm_rosetta.shims.auth_exchange`
- `llm_rosetta.shims.quota_probe`

`llm_rosetta.api` 有一份字面量、分组的 `__all__`，共 62 个名字：聊天转换、类型化管道、格式目录与模型类型词汇表、提供方与 shim 名称、shim 注册表（读侧与写侧）与加载器、清单词汇表与叶子合并、脚手架入口、保真度与性能分析，以及宿主要检查的 IR 类型。core 自己从不导入 `llm_rosetta.api`：依赖方向是从宿主指向 core。

### 网关规则

网关只通过这五条路径访问 core。规则是严格的，没有白名单：网关源码中导入任何其他 core 模块或私有（以 `_` 开头）core 名称，都会使 `tests/gateway/test_core_surface_boundary.py` 失败。让迁移分步落地的遗留与私有白名单已删除。

### `SURFACE_VERSION`

`llm_rosetta.api.SURFACE_VERSION` 是整数，现为 `1`。当接口中的名字被移除或含义改变时它递增，新增名字时不递增，因此接口在一个次版本内只增不减。0.14.0 是该接口的发布版本。网关声明自己能运行的最低级别（`llm_rosetta_gateway.surface.REQUIRED_SURFACE_VERSION`），`create_app` 在启动时断言它：core 过旧时网关以 `RuntimeError` 停止，信息写出两个级别，而不是在第一次请求时报缺少属性。早于该常量的 core 视为级别 0。网关对 core 的版本约束保持为 `llm-rosetta>=X.Y.0.dev0,<X.(Y+1)`。

### 随严格规则新增的名字

| 名字 | 用途 |
| --- | --- |
| `plugin_source`、`builtin_providers_dir` | 管理面板的 shim 文件路由 |
| `add_scaffold_arguments`、`run_scaffold`、`run_scaffold_check`、`run_scaffold_migrate` | 网关的 `shims new`、`shims check` 与 `shims migrate` 命令 |
| `leaf_connection`、`manifest_vocabulary`、`merge_connection` | 厂商清单词汇表，以及把清单的连接配置合并进叶子 |
| `SURFACE_VERSION` | 上文所述的接口级别 |

`manifest_vocabulary()` 返回 `ManifestVocabulary`；该类型本身不从 `api` 导出，因为调用方只读它的属性，没有地方需要标注它。已移除的名字列在 [迁移到 0.14](../guide/migrating-to-0.14.md) 中。
