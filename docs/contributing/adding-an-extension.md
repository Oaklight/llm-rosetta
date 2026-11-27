---
title: 添加扩展
---

# 添加扩展

一个扩展就是一个文件夹，提供 shim、transport、命令，或它们的任意组合。本页讲如何创建、检查和发布扩展。关于扩展是什么、如何安装、`plugin.yaml` 里有什么，请先读[扩展](../guide/extensions.md)。

属于 core 的服务方（模型厂商自己的 API，使用静态 API key，没有 `auth.py`）不是扩展：它放在内置目录树 `packages/llm-rosetta/src/llm_rosetta/shims/builtin/<vendor>/` 中，用 `scripts/dev/new_shim.py` 生成脚手架。见 [Shim 的放置与成熟度](../guide/shim-placement.md)。

## 创建：`ext new` { #create-it-ext-new }

```bash
llm-rosetta ext new acme                                  # 一个 chat shim
llm-rosetta ext new acme --provides shims,transports,commands --verb hello --verb status
llm-rosetta ext new acme --format messages --name "Acme AI" --dest ./packages
```

| 参数 | 含义 |
|----------|---------|
| `ext_id` | 扩展 id：小写字母、数字和下划线，以字母开头。它同时是文件夹名、导入名（`llm_rosetta_ext.<id>`）和服务方键。 |
| `--provides KINDS` | 要生成脚手架的类别，逗号分隔：`shims`、`transports`、`commands`。默认 `shims`。 |
| `--format` | 第一条通道的 chat 格式，仅用于 shims 类别。默认 `chat_completions`。未提供 shims 类别时会被拒绝。 |
| `--verb VERB` | commands 类别的命令动词；可重复。默认 `run`。未提供 commands 类别时会被拒绝。 |
| `--name` | 显示名称；默认等于 id。 |
| `--dest DIR` | 创建文件夹的目录。默认 `packages`。 |

命令会拒绝已存在的文件夹、未知的类别、指向你没要求的类别的选项，以及目录中不是 chat 格式的格式。每个类别会在公共基础（`plugin.yaml`、`pyproject.toml`、`README.md`、`__init__.py`、`tests/test_extension.py`）之上添加自己的文件：

| 类别 | 添加 |
|------|------|
| shims | `provider.yaml`（服务方清单）和 `llm/provider.yaml`（第一条通道，命名为 `<id>--<format>`） |
| transports | 暴露 `TRANSPORTS` 的 `transport/` 包 |
| commands | 暴露 `COMMANDS` 的 `commands.py`，每个动词一个函数 |

示例目录树由 `llm_rosetta/authoring/_ext_template/` 中的模板文件夹渲染而成，shim 文件来自 `scripts/dev/new_shim.py` 为内置目录树使用的同一个生成器，所以两者不会写出不同的形状。`ext new` 结束时会打印 `next: llm-rosetta ext check <folder>`。

## 目录树契约 { #the-tree-contract }

服务方目录放清单而不是叶子，每个叶子都在模型类型目录之下，只有当某个类型有多于一条通道时才出现格式层。在扩展中，文件夹就是那一个服务方：

```
<extension>/                 provider.yaml 是服务方清单，永远不是叶子
├── auth.py                  可选，所有通道共用的凭据交换
├── quota.py                 可选
└── [<lane>/]                可选：同一服务方的命名实例
    └── <type>/              llm | embeddings | rerank | decisions
        └── [<format>/]      仅当该类型有多于一条通道时才有
            ├── provider.yaml
            ├── transforms.py
            ├── auth.py
            └── quota.py
```

规则简述：服务方目录永远不是叶子；类型目录必须存在，即使只有一条通道；格式目录只出现在该类型的第二条通道旁边。文件白名单及其理由见 `llm_rosetta/authoring/_shim_template/README_en.md`，它是 shim 形状的唯一事实来源。附属文件只能通过 `llm_rosetta.contracts` 导入 `llm_rosetta`（见 [shim 可以导入什么](../guide/shims.md#what-a-shim-may-import)）。

## 检查：`ext check` { #check-it-ext-check }

```bash
llm-rosetta ext check packages/acme       # 一个扩展
llm-rosetta ext check .                   # 根目录下的每个 packages/*
llm-rosetta ext check packages/acme --no-build
```

`ext check` 会应用清单模式、文件夹规则、目录树契约以及各类别的规则，且从不导入扩展。每个缺陷输出一行 `problem <id>: <sentence>`，无法运行的检查输出 `skipped <id>: <reason>`，文件夹没有问题时输出 `ok <id>`。任一文件夹有问题时退出码为 `1`。跳过的检查不会算作通过。

| 范围 | 检查内容 |
|------|----------------|
| 文件夹 | 文件夹名是合法的 id 且等于 `plugin.yaml` 中的 `id`；`README.md`、`__init__.py` 和 `tests/` 存在；`pyproject.toml` 可解析，且其名称、版本、包映射和 `llm_rosetta.extensions` entry point 与清单一致。 |
| Shims | 目录树遵循契约，能通过真实加载器加载（在实时注册表之外进行），且注册的名称恰好是 `provides.shims` 中的那些。 |
| Transports | `transport/` 暴露 `TRANSPORTS`，其 id 恰好是 `provides.transports` 中的那些。 |
| Commands | `commands.py` 暴露 `COMMANDS`，其动词恰好是 `provides.commands` 中的那些。 |
| Wheel | 从干净副本构建的 wheel 包含文件夹里的文件，且仅此而已。使用 `--no-build` 或 pip 不可用（例如在二进制中）时会跳过。 |

## 迁移旧目录树：`ext migrate` { #move-an-older-tree-ext-migrate }

在目录树契约出现之前写的扩展，可能有叶子的层级不对。`ext migrate` 会移动它：

```bash
llm-rosetta ext migrate packages/acme            # 打印计划；不移动任何东西
llm-rosetta ext migrate packages/acme --write    # 执行移动
```

它会重命名目录，包括第二条通道的情形：已有叶子下移到 `<type>/<its format>/`。无法在不猜测的情况下移动的叶子会被报告并保持原样。除此之外它什么也不改：不改 `plugin.yaml`，不改附属文件的内容。移动之后 `pyproject.toml` 的 package-data 可能仍指向旧路径，`transforms.py` 中的相对导入也可能要多加一个点；之后请运行 `ext check`（wheel 检查能发现前者，扩展自己的测试能发现后者）。试运行发现有工作要做时退出码为 `1`。

## 测试与发布 { #test-and-publish }

1. 运行扩展自己的测试：`ext new` 会写出 `tests/test_extension.py`，它从 `llm_rosetta.api` 导入 `run_ext_check`。
2. 构建 wheel 并安装试用：`llm-rosetta ext install ./dist/<wheel>.whl`，然后 `llm-rosetta ext doctor acme`。
3. 发布时，扩展仓库把每个 wheel 附在 release 上，并在 wheel 旁边写 `index.json`（[扩展索引](../guide/extensions.md#the-extension-index)）。网关随后在 `shims.sources` 中列出该仓库，并在 `shims.install` 中选择该扩展。两个官方仓库是 `Oaklight/llm-rosetta-ext-contrib` 和 `Oaklight/llm-rosetta-ext-argonne`。

扩展必须能在没有 pip 的环境里运行：只依赖标准库和 core。

## 分层规则在哪里 { #where-the-layering-rules-live }

让 core 和网关保持分层的导入规则就是测试，表格就在测试里：

| 表 | 文件 |
|-------|------|
| core 的包级边（`CORE_EDGES`） | `tests/test_core_layering.py` |
| 转换层不导入 shim 层或扩展层 | `tests/test_core_layering.py` |
| 网关的层与边（`GATEWAY_EDGES`、`LAZY_UPWARD`） | `tests/gateway/test_layers.py` |
| 网关只通过 `llm_rosetta.api` 和 `llm_rosetta.contracts` 访问 core | `tests/gateway/test_core_surface_boundary.py` |
| shim 附属文件只导入 `llm_rosetta.contracts` | `tests/test_shim_sidecar_boundary.py` |
| core 从不导入网关或 `llm_rosetta.api` | `tests/test_core_import_boundary.py` |

表中不允许的新导入会连同其位置一起失败；已不存在的边必须从表中删除。见[架构指南](architecture.md#layering)。
