---
title: 扩展
---

# 扩展

**扩展**（extension）是所有不随 core 一起发布的内容的可安装单元：一组提供方 shim、一个连接网关没有内置客户端的通道的 transport、供 `llm-rosetta` 命令行使用的命令，或者三者的任意组合。一个扩展就是一个文件夹，清单文件 `plugin.yaml` 与代码放在一起，文件夹以 `llm_rosetta_ext.<id>` 的名字导入。

扩展取代了 0.13 的插件机制：`llm_rosetta.shim_providers` entry point 组、网关的本地 shim 目录（`shims.paths`、`shims.yaml`）以及脚手架的“包外壳”均已移除，见[迁移到 0.14](migrating-to-0.14.md#retired-extension-paths)。

## 文件夹 { #the-folder }

```
acme/                         # 扩展 id；以 llm_rosetta_ext.acme 导入
├── plugin.yaml               # 清单，先于任何代码被读取
├── pyproject.toml            # 打包；文件夹本身就是包
├── README.md
├── __init__.py
├── provider.yaml             # 服务方清单            （shims 类别）
├── llm/chat_completions/provider.yaml                （shims 类别）
├── transport/                # TRANSPORTS 映射       （transports 类别）
├── commands.py               # COMMANDS 映射         （commands 类别）
└── tests/
```

`llm_rosetta_ext` 是 PEP 420 命名空间：没有 `__init__.py`，也没有哪个发行版拥有它，因此 `sys.path` 上的任何目录都可以向它贡献一个文件夹。发现过程只读取目录和包元数据，不导入任何扩展代码。

## `plugin.yaml` { #plugin-yaml }

清单按 `llm_rosetta/contracts/extension_manifest.schema.json` 校验，宿主在导入扩展的任何一行代码之前就会读取它。core 无法运行某个扩展时，会用一句话跳过它，而不是在导入时失败。

```yaml
id: acme
name: Acme
version: 0.1.0
maturity: experimental        # experimental | beta | stable
requires:
  surface: 1                  # 必须等于 core 的 SURFACE_VERSION
  ir: 1                       # 不得超过 core 的 IR_VERSION
provides:
  shims: [acme--chat_completions]
  transports: [acme_cli]
  commands:
    hello:
      en: "Say hello"
      zh: "打个招呼"
summary:
  en: "Acme extension for llm-rosetta."
  zh: "llm-rosetta 的 Acme 扩展。"
```

| 键 | 含义 |
|-----|---------|
| `id`、`name`、`version`、`maturity`、`requires`、`provides` | 必填。`provides` 至少要列出 `shims`、`transports`、`commands` 之一。 |
| `provides.shims` | shim 名称：id 本身、`<id>--<name>`，或 `<id>_<lane>--<name>`。 |
| `provides.transports` | 扩展实现的 transport id。shim 的 `connection.transport` 先解析为自己所属扩展的 transport，其次是宿主内置的，最后是 `requires.extensions` 中所列扩展的 transport。 |
| `provides.commands` | 扩展新增的动词，每个带一行按语言给出的摘要（必须有 `en`）。`llm-rosetta <id> <verb>` 运行它；id 中的下划线写作连字符。 |
| `requires.extensions` | 本扩展所引用 transport 的其他扩展的 id。 |
| `external` | 扩展驱动但不自带的程序（`name`、`check`、`minimum`、`install_hint`）；`ext doctor` 会报告它们。 |
| `platforms` | `linux`、`darwin`、`windows` 的子集。 |
| `tags`、`homepage`、`summary` | 列表展示用的元数据。 |

兼容性只有两个比较：`requires.surface` 等于 `SURFACE_VERSION`，且 `requires.ir` 不超过 `IR_VERSION`，再加上 `platforms`。见[版本常量](../api/index.md#version-constants)。

## 各类别的含义 { #what-each-kind-means }

| 类别 | 加载器读取 | 由谁注册 |
|------|------------------|---------------|
| `shims` | shim 目录树：`provider.yaml` 文件及 `auth.py` / `quota.py` / `transforms.py` 附属文件。目录树契约见[提供方 Shim](shims.md#declarative-provider-directory)。 | shim 层 |
| `transports` | `llm_rosetta_ext.<id>.transport`，其中暴露 `TRANSPORTS`，即 transport id 到工厂的映射。 | 网关 |
| `commands` | `llm_rosetta_ext.<id>.commands`，其中暴露 `COMMANDS`，即动词到可调用对象的映射。 | 命令行 |

core 只管理包。`provides` 各类别的含义属于拥有该类别的那一层，该层为这个类别注册一个处理器（`register_kind_handler`）。首次加载之后才注册的处理器，会应用到已经加载的扩展上。有问题的扩展会被跳过并点名，不会影响其他扩展。

## 扩展装在哪里 { #where-extensions-live }

扩展目录按以下顺序确定：

1. 拥有数据目录的宿主指定的目录（网关使用 `<data dir>/extensions`）；
2. 环境变量 `LLM_ROSETTA_EXTENSIONS_DIR`；
3. 按用户的默认目录：Linux 上为 `$XDG_DATA_HOME/llm-rosetta/extensions` 或 `~/.local/share/llm-rosetta/extensions`，macOS 上为 `~/Library/Application Support/llm-rosetta/extensions`，Windows 上为 `%LOCALAPPDATA%\llm-rosetta\extensions`。

默认值绝不会回退到 `/` 之下的路径：没有 passwd 条目的容器 uid 会得到 `HOME=/`，此时 core 会拒绝并给出点名 `LLM_ROSETTA_EXTENSIONS_DIR` 的提示。两个 Docker 入口脚本在该变量未设置时会把它设为 `$CONFIG_DIR/extensions`（`/config/extensions`），因此扩展不会随容器消失。该目录必须可被容器的 uid 写入。

扩展既可以在扩展目录中发现，也可以来自声明了 `llm_rosetta.extensions` 组 entry point 的已安装发行版（名称为 id，值为 `llm_rosetta_ext.<id>`）。同一个扩展经两条路径到达时只加载一次。仍然在已退役的 `llm_rosetta.shim_providers` 组声明 entry point 的发行版会被报告，但不会运行。

## 安装 { #installing }

安装不需要 pip：wheel 是一个 zip，成员是 `llm_rosetta_ext/<id>/...` 与 `.dist-info`，安装器只解包这些。所有拒绝条件（成员名、大小上限、清单、兼容性、SHA-256）都在向目录写入任何字节之前检查；wheel 先解包到临时目录，再整体换入。成员名带盘符或流分隔符，或包含 Windows 无法保存的名字，都会被拒绝。

```bash
llm-rosetta ext install kilo_free              # 一个 id，取自扩展索引
llm-rosetta ext install ./acme-0.1.0-py3-none-any.whl
llm-rosetta ext install https://example.org/acme.whl --sha256 <hex>
llm-rosetta ext update                         # 每个扩展装上最新的兼容版本
llm-rosetta ext doctor                         # 哪里有问题，缺什么
```

### 扩展索引 { #the-extension-index }

扩展仓库把每个 wheel 作为 release 资源发布，并在旁边写一个 `index.json`。该文件由各扩展自己的文件生成。第 1 版：

```json
{
  "index_version": 1,
  "source": "contrib",
  "generated_at": "2026-11-26T08:00:00Z",
  "extensions": {
    "kilo_free": [
      {
        "version": "0.4.1",
        "url": "llm_rosetta_ext_kilo_free-0.4.1-py3-none-any.whl",
        "sha256": "<64 hex digits>",
        "surface": 1,
        "ir": 1,
        "requires_python": ">=3.10",
        "requires_extensions": [],
        "name": "Kilo (Free)",
        "maturity": "beta",
        "tags": ["tier:free"],
        "shims": ["kilo_free--chat_completions"],
        "logo": "https://…/kilocode.svg",
        "commit": "<git sha>"
      }
    ]
  }
}
```

安装只需要 `version`、`url`（绝对地址，或相对索引的地址）和 `sha256`。`surface`、`ir`、`requires_python` 与 `requires_extensions` 说明当前宿主能否运行这个 wheel。`name`、`maturity`、`tags`、`shims` 与 `logo` 用于列出尚未安装的扩展。`index_version` 是整数 `1`。安装器挑选当前解释器和 core 都接受的最新版本，共享的依赖只装一次，并在写入任何内容之前校验摘要。

索引位置依次为 `--index`、`$LLM_ROSETTA_EXTENSION_INDEX`、contrib 仓库的 `releases/latest/download/index.json`。它是一个 `https` URL 或本地文件路径。索引的 `source` 头与它被当作的目录 id 不一致时，网关会拒绝该索引；`source` 头只是索引自己的声明，不能证明发布者是谁。

!!! warning "信任"
    已安装的扩展是加载时就会运行的代码。索引意味着以发布它的仓库的信任级别执行代码。摘要只能证明 wheel 就是索引所列的那个，不能证明二者安全。

### `ext` 动词 { #the-ext-verbs }

| 动词 | 作用 |
|------|------|
| `list [--dir D] [--json]` | 已安装或已发现的扩展及其状态（`loaded`、`skipped`、`failed`、`not loadable`）。 |
| `show <id> [--json]` | 某个扩展的清单和状态。 |
| `install <wheel \| URL \| id> [--sha256 H] [--version V] [--index I] [--pip SPEC] [--dry-run]` | 安装。`--pip` 用 pip 安装，仅在 pip 可用时有效。 |
| `remove <id> [--dry-run]` | 移除已安装的扩展。 |
| `update [<id>] [--index I] [--dry-run]` | 安装索引中最新的兼容版本。 |
| `doctor [<id>] [--no-import] [--json]` | 检查兼容性、缺失的所需扩展、外部程序，并在子进程中做导入检查。 |
| `new`、`check`、`migrate` | 开发扩展；见[添加扩展](../contributing/adding-an-extension.md)。 |

`list`、`show`、`doctor` 接受 `--json`；`install`、`remove`、`update` 接受 `--dry-run`，只说明会改什么，不做任何改动。退出码：`0` 成功，`1` 失败或发现问题，`2` 用法错误。

## 网关与扩展 { #the-gateway-and-extensions }

网关的 `shims` 配置段负责从扩展索引中选择要安装的内容以及要关闭的内容，见[外部 Shim 源](../gateway/configuration.md#external-shim-sources)。已安装的扩展无论 `shims.install` 是否仍选中它都会加载。core 已经注册的 shim 名称优先：不会被替换，扩展的状态会说明是哪个持有者占用了该名称。已迁出 core 的服务方，在配置里提到它的 shim 时会被自动安装（见[迁移到 0.14](migrating-to-0.14.md#moved-to-contrib)）。

扩展可以新增命令：`llm-rosetta <id> <verb>`（例如 `llm-rosetta-gateway argo tunnel` 通过网关入口到达同一个区域）。`ext`、`help`、`gateway` 和 `serve` 这几个名字不能用作扩展 id。

## 另见

- [添加扩展](../contributing/adding-an-extension.md)：`ext new`、`ext check`、`ext migrate` 与目录树契约。
- [提供方 Shim](shims.md)：shims 类别的详细说明。
- [Shim 的放置与成熟度](shim-placement.md)：哪些内容在 core，哪些在扩展仓库。
