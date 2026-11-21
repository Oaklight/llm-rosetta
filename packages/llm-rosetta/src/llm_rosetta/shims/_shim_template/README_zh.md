# shim 模板

这个目录是 provider shim 形状的唯一出处。`llm-rosetta-gateway shims new`
按这里的文件渲染；读取结果的 loader 在 `llm_rosetta/shims/_loader/`。这里
本身不是 shim：目录名以 `_` 开头，任何扫描都会跳过。

手写 shim 之前先读这份文件。模板和某个 shim 不一致时，以模板为准，那个 shim
是缺陷。

## 一个 shim 允许有哪些文件

loader 只打开下面这些名字。其余文件本身不产生任何效果——但有一个区别要记住：
helper 模块在被其中某个文件 import 之前是惰性的。`transforms.py` 可以 import，
所以它旁边的 `_helpers.py` 是可用的私有模块；真正不会被读的是 `__init__.py`，
在里面写 `raise` 永远不会执行。

| 文件 | 位置 | 必需 | 读取为 |
|:--|:--|:--|:--|
| `provider.yaml` | vendor 根 | 否 | vendor manifest：`connection:`、`logo`、`group`、`group_label`、`fields:` |
| `provider.yaml` | 叶子 | 是 | shim 自己的声明 |
| `auth.py` | vendor 根 | 否 | `EXCHANGE`，该 vendor 所有 lane 的凭据交换 |
| `auth.py` | 叶子 | 否 | 该叶子的交换；vendor 已有时报错 |
| `quota.py` | vendor 根或叶子 | 否 | `PROBES`，额度读取器 |
| `transforms.py` | 叶子 | 否 | 字段级 transform 元组 |
| `shims.yaml` | gateway 扫描的那个目录 | 否 | 该包声明加载的 shim 集合 |

`auth.py` 和 `quota.py` 以独立模块执行：不能用相对 import，必须自包含
（标准库总是可用）。`transforms.py` 可以用相对 import——loader 会为它注册
包链——所以它旁边的 helper 模块能被导入。

`shims.yaml` 与各 vendor 目录并排，放在 gateway 被指向的那个目录里——本地树是
`shims.paths` 指的目录，拉取来的包是 remote 的 `path`（本仓库发布的包里是
`src/<package>/providers`）。往上放一层的 `src/<package>/shims.yaml` 永远不会被读：
源照样加载，里面的 core 下限不会被检查，而且没有任何提示。

## 叶子放在哪里

```
<vendor>/                    provider.yaml 是 manifest，永远不是叶子
├── auth.py                  可选
├── quota.py                 可选
└── [<lane>/]                可选：同一 vendor 的多个命名实例
    └── <type>/              llm | embeddings | rerank | decisions
        └── [<format>/]      仅当该 type 下有多个 lane
            ├── provider.yaml
            ├── transforms.py
            ├── auth.py
            └── quota.py
```

三条规则决定路径。loader 另外还能读两种旧形状——叶子直接放在 vendor 下，以及
叶子放在 `<vendor>/<format>/` 下——已有的包因此照旧能加载；它们是兼容，不是
约定，`shims check` 会报出这两类。

1. **vendor 目录永远不是叶子。** 那里的 `provider.yaml` 是 manifest，所有
   叶子都在它下面。manifest 只写 `connection:`、`logo`、`group`、
   `group_label`、`fields:`，不写任何标识某个 shim 的字段——加上 `name`、`model_type`
   或 `format` 就把 vendor 变成了叶子，它下面的 shim 会以零个加载，只留一行
   "missing 'name' or 'format'"。
2. **`<type>` 目录永远存在**，即使里面只有一个 lane。
3. **`<format>` 目录只在该 type 出现第二个 lane 时存在。** 第一个 lane 写成
   `<vendor>/<type>/provider.yaml`；第二个到达时，把第一个移进
   `<vendor>/<type>/<它的 format>/`，自己放在旁边。这个移动由脚手架完成。

lane 是同一 vendor 的命名实例——ALCF 的 `metis`、`minerva`、`sophia` 是同一
账号的三个集群，共享一个 manifest 和一份凭据。lane 自己没有
`provider.yaml`，它继承 vendor manifest。

## 声明凭据输入

如果凭据除 key 之外还需要别的输入，就在 `fields:` 里声明：所有 lane 共享的
写在 vendor manifest 上，某个 lane 要新增或改写就写在该叶子上。管理面板的服务
方表单会按条目逐个渲染输入框，一次保存就把 key 和这些值一起写进服务方条目
（`auth_fields` 之下），shim 的 `auth.py` 再从那里读回：

```yaml
fields:
  - key: email            # 记录键，纯标识符
    label: 账号邮箱        # 必填
    placeholder: you@example.gov
    help: 该 AskSage 账号 key 对应的邮箱地址。
    required: true
```

这个块是数据：为某个 vendor 改表单就是改 YAML，不需要动 gateway 或面板。
条目在加载时校验——缺 `key` 或 `label`、key 重复、`required` 不是布尔值——
都会带着文件名被拒绝；`shims check` 会在安装这棵树之前报出同样的发现。
这里声明的都不是机密：值存在服务方条目里、会回传给已认证的面板，因此真正的
凭据应放在 `api_key` 或 exchange 自己的记录里。

## 创建一个 shim

```bash
# 已安装的 gateway，交互式
llm-rosetta-gateway shims new acme --format chat_completions

# 给已有 vendor 增加一个 lane
llm-rosetta-gateway shims new acme --lane metis --format chat_completions

# 生成一个完整的外部包
llm-rosetta-gateway shims new acme --format chat_completions \
    --package llm-rosetta-shim-acme

# 在仓库检出内，写入内置树
python scripts/dev/new_shim.py acme --format chat_completions
```

每次运行都会用真实 loader 加载自己的产物，再报告成功，所以 gateway 读不了的
形状在创建时就失败，而不是等到启动。检查已有树：

```bash
llm-rosetta-gateway shims check --root src/llm_rosetta_shim_acme/providers
```

## 模板目录如何对应到 shim

`tree/` 下的目录用固定名字，方便编辑器打开：`tree/vendor/` 对应
`<root>/<vendor>/`，`tree/vendor/type/` 对应 `<vendor>/<type>/`，
`tree/vendor/type/format/` 对应 `<vendor>/<type>/<format>/`——该 type 只有
一个 lane 时去掉 format 层，写成 `<vendor>/<type>/`。
`package/src/package/` 对应 `src/<package>/`。

`tree/` 和 `package/` 下的每个文件都带 `.tmpl` 后缀，脚手架写入时去掉。没有
这个后缀，ruff、setuptools 和 loader 会把占位文件当成真文件——
`package/pyproject.toml` 在还写着 `{{vendor}}` 时不是合法 TOML。

## 占位符

脚手架渲染以下名字。某行占位符渲染为空时整行被删掉，所以可选字段可以在模板里
占一行。

`{{vendor}}`、`{{group}}`、`{{group_label}}`、`{{logo}}`、`{{api_key_env}}`、
`{{name}}`、`{{model_type}}`、`{{format}}`、`{{base_url}}`、
`{{distribution}}`、`{{package}}`、`{{core_floor}}`。

`{{core_floor}}` 填入正在运行的 core 版本，所以今天生成的包一定锁在能读它
的 core 上。
