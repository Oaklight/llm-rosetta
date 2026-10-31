---
title: 配置
---

# 配置

网关的配置文件格式详解。

!!! tip "配置重载"
    网关**不会**自动监视配置文件。改了配置后，要么重启进程，要么通过[管理面板](admin-panel.md) / [Admin API](../api/admin.md) 操作——管理界面的改动会热重载并自动写回配置文件。

## 提供方

每个提供方需要 `api_key`、`base_url`，以及可选的 `type` 来指定 API 标准：

```jsonc
"providers": {
  "my-openai":   { "type": "chat_completions",      "api_key": "sk-...",     "base_url": "https://api.openai.com/v1" },
  "my-anthropic": { "type": "messages",        "api_key": "sk-ant-...", "base_url": "https://api.anthropic.com" },
  "my-google":   { "type": "google_generate",            "api_key": "AIza...",    "base_url": "https://generativelanguage.googleapis.com" }
}
```

提供方名称是用户自定义的字符串（如 `"my-openai"`、`"prod-claude"`）。`type` 字段指定使用哪种 API 标准。

可用类型：`chat_completions`、`open_responses`、`messages`、`google_generate`、`google_interactions`。

### 使用 Shim

除了 `type`，还可以使用 `shim` 字段引用已注册的提供方 shim。Shim 是一个轻量级身份卡片，声明了提供方使用的基础 API 标准、连接默认值和字段级转换规则。

```jsonc
"providers": {
  "my-deepseek":   { "shim": "deepseek--chat_completions",   "api_key": "${DEEPSEEK_API_KEY}" },
  "my-volcengine": { "shim": "volcengine--chat_completions",  "api_key": "${VOLCENGINE_API_KEY}", "base_url": "https://ark.cn-beijing.volces.com/api/v3" }
}
```

指定 `shim` 时：

- **基础类型**会自动解析（如 `deepseek--chat_completions` → `chat_completions`）
- 如果配置中未设置，会从 shim 填充默认的 **`base_url`** 和 **`api_key` 环境变量**
- 请求/响应转换时会应用**字段级转换**（如 Volcengine 的 shim 会剥离其 API 不支持的 `logprobs` 和 `top_logprobs` 字段）

内置 shim：`openai--chat_completions`、`openai--open_responses`、`anthropic--messages`、`google--google_generate`、`deepseek--chat_completions`、`volcengine--chat_completions`。

也可以通过 `register_shim()` 以编程方式注册自定义 shim。

### 外部 Shim 源 { #external-shim-sources }

shim 不必放在包内。顶层 `shims` 段可以让 gateway 额外扫描若干目录，以及从远程归档拉取
并缓存到本地 —— 于是一套 shim 可以维护在独立仓库里，直接消费，无需往 gateway 镜像里
安装任何东西。

```jsonc
"shims": {
  // 按名称预载 catalog 包：alcf | argo | asksage。每个都会展开为下方的 remote；
  // 同一 URL 的显式 remote 会覆盖对应的预设。
  "presets": ["alcf"],

  // 扫描 <vendor>/<type>/provider.yaml（内置布局），以及旧的
  // <name>/provider.yaml（平铺）或 <group>/<leaf>/provider.yaml（分组）。
  // 相对路径以本配置文件所在目录为基准解析。
  "paths": ["./local-shims", "../shared/rosetta-shims"],

  "remotes": [
    {
      "name": "alcf",                                  // 可选；缺省时从 URL 推导
      "url": "https://github.com/Oaklight/llm-rosetta-shim-alcf",
      "ref": "v0.1.0",                                 // 可选；见下文「ref 解析」
      "path": "src/llm_rosetta_shim_alcf/providers",   // 归档内的子路径
      "archive": "auto",                               // auto | tar.gz | zip
      "sha256": "0123…abcd",                           // 可选；归档校验
      "token_env": "GITHUB_TOKEN",                     // 可选；仅私有镜像需要
      "refresh": "auto"                                // auto | always | never
    }
  ],

  "cache_dir": "./shims-cache",   // 可选；默认 <config_dir>/shims-cache
  "remote_ttl": 86400,            // 可选；未固定 ref 的重新拉取间隔（秒）
  "strict": false                 // 可选；true 表示某个源失败即拒绝启动
}
```

**目录布局。** `paths` 里的目录可以按模型类型分组
（`vendor/embeddings/provider.yaml`、`vendor/llm/chat_completions/provider.yaml`），
与内置 shim 的布局一致；旧布局仍然原样加载 —— 子目录直接放 `provider.yaml`
（`my-provider/provider.yaml`），或只分组一层（`vendor/messages/provider.yaml`、
`vendor/chat_completions/provider.yaml`）。

**远程拉取不依赖 git。** GitHub 仓库 URL 会被改写到 codeload 的 tarball 端点；其它站点
给出显式归档 URL 即可（可含 `{ref}` 占位符），所以 GitLab 和自建 forge 都不需要按站点
写代码。只接受 HTTPS（回环地址可用 HTTP）。

**ref 解析。** 给了 `ref`（tag / 分支 / commit SHA）就拉该 ref。省略 `ref` 时，会通过
host API（GitHub）把 `HEAD` 解析成具体 commit SHA 并记录进缓存 stamp，后续启动复用同一
commit；配置文件本身不会被改写。要可复现部署请固定 tag 或 SHA。

**缓存与刷新。** 归档解压到 `cache_dir`（默认 `<config_dir>/shims-cache`，与配置文件同级）。
固定 ref、给出 `sha256`、或 `refresh: "never"` 都只拉取一次；`refresh: "always"` 每次都拉；
默认 `auto` 在缓存超过 `remote_ttl` 时重新拉取。拉取失败但已有缓存时，会告警并使用旧缓存；
只有在无缓存可用时该源才算失败。

**优先级。** shim 同名时后者覆盖前者：内置 → entry point 插件 → `paths` 按顺序 →
`remotes` 按顺序。

**失败处理。** 默认情况下坏的源只记录日志、在管理面板中报告并跳过，gateway 照常启动；
设置 `shims.strict: true` 可改为启动失败。

**编程调用。** `llm_rosetta.shims.rebuild_shims()` 可直接加载目录与远程源（不需要配置文件）；
`llm_rosetta_gateway.apply_shim_sources(raw, config_path)` 应用一个配置字典。
`create_app` 也会自动应用该段。

**独立包与预设。** ALCF、AskSage 与 Argo 的 shim 维护在各自的仓库中。`presets`
命名其中之一（`alcf`、`argo`、`asksage`），并展开为 catalog 中的 remote —— 也就是上面
`remotes` 示例里的那条条目。预设先被应用，同一 URL 的显式 remote 会合并到对应预设上
（显式键优先），因此随后叠加固定的 `ref` 或 `sha256` 时不会丢掉 catalog 中的 `path`。发布的
仓库都是公开的，预设不带 `token_env`。若提供方配置仍使用这些 shim 却没有声明源，启动时会得到
一条错误，指明包名、预设与展开后的 remote。管理面板的预设目录（见
[管理面板 → Shim 源](admin-panel.md#shim-sources)）可一键添加同样的条目。

!!! warning "声明的源会执行代码"
    shim 目录可以携带 `transforms.py`，它会在 shim 加载时被执行 —— 与 `pip install`
    安装插件包是同一信任级别。建议固定 `ref`（必要时再给 `sha256`），并且只添加你信任的源。
    信任边界是「谁能修改配置文件」。

### 非 LLM 端点 { #non-llm-endpoints }

一个提供方条目也可以承载 embedding、rerank 或 decision 流量。可以直接写方言，也可以指向该能力对应的 shim：

```jsonc
"providers": {
  // 显式格式：路径默认为 /v1/embeddings
  "openai": {
    "type": "openai",
    "api_key": "${OPENAI_API_KEY}",
    "embeddings_format": "openai"
  },
  // 端点 shim：格式、路径与连接默认值都来自该 shim
  "acme-embed":  { "embeddings_shim": "acme--embedding" },
  "google-embed": { "embeddings_shim": "google--embedding" },
  "acme-rerank": { "rerank_shim": "acme--rerank", "rerank_path": "/v1/rerank" }
}
```

端点键为 `embeddings_format` / `embeddings_path` / `embeddings_shim`、`rerank_format` / `rerank_path` / `rerank_shim`，以及 `decisions_format` / `decisions_path` / `decisions_shim`。每个端点的优先级：

1. **shim**：`<type>_shim`；否则使用提供方级的 `shim` 或 `type`，但仅当该 shim 的 `model_type` 与端点一致；
2. **格式**：显式的 `<type>_format` 优先于 shim 的格式；
3. **路径**：显式的 `<type>_path` 优先于 shim 的 `connection.default_path`（仅当最终格式就是该 shim 的格式时才使用），其次是该格式自身的默认路径（Cohere 使用 `/v2/...`；Google 的 embedding 路径携带模型）；
4. **连接**：条目中非空的 `base_url` / `api_key` 优先于 shim 默认值。

`<type>_shim` 若不属于该能力，启动时会报错并指明该 shim 的类别。gateway 把每个提供方的身份记录为 `(model_type, format)` 对；LLM 模型若被路由到身份为 `embeddings`、`rerank` 或 `decisions` 的提供方，会被拒绝。

旧的 `embeddings_providers` / `rerank_providers` / `decisions_providers` 段仍然可用，并与统一条目合并；改名前写作 `embedding_providers`、`decision_providers` 的段按同一段读取。

模型类型与它服务的端点拼写一致：`embeddings` 与 `decisions`。改名前写下的配置 —— `embedding_format` / `embedding_path` / `embedding_shim`、`default_embedding_format`、`decision_format` / `decision_path` / `decision_shim`、`default_decision_format`，或标记为 `"type": "embedding"` / `"type": "decision"` 的模型 —— 会在加载时改写，原文件另存为 `.pre-migration.bak`（文档版本升到 `config_version` 3）。若文件无法写回（只读挂载），旧键仍会被读取。

### 图片数量限制

Shim 支持 `max_images` 和 `max_images_pattern` 字段，用于对每次请求中的图片数量设置上限：

| 字段 | 类型 | 说明 |
|------|------|------|
| `max_images` | int | 每次请求允许的最大图片数量 |
| `max_images_pattern` | str | 应用于模型名称的正则表达式；仅对匹配模型执行限制 |

当请求超出上限时，最旧的图片将被替换为文本占位符。如果设置了 `max_images_pattern`，仅匹配的模型名称受限——其他模型不受影响，直接透传。

**示例 — 独立的 Argo OpenAI shim：**

Argo OpenAI shim 声明了 `max_images: 50`，`max_images_pattern: "^(gpt|o\d)"`，效果如下：

- GPT 系列和 o 系列模型：图片截断至 50 张
- 通过同一服务方路由的 Gemini 和 Claude 模型：不受限制，直接透传

可通过 `register_shim()` 注册自定义 shim 实现同样的限制逻辑。

!!! tip "解析优先级"
    提供方类型解析顺序为：`shim` → `type` → 提供方名称（兜底）。

!!! note "向后兼容"
    如果 `shim` 和 `type` 都省略，提供方名称本身将用作类型。配置中 `shim` / `type` 里的旧写法（如 `openai_chat`、`openai_responses`、`anthropic`、`google` 等）会在启动时一次性改写为规范名（幂等，并保存 `.pre-migration.bak`），因此旧配置仍可继续使用；无法映射的名称会以指向文档的错误拒绝。

### 启用 / 禁用提供方

每个提供方支持 `enabled` 字段（默认 `true`）。禁用的提供方及其关联模型将从路由中静默排除：

```jsonc
"my-openai": { "type": "chat_completions", "api_key": "sk-...", "base_url": "https://api.openai.com/v1", "enabled": false }
```

这在需要临时下线提供方但不删除配置时很有用。[管理面板](admin-panel.md)提供了切换开关来操作此功能。

### API 密钥轮转

每个提供方支持通过逗号分隔配置多个 API 密钥，网关以轮询方式依次使用：

```jsonc
"my-openai": { "type": "chat_completions", "api_key": "sk-key1,sk-key2,sk-key3", "base_url": "https://api.openai.com/v1" }
```

### 环境变量替换

API 密钥支持 `${ENV_VAR}` 语法 — 启动时从环境变量读取：

```jsonc
"my-openai": { "type": "chat_completions", "api_key": "${OPENAI_API_KEY}", "base_url": "https://api.openai.com/v1" }
```

### 动态令牌刷新（`token_command`）

对于使用短期令牌的提供方（如 ALCF 推理服务的 Globus OAuth），网关可以运行外部命令来获取并定期刷新 API 密钥：

```jsonc
"my-lane": {
  "shim": "vendor_lane--chat_completions",
  "token_command": ["python3", "/opt/vendor/token.py"],
  "token_refresh_interval": 3600
}
```

shim 可以自己声明这条命令 —— `provider.yaml` 里的 `connection.token_command` —— 此时提供方完全不必写 `token_command`。ALCF 与 AskSage 的 shim 都随包发布了该命令，因此它们的提供方只需要 `{"shim": "..."}`；在提供方上自己写的 `token_command` 仍然优先。

提供方类型取自 `shim`；`type`，然后是提供方自身的键名，依次作为兜底。（`provider`
并不是配置键 —— 以这种拼写写出的提供方条目从未能解析到它的 shim。）

| 字段 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `token_command` | `list[str]` | — | 命令参数列表；stdout 作为 API 密钥 |
| `token_refresh_interval` | `int` | `3600` | 定时刷新间隔秒数（最小 60） |

设置 `token_command` 后：

1. 命令在**启动时**运行以获取初始 API 密钥
2. 后台任务每隔 `token_refresh_interval` 秒重新运行
3. 收到上游 **401 响应**时，网关立即触发非周期性刷新（带防抖避免风暴）
4. 命令的 stdout 作为新密钥 — 支持逗号分隔的多密钥轮转

!!! note "`token_command` 和 `api_key` 互斥"
    如果在同一个提供方条目里同时指定两者，网关拒绝启动。只能使用其中之一。shim 的 `api_key_env` 不算“指定了密钥”：它只是默认凭据，配置了命令时便由命令取代——因此在 `token_command` 提供方旁边导出 `ALCF_API_KEY` 是允许的。

#### 需要写代码的交换（`auth.py`）

`token_command` 适合用一条程序就能表达的交换。OAuth 配置不行：首次登录是交互式的，refresh
token 每次使用都会轮换，而交换保存下来的是一个记录而不是一个字符串。这类 shim 在
`provider.yaml` 旁边附带 `auth.py`，网关改为调用它而不是命令 —— 同样在启动时填充、同样按
后台间隔刷新、同样在上游 `401` 后被动刷新。提供方配置不需要任何改动，因此 shim 附带
`auth.py` 的提供方依然只写 `{"shim": "..."}`。

两种形式互斥：shim 只声明其一，而在提供方上写的 `token_command` 优先级高于任一形式。模块
契约见 [提供方 Shim](../guide/shims.md)。

在交换能产出凭据之前 —— 对 Sign in with ChatGPT 而言，就是人工完成登录之前 —— 提供方会在
`token_status` 中被标记为需要交互式登录，且不会向上游发出请求，而不是让每个请求都收到
`401`。

#### Docker 中使用 ALCF 的示例

ALCF 提供方使用 Globus OAuth 令牌，其交换逻辑随 `llm-rosetta-shim-alcf` 发布（镜像已安装该包），因此只需要挂载令牌文件：

```yaml
# docker-compose.yaml
volumes:
  - ./config:/config
  - ~/.globus:/home/appuser/.globus              # Globus 令牌文件（需读写权限以刷新）
```

操作步骤：

1. **在宿主机上登录**（仅需一次）：`python3 -m llm_rosetta_shim_alcf.providers.alcf.token --login`
2. 使用上述 volume 挂载**启动容器**
3. 通过管理面板或 `config.jsonc` **添加提供方** —— 不需要 `token_command`

容器内的 `appuser`（uid 1000）需要对 `~/.globus` 目录有读写权限，以便就地更新刷新令牌。

### 逐提供方代理

可为单个提供方指定代理：

```jsonc
"my-anthropic": { "type": "messages", "api_key": "sk-ant-...", "base_url": "https://api.anthropic.com", "proxy": "http://proxy:8080" }
```

## 代理配置

可在 `server` 部分设置全局代理，适用于所有提供方（除非逐提供方覆盖）：

```jsonc
{
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "proxy": "http://proxy.example.com:8080"
  }
}
```

同时支持 HTTP 和 SOCKS5 代理：

```jsonc
// HTTP 代理
"proxy": "http://proxy.example.com:8080"

// SOCKS5 代理（无认证）
"proxy": "socks5://proxy.example.com:1080"

// SOCKS5 代理（用户名/密码认证）
"proxy": "socks5://username:password@proxy.example.com:1080"
```

CLI `--proxy` 参数会覆盖配置文件中的全局代理设置。

## Unix 域套接字

Gateway 可以监听 Unix 域套接字而非 TCP。适用于共享多用户主机（如 HPC 登录节点），在这些环境中绑定 `127.0.0.1` 仍会暴露服务给所有本地用户：

```jsonc
{
  "server": {
    "socket": "/run/user/1000/rosetta.sock"
  }
}
```

或通过 CLI：

```bash
llm-rosetta-gateway --socket /run/user/$(id -u)/rosetta.sock
```

设置 `socket` 后，`host` 和 `port` 将被忽略。套接字文件：

- 创建时设置为**仅所有者可访问**（`0600`）——主机上的其他用户无法连接
- 关闭时**自动删除**
- 启动时**自动清理残留套接字**（如果上一个实例崩溃）

结合 SSH `LocalForward`，可实现端到端的访问控制。

## 模型路由

`models` 部分将模型名称映射到提供方：

```jsonc
"models": {
  "gpt-4o": "my-openai",
  "claude-sonnet-4-20250514": "my-anthropic",
  "gemini-2.0-flash": "my-google"
}
```

当请求包含 `"model": "claude-sonnet-4-20250514"` 时，网关查找到 `my-anthropic` 并相应转发。

### 模型能力声明

模型可以使用字典格式声明能力：

```jsonc
"models": {
  "gpt-4o": { "provider": "my-openai", "capabilities": ["text", "vision", "tools"] },
  "gemini-2.0-flash": { "provider": "my-google", "capabilities": ["text", "tools"] }
}
```

可用能力：`text`、`vision`、`tools`、`embedding`、`reasoning`。如未指定，默认为 `["text"]`。注意 `embedding` 与 `vision`/`tools` 互斥，`reasoning` 与 `embedding` 互斥。

能力信息显示在[管理面板](admin-panel.md)中，也可在面板中编辑。

### 路由策略

多提供方模型条目支持 `strategy` 字段来控制网关如何在提供方之间选择：

```jsonc
"models": {
  "gpt-4o": {
    "providers": [
      {"name": "openai-prod", "weight": 5},
      {"name": "openai-backup", "weight": 1}
    ],
    "strategy": "weighted_round_robin"
  }
}
```

| 策略 | 描述 |
|------|------|
| `weighted_round_robin` | **（默认）** 平滑的 nginx 风格加权轮询。权重 [5, 1] 产生序列 A A A A A B，而非突发切换。 |
| `affinity_round_robin` | 对客户端标识（API key 或 IP）做 SHA-256 哈希，确定性地选择首选提供方，实现缓存亲和。无标识或仅单提供方时回退到加权轮询。 |

策略按模型设置。单提供方模型忽略此字段。

### 同格式偏好

`server.prefer_same_format`（默认 `false`）让 provider 选择偏向已经在讲客户端 API 格式的上游，多格式的模型因此只在必要时才做转换。选择先确定一个**锚点**——即关闭该开关时路由会挑中的那个 provider，因此全量 route 的计数推进与关闭时完全一致——再在这个锚点范围内把格式对齐到客户端，分三层：

1. 与锚点**同 shim、同 endpoint**（通常就是锚点自身）。shim 已经固定了格式，所以这一层只决定同 endpoint 之间的故障转移顺序。
2. **同一上游账号**的其它格式——锚点所属账号也能提供客户端的格式，请求因此停留在同一账号，而不必换到另一个账号。
3. 客户端格式下的**任意 provider**。
4. 都不满足则回到锚点，经 IR 转换。

账号身份是推导出来的，不需要声明。两个 provider 携带同一份凭据即视为同一账号——同一个 `token_command`，或同一个 key 值（字面 key 与由环境变量解析出的同值 key 视为同一个 key）。比较的只是凭据的哈希，keyless provider 永不归组。`base_url` 被刻意排除在外，因此用同一个 key、从不同路径提供多种格式的厂商仍算作一个账号。

该开关只改变*由哪个*上游来服务请求。转换依然始终经过 IR——包括第 1 层的情况。参见[架构指南](../contributing/architecture.md)。

### 配额故障转移 { #quota-failover }

`failover_on_quota`（按模型配置，默认 `false`）让请求在服务方因 in-band 配额拒绝而没有任何可用凭据时，改由同一模型的另一个 provider 来服务：

```jsonc
"models": {
  "gpt-5": {
    "providers": ["openai-prod", "azure-backup"],
    "failover_on_quota": true
  }
}
```

触发条件刻意很窄：必须是 provider 的 shim 归类为配额拒绝的上游 `429`（部分 shim 也会归类 `403`），且该 provider 已没有凭据能服务这次请求。普通的 `5xx`、`401`，以及网关无法归类的 `429`，都照旧原样返回。预算是**一次故障转移** —— 不是一次上游调用：如果同一 provider 还有一个可用凭据，会先花掉原有的同 provider 重试，只有整个 provider 都用尽才会转移。拒绝的 provider 在本次请求余下部分被排除，改试另一个 provider；如果它也拒绝，客户端收到的就是它这次的拒绝。

默认关闭，是因为它真的有所取舍：换一个 provider 就是换一份 prompt 缓存、换一个限流桶，甚至可能是另一个模型构建。真的发生时，请求的 profile 会记录 `failover: {from, to, reason}`，请求日志行也指向实际服务的 provider。

流式请求同样会故障转移，但只在第一个事件之前：处理程序在写出任何内容之前就打开上游连接，因此这一刻的拒绝仍可由别处作答。字节一旦发给客户端，响应就无法重放，流中途的失败会以对应代码结束客户端的流。

## 熔断器 { #circuit-breaker }

可选的熔断器会在提供方持续失败时直接短路请求，而不是把每个请求都发到上游再失败一次。默认关闭。

```jsonc
{
  "server": {
    "circuit_breaker": {
      "enabled": true,
      "error_threshold": 5,      // 连续失败多少次后 CLOSED -> OPEN
      "cooldown_seconds": 30,    // OPEN 状态等待多久后允许探测
      "half_open_max_probes": 1  // HALF_OPEN 状态允许的并发探测数
    }
  }
}
```

每个提供方都可以用自己的 `circuit_breaker` 对象覆盖任意字段。熔断器按提供方统计连续失败：
达到 `error_threshold` 后打开，请求在本地以 `503` 与 `Retry-After` 拒绝；经过
`cooldown_seconds` 后放行少量探测，探测成功则关闭，失败则重新打开。管理台仪表盘会报告每个
已启用熔断器的状态。熔断器负责“处置”失败，健康指标仍继续“报告”失败。

## 网关 API Key { #gateway-api-keys }

网关 API Key 通过**管理面板**管理——配置文件中不再设置 API Key。

在 `/admin` → **Keys** 标签页中生成、轮换和删除 Key，或通过
[Admin API](../api/admin.md#api-key-management) 操作。所有 Key 存储在 SQLite
keystore（`keys.db`）中。

配置了 Key 后，所有 `/v1/*` 端点需要使用对应 API 标准的原生格式进行认证：

| API 标准 | 凭证格式 |
|---------|---------|
| OpenAI Chat / Responses | `Authorization: Bearer <key>` |
| Anthropic | `x-api-key: <key>` |
| Google GenAI | `x-goog-api-key: <key>` 或 `?key=<key>` 查询参数 |

!!! note "管理面板"
    管理面板（`/admin/*`）**不需要**网关 API Key。可以使用内置的 `admin_password` 选项保护（见下文），也可以通过反向代理实现（如 Caddy 的 `basicauth`、Nginx 的 `auth_basic`）。

未配置任何 Key 时，行为取决于 `open_on_no_keys`（默认：`false`——所有 `/v1/*` 请求被阻止）。

## 管理面板安全 { #admin-panel-security }

### `admin_password`

可选。设置后，访问管理面板（`/admin/*`）前需要密码登录。会话通过服务端内存 session store 追踪。

支持 `${ENV_VAR}` 替换：

```jsonc
{
  "server": {
    "admin_password": "${ADMIN_PASSWORD}"
  }
}
```

!!! tip
    如果网关对公网开放，强烈建议设置 `admin_password`，防止未授权访问提供方配置和请求日志。

!!! warning "未解析的占位符"
    如果 `admin_password` 包含未解析的 `${ENV_VAR}` 占位符（即环境变量未在启动时设置），网关会**拒绝启动**并输出清晰的错误信息，防止将字面量字符串 `${ADMIN_PASSWORD}` 作为密码使用。

!!! info "会话认证与内部令牌"
    浏览器认证使用**基于会话的 cookie**（HttpOnly + SameSite=Lax，
    30 分钟无活动超时）。`X-Admin-Token` 请求头是一个独立的机制，
    用于编程式 API 访问——它通过 HMAC 比较内部代理令牌进行验证，
    而非管理密码。

### `credential_visible`

布尔值，默认 `true`。设为 `false` 后，管理界面中所有 API 密钥的值将被隐藏——复制和查看控件均会禁用。适用于网关被多个用户共用、不希望密钥从面板中直接读取的场景。

```jsonc
{
  "server": {
    "credential_visible": false
  }
}
```

!!! note
    此设置仅控制界面可见性。密钥仍会被网关用于上游请求，只是不在管理界面中展示。

### `admin_cors_origins`

允许跨域访问管理 API（`/admin/api/*`）的来源列表。默认为空列表，表示不发送 `Access-Control-Allow-Origin` 响应头，仅允许同源请求。

如需允许特定来源：

```jsonc
{
  "server": {
    "admin_cors_origins": ["https://my-dashboard.example.com"]
  }
}
```

!!! note
    CORS 收紧仅对 `/admin/api/*` 端点生效，`/v1/*` 代理端点不受影响。

## 限流

网关支持基于客户端的单窗口或多窗口滑动限流。在 `server` 段中配置：

```jsonc
{
  "server": {
    "rate_limit": "10/m"        // 单窗口：每分钟 10 个请求
  }
}
```

多窗口限流使用逗号分隔的窗口规格：

```jsonc
{
  "server": {
    "rate_limit": "10/m, 100/h"  // 每分钟 10 个 且 每小时 100 个
  }
}
```

| 窗口后缀 | 含义 |
|---------|------|
| `/s` | 每秒 |
| `/m` | 每分钟 |
| `/h` | 每小时 |

每个客户端（通过 API key 或 IP 识别）独立追踪。当任一窗口耗尽时，请求收到 429 响应，带有 `Retry-After` header。

限流状态可通过 admin 面板的 `GET /admin/api/rate-limits` 查看。

## 路由环路检测

当多个网关实例链式连接时（如校园网关转发到云端网关），可能出现路由环路。网关通过跳数 header 检测：

```jsonc
{
  "server": {
    "max_hops": 4              // 默认值：4
  }
}
```

每个网关递增 `X-Rosetta-Hops` header。当计数超过 `max_hops` 时，请求被拒绝并返回 508（Loop Detected）响应。

## 软错误检测

某些上游 Provider 返回 HTTP 200 但 body 包含错误内容（如限流 HTML 页面、JSON 错误对象）。网关可通过 shim 配置的正则模式检测并重新包装为正确的错误响应：

```jsonc
// 在 provider shim YAML 中：
soft_error_patterns:
  - pattern: "rate limit exceeded"
    status: 429
  - pattern: "internal server error"
    status: 500
```

当 200 响应 body 匹配到模式时，将使用配置的状态码和正确的错误信封重新包装。

## 保真度验证

网关对**同格式路由**（如 OpenAI → OpenAI 经不同提供方）验证转换保真度，
通过比较原始请求/响应体与往返转换后的结果。

- **同格式影子对比始终运行** — 无需配置。当源和目标使用相同的 API
  格式时，网关自动对比原始和转换后的请求与响应。
- **严重级别为 critical 的差异**会自动触发错误转储，可在管理面板的
  错误转储区域查看。
- 保真度差异结果存储在请求的 profile 数据中，位于 `"fidelity"` 键下。

跨格式路由不运行保真度对比，因为源和目标格式本质不同。

## 延迟启动

默认情况下，网关将阻塞性启动工作（Provider 连通性检查、模型列表拉取）推迟到后台任务。网关立即开始接受请求，同时这些任务在后台完成。启动进度在 admin 面板中可见。

## 错误转储保留

错误转储通过 `server` 段中的 `error_dump_cap` 限制上限：

```jsonc
{
  "server": {
    "error_dump_cap": 500      // 默认值：500
  }
}
```

!!! warning "已废弃的 key"
    之前的 `error_max` key 已废弃，启动时会发出警告。请使用 `error_dump_cap` 替代。

## 调试选项

```jsonc
{
  "debug": {
    "verbose": true,       // 启用 DEBUG 级别日志
    "log_bodies": true     // 记录完整的请求/响应体
  }
}
```

也可以通过环境变量设置：`LLM_ROSETTA_VERBOSE=1`、`LLM_ROSETTA_LOG_BODIES=1`。

## 请求追踪

每个代理请求都会携带 `X-Request-ID` 请求头。若来源请求已包含该头，则保留其值；否则自动生成新的 UUID。该请求头会：

- 转发给上游服务方
- 出现在所有响应头中（包括错误响应）
- 以 `[request_id]` 前缀写入日志，实现端到端可追溯

无需任何配置——请求 ID 传播始终生效。

## 健康检查端点

网关提供三个健康检查端点：

| 端点 | HTTP 状态码 | 说明 |
|------|-----------|------|
| `/health` | 始终 200 | 网关状态：运行时长、请求总数、最近一小时错误数、各服务方健康状况 |
| `/health/live` | 始终 200 | Kubernetes 存活探针——确认进程正在运行 |
| `/health/ready` | 200 / 503 | Kubernetes 就绪探针——启动期工作尚未完成、或任意服务方降级时返回 503 |

`/health/ready` 还会用一个 `maintenance` 对象报告启动后的数据修复做了什么：
`state` 取 `not_started`、`running`、`done`，或带失败步骤名的 `partial`。
某一步修复失败**不会**让探针返回 503——每一步都可以重试，为它挡掉流量只会把
一次迟到的修复变成一次故障。详见[管理 API
参考](../api/admin.md#maintenance)。

`/health` 响应示例：

```json
{
  "status": "ok",
  "uptime": 3600.5,
  "requests_total": 1234,
  "errors_last_hour": 2,
  "providers": {
    "openai-prod":    { "status": "ok" },
    "anthropic-prod": { "status": "ok" }
  }
}
```

`status` 字段取值：所有服务方正常时为 `"ok"`，一个或多个服务方异常时为 `"degraded"`。

## Embedding 提供方

网关可以代理 `/v1/embeddings` 请求，支持跨格式 IR 转换（OpenAI ↔ Cohere ↔ Jina ↔ Voyage ↔ Google GenAI）。统一提供方条目可以用 `embeddings_format` 或 `embeddings_shim` 声明该端点（见[非 LLM 端点](#non-llm-endpoints)）；下面的旧格式仍然受支持。Embedding 提供方和模型独立于 chat 提供方配置：

```jsonc
{
  "embeddings_providers": {
    "jina-embed": { "type": "jina", "api_key": "${JINA_API_KEY}", "base_url": "https://api.jina.ai" },
    "voyage-embed": { "type": "voyage", "api_key": "${VOYAGE_API_KEY}", "base_url": "https://api.voyageai.com" }
  },
  "embeddings_models": {
    "jina-embeddings-v3": "jina-embed",
    "voyage-3-large": "voyage-embed"
  },
  "default_embeddings_format": "openai"  // 自动检测失败时的源格式
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `embeddings_providers` | `dict` | `{}` | Embedding 上游的提供方配置（格式同 `providers`） |
| `embeddings_models` | `dict` | `{}` | 模型 → 提供方映射 |
| `default_embeddings_format` | `str` | `"openai"` | 源格式兜底。选项：`openai`、`cohere`、`jina`、`voyage`、`google_generate`、`minimax` |

Embedding 流量**只**通过 embedding 端点路由。没有 embedding 路由的模型——即未列入
`embeddings_models`，或统一提供方上没有 `"type": "embeddings"`——会返回 `404`，网关不会
把它转发给 chat 提供方。Google 端点把模型放在 URL 中
（`/v1beta/models/{model}:embedContent`），当转换后的请求包含多个输入时改用
`:batchEmbedContents`。

## Rerank 提供方

网关可以代理 `/v1/rerank` 和 `/v2/rerank` 请求，支持跨格式 IR 转换（Jina ↔ Cohere ↔ Voyage）。统一条目使用 `rerank_format` 或 `rerank_shim`（见[非 LLM 端点](#non-llm-endpoints)）；下面的旧格式仍然受支持：

```jsonc
{
  "rerank_providers": {
    "jina-rerank": { "type": "jina", "api_key": "${JINA_API_KEY}", "base_url": "https://api.jina.ai" },
    "cohere-rerank": { "type": "cohere", "api_key": "${COHERE_API_KEY}", "base_url": "https://api.cohere.com" }
  },
  "rerank_models": {
    "jina-reranker-v2-base-multilingual": "jina-rerank",
    "rerank-v3.5": "cohere-rerank"
  },
  "default_rerank_format": "jina"  // 自动检测失败时的源格式
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rerank_providers` | `dict` | `{}` | Rerank 上游的提供方配置 |
| `rerank_models` | `dict` | `{}` | 模型 → 提供方映射 |
| `default_rerank_format` | `str` | `"jina"` | 源格式兜底。选项：`jina`、`cohere`、`voyage` |

`/v2/rerank` 端点会根据 URL 路径自动检测 Cohere 源格式。

## Decision 提供方

网关可代理 `/v1/decision` 和 `/v1/systemone` 请求，用于概率化结构决策模型。统一条目使用 `decisions_format` 或 `decisions_shim`（见[非 LLM 端点](#non-llm-endpoints)）；下面的旧格式仍然受支持：

```jsonc
{
  "decisions_providers": {
    "jev-prod": { "type": "typesafe", "api_key": "${JEV_API_KEY}", "base_url": "https://api.typesafe.ai" }
  },
  "decisions_models": {
    "jev-1": "jev-prod"
  },
  "default_decisions_format": "systemone"
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `decisions_providers` | `dict` | `{}` | Decision 上游的 provider 配置 |
| `decisions_models` | `dict` | `{}` | 模型 → provider 映射 |
| `default_decisions_format` | `str` | `"systemone"` | 源格式回退 |

旧式条目的 `format` 或 `type` 可以写 decision format（`systemone`、`openai_decisions`），也可以写已注册的 decision shim，由其提供 format 与连接默认值（例如 `"type": "typesafe"` → `systemone`）。`decisions_path` 默认取该 format 的上游路径（`openai_decisions` → `/v1/decisions`，其余为 `/v1/systemone`）。

## 完整示例

```jsonc
{
  "providers": {
    "openai-prod":    { "type": "chat_completions",      "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "openai-resp":    { "type": "open_responses",  "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "anthropic-prod": { "type": "messages",         "api_key": "${ANTHROPIC_API_KEY}",  "base_url": "https://api.anthropic.com" },
    "google-prod":    { "type": "google_generate",            "api_key": "${GOOGLE_API_KEY}",     "base_url": "https://generativelanguage.googleapis.com" },
    // 基于 Shim 的提供方 — base_url 和 transforms 自动解析
    "deepseek":       { "shim": "deepseek--chat_completions",          "api_key": "${DEEPSEEK_API_KEY}" },
    "volcengine":     { "shim": "volcengine--chat_completions",         "api_key": "${VOLCENGINE_API_KEY}", "base_url": "https://ark.cn-beijing.volces.com/api/v3" }
  },
  "models": {
    "gpt-4o":                     { "provider": "openai-prod",    "capabilities": ["text", "vision", "tools"] },
    "claude-sonnet-4-20250514":   { "provider": "anthropic-prod", "capabilities": ["text", "vision", "tools"] },
    "gemini-2.0-flash":           { "provider": "google-prod",    "capabilities": ["text", "tools"] },
    "deepseek-r1":                { "provider": "deepseek",       "capabilities": ["text", "tools"] }
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "api_key": "${GATEWAY_API_KEY}",
    "admin_password": "${ADMIN_PASSWORD}",
    "credential_visible": false,
    "admin_cors_origins": []
  }
}
```
