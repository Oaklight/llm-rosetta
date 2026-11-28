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

shim 不必放在包内。顶层 `shims` 段会读取**扩展索引**并从中安装服务方，于是一套 shim 可以维护在独立仓库里，无需重新构建 gateway 镜像就能使用。源是扩展仓库发布的索引（`index.json`）；安装就是把一个[扩展](../guide/extensions.md) wheel 放进扩展目录。

```jsonc
"shims": {
  // 扩展索引（catalog id 或索引 URL）以及从中安装的服务方：服务方名、
  // tag:<标签>、*，均可加 "<source>:" 前缀；!<source>:<vendor> 表示排除某个服务方。
  // 只写 install 时读取 contrib 与 argonne。
  "sources": ["contrib", "argonne"],
  "install": ["opendesign", "tag:tier:free", "argonne:*"],

  // 要关闭的 shim 名称（管理面板里每个 shim 的开关）。
  "disabled": ["argo--messages"],

  "cache_dir": "./shims-cache",   // 可选；默认 <config_dir>/shims-cache
  "remote_ttl": 86400,            // 可选；索引的新鲜期（秒）
  "strict": false                 // 可选；true 表示某个源失败即拒绝启动
}
```

网关只读取这些键：`sources`、`install`、`disabled`、`cache_dir`、`remote_ttl`、`strict`。

!!! danger "0.15.0 已移除"
    `shims.presets`、`shims.include`、`shims.exclude`、`shims.remotes` 与 `shims.paths` 不再被读取，
    网关也不再读取目录里的 `shims.yaml`。配置里只要还带着其中任何一个非空的键，
    **加载配置时就会被拒绝**，错误为 `config: 'shims.<key>' was removed in 0.15.0: <建议>`。
    运行 `llm-rosetta gateway migrate --config <path>`（先干跑）再加 `--write`，即可把 `presets`、`include`、`exclude`
    改写成新形式，见 [已移除的键](../guide/shims.md#retired-keys)。`remotes` 条目（显式归档 URL）和 `paths` 目录
    没有对应的改写：请发布一个扩展索引并写进 `shims.sources`，或把该目录变成
    [扩展文件夹](../contributing/adding-an-extension.md)并安装到扩展目录里。

**扩展源。** `sources` 列出扩展索引：catalog id（`contrib`、`argonne`）、索引 URL，
或 `{"url", "fallback", "token_env"}`。catalog id 解析为 `Oaklight/llm-rosetta-ext-contrib` 或
`Oaklight/llm-rosetta-ext-argonne` 的 `releases/latest/download/index.json`。`install` 从中选择服务方：`<vendor>`、
`tag:<标签>`（整体匹配，如 `tag:tier:free`）、`*`，均可加 `<source>:` 前缀；`!<vendor>` 与
`!<source>:<vendor>` 排除某个服务方，`<source>:<vendor>@<version>` 固定某个版本。网关为每个服务方
安装要求与本 core 和解释器匹配的最新稳定版本（`surface` 等于 `SURFACE_VERSION`，`ir` 不高于
`IR_VERSION`；只有在没有可安装的稳定版本时才安装预发布版本），下载 wheel，按索引条目校验其 sha256，
再用 core 的标准库安装器装入扩展目录。只在出现更新的可安装版本时升级。在已应答的
源中没有匹配结果的条目会记录日志，并报告为 *不在 manifest 中* 的条目。索引在
`remote_ttl` 内复用；源不可达时保留缓存的索引与上一次成功的安装。core 按 shim 优先：
core（或其他持有者）已注册的 shim 名称不会被替换，扩展的状态会说明是谁持有该名称。
见 [Manifest 源](../guide/shims.md#manifest-sources)。

已安装的扩展无论 `shims.install` 是否仍选中它都会加载：选择决定安装什么、保持更新什么，`shims.disabled` 决定关闭什么。
从配置里去掉一个源或整个 `shims` 段，不会卸载任何东西。

**扩展装在哪里。** 网关装到 `<data dir>/extensions`。设置了 `LLM_ROSETTA_EXTENSIONS_DIR` 时它优先，
`llm-rosetta ext` 命令读取同一个变量，因此用命令安装和由网关提供服务会落在同一个位置。
Docker 镜像把它设为 `/config/extensions`。

**独立包。** ALCF、AskSage、Argo、免费端点与 Open Design 的 shim 位于 core 之外，在 `argonne` 与
`contrib` 源里。用 `shims.install` 声明服务方：`argonne:alcf`、`argonne:argo`、`argonne:asksage`、
`contrib:tag:tier:free`（免费端点）、`contrib:opendesign`。若提供方配置仍使用这些 shim 却没有声明源，
启动时会得到一条错误，指明服务方与应加入的 `shims.install` 条目。已迁出 core 的服务方，在配置里提到其 shim 时会被自动安装。

**缓存与刷新。** 索引与安装记录存放在 `cache_dir`（默认 `<config_dir>/shims-cache`，与配置文件同级），
索引在 `remote_ttl` 秒内复用。热缓存覆盖传输失败、HTTP 错误、磁盘写不进去和 `token_env` 未设置：
改用缓存的索引并标记为 stale。`sha256` 不符永远不会被掩盖，`generated_at` 早于缓存副本的索引也永远不会替换缓存。

**优先级。** shim 同名时 core 优先：先内置，再已安装的扩展。所有 shim 都已被占用的扩展会在下载之前就被跳过。

**失败处理。** 默认情况下坏的源只记录日志、在管理面板中报告并跳过，gateway 照常启动；
设置 `shims.strict: true` 可改为启动失败（`install` 里出现未知的 `<source>:` 限定符也会被拒绝）。

**编程调用。** `llm_rosetta_gateway.apply_shim_sources(raw, config_path)` 应用一个配置字典；
`create_app` 也会自动应用该段。

!!! warning "声明的源会执行代码"
    已安装的扩展可能带有 `transforms.py` 或 transport，扩展加载时会执行它 —— 与 `pip install` 一个包同一信任级别。
    索引意味着以发布它的仓库的信任级别执行代码。只添加你信任的源。信任边界就是谁能编辑配置文件。

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

每个端点的设置也可以写进以该端点命名的块里。两种写法是同一个设置；管理面板写出的是块。

```jsonc
"providers": {
  // 一个账号、两家服务方：聊天在一家，embedding 在另一家。
  "frankenservice": {
    "type": "google_generate",
    "base_url": "https://generativelanguage.googleapis.com",
    "api_key": "${FRANKEN_KEY}",
    "embeddings": {
      "shim": "cohere--embedding",
      "base_url": "https://api.cohere.com/v2"
    }
  }
}
```

一个提供方条目是一个**账号**：一份凭据、一个 key ring、一次刷新。每个类型端点都是它的一个**面**，可以落在另一家服务方的主机上 —— 所以 `base_url` 属于面，而不只属于条目。块内的字段是 `shim`、`format`、`path`、`enabled` 与 `base_url`。

管理面板还会在条目上存一个 `preset`：Basic 页「服务方预设」所选的服务方（一个 shim 名，例如 `openai--chat_completions`）。运行时没有任何代码读它 —— 每个面仍带着自己的 `shim`，因为网关不会按服务方推断面。编辑器用 `preset` 给尚未手动设置的面填默认值，并区分「跟随预设的面」与「改成了别家服务方的面」；没有它的条目按其 LLM `type` 打开。

`base_url` 是主机，`path` 是该主机上的路径 —— 请求发往 `base_url` + `path`。版本段属于 `path`（或 shim 的默认路径），不属于 `base_url`：写成 `base_url: "https://api.cohere.com/v2"` 而端点路径是 `/v1/embeddings`，实际请求的是 `/v2/v1/embeddings`，真实服务方会返回 404。扁平的 `embeddings_format` / `embeddings_path` / `embeddings_shim` / `embeddings_enabled`（以及 `rerank_*`、`decisions_*` 两组）仍然可读，加载时会改写成块（`config_version` 5）。

每个端点的优先级：

1. **shim**：块内的 `shim`（或扁平的 `<type>_shim`）；否则使用提供方级的 `shim` 或 `type`，但仅当该 shim 的 `model_type` 与端点一致；
2. **格式**：显式的格式优先于 shim 的格式；
3. **路径**：显式的路径优先于 shim 的 `connection.default_path`（仅当最终格式就是该 shim 的格式时才使用），其次是该格式自身的默认路径（Cohere 使用 `/v2/...`；Google 的 embedding 路径携带模型）；
4. **主机**：端点自己的 `base_url`；否则条目的 `base_url`；否则 shim 的 `connection.base_url`。若端点继承条目的主机、而没有为它服务的 shim 声明该主机，启动时会写一条日志，点出相关主机和修法 —— 这种组合几乎总是块出现之前写下的条目，请求会发往错误的服务方。多家服务方共用的方言不等于某一家服务方的主机：SiliconFlow 上的 `embeddings: {format: "openai"}` 是被接受的，因为 `siliconflow--embedding` 为它服务并声明了 SiliconFlow 的主机；
5. **凭据**：条目的（`base_url` / `api_key` 默认值来自 shim）。一个条目就是一份凭据；两家服务方两把 key 就是两个条目。

`<type>_shim` 若不属于该能力，启动时会报错并指明该 shim 的类别。gateway 把每个提供方的身份记录为 `(model_type, format)` 对；LLM 模型若被路由到身份为 `embeddings`、`rerank` 或 `decisions` 的提供方，会被拒绝。

声明在 `providers:` 条目上的端点转发**该条目自己的凭据**。当该条目自行铸造凭据 —— `token_command`，或 shim 的 `auth.py` 用已保存的记录换取 —— 端点读取的是聊天通道使用的同一个 key ring：一次铸造、一次启动填充、一份刷新循环即可服务该提供方的全部端点；铸造失败时，提供方卡片会写明是哪条命令、什么错误，而不是把空凭据发往上游。配额状态同样共享：被探测或 429 降级的 key，会被该提供方的每个端点跳过，因为额度属于凭据，不属于端点。端点收到 401 时会经同一来源重铸一次，并重试请求一次；若再次被拒则如实返回，而会话策略 401（需要人工重新登录）永不重试。

共享止于「测量」这一层：端点不轮询额度，因此不参与余量加权，也不参与「所有凭据都已用尽」的排除判定 —— 它只被凭据自身报告的状态排除（key 被降级、提供方被阻塞、或限流停用）。

旧的 `embeddings_providers` / `rerank_providers` / `decisions_providers` 段仍然可用，并与统一条目合并；改名前写作 `embedding_providers`、`decision_providers` 的段不再被读取：含有它们的配置会在加载时失败，并写出替代的段名。旧段中的条目在 `providers:` 里没有可共享铸造凭据的对应条目，因此其中自行铸造凭据的条目仍会被停用，理由会指明该段 —— 把它移到 `providers:` 即可使用。

模型类型与它服务的端点拼写一致：`embeddings` 与 `decisions`。配置加载器不再读取改名前的写法：`embedding_format` / `embedding_path` / `embedding_shim` / `embedding_enabled`、`default_embedding_format`、`embedding_models`、`decision_format` / `decision_path` / `decision_shim`、`default_decision_format`、提供方的 `embedding:` 或 `decision:` 块，以及标记为 `"type": "embedding"` / `"type": "decision"` 的模型。每一项都会在加载时失败，输出一行 `llm-rosetta-gateway: error: ...`，写出文件、键名与替代写法，并以退出码 1 结束。`llm-rosetta gateway migrate --config <path> --write` 可改写它们；参见 [迁移到 0.14](../guide/migrating-to-0.14.md#llm-rosetta-migrate)。比当前 `config_version` 更旧的文档在加载时仍会被升级（原文件另存为 `.pre-migration.bak`）；只有已经处于当前版本的文档才会因旧写法被拒绝。

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
    如果 `shim` 和 `type` 都省略，提供方名称本身将用作类型。`shim` / `type` 里的已退役写法（如 `openai_chat`、`openai_responses`、`anthropic`、`google` 等）会在启动时被拒绝，错误信息写出文件、键名与替代名称，网关以退出码 1 结束；不再被改写。运行 `llm-rosetta gateway migrate --config <path> --write` 改写配置，参见 [迁移到 0.14](../guide/migrating-to-0.14.md#retired-names)。

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

ALCF 提供方使用 Globus OAuth 令牌，其交换逻辑随 `alcf` 扩展发布（来自 `argonne` 源；声明 `shims.install: ["argonne:alcf"]`），因此只需要挂载令牌文件：

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

容器以 `${PUID:-1000}` 运行（compose 的 `user:` 指令），挂载的 `~/.globus` 必须对该 uid 可写，以便就地更新刷新令牌。

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

## 请求体大小 {#request-body-size}

入站服务器默认接受最大 **50 MB** 的请求体。要转发很长的工具密集对话，或把一份文档
放进视觉请求的部署，可以调高上限：

```jsonc
{
  "server": {
    "max_body_size": 100000000
  }
}
```

值以**字节**为单位，限制的是整个请求体，在解析之前生效：超过上限的请求直接返回 `413`，
不做转换。下限是 **1 MB** —— 更小的值会被夹到下限，因为没有任何提供方的载荷会这么小，
这样的值只可能是笔误。不是数字的值是加载错误，会点名这个键，而不是悄悄退回默认值：
以为调高了上限的部署，不能继续用着旧的那一个。

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

多提供方模型通过 `routing` 块在提供方之间选择：

```jsonc
"models": {
  "gpt-4o": {
    "providers": [
      {"name": "openai-prod", "weight": 5},
      {"name": "openai-backup", "weight": 1}
    ],
    "routing": {"strategy": "weighted", "affinity": "off", "sink": true}
  }
}
```

| 策略 | 描述 |
|------|------|
| `weighted` | **（默认）** 平滑的 nginx 风格加权轮询。权重 [5, 1] 产生序列 A A A A A B，而非突发切换。 |
| `ordered` | 第一个可用的提供方服务每个请求；列表顺序就是策略，权重被忽略。 |
| `rotate` | 普通轮询 —— 每个提供方轮流服务，与其权重无关。 |
| `least_used` | 自路由建立以来服务次数最少的提供方。它是进程内的平局裁决，不是负载监控。 |
| `pace` | 只按剩余额度做加权轮询：配置权重被忽略，剩余额度最多的提供方先用。 |

所有策略都仍会乘以额度权重（`weight × (0.25 + 0.75 × headroom)`）；`pace` 是其中唯一把配置权重
从乘积中去掉的那个。

订阅通道加付费兜底用 `ordered`：订阅排在第一位，付费提供方按尝试顺序排在后面。每个请求由第一个能接的通道服务，因此订阅耗尽、被阻塞或被限流时自动交给下一条，兜底数量不限。路由页的列表显示这个顺序，拖动一行即可写入。

| 亲和 | 描述 |
|------|------|
| `off` | **（默认）** 每个请求独立选择。 |
| `session` | 同一客户端固定在一个提供方（按客户端 key 做加权 rendezvous），保全其提示缓存。只能与 `strategy: "weighted"` 搭配，因为身份取代了逐请求的选择。 |
| `turn`、`strict` | 尚未实现。加载时会按名称拒绝，而不是接受一个什么都不做的值。 |

| 字段 | 描述 |
|------|------|
| `strategy` | 上表中的一个值。 |
| `affinity` | `off` 或 `session`。 |
| `sink` | **（默认 `true`）** 最近一次 429 是否让该模型的提供方排到最后。提供方的*禁用*是进程级的；这里只决定该模型是否在意较轻的下沉。 |
| `first_token_ms` | 预留给首个 token 的截止时间。会被解析并保留；选择器尚未据此行动。 |
| `rules` | 按请求的规则，按顺序求值；第一条匹配的生效（见下文“规则”）。 |

#### 规则

规则在模型自己的路由内移动单个请求：

```jsonc
"routing": {
  "rules": [
    {"when": {"images": true}, "use": "openrouter"},
    {"when": {"tokens_gte": 120000}, "use": "argo"},
    {"when": {"effort_gte": "high"}, "use": "argo"},
    {"when": {"agents": ["codex"]}, "use": "openrouter"},
    {"when": {"time": {"tz": "Asia/Shanghai", "from": "09:00", "to": "18:00"}}, "pause": "argo"}
  ]
}
```

| 条件 | 成立条件 |
|:---|:---|
| `images` | 请求带有图片（为 `false` 时不带）。 |
| `tokens_gte` | 请求的估算输入规模不小于该 token 数。 |
| `effort_gte` | 请求要求的推理强度不低于该级别，梯级为 `minimal < low < medium < high < xhigh < max`。未指定强度的请求不匹配。 |
| `agents` | 名称出现在客户端的 `x-rosetta-agent` 标签或其 `User-Agent` 中（不区分大小写）。值是非空数组。 |
| `time` | `tz` 下的当前时间落在 `[from, to)` 内；跨夜的窗口（`from` 晚于 `to`）会越过午夜。 |

同一个 `when` 中的每个条件都必须成立；空的 `when` 总是匹配。

动作只能二选一：`use`（为本请求提升该提供方）或 `pause`（为本请求丢弃它）。第一条匹配的规则生效；
没有规则匹配的请求照常落到策略。规则只能点名模型已经列出的提供方 —— 它提升或丢弃成员，绝不引入
成员 —— 点名其他提供方会让配置加载失败。

规则在锚点选定之后运行，因此它与同格式分层是组合关系而不是替代关系：规则从路由本会选中的提供方
出发。`use` 规则点名的提供方如果本次无法服务，规则被忽略，锚点保持不变。

`pause` 规则同样是偏好而不是否决。它对匹配的请求丢弃该提供方，但如果这样会让路由一个候选也不剩，
规则之前选定的锚点保持不变，仍然服务该请求。因此在只列出 `p` 一个提供方的模型上写 `pause p` 不会
改变任何结果：暂停永远不会让提供方下线。要让某个提供方停止服务，请把它从模型的路由中移除，或把
该提供方关闭。

条件读取请求的 **IR**，绝不读取源请求体，因此同一条规则在客户端使用任何方言时含义相同。有两个
事实是近似值，值得知道：

- `tokens_gte` 比较的是序列化后 IR 的大小除以四，并排除图片负载。网关没有分词器；把这个数字当作
  数量级看待。
- `agents` 按子串匹配，因为各 agent 的写法并不统一。

没有规则的模型不会为这个特性付出任何代价：只有模型至少有一条规则时，才会为规则求值构建 IR。

未知键会让配置加载失败，管理面板保存会报同样的消息，因为两者都通过同一个解析器构建配置。

块之前的写法 —— 提供方列表旁的单个 `strategy` 字段 —— 仍会被读取：
`weighted_round_robin` 表示默认值，`affinity_round_robin` 表示 `affinity: "session"`。
在同一模型上同时写 `strategy` 和 `routing` 是错误，而不是谁优先的规则。

单提供方模型忽略此块。

### 同格式偏好

`server.prefer_same_format`（默认 `false`）让 provider 选择偏向已经在讲客户端 API 格式的上游，多格式的模型因此只在必要时才做转换。选择先确定一个**锚点**——即关闭该开关时路由会挑中的那个 provider，因此全量 route 的计数推进与关闭时完全一致——再在这个锚点范围内把格式对齐到客户端，分三层：

1. 与锚点**同 shim、同 endpoint**（通常就是锚点自身）。shim 已经固定了格式，所以这一层只决定同 endpoint 之间的故障转移顺序。
2. **同一上游账号**的其它格式——锚点所属账号也能提供客户端的格式，请求因此停留在同一账号，而不必换到另一个账号。
3. 客户端格式下的**任意 provider**。
4. 都不满足则回到锚点，经 IR 转换。

账号身份是推导出来的，不需要声明。两个 provider 携带同一份凭据即视为同一账号——同一个 `token_command`，或同一个 key 值（字面 key 与由环境变量解析出的同值 key 视为同一个 key）。比较的只是凭据的哈希，keyless provider 永不归组。`base_url` 被刻意排除在外，因此用同一个 key、从不同路径提供多种格式的服务方仍算作一个账号。

该开关只改变*由哪个*上游来服务请求。转换依然始终经过 IR——包括第 1 层的情况。参见[架构指南](../contributing/architecture.md)。

### 配额故障转移 { #quota-failover }

`failover_on_error`（按模型配置，默认 `true`）在服务方拒绝时，让请求改由同一模型的另一个 provider 来服务；设为 `false` 可为该模型关闭。`failover_on_quota` 是同一开关的旧拼写，仍然会被读取，已有配置不受影响；但两者同时写出不同的值，或任一键的值不是布尔，会让配置加载失败，而不是静默取舍：

```jsonc
"models": {
  "gpt-5": {
    "providers": ["openai-prod", "azure-backup"],
    "failover_on_error": false
  }
}
```

两类拒绝会触发故障转移：

- **配额。** 上游 `402`，或 provider 的 shim 归类为配额拒绝的 `429`（部分 shim 也会归类 `403`）。网关能降级凭据时会先降级；只有该 provider 已没有凭据能服务这次请求时才会转移。
- **瞬时故障。** `408`、`500`、`502`、`503`、`504`、`529`，以及 Cloudflare 源站错误 `520`-`524`。它们表示 provider 现在无法服务，而不是凭据有问题，因此不会降级任何凭据。

`400`、`403`、`404`、`413`、`422` 会把上游的答复原样返回：换一个 provider 也会拒绝同一个请求。

一次拒绝排除掉什么，取决于它说的是什么事：

- **配额类拒绝是凭据级的。** 同一提供方的所有车道共用一个凭据，因此整个 provider 在本次请求余下部分被排除，改试另一个 provider。
- **瞬时故障是车道级的。** 它说的是作答的那个上游模型服务不了，因此只有该车道离开路线，**同一提供方的其他车道先被尝试** —— 同一账号、同一 base URL，是代价最小的重试。只有在拒绝方已无车道可用时，才会去找另一个 provider。

因此瞬时的预算是一次**兄弟车道**重试加一次**跨 provider** 重试，最多三次上游尝试；配额类拒绝仍然只花一次。请求真的移动过时，profile 记录 `failover: {from, to, from_lane, to_lane, scope, reason}` —— 两跳会在第一跳下嵌套第二跳 —— 请求日志行指向实际服务的车道。

默认开启，因为它响应的拒绝本来就是客户端会收到的错误；只有一个提供方的模型也没有第二次尝试可言。关闭它才是真正的取舍：换一条车道或一个 provider 就是换一份 prompt 缓存，甚至可能是另一个模型构建。

容量类拒绝本身也是路由信号，而它喂的那本账属于车道。一次 `503`、`529` 或 `520`-`524` 会把作答的那条车道排到同 provider 其它车道之后（下沉），但它仍是候选 —— 单车道 provider 没有可排在后面的对象，照常服务。五分钟内同一条车道出现三次，就把它禁用一分钟，重复触发时翻倍。只有**该车道自己**的成功会清零它的计数：兄弟车道服务成功只说明这个 provider 能服务，不说明那条拒绝过的上游模型已恢复。没有任何探测能解释的 `402`/`429` 仍是凭据级事实，按 provider 计数，任意一次成功即可清零。`500`、`502`、`504` 或 `408` 会为本次请求故障转移，但单凭它不会进入阶梯。达到阈值的那几个请求照样被服务 —— 由兄弟车道或故障转移服务，而不是那条失败的车道。当阶梯让该模型一条候选都不剩（单车道、且已被禁用）时，客户端会被告知真实原因：`503` 带 `rate limited until <time>`，并在 `Retry-After` 中给出该结束时间，而不是「initializing」。

流式请求同样会故障转移，但只在第一个事件之前：处理程序在写出任何内容之前就打开上游连接，因此这一刻的拒绝仍可由别处作答。字节一旦发给客户端，响应就无法重放，流中途的失败会以对应代码结束客户端的流。

### 提供方条目级能力 { #provider-capabilities }

`providers[]` 的每个条目可以声明该提供方**在这个模型上**提供什么，从而按请求缩小路由范围：

```jsonc
"models": {
  "gemini-3.5-flash": {
    "providers": [
      {"name": "gemini-text", "capabilities": ["text"]},
      {"name": "gemini-vision", "capabilities": ["text", "vision"]}
    ]
  }
}
```

带图片的请求需要 `vision`，声明了工具的请求需要 `tools`，指定推理强度的请求需要 `reasoning`。条目声明的列表缺少所需能力时，它**对该请求**不是候选，对其他请求仍是候选。未声明任何能力的条目继承提供方自身的能力，因此配置里没有该字段时行为完全不变。所有候选都缺少所需能力时，路由返回 `400` 并写明缺少的能力 —— 重试无法补上。模型抽屉的每个提供方行都有同一字段，留空表示继承（[管理面板](admin-panel.md)）。

### 同一提供方的多个上游模型 { #provider-lanes }

一个模型可以多次列出同一个提供方，每个条目写不同的 `upstream_model`：

```jsonc
"models": {
  "deepseek-v4.1-flash": {
    "providers": [
      {"name": "opendesign-go", "upstream_model": "deepseek-v4.1-flash", "weight": 3},
      {"name": "opendesign-go", "upstream_model": "deepseek-v4.1-flash-eco", "weight": 1}
    ]
  }
}
```

每个条目都是独立的候选：策略分别计数，资格判定与能力过滤分别判断，服务记录里写明真正服务的 `upstream_model`。同一提供方且同一 `upstream_model` 写两遍等于同一个候选写了两遍，配置会拒绝加载。模型编辑器追加车道的方式与追加第二个提供方相同：写上提供方与新的 `upstream_model` 并确认合并，条目就会多出一条车道，原有的车道保持不变。抽屉按车道逐行列出，开关、权重、能力集合与删除都只作用于所在的那条车道；**+ lane** 以未选提供方的状态打开编辑器，新车道确认后才会加入条目。写在 `providers[]` 旁的条目级 `upstream_model` 仍是未声明上游车道的兜底值；抽屉以“兜底上游”只读显示，只能通过它自己的删除动作移除。配额耗尽与 `429` 仍以提供方为单位：同一提供方的所有条目共用凭据，因此其中一次拒绝会同时移除它们。瞬时上游故障则以车道为单位 —— 它说的是那个上游模型服务不了，其他车道仍然可试，而且先于另一个 provider 被尝试（见[配额故障转移](#quota-failover)）。

### 非 LLM 路由 { #typed-routes }

在统一 `models` 段中声明的嵌入、重排与决策模型，与聊天路由走同样的两个阶段 —— 先资格判定，再按声明权重选择锚点。因此一个有多提供方的非 LLM 模型会按权重分配流量，并跳过状态判定已移除的提供方（配额耗尽、被阻塞，或连续 429 导致停用）：

```jsonc
"models": {
  "jina-embeddings-v3": {
    "type": "embeddings",
    "providers": [{"name": "jina-embed", "weight": 3}, {"name": "voyage-embed"}],
    "failover_on_quota": true
  }
}
```

没有声明该类型端点的提供方不进入候选。所有候选都无法服务时，该路由返回 `503`（启动失败后为 `502`），已知恢复时间时带 `Retry-After`，而不是 `404`。开启 `failover_on_error`（或旧拼写 `failover_on_quota`）后，上游 `429`、`402` 或瞬时 `5xx` 会把请求移到下一个提供方，并计入聊天路由使用的同一条连续 429 规则。旧版 `<type>_models` 段每个模型只写一个提供方，仍是单提供方行为。

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

### `update_check` 与 `self_update`

管理面板可以提示有新版本，也可以直接安装它。两者是 `server` 下的两个独立开关，默认都是 `true`：

| 键 | 作用 |
|-----|--------|
| `update_check` | 网关向包索引查询最新的 `llm-rosetta` / `llm-rosetta-gateway` 版本，结果缓存六小时；有更新的版本时面板右上角出现气泡。设为 `false` 则完全不发外部请求。 |
| `self_update` | 面板里的 **立即更新** 会在当前环境执行 `pip install --upgrade`，然后用同一命令行替换进程。设为 `false` 时气泡和更新日志仍在，只禁用按钮。 |

```jsonc
{
  "server": {
    "update_check": true,
    "self_update": false
  }
}
```

只有字面量 `true` 才会打开这两个开关：手写文件里带引号的 `"true"` 视为关闭，这样一次外部查询 —— 或一次会替换正在运行代码的 pip 安装 —— 不会因为笔误而打开。

!!! warning "可编辑安装与容器"
    从源码目录（可编辑安装）运行的网关会拒绝自更新：pip 会把工作树里的包换成索引上的发行版，之后的修改将不再生效。在容器里，安装结果会保留到容器被重建为止；重建新镜像才是长期做法。

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

## 可信代理 { #trusted-proxies }

`server.trusted_proxies` 以 CIDR 列出网关前面的反向代理（单个地址等同于一个主机）：

```jsonc
{
  "server": {
    "trusted_proxies": ["10.0.0.0/8", "172.16.0.0/12", "127.0.0.1/32"]
  }
}
```

每个请求的客户端地址只解析一次；按 IP 限流、admin 登录锁定、请求日志和被拒请求日志都用它：

- 什么都不列（默认）时，客户端就是 TCP 对端，所有转发头都被忽略。
- 对端不在列表里，不论它发什么头，对端就是客户端。
- 对端在列表里时，才读 `X-Forwarded-For`。链**从右往左**读，跳过列表里的地址，第一个其他地址就是客户端；它左边的项是原始调用方自己写的，不使用。
- 链为空、全部受信，或含有不是 IP 地址的项时，不能指明客户端，使用对端。
- `X-Real-IP` 只在受信对端没有可用 `X-Forwarded-For` 时读取，且不会覆盖由链得出的结论。
- `X-Forwarded-Proto: https`（决定 admin 会话 cookie 的 `Secure` 标志）只在对端受信时算数。

条目非法时启动失败并指出是哪一项，例如 `server.trusted_proxies[1]`。配置重载时会重新应用这个列表。

!!! warning "在反向代理后面"
    不列代理时，代理后面的所有客户端都是代理的地址：它们共用一个按 IP 限流和一个 admin 登录计数，输错五次密码会把所有人锁住。请列出代理，地址取网关实际看到的 TCP 对端地址。

!!! note "迁移"
    `rate_limit.trust_proxy` 已删除。配置里还有它时网关拒绝启动；请改用 `server.trusted_proxies`。

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
