---
title: 配置
---

# 配置

本页详细介绍网关的配置文件格式。

## 提供商

每个提供商条目需要 `api_key`、`base_url`，以及可选的 `type` 字段指定 API 标准：

```jsonc
"providers": {
  "my-openai":   { "type": "openai_chat",      "api_key": "sk-...",     "base_url": "https://api.openai.com/v1" },
  "my-anthropic": { "type": "anthropic",        "api_key": "sk-ant-...", "base_url": "https://api.anthropic.com" },
  "my-google":   { "type": "google",            "api_key": "AIza...",    "base_url": "https://generativelanguage.googleapis.com" }
}
```

提供商名称是用户自定义的字符串（如 `"my-openai"`、`"prod-claude"`）。`type` 字段指定使用哪种 API 标准。

可用类型：`openai_chat`、`openai_responses`、`anthropic`、`google`。

### 使用 Shim

除了 `type`，还可以使用 `shim` 字段引用已注册的提供商 shim。Shim 是一个轻量级身份卡片，声明了提供商使用的基础 API 标准、连接默认值和字段级转换规则。

```jsonc
"providers": {
  "my-deepseek":   { "shim": "deepseek",   "api_key": "${DEEPSEEK_API_KEY}" },
  "my-volcengine": { "shim": "volcengine",  "api_key": "${VOLCENGINE_API_KEY}", "base_url": "https://ark.cn-beijing.volces.com/api/v3" }
}
```

指定 `shim` 时：

- **基础类型**会自动解析（如 `deepseek` → `openai_chat`）
- 如果配置中未设置，会从 shim 填充默认的 **`base_url`** 和 **`api_key` 环境变量**
- 请求/响应转换时会应用**字段级转换**（如 Volcengine 的 shim 会剥离其 API 不支持的 `logprobs` 和 `top_logprobs` 字段）

内置 shim：`openai`、`openai_responses`、`anthropic`、`google`、`deepseek`、`volcengine`。

也可以通过 `register_shim()` 以编程方式注册自定义 shim。

### 图片数量限制

Shim 支持 `max_images` 和 `max_images_pattern` 字段，用于对每次请求中的图片数量设置上限：

| 字段 | 类型 | 说明 |
|------|------|------|
| `max_images` | int | 每次请求允许的最大图片数量 |
| `max_images_pattern` | str | 应用于模型名称的正则表达式；仅对匹配模型执行限制 |

当请求超出上限时，最旧的图片将被替换为文本占位符。如果设置了 `max_images_pattern`，仅匹配的模型名称受限——其他模型不受影响，直接透传。

**示例 — 内置 Argo OpenAI shim：**

内置的 Argo OpenAI shim 声明了 `max_images: 50`，`max_images_pattern: "^(gpt|o\d)"`，效果如下：

- GPT 系列和 o 系列模型：图片截断至 50 张
- 通过同一服务方路由的 Gemini 和 Claude 模型：不受限制，直接透传

可通过 `register_shim()` 注册自定义 shim 实现同样的限制逻辑。

!!! tip "解析优先级"
    提供商类型解析顺序为：`shim` → `type` → 提供商名称（兜底）。

!!! note "向后兼容"
    如果 `shim` 和 `type` 都省略，提供商名称本身将用作类型。这意味着使用旧格式（提供商名称为 `openai_chat`、`anthropic` 等）的配置无需修改即可继续使用。

### 启用 / 禁用提供商

每个提供商支持 `enabled` 字段（默认 `true`）。禁用的提供商及其关联模型将从路由中静默排除：

```jsonc
"my-openai": { "type": "openai_chat", "api_key": "sk-...", "base_url": "https://api.openai.com/v1", "enabled": false }
```

这在需要临时下线提供商但不删除配置时很有用。[管理面板](admin-panel.md)提供了切换开关来操作此功能。

### API 密钥轮转

每个提供商支持通过逗号分隔配置多个 API 密钥，网关以轮询方式依次使用：

```jsonc
"my-openai": { "type": "openai_chat", "api_key": "sk-key1,sk-key2,sk-key3", "base_url": "https://api.openai.com/v1" }
```

### 环境变量替换

API 密钥支持 `${ENV_VAR}` 语法 — 启动时从环境变量读取：

```jsonc
"my-openai": { "type": "openai_chat", "api_key": "${OPENAI_API_KEY}", "base_url": "https://api.openai.com/v1" }
```

### 逐提供商代理

可为单个提供商指定代理：

```jsonc
"my-anthropic": { "type": "anthropic", "api_key": "sk-ant-...", "base_url": "https://api.anthropic.com", "proxy": "http://proxy:8080" }
```

## 代理配置

可在 `server` 部分设置全局代理，适用于所有提供商（除非逐提供商覆盖）：

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

`models` 部分将模型名称映射到提供商：

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

## 网关 API Key

通过网关级 API Key 保护 AI 请求端点：

```jsonc
{
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "api_key": "my-secret-gateway-key"
  }
}
```

配置后，所有 `/v1/*` 端点需要使用对应 API 标准的原生格式进行认证：

| API 标准 | 凭证格式 |
|---------|---------|
| OpenAI Chat / Responses | `Authorization: Bearer <key>` |
| Anthropic | `x-api-key: <key>` |
| Google GenAI | `x-goog-api-key: <key>` 或 `?key=<key>` 查询参数 |

API Key 也支持 `${ENV_VAR}` 替换：

```jsonc
"api_key": "${GATEWAY_API_KEY}"
```

!!! note "管理面板"
    管理面板（`/admin/*`）**不需要**网关 API Key。可以使用内置的 `admin_password` 选项保护（见下文），也可以通过反向代理实现（如 Caddy 的 `basicauth`、Nginx 的 `auth_basic`）。

未配置 `api_key` 时，所有请求无需认证直接通过（向后兼容）。

## 管理面板安全

### `admin_password`

可选。设置后，访问管理面板（`/admin/*`）前需要密码登录。会话通过 HMAC token 追踪，无需外部 session 存储。

支持 `${ENV_VAR}` 替换：

```jsonc
{
  "server": {
    "admin_password": "${ADMIN_PASSWORD}"
  }
}
```

!!! tip
    如果网关对公网开放，强烈建议设置 `admin_password`，防止未授权访问提供商配置和请求日志。

!!! warning "未解析的占位符"
    如果 `admin_password` 包含未解析的 `${ENV_VAR}` 占位符（即环境变量未在启动时设置），网关会**拒绝启动**并输出清晰的错误信息，防止将字面量字符串 `${ADMIN_PASSWORD}` 作为密码使用。

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
| `/health/ready` | 200 / 503 | Kubernetes 就绪探针——任意服务方降级时返回 503 |

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

## 完整示例

```jsonc
{
  "providers": {
    "openai-prod":    { "type": "openai_chat",      "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "openai-resp":    { "type": "openai_responses",  "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "anthropic-prod": { "type": "anthropic",         "api_key": "${ANTHROPIC_API_KEY}",  "base_url": "https://api.anthropic.com" },
    "google-prod":    { "type": "google",            "api_key": "${GOOGLE_API_KEY}",     "base_url": "https://generativelanguage.googleapis.com" },
    // 基于 Shim 的提供商 — base_url 和 transforms 自动解析
    "deepseek":       { "shim": "deepseek",          "api_key": "${DEEPSEEK_API_KEY}" },
    "volcengine":     { "shim": "volcengine",         "api_key": "${VOLCENGINE_API_KEY}", "base_url": "https://ark.cn-beijing.volces.com/api/v3" }
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
