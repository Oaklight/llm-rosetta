---
title: Admin API
---

# Admin API 参考

网关在 `/admin/api/` 路径下提供内置 REST API，为
[管理面板](../gateway/admin-panel.md)提供数据支持，也可供脚本或外部工具直接调用。
除特别说明外，所有端点均返回 JSON。

!!! tip "基础路径"
    以下所有端点均相对于网关监听地址，例如
    `http://localhost:8765`。

---

## 认证

### 检查认证要求

检查管理面板是否需要密码。

```
GET /admin/api/auth-check
```

**响应**

```json
{"requires_auth": true}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `requires_auth` | `bool` | 当配置中设置了 `server.admin_password` 时为 `true`。 |

---

### 登录

验证管理员密码并获取会话令牌。每个 IP 最多允许 5 次失败尝试，超过后锁定 5 分钟。

```
POST /admin/api/login
```

**请求体**

```json
{"password": "my-admin-password"}
```

**成功响应**

```json
{"ok": true, "token": "<session-token>"}
```

**错误响应**

| 状态码 | 响应体 | 触发条件 |
|--------|--------|---------|
| 400 | `{"error": "Admin password not configured"}` | 配置中未设置 `server.admin_password` |
| 400 | `{"error": "Invalid JSON body"}` | 请求体格式错误 |
| 401 | `{"error": "Invalid password"}` | 密码错误 |
| 429 | `{"error": "Too many failed attempts. Try again in Ns."}` | 触发速率限制 |

---

## 配置管理

### 获取配置

返回完整的网关配置，API Key 已脱敏。

```
GET /admin/api/config
```

**响应**

```json
{
  "config_path": "/home/user/.config/llm-rosetta-gateway/config.jsonc",
  "providers": {
    "openai": {
      "api_key": "sk-a***abcd",
      "base_url": "https://api.openai.com/v1",
      "type": "openai_chat"
    }
  },
  "models": {
    "gpt-4o": {
      "provider": "openai",
      "capabilities": ["text", "vision", "tools"]
    }
  },
  "server": {"proxy": "http://proxy:8080"},
  "credential_visible": false,
  "known_provider_types": ["openai_chat", "openai_responses", "anthropic", "google"],
  "registered_shims": [
    {
      "name": "openai",
      "base": "openai_chat",
      "logo": "https://...",
      "default_base_url": "https://api.openai.com/v1",
      "default_api_key_env": "OPENAI_API_KEY"
    }
  ]
}
```

!!! note "Key 脱敏"
    明文 API Key 会被脱敏显示（`sk-a***abcd`）。环境变量占位符（如
    `${OPENAI_API_KEY}`）保持原样返回。

---

### 添加 / 更新提供商

创建新提供商或更新已有提供商。支持重命名（自动更新关联的模型引用）。

```
PUT /admin/api/config/providers/<name>
```

**请求体**

| 字段 | 类型 | 是否必填 | 说明 |
|------|------|---------|------|
| `api_key` | `string` | 是* | 提供商 API Key。编辑时可省略或留空以保留原有 Key。 |
| `base_url` | `string` | 是 | 提供商 Base URL。 |
| `type` | `string` | 否 | API 标准或 shim 名称（如 `openai_chat`、`anthropic`）。 |
| `proxy` | `string` | 否 | 提供商级别的代理 URL。 |
| `rename_from` | `string` | 否 | 重命名时的原名称——旧条目将被删除，模型引用自动更新。 |

**成功响应**

```json
{
  "ok": true,
  "provider": "openai",
  "providers": ["openai", "anthropic"]
}
```

**错误**：400（缺少字段、JSON 格式错误）、404（重命名源不存在）、409（重命名目标已存在）、500（磁盘 / 重载失败）。

---

### 删除提供商

```
DELETE /admin/api/config/providers/<name>?cascade=true
```

| 查询参数 | 类型 | 默认值 | 说明 |
|---------|------|-------|------|
| `cascade` | `"true"` \| `"1"` | — | 同时删除引用该提供商的模型。 |

**成功响应**

```json
{
  "ok": true,
  "deleted": "openai",
  "providers": ["anthropic"],
  "cascade_deleted_models": ["gpt-4o"]
}
```

**错误**：404（提供商不存在）、409（仍有模型引用且未设置 `cascade`）。

---

### 切换提供商状态

启用或禁用提供商，无需删除。

```
POST /admin/api/config/providers/<name>/toggle
```

**响应**

```json
{"ok": true, "provider": "openai", "enabled": false}
```

---

### 添加 / 更新模型

```
PUT /admin/api/config/models/<name>
```

**请求体**

| 字段 | 类型 | 是否必填 | 说明 |
|------|------|---------|------|
| `provider` | `string` | 是 | 目标提供商名称（必须已存在）。 |
| `capabilities` | `string[]` | 否 | 能力列表。默认为 `["text"]`。 |
| `upstream_model` | `string` | 否 | 上游模型 ID（当与网关侧名称不同时使用）。 |
| `rename_from` | `string` | 否 | 重命名时的原名称。 |

**成功响应**

```json
{
  "ok": true,
  "model": "gpt-4o",
  "provider": "openai",
  "capabilities": ["text", "vision", "tools"],
  "models": {"gpt-4o": "openai"}
}
```

---

### 删除模型

```
DELETE /admin/api/config/models/<name>
```

**响应**

```json
{"ok": true, "deleted": "gpt-4o", "models": {}}
```

---

### 更新服务器设置

目前支持全局代理设置。

```
PUT /admin/api/config/server
```

**请求体**

```json
{"proxy": "http://proxy:8080"}
```

发送空字符串（`""`）可移除代理设置。

**响应**

```json
{"ok": true, "server": {"proxy": "http://proxy:8080"}}
```

---

### 重载配置

从磁盘强制热重载配置，无需重启网关。

```
POST /admin/api/config/reload
```

**响应**

```json
{
  "ok": true,
  "providers": ["openai", "anthropic"],
  "models": {"gpt-4o": "openai"}
}
```

---

### 获取上游模型列表

查询提供商的 `/v1/models` 端点以发现可用模型。

```
GET /admin/api/config/providers/<name>/models
```

**响应**

```json
{
  "provider": "openai",
  "api_standard": "openai_chat",
  "models": ["gpt-4o", "gpt-4o-mini", "o3-mini"]
}
```

**错误**：404（提供商不存在）、502（上游连接或 HTTP 错误）。

---

### 批量添加模型

一次为指定提供商添加多个模型。

```
POST /admin/api/config/models
```

**请求体**

| 字段 | 类型 | 是否必填 | 说明 |
|------|------|---------|------|
| `provider` | `string` | 是 | 目标提供商名称。 |
| `models` | `string[]` | 是 | 要添加的模型 ID 列表。 |
| `prefix` | `string` | 否 | 添加到每个模型名称前的前缀。使用前缀时，原始 ID 存储为 `upstream_model`。 |
| `capabilities` | `string[]` | 否 | 所有添加模型的能力列表。默认为 `["text", "vision", "tools"]`。 |

**响应**

```json
{
  "ok": true,
  "added": ["my/gpt-4o", "my/gpt-4o-mini"],
  "skipped": ["my/o3-mini"],
  "models": {"my/gpt-4o": "openai", "my/gpt-4o-mini": "openai"}
}
```

---

### 查看提供商 API Key

返回提供商未脱敏的 API Key。仅在配置中 `server.credential_visible` 为 `true`
时可用。

```
GET /admin/api/config/providers/<name>/key
```

**响应**

```json
{"api_key": "sk-abc123..."}
```

**错误**：403（凭证可见性已禁用）、404（提供商不存在）。

---

## API Key 管理

### 列出所有 Key

返回所有网关 API Key（值已脱敏）。

```
GET /admin/api/keys
```

**响应**

```json
{
  "keys": [
    {
      "id": "a1b2c3d4",
      "key": "rsk-***abcd",
      "label": "production",
      "created": "2025-01-15T10:30:00+00:00"
    }
  ]
}
```

!!! note "旧版单 Key 兼容"
    如果配置使用旧版 `server.api_key` 字段而非 `server.api_keys`，
    该 Key 会作为 `id: "default"` 的合成条目返回。

---

### 创建 Key

生成新的网关 API Key（或注册手动指定的 Key）。

```
POST /admin/api/keys
```

**请求体**

| 字段 | 类型 | 是否必填 | 说明 |
|------|------|---------|------|
| `label` | `string` | 否 | 可读标签。 |
| `key` | `string` | 否 | 手动指定的 Key 值。省略时自动生成 `rsk-<hex>` 格式的随机 Key。 |

**响应**

```json
{
  "ok": true,
  "key": {
    "id": "e5f6a7b8",
    "key": "rsk-0123456789abcdef0123456789abcdef",
    "label": "staging",
    "created": "2025-01-15T12:00:00+00:00"
  }
}
```

!!! warning
    完整 Key 值**仅在此响应中返回一次**。后续 `GET /admin/api/keys`
    调用返回的是脱敏值。

---

### 更新 Key 标签

```
PUT /admin/api/keys/<key_id>
```

**请求体**

```json
{"label": "new-label"}
```

**响应**

```json
{"ok": true, "id": "a1b2c3d4", "label": "new-label"}
```

---

### 删除 Key

```
DELETE /admin/api/keys/<key_id>
```

**响应**

```json
{"ok": true, "deleted": "a1b2c3d4"}
```

---

### 查看 Key 明文

返回指定 API Key 的未脱敏值。需要 `server.credential_visible: true`。

```
GET /admin/api/keys/<key_id>/reveal
```

**响应**

```json
{"key": "rsk-0123456789abcdef0123456789abcdef"}
```

**错误**：403（凭证可见性已禁用）、404（Key 不存在）。

---

### 获取内部令牌

返回管理面板用于测试请求的临时内部令牌。

```
GET /admin/api/internal-token
```

**响应**

```json
{"token": "internal-abc123"}
```

---

## 可观测性

### 指标快照

返回实时网关指标，包括计数器、分类统计和滚动时间序列。

```
GET /admin/api/metrics?seconds=60
```

| 查询参数 | 类型 | 默认值 | 范围 | 说明 |
|---------|------|-------|------|------|
| `seconds` | `int` | `60` | 1–300 | 时间序列窗口长度。 |

**响应**

```json
{
  "uptime_seconds": 3600.5,
  "total_requests": 1234,
  "total_errors": 12,
  "total_streams": 456,
  "error_rate": 0.0097,
  "active_streams": 3,
  "by_model": {"gpt-4o": 800, "claude-sonnet-4-20250514": 434},
  "by_source_provider": {"openai_chat": 1000, "anthropic": 234},
  "by_target_provider": {"openai": 800, "anthropic": 434},
  "by_status_code": {"200": 1222, "500": 12},
  "series": [
    {"t": -59, "count": 2, "avg_ms": 450.12, "errors": 0},
    {"t": -58, "count": 0, "avg_ms": 0, "errors": 0}
  ]
}
```

| 时间序列字段 | 说明 |
|-------------|------|
| `t` | 负偏移秒数（如 `-59` 表示 59 秒前）。 |
| `count` | 该秒内的请求数。 |
| `avg_ms` | 平均延迟（毫秒）。 |
| `errors` | 错误响应数（状态码 >= 400）。 |

---

### 请求日志

返回分页、可过滤的请求日志条目（最新优先）。

```
GET /admin/api/requests?limit=50&offset=0&model=gpt-4o&provider=openai&status=ok
```

| 查询参数 | 类型 | 默认值 | 说明 |
|---------|------|-------|------|
| `limit` | `int` | `50` | 每页条数。 |
| `offset` | `int` | `0` | 跳过的条目数。 |
| `model` | `string` | — | 按模型名称过滤。 |
| `provider` | `string` | — | 按目标提供商过滤。 |
| `status` | `"ok"` \| `"error"` | — | `ok` = 2xx/3xx，`error` = 4xx/5xx。 |

**响应**

```json
{
  "entries": [
    {
      "id": "a1b2c3d4e5f6...",
      "timestamp": "2025-01-15T12:00:00+00:00",
      "model": "gpt-4o",
      "source_provider": "openai_chat",
      "target_provider": "openai",
      "is_stream": false,
      "status_code": 200,
      "duration_ms": 1234.56,
      "api_key_label": "production"
    }
  ],
  "total": 100
}
```

---

### 清除请求日志

```
DELETE /admin/api/requests
```

**响应**

```json
{"ok": true}
```

---

### 网络诊断

通过网关配置的代理（如有）执行 IP 地理定位和 Google 连通性检查。

```
GET /admin/api/diagnostics/network
```

**响应**

```json
{
  "proxy": "http://proxy:8080",
  "ip": {
    "ok": true,
    "ip": "203.0.113.42",
    "country": "United States",
    "city": "San Francisco",
    "isp": "Example ISP"
  },
  "google": {
    "ok": true,
    "status": 204
  }
}
```

`proxy` 字段仅在配置了全局代理时出现。
每个子对象在失败时返回 `"ok": false` 和 `"error"` 消息。

---

### 重建指标

从持久化的请求日志重建所有指标计数器。适用于修复计数器 bug 后或内存中的计数器与
实际日志数据不一致的情况。

```
POST /admin/api/metrics/rebuild
```

**响应**

```json
{
  "ok": true,
  "rebuilt_from": 1234,
  "counters": {
    "total_requests": 1234,
    "total_errors": 12,
    "total_streams": 456
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `rebuilt_from` | `int` | 用于重建计数器的日志条目数。 |
| `counters` | `object` | 重建后的计数器值（已立即持久化）。 |

**错误**：400（未配置持久化——网关以纯内存模式运行）。

---

## 性能分析

基于 [pyinstrument](https://github.com/joerick/pyinstrument) 的按需深度性能分析。
启用后，网关会对接下来的 *N* 个请求进行性能分析，并保存交互式火焰图结果。

!!! note "可选依赖"
    性能分析需要 `pyinstrument`。安装方式：
    `pip install llm-rosetta[profiling]`

### 获取状态

返回当前的性能分析状态。

```
GET /admin/api/profiling/status
```

**响应**

```json
{
  "enabled": false,
  "remaining": 0,
  "results_count": 3,
  "max_results": 20
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `enabled` | `bool` | 性能分析是否已启用。 |
| `remaining` | `int` | 剩余待分析的请求数。 |
| `results_count` | `int` | 已存储的分析结果数量。 |
| `max_results` | `int` | 保留的最大结果数（超出时裁剪最旧的）。 |

---

### 启用性能分析

为接下来的 *N* 个请求启用性能分析。

```
POST /admin/api/profiling/enable
```

**请求体**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `requests` | `int` | `5` | 要分析的请求数。取值范围 **1–100**。 |

**响应**

```json
{
  "enabled": true,
  "remaining": 10,
  "results_count": 0,
  "max_results": 20
}
```

**错误**：400（未安装 `pyinstrument`）。

---

### 禁用性能分析

在剩余计数归零之前手动禁用性能分析。

```
POST /admin/api/profiling/disable
```

**响应**

```json
{
  "enabled": false,
  "remaining": 0,
  "results_count": 3,
  "max_results": 20
}
```

---

### 列出分析结果

返回所有已存储分析结果的摘要（不含完整的 HTML/text 内容以保持响应紧凑）。

```
GET /admin/api/profiling/results
```

**响应**

```json
{
  "results": [
    {
      "timestamp": "2025-01-15T12:00:00+00:00",
      "request_id": "a1b2c3d4",
      "model": "gpt-4o",
      "source": "openai_chat",
      "target": "openai",
      "is_stream": false,
      "duration_ms": 1234.56
    }
  ],
  "total": 1
}
```

---

### 获取单个结果

按索引（从 0 开始）返回单个分析结果。

```
GET /admin/api/profiling/results/<index>
```

| 查询参数 | 类型 | 默认值 | 说明 |
|---------|------|-------|------|
| `format` | `"html"` \| `"json"` | `"json"` | `html` 直接返回交互式火焰图，Content-Type 为 `text/html`。 |

**响应（JSON，默认）**

返回完整结果对象，包含 `html` 和 `text` 字段。

**响应（HTML）**

直接返回 pyinstrument 火焰图 HTML 页面（Content-Type 为 `text/html`），可在浏览器中查看。

**错误**：400（无效索引）、404（索引超出范围）。

---

### 下载全部结果

将所有分析结果下载为 ZIP 压缩包。每个结果保存为 HTML 火焰图文件，
命名格式为 `profile-<index>-<model>-<timestamp>.html`。

```
GET /admin/api/profiling/results/download
```

返回 `application/zip` 响应，带 `Content-Disposition: attachment` 头。

**错误**：404（无结果可下载）。

---

### 清除分析结果

删除所有已存储的分析结果。

```
DELETE /admin/api/profiling/results
```

**响应**

```json
{"ok": true}
```

---

## 错误转储

当网关在请求处理过程中遇到错误时，可以持久化详细的诊断转储，包括原始请求体、
转换后的请求体、错误详情和计时信息。错误转储功能需要配置持久化。

### 列出错误转储

返回分页、可过滤的错误转储条目（最新优先）。

```
GET /admin/api/error-dumps?limit=50&offset=0
```

| 查询参数 | 类型 | 默认值 | 说明 |
|---------|------|-------|------|
| `limit` | `int` | `50` | 每页条数。 |
| `offset` | `int` | `0` | 跳过的条目数。 |
| `model` | `string` | — | 按模型名称过滤。 |
| `error_phase` | `string` | — | 按错误发生阶段过滤。 |
| `provider` | `string` | — | 按目标提供商过滤。 |

**响应**

```json
{
  "entries": [
    {
      "id": "d1e2f3a4",
      "timestamp": "2025-01-15T12:00:00+00:00",
      "model": "gpt-4o",
      "provider": "openai",
      "error_phase": "upstream",
      "error_message": "Connection refused",
      "status_code": 502,
      "body_hash": "abc123..."
    }
  ],
  "total": 42
}
```

**错误**：400（未配置持久化）。

---

### 获取错误转储详情

返回单个错误转储，包含解压后的请求体和转换后的请求体。

```
GET /admin/api/error-dumps/<dump_id>
```

**响应**

```json
{
  "id": "d1e2f3a4",
  "timestamp": "2025-01-15T12:00:00+00:00",
  "model": "gpt-4o",
  "provider": "openai",
  "error_phase": "upstream",
  "error_message": "Connection refused",
  "status_code": 502,
  "body_hash": "abc123...",
  "request_body": {"model": "gpt-4o", "messages": [...]},
  "converted_body_hash": "def456...",
  "converted_body": {"model": "gpt-4o", "messages": [...]}
}
```

`request_body` 和 `converted_body` 字段为解压后的 JSON 对象。
如果解压失败或未存储请求体，这些字段为 `null`。

**错误**：400（未配置持久化）、404（转储不存在）。

---

### 下载请求体

下载错误转储的原始解压请求体 JSON 文件。

```
GET /admin/api/error-dumps/<dump_id>/body
```

返回 `application/json` 响应，带
`Content-Disposition: attachment; filename="error-dump-<dump_id>.json"` 头。

**错误**：400（未配置持久化）、404（转储不存在或未存储请求体）。

---

### 清除错误转储

删除所有已存储的错误转储。

```
DELETE /admin/api/error-dumps
```

**响应**

```json
{"ok": true}
```

**错误**：400（未配置持久化）。

---

## 模型测试

Admin API 以异步方式执行模型测试。`POST` 启动任务后，通过 `GET` 轮询结果。

### 启动测试

```
POST /admin/api/test
```

**请求体**

| 字段 | 类型 | 是否必填 | 说明 |
|------|------|---------|------|
| `endpoint` | `string` | 是 | 要调用的网关端点，如 `/v1/chat/completions`。 |
| `payload` | `object` | 是 | 转发到端点的请求体。 |

**响应**

```json
{"task_id": "a1b2c3d4e5f6"}
```

测试请求通过网关自身路由，使用临时内部令牌，因此会经过完整的代理管道。

---

### 轮询测试结果

```
GET /admin/api/test/<task_id>
```

**响应（进行中）**

```json
{"status": "pending"}
```

**响应（已完成）**

```json
{
  "status": "done",
  "status_code": 200,
  "body": {"id": "chatcmpl-...", "choices": [...]}
}
```

**响应（出错）**

```json
{"status": "error", "error": "Connection refused"}
```

| 状态值 | 含义 |
|--------|------|
| `pending` | 任务仍在执行中。 |
| `done` | 上游已响应；检查 `status_code` 和 `body`。 |
| `error` | 请求过程中发生异常。 |
| `cancelled` | 任务已通过 `DELETE` 取消。 |

任务在 5 分钟后自动清理。

---

### 取消测试

```
DELETE /admin/api/test/<task_id>
```

**响应**

```json
{"ok": true}
```

---

## 通用错误格式

所有端点使用一致的错误信封：

```json
{"error": "可读的错误信息"}
```

写入磁盘的端点在配置保存成功但热重载失败时，可能返回部分成功响应：

```json
{
  "error": "Config saved but reload failed: ...",
  "saved": true,
  "reloaded": false
}
```
