---
title: 管理面板
---

# 管理面板（Admin Panel）

网关内置了基于 Web 的管理面板，用于管理配置、监控流量和查看请求日志——无需编辑配置文件或重启服务器。

访问地址：**`http://localhost:8765/admin/`**

## 配置管理

**Configuration** 标签页可通过可视化界面管理提供商、模型和服务器设置。

![配置管理标签页](images/admin-configuration.png)

### 服务器设置

设置全局代理 URL，适用于所有提供商（除非在提供商级别单独覆盖）。

### 提供商管理

每个提供商卡片显示名称、API 标准类型、Base URL、脱敏后的 API Key、可选代理，以及**提供商 Logo**（如果在提供商 shim 中声明）。当提供商配置的 shim 包含 `to_transforms` 或 `from_transforms` 时，活跃的转换规则会显示在卡片上。支持以下操作：

- **添加**新提供商（"+ Add Provider" 按钮）
- **编辑**现有提供商（名称、类型/shim、Base URL、API Key、代理）
- **重命名**提供商——所有关联模型引用自动更新
- **删除**提供商

编辑时，API Key 显示为密码字段，支持可见性切换和复制按钮。卡片上显示的脱敏 Key 不会被写回配置文件。

### 模型路由

提供商区域下方是模型路由表，列出所有已配置的模型及其目标提供商和能力标记。支持以下操作：

- **搜索**模型名称或通过下拉框**过滤**提供商
- **排序**：点击列标题按模型名或提供商排序
- **添加**新模型并指定提供商
- **从提供商获取** —— 查询上游 `/v1/models` 端点，浏览可用模型并通过复选框批量添加，支持可选前缀
- **编辑**模型能力（内联编辑）
- **删除**模型路由条目
- **测试**：直接在管理面板中测试模型

### 模型能力

每个模型可声明能力，决定可用的测试类型并在 `/v1/models` 响应中返回：

| 能力 | 说明 | 徽章颜色 |
|------|------|---------|
| `text` | 文本生成 | 蓝色 |
| `vision` | 图像理解 | 绿色 |
| `tools` | 函数/工具调用 | 橙色 |
| `embedding` | 文本嵌入（与 vision/tools 互斥） | 青色 |
| `reasoning` | 扩展思考/思维链（与 embedding 互斥） | 紫色 |

### 模型测试

点击模型行中的 **Test** 按钮可运行快速测试。可用的测试类型取决于模型声明的能力：

| 测试类型 | 说明 | 需要能力 |
|---------|------|---------|
| Text | 简单文本查询 | `text` |
| Vision | 图像理解（内嵌测试图片） | `vision` |
| Stream | 流式文本响应 | `text` |
| Tools | 函数调用（天气工具） | `tools` |
| Embedding | POST 到 `/v1/embeddings`，显示维度数量 | `embedding` |
| Reasoning | 文本查询，附带 `reasoning_effort: "low"` | `reasoning` |

Embedding 模型显示单一的 Test 按钮（无下拉菜单），因为只有一种适用的测试类型。

测试结果在主输出区域显示提取的模型回复，可折叠区域包含原始请求负载、原始响应体，以及（视觉测试时）测试图片预览。

## 仪表盘

**Dashboard** 标签页提供网关流量的实时指标。

![仪表盘标签页](images/admin-dashboard.png)

### 摘要卡片

- **Total Requests** — 自启动以来的累计请求数（或自上次持久化加载）
- **Error Rate** — 非 2xx 响应的百分比
- **Active Streams** — 当前活跃的流式连接数
- **Uptime** — 网关启动以来的运行时间

### 时间序列图表

两个滚动 60 秒窗口的图表：

- **Throughput (req/s)** — 每秒请求率
- **Latency (ms)** — 每秒平均响应时间

### 按提供商分类

按目标提供商分组的请求计数表，便于了解流量分布。

## 请求日志

**Request Log** 标签页显示通过网关的每个请求。

![请求日志标签页](images/admin-request-log.png)

每条记录包含：

| 列 | 说明 |
|----|------|
| Time | 请求时间戳 |
| Model | 请求中的模型名称 |
| Source -> Target | 来源 API 格式和目标提供商 |
| Mode | 流式或非流式 |
| Status | HTTP 状态码（颜色标记） |
| Duration | 端到端延迟 |

### 过滤

使用顶部的下拉筛选器可按以下条件过滤：

- **Model** — 按特定模型筛选
- **Provider** — 按目标提供商筛选
- **Status** — 仅显示成功（2xx/3xx）或错误（4xx/5xx）响应

点击 **Clear Log** 清除当前视图中的所有条目。

## 主题

管理面板提供 8 种主题，可通过右上角的下拉菜单切换：

| 主题 | 风格 |
|------|------|
| Light | 默认，简洁白色背景 |
| Indigo Dark | 深色，靛蓝色调 |
| Dracula | 经典深色主题 |
| Nord | 北极风格的柔和色调 |
| Solarized | Ethan Schoonover 配色方案 |
| Osaka Jade | 深色，翡翠绿色调 |
| One Dark | Atom 编辑器深色主题 |
| Rosé Pine | 柔和的玫瑰与松木色调 |

主题选择存储在 `localStorage` 中，浏览器关闭后依然保留。

## 认证

管理面板**不需要**网关 API Key。

可在配置文件中设置 `server.admin_password` 启用内置密码保护：

```jsonc
{ "server": { "admin_password": "${ADMIN_PASSWORD}" } }
```

详见[配置 — 管理面板安全](configuration.md#管理面板安全)。

启用密码保护后：

- 登录令牌存储在 `localStorage` 中，会话在浏览器重启后依然保留，而非仅限于当前标签页。
- 右上角显示**登出**按钮，可手动结束会话。
- 会话在 **30 分钟无操作**后自动过期。鼠标移动、键盘输入、滚动和点击均视为活跃操作。
- 登录表单使用标准 HTML 表单语义，浏览器密码管理器可保存和自动填充凭据。

也可以通过反向代理保护管理面板：

- **Caddy**：使用 `basicauth` 指令
- **Nginx**：使用 `auth_basic` 指令
- **Traefik**：使用 BasicAuth 中间件

网关 API Key（通过 `server.api_key` 配置）仅保护 AI 请求端点（`/v1/*`）。详见[配置 — 网关 API Key](configuration.md#网关-api-key)。

## 国际化

管理面板支持 English 和中文。通过右上角的语言下拉菜单切换。选择同样保存在 `localStorage` 中。

## 数据持久化

指标和请求日志数据持久化到配置文件旁的单个 SQLite 数据库中：

```
~/.config/llm-rosetta-gateway/
    config.jsonc
    data/
        gateway.db    # SQLite 数据库 — 请求日志 + 指标（WAL 模式）
```

### 工作原理

- **请求日志**条目在每次请求时直接写入 `gateway.db`
- **指标计数器**在每次更新后保存到 `gateway.db`
- **启动时**加载持久化数据——指标和日志在重启后恢复
- **WAL 模式**确保写入崩溃安全，同时不阻塞读取

### 数据保留

最多保留 **5,000** 条请求日志。超出上限时，最旧的条目会自动删除。

### 历史数据迁移

首次启动时，如果数据目录中存在旧版的 `request_log.jsonl` 或 `metrics.json` 文件，会自动导入 `gateway.db` 并重命名为 `*.migrated`。

## 缓存 / 反向代理

管理面板在所有 `/admin/api/*` 响应上设置 `Cache-Control: no-cache, no-store, must-revalidate`。这可以防止激进的反向代理缓存（例如 Caddy 搭配 Souin 插件）返回过期的仪表盘指标或请求日志数据。

模型测试轮询使用 POST 请求而非 GET，以避免被中间代理缓存。
