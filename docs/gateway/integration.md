---
title: 网关集成指南
---

# 网关集成指南

llm-rosetta 网关可以作为代理管道 + 管理面板嵌入到下游项目中。与运行独立的 `llm-rosetta-gateway` CLI 不同，你的项目直接导入网关内部模块，将其接入自己的 HTTP 应用、配置系统和启动流程。

本指南涵盖集成接口：`ConfigIO` 协议、`setup_admin()` 参数、遥测对接和品牌定制。典型的集成示例是 [argo-proxy](https://github.com/Oaklight/argo-proxy)，它在网关之上封装了自己的 YAML 配置、模型注册表和认证层。

## ConfigIO 协议

### 为什么需要

独立网关读写 JSONC 配置文件。下游项目通常有自己的配置格式——YAML、TOML、环境变量，甚至数据库。`ConfigIO` 协议抽象了文件 I/O，使管理面板无需了解底层格式即可读写配置。

### 协议定义

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class ConfigIO(Protocol):
    def load(self, path: str) -> dict[str, Any]:
        """读取配置，执行环境变量替换（用于运行时）。"""
        ...

    def load_raw(self, path: str) -> dict[str, Any]:
        """读取配置，不执行环境变量替换（用于编辑往返）。"""
        ...

    def save(self, path: str, data: dict[str, Any]) -> None:
        """将配置写回磁盘。"""
        ...
```

- `load()` 返回解析了 `${ENV_VAR}` 占位符的配置——用于运行时。
- `load_raw()` 返回保留占位符的原始配置——用于管理面板读取配置进行编辑时，确保写回时不丢失占位符。
- `save()` 将管理面板的编辑持久化到磁盘。

### 注入运行时状态

如果你的项目在运行时生成 provider 或 model（例如从上游 API 发现），在 `load_raw()` 中注入它们：

```python
class MyConfigIO:
    def __init__(self, config, model_registry):
        self._config = config
        self._registry = model_registry

    def load(self, path: str) -> dict[str, Any]:
        raw = load_my_config(path)
        return self._inject_runtime_state(raw)

    def load_raw(self, path: str) -> dict[str, Any]:
        raw = load_my_config(path)
        return self._inject_runtime_state(raw)

    def _inject_runtime_state(self, data: dict[str, Any]) -> dict[str, Any]:
        # 注入由项目管理的 provider 和 model，而非存储在配置文件中的
        data["providers"] = build_providers_from_registry(self._registry)
        data["models"] = build_models_from_registry(self._registry)
        data.setdefault("server", {}).update({
            "host": self._config.host,
            "port": self._config.port,
        })
        return data

    def save(self, path: str, data: dict[str, Any]) -> None:
        # 剥离运行时注入的字段，不应持久化
        data.pop("providers", None)
        data.pop("models", None)
        write_my_config(path, data)
```

!!! warning "在 `save()` 中剥离运行时字段"
    如果你的 `load_raw()` 注入了由上游管理（非用户可编辑）的 provider/model，`save()` **必须**在写入前将其剥离。否则管理面板的"保存"按钮会将运行时生成数据的过时副本持久化到配置文件中。

## setup_admin() 参数

`setup_admin()` 函数在你的 app 实例上初始化管理面板状态。通常在应用启动期间，在你自己的配置和传输层准备就绪后调用。

```python
from llm_rosetta.gateway.admin import setup_admin

def setup_admin(
    app,
    config: GatewayConfig,
    config_path: str | None,
    config_io: ConfigIO | None = None,
    custom_head: str | None = None,
    branding: dict[str, Any] | None = None,
    disabled_tabs: list[str] | None = None,
) -> None:
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `app` | `App` | 你的 HTTP 应用实例 |
| `config` | `GatewayConfig` | 已解析的网关配置 |
| `config_path` | `str | None` | 磁盘上的配置文件路径。`None` 禁用持久化和配置编辑。 |
| `config_io` | `ConfigIO | None` | 自定义配置适配器。为 `None` 时默认使用 `JsoncConfigIO`。 |
| `custom_head` | `str | None` | 注入管理页面 `</head>` 前的 HTML 片段 |
| `branding` | `dict | None` | 用于自定义管理面板外观的字典 |
| `disabled_tabs` | `list[str] | None` | 要隐藏的管理面板标签页 ID（如 `["metrics"]`）。`None` 显示所有标签页。 |

### 完整示例

```python
from llm_rosetta.gateway.admin import setup_admin
from llm_rosetta.gateway.admin.routes import register_admin_routes

# 1. 先注册管理路由
register_admin_routes(app)

# 2. 初始化管理状态
config_io = MyConfigIO(my_config, model_registry)
setup_admin(
    app,
    gateway_config,
    str(config_path) if config_path else None,
    config_io=config_io,
    custom_head='<script>console.log("custom admin");</script>',
    branding={
        "title": "My Project",
        "subtitle": "gateway admin",
        "version": __version__,
        "links": [
            {"label": "GitHub", "url": "https://github.com/me/myproject", "icon": "github"},
            {"label": "PyPI", "url": "https://pypi.org/project/myproject/", "icon": "pypi"},
            {"label": "Docs", "url": "https://myproject.readthedocs.io", "icon": "docs"},
        ],
        "attribution": "Powered by llm-rosetta gateway",
    },
)
```

## 遥测集成

网关的 `handle_streaming()` 和 `handle_non_streaming()` **不会**记录遥测数据——它们只处理请求转换和上游转发。调用者负责记录指标和请求日志条目。

### 记录指标和请求日志

每次代理调用后，记录结果：

```python
import time
from llm_rosetta.observability import MetricsCollector, RequestLog, RequestLogEntry

# 这些由 setup_admin() 创建并挂载到 app 上
metrics: MetricsCollector = app.metrics
request_log: RequestLog = app.request_log

t0 = time.monotonic()
response, profile = await handle_non_streaming(
    route, provider_info, body,
    transport=transport,
    metadata_store=store,
    extra_headers=extra_headers,
    capture_state=app.capture_state,  # 用于内容捕获
    persistence=app.persistence,       # 用于错误转储
)
duration_ms = (time.monotonic() - t0) * 1000

# 记录指标
metrics.record_request(
    model=model,
    source=source_provider,
    target=route.target_provider,
    status_code=response.status_code,
    duration_ms=duration_ms,
    is_stream=False,
    provider_name=route.provider_name,
    error_detail=error_detail,
)

# 记录请求日志条目
entry = RequestLogEntry.create(
    model=model,
    source_provider=source_provider,
    target_provider=route.target_provider,
    target_provider_name=route.provider_name,
    is_stream=False,
    status_code=response.status_code,
    duration_ms=duration_ms,
    error_detail=error_detail,
)
request_log.add(entry)
```

### 传递 capture_state 和 persistence

`capture_state` 和 `persistence` 参数分别启用内容捕获（用于管理面板的请求检查器）和错误转储（用于调试上游故障）：

```python
response, profile = await handle_streaming(
    route, provider_info, body,
    transport=transport,
    metadata_store=store,
    extra_headers=extra_headers,
    capture_state=app.capture_state,   # 内存中的内容捕获
    entry_id=pre_entry_id,             # 用于流式 profile 回写
    request_log=app.request_log,       # 用于流式 profile 回写
)
```

两者都由 `setup_admin()` 自动创建并挂载到 `app` 上。

### 管理面板测试按钮

管理面板有一个"测试"按钮，通过网关发送请求以验证 provider 连通性。为使其正常工作，app 必须暴露其绑定地址：

```python
app._bind_host = host
app._bind_port = port
```

在调用 `setup_admin()` 之后、服务器开始接受连接之前设置这些属性。

## 品牌定制

### branding 字典

`branding` 字典映射到管理面板各处的 UI 元素：

| 键 | 显示位置 | 默认值 |
|----|---------|--------|
| `title` | 顶部栏、浏览器标签页 | `"llm-rosetta"` |
| `subtitle` | 顶部栏（较小文字） | `"gateway admin"` |
| `version` | 设置页脚 | _(无)_ |
| `links` | 设置页脚（图标按钮） | _(无)_ |
| `attribution` | 设置页脚（"Powered by" 行） | _(无)_ |

### 页脚链接

`links` 列表中的每个条目是包含 `label`、`url` 和可选 `icon` 的字典：

```python
"links": [
    {"label": "GitHub",  "url": "https://github.com/me/project", "icon": "github"},
    {"label": "PyPI",    "url": "https://pypi.org/project/pkg/",  "icon": "pypi"},
    {"label": "Docker",  "url": "https://hub.docker.com/r/me/img","icon": "docker"},
    {"label": "Docs",    "url": "https://docs.example.com",       "icon": "docs"},
    {"label": "Custom",  "url": "https://example.com"},  # 无图标
]
```

可用图标名称：`github`、`pypi`、`docker`、`docs`。省略 `icon` 则显示为纯文本链接。

### 使用 custom_head 进行深度 UI 定制

`custom_head` 参数在管理页面的 `</head>` 前注入原始 HTML。适用于 CSS 覆盖、JavaScript 补丁或基于 MutationObserver 的 UI 修改。

示例——为托管的 provider 隐藏编辑/删除按钮：

```python
_ADMIN_CUSTOM_HEAD = """\
<script>
document.addEventListener('DOMContentLoaded', function(){
  var _managed = {'my-upstream-openai':1, 'my-upstream-anthropic':1};
  new MutationObserver(function(){
    var cards = document.querySelectorAll('.provider-card');
    if (!cards.length) return;
    cards.forEach(function(card){
      var nm = card.querySelector('.name');
      if (!nm) return;
      var pName = nm.textContent.trim();
      if (!_managed[pName]) return;
      // 隐藏托管 provider 的开关和操作按钮
      var toggle = card.querySelector('.toggle');
      if (toggle) toggle.style.display = 'none';
      var actions = card.querySelector('.actions');
      if (actions) actions.style.display = 'none';
    });
    // 完全隐藏"添加 Provider"按钮
    var addBtn = document.querySelector('button[onclick*="openProviderModal"]');
    if (addBtn) addBtn.style.display = 'none';
  }).observe(document.body, {childList: true, subtree: true});
});
</script>"""
```

!!! note
    `admin.html` 中的 `configData` 是 `let` 作用域的——你无法在注入的脚本中通过 `window` 访问它。如果需要读取配置状态，请使用管理 API 端点（`/admin/api/config`）。

## Provider 配置

### 只读 Provider

当 provider 由你的项目管理（例如从上游服务自动发现）时，将其标记为只读：

```python
def build_providers(config):
    return {
        "my-openai": {
            "type": "openai_chat",
            "api_key": config.api_key,
            "base_url": config.openai_base_url,
            "readonly": True,  # 阻止管理面板编辑
        },
    }
```

`readonly: True` 标志告诉管理面板禁用该 provider 的编辑控件。

### Model 能力声明

每个 model 条目可以声明其能力，这些能力会显示在管理面板的 model 表格中：

```python
def build_models(registry):
    models = {}
    for alias, model_id in registry.available_models.items():
        models[alias] = {
            "provider": resolve_provider(model_id),
            "capabilities": ["text", "vision", "tools", "reasoning"],
        }
        # 可选：将网关别名映射到不同的上游 model 名称
        if model_id != alias:
            models[alias]["upstream_model"] = model_id
    return models
```

可用的能力值：`text`、`vision`、`tools`、`reasoning`（或任何自定义字符串——它们在管理 UI 中原样显示）。

### 保存时剥离 Provider/Model 变更

对于 provider 和 model 由上游控制的托管部署，你的 `ConfigIO.save()` 应剥离管理面板对这些部分的任何编辑：

```python
def save(self, path: str, data: dict[str, Any]) -> None:
    # 移除运行时管理的部分
    data.pop("providers", None)
    data.pop("models", None)

    # 将网关格式的键映射回你项目的配置格式
    if server := data.pop("server", None):
        if "host" in server:
            data["host"] = server["host"]
        if "port" in server:
            data["port"] = server["port"]

    write_my_config(path, data)
```

## 常见陷阱

### 循环导入

如果你的 `ConfigIO` 从 bridge 模块导入，而 bridge 模块又从 config 导入，启动时会遇到循环导入错误。在方法体内使用延迟导入：

```python
class MyConfigIO:
    def _inject_runtime_state(self, data):
        # 延迟导入以避免循环依赖
        from ..bridge import build_providers, build_models
        data["providers"] = build_providers(self._config)
        data["models"] = build_models(self._registry)
        return data
```

### config_path 的 Path vs str

`setup_admin()` 的 `config_path` 参数必须是 `str`，而非 `pathlib.Path`。管理面板会将其序列化为 JSON 发送给前端。如果你有 `Path` 对象，传入 `str(config_path)`：

```python
setup_admin(
    app,
    gateway_config,
    str(config_path) if config_path else None,  # 不要传 Path 对象
    config_io=config_io,
)
```

### anthropic_stream_mode 与管理面板测试按钮

管理面板的测试按钮发送**非流式**请求。如果你的项目强制 Anthropic 使用流式（例如 `anthropic_stream_mode = "force"`），测试按钮可能表现异常。确保非流式请求仍然可以通过以用于测试目的，或者处理 `"retry"` 模式——该模式在上游返回"需要流式"错误时自动回退到流式。

### admin.html 中 configData 的作用域

管理页面的 `configData` 变量在 IIFE 内以 `let` 声明。通过 `custom_head` 注入的脚本运行在全局作用域中，无法直接访问 `configData`。如果你需要在注入的脚本中获取配置状态，从管理 API 获取：

```javascript
// 在 custom_head 脚本中——不要尝试直接读取 configData
fetch('/admin/api/config')
  .then(r => r.json())
  .then(cfg => {
    // 现在你有了配置
  });
```

### 先注册路由再调用 setup_admin

调用 `register_admin_routes(app)` 应在 `setup_admin()` **之前**。路由注册设置 HTTP 端点，而 `setup_admin()` 初始化这些端点依赖的状态。颠倒顺序不会崩溃，但在注册和初始化之间的短暂窗口内，路由会引用未初始化的状态。
