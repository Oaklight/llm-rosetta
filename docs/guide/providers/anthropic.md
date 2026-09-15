# Anthropic

Anthropic 通过专用的 Messages API shim 提供支持，并针对 Claude 模型系列的思考/推理行为提供模型级别的覆盖配置。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `anthropic` | `anthropic` | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |

## 推理支持

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `output_config.effort` |
| Effort 范围 | `low` → `max` |
| 可见性模式 | `auto` → `summarized`、`concise` → `summarized`、`detailed` → `summarized`、`none` → `omitted` |

**思考模式：**

| IR 模式 | Anthropic 值 |
|:---|:---|
| `auto` | `adaptive` |
| `enabled` | `enabled` |
| `disabled` | `disabled` |

### 模型覆盖

不同的 Claude 模型对思考功能的支持不同。Shim 会自动应用模型级别的覆盖配置：

| 模型 | 思考模式 | Budget 比率 | Effort |
|:---|:---|:---:|:---|
| `claude-haiku-4-5-20251001` | 仅 `enabled`、`disabled` | 0.8 | 禁用（Haiku 拒绝 effort 参数） |
| `claude-opus-4-7` | 仅 `adaptive`（enabled → adaptive） | — | 支持 |
| `claude-opus-4-8` | 仅 `adaptive`（enabled → adaptive） | — | 支持 |
| 其他（如 Sonnet 4.6、Opus 4.6） | `adaptive`、`enabled`、`disabled` | — | 支持 |

!!! note "Haiku Effort 限制"
    Claude Haiku 4.5 在收到 `effort` 参数时会返回 400 错误。Shim 会自动为该模型抑制 effort 参数的发送。

## 转换规则

| 类型 | 转换 | 用途 |
|:---|:---|:---|
| IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头 |
| IR | `auto_cache_breakpoints()` | 自动插入缓存断点以支持 prompt 缓存 |

## 网关配置

```jsonc
{
  "providers": {
    "anthropic": {
      "shim": "anthropic",
      "api_key": "${ANTHROPIC_API_KEY}"
    }
  }
}
```
