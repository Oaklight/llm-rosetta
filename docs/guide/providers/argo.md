# Argo

Argo 是美国阿贡国家实验室的内部 API 网关，通过统一端点提供对多种 LLM 模型的访问。它通过两个 shim 端点提供支持。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `argo--openai_chat` | `openai_chat` | `https://apps.inside.anl.gov/argoapi/v1` | `ARGO_API_KEY` |
| `argo--anthropic` | `anthropic` | `https://apps.inside.anl.gov/argoapi` | `ARGO_API_KEY` |

!!! info "模型 ID 字段"
    两个 Argo shim 均使用 `model_id_field: internal_id`，即模型标识符在请求体中以 `internal_id` 字段发送，而非标准的 `model` 字段。

## 推理支持

**Chat Completions (`argo--openai_chat`)：**

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `reasoning_effort` |
| Effort 范围 | 完整 IR 阶梯（`minimal` → `max`） |

**Anthropic (`argo--anthropic`)：**

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `output_config.effort` |
| Effort 范围 | `low` → `max` |
| Budget 比率 | 0.8 |
| 思考默认值 | `auto` |
| 未签名块 | `preserve` |
| 可见性模式 | `auto`/`concise`/`detailed` → `summarized`、`none` → `omitted` |

**思考模式：**

| IR 模式 | Argo Anthropic 值 |
|:---|:---|
| `auto` | `adaptive` |
| `enabled` | `enabled` |
| `disabled` | `disabled` |

### 模型覆盖

| 模型 | 思考模式 | Budget 比率 |
|:---|:---|:---:|
| `claudehaiku45` | 仅 `enabled`、`disabled` | 0.8 |
| `claudesonnet4` | 仅 `enabled`、`disabled` | 0.8 |
| `claudeopus47` | 仅 `adaptive`（enabled → adaptive） | — |
| `claudeopus48` | 仅 `adaptive`（enabled → adaptive） | — |

## 能力标志

| 标志 | `argo--openai_chat` | `argo--anthropic` |
|:---|:---:|:---:|
| 自定义工具 | ✅ | — |
| 最大工具描述长度 | 1024 字符 | — |

## 转换规则

**`argo--openai_chat`：**

| 类型 | 转换 | 用途 |
|:---|:---|:---|
| Post-IR | `rename max_tokens → max_completion_tokens` | 适配 Argo 的字段命名 |
| Post-IR | `downgrade developer → system` | 将 developer 角色转换为 system 角色 |
| Post-IR | `default null content → ""` | 防止空内容错误 |
| Post-IR | `strip temperature`（claudeopus47\*） | 为 Opus 4.7 模型移除 temperature |
| Post-IR | `flatten system content arrays`（gemini\*） | 为 Gemini 模型展平系统内容数组 |
| IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头 |
| IR | `truncate_images(50)`（gpt/o\*） | 限制 GPT/o 系列每次请求最多 50 张图片 |
| IR | `unwind_parallel_tool_calls()`（gemini\*） | 为 Gemini 模型串行化并行工具调用 |

**`argo--anthropic`：**

| 类型 | 转换 | 用途 |
|:---|:---|:---|
| Pre-IR | `normalize_openai_response()` | 将 OpenAI 格式的响应转换为 Anthropic 格式（处理 Argo 不一致的响应格式） |
| IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头 |
| IR | `auto_cache_breakpoints()` | 插入缓存断点以支持 prompt 缓存 |

## 网关配置

=== "Chat Completions"

    ```jsonc
    {
      "providers": {
        "argo": {
          "shim": "argo--openai_chat",
          "api_key": "${ARGO_API_KEY}"
        }
      }
    }
    ```

=== "Anthropic"

    ```jsonc
    {
      "providers": {
        "argo-anthropic": {
          "shim": "argo--anthropic",
          "api_key": "${ARGO_API_KEY}"
        }
      }
    }
    ```

## 备注

- Argo 作为多个上游提供方（OpenAI、Anthropic、Google）的代理。Shim 透明地处理提供方特定的行为差异，包括响应格式标准化和模型特定的字段调整。
- Anthropic 端点上的 `unsigned_blocks: preserve` 设置保留响应中未签名的推理块，这是因为 Argo 的代理层不会对思考块进行签名。
- 两个 shim 均导出 `model_list_transform`，用于标准化模型列表的响应格式。
