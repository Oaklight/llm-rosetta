# Google

Google 通过两个 shim 端点提供支持：一个用于 generateContent API，另一个用于较新的 Interactions API。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `google` | `google_generate` | `https://generativelanguage.googleapis.com` | `GOOGLE_API_KEY` |
| `google_interactions` | `google_interactions` | `https://generativelanguage.googleapis.com` | `GOOGLE_API_KEY` |

## 推理支持

两个端点共享相同的推理配置：

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `thinking_level` |
| Effort 范围 | `minimal` → `high` |

## 转换规则

| Shim | 类型 | 转换 | 用途 |
|:---|:---|:---|:---|
| `google` | IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头 |
| `google_interactions` | — | — | 无转换规则 |

## 网关配置

=== "generateContent"

    ```jsonc
    {
      "providers": {
        "google": {
          "shim": "google",
          "api_key": "${GOOGLE_API_KEY}"
        }
      }
    }
    ```

=== "Interactions"

    ```jsonc
    {
      "providers": {
        "google-interactions": {
          "shim": "google_interactions",
          "api_key": "${GOOGLE_API_KEY}"
        }
      }
    }
    ```

## 备注

- `google` shim 面向 generateContent 端点，即标准的 Gemini API。`google_interactions` shim 面向较新的 Interactions API 格式。
- Google 通过 `key` 查询参数进行 API Key 认证，而非大多数其他提供方使用的 `Authorization` 请求头。
