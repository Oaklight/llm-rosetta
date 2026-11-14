# ModelScope

魔搭（ModelScope）是阿里的模型社区。其推理网关使用 OpenAI Chat Completions 结构，
对注册用户免费。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `modelscope--chat_completions` | `chat_completions` | `https://api-inference.modelscope.cn/v1` | `MODELSCOPE_API_KEY` |

## 模型目录

`GET /v1/models` 返回 OpenAI 结构，共 38 条，且没有类型字段，
因此整份目录都会提供给选择器。

## 开通与额度

需要将阿里云账号与 ModelScope 账号绑定，并完成实名认证。额度是动态的：
每用户每天共 2,000 次请求，单模型上限 500 次，并发也动态限流。

## 网关配置

```jsonc
{
  "providers": {
    "modelscope": {
      "type": "modelscope--chat_completions",
      "api_key": "${MODELSCOPE_API_KEY}"
    }
  }
}
```
