# NVIDIA NIM

NVIDIA NIM 在 `build.nvidia.com` 上托管开源权重模型，接口兼容 OpenAI，
使用开发者计划的密钥签名。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `nvidia--chat_completions` | `chat_completions` | `https://integrate.api.nvidia.com/v1` | `NVIDIA_API_KEY` |

## 模型目录

`GET /v1/models` 返回 OpenAI 结构，且没有类型字段，因此整份目录都会提供给选择器。
NVIDIA 不公布 RPM/RPD 表：速率按模型、并随并发用户数变化，应从账户页面读取。

!!! warning "仅限原型验证"
    NVIDIA 的托管端点是用于原型、研究、开发与测试的。生产使用需要
    NVIDIA AI Enterprise 授权。

## 网关配置

```jsonc
{
  "providers": {
    "nvidia": {
      "type": "nvidia--chat_completions",
      "api_key": "${NVIDIA_API_KEY}"
    }
  }
}
```
