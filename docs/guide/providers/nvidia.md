---
title: NVIDIA NIM
---

# NVIDIA NIM

NVIDIA NIM hosts open-weight models on `build.nvidia.com` behind an
OpenAI-compatible endpoint, signed with a developer-program key.

## Endpoint

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `nvidia--chat_completions` | `chat_completions` | `https://integrate.api.nvidia.com/v1` | `NVIDIA_API_KEY` |

## Model roster

`GET /v1/models` answers the OpenAI shape with no type field, so the whole
catalogue is offered to the picker. NVIDIA publishes no RPM/RPD table: the
rate varies per model and with the number of concurrent users, and it is
read from the account page.

!!! warning "Prototyping only"
    NVIDIA's hosted endpoints are for prototyping, research, development and
    testing. Production use requires an NVIDIA AI Enterprise licence.

## Gateway configuration

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
