---
title: 提供方额度
---

# 提供方额度

订阅型通道——OpenCode Go、ChatGPT 计划、各类 Coding Plan——按窗口计量凭据，
而不是按请求计价。网关可以读取这些窗口，在 provider 视图里显示，并停止向窗口已耗尽的
凭据路由。

## 各层各管什么

分层的目的，是让每一类事实只存在于维护它的地方：

| 层 | 持有 | 声明位置 |
|:---|:---|:---|
| **Shim** | 这个 vendor 的额度从哪里读、怎么解码 | `provider.yaml` → `quota:`，以及 `quota.py` 里的 probe |
| **Provider** | 有哪些凭据，以及对 shim 的覆盖 | `config.jsonc` → `quota:` |
| **Quota 状态** | 每个凭据的最后一次读数与降级 | 仅运行时 |

网关侧的 `llm_rosetta_gateway.quota` 只负责**运行时**：按凭据的状态、探测执行与
管理接口。**契约与解析**——`QuotaProbe` 接口、窗口与带内信号类型、
`resolve_probe`——位于 `llm_rosetta.shims.quota_probe`，因为 shim 的 `quota.py`
实现该契约，而翻译核心绝不能导入 gateway。这里不包含任何 vendor 代码。
Shim **自带 probe**：`quota.py` 与 `provider.yaml` 放在一起（跨多个格式的
vendor 放组目录，一次性 lane 放叶目录），由 `quota.probe` 按名称选中。端点、
认证头、响应解码和报错措辞都是 vendor 事实，留在 vendor 旁边；vendor 改变
响应格式时只改那一个 shim 模块。

## 状态属于凭据

读数属于账户，而不属于 provider 条目。状态键是
`(probe, sha256(key)[:16])`：

- 两个 provider 配置同一个 key 时共享一份读数；
- 同一个 key 用在两条通道（Go 与 Zen）上是两份读数，因为 probe 不同；
- 原始 key 不会被存储、记录或返回——admin 响应里只有指纹。

状态是进程内的。若部署运行多个 gateway worker，每个 worker 会独立探测与降级；
上游额度仍然共享，读数最终会收敛，但某个 worker 学到的降级在它自己探测或收到
429 之前不会传播到其他 worker。

## 轮换

`KeyRing` 保留原有的 round-robin 与 rendezvous affinity，只多一个谓词：
状态已降级的凭据会被跳过。若所有 key 都被降级，ring 退回普通轮换而不是失败——
降级只是建议，最终以上游为准。

耗尽的窗口会把凭据降级到其 `resetsAt`。若 provider 在 429 响应体里点名了耗尽的
窗口，则立即在带内降级，不必等下一次探测。相反，**探测失败**不构成任何证据，
永远不会导致降级。

## 支持的 probe

| Probe | 端点 | 凭据 | 所在模块 |
|:---|:---|:---|:---|
| `opencode_go` | `/zen/go/v1` 上的 `{base_url}/usage` | Go key，作为 `Authorization: Bearer` | `providers/opencode/quota.py` |
| `openai_codex` | `https://chatgpt.com/backend-api/wham/usage` | 计划 access token，外加从 JWT claim 取出的 `chatgpt-account-id` | `providers/openai/quota.py` |

Zen 通道（`/zen/v1`）没有按 key 的用量端点；查询它的额度会直接报告这一点，
而不是去探测一个必然 404 的 URL。按量付费余额属于 console session 读取，不在范围内。

## 配置

OpenCode 的 shim 声明了 probe 和五分钟后台刷新：

```yaml
quota:
  probe: opencode_go
  interval: 300
```

`opencode_go` 实现在 `providers/opencode/quota.py`；shim loader 对组目录执行一次，
并把它附加到该组的每个 OpenCode 叶 shim 上——声明只负责按名称选中。
需要指定 vendor 的 provider 可以显式给出 probe 名和 URL：

```jsonc
{
  "providers": {
    "opencode-go": {
      "type": "opencode--chat_completions",
      "api_key": "${OPENCODE_API_KEY}"
    }
  }
}
```

| 字段 | 含义 |
|:---|:---|
| `probe` | `opencode_go` 或 `openai_codex` |
| `url` | 显式用量 URL，用于无法从 `base_url` 推导的情况 |
| `interval` | 后台刷新秒数；`0` 表示只在需要时读取 |
| `demote_at` | 已用比例达到多少即视为耗尽；`100` 沿用 vendor 自己的规则 |

## 读取

管理面板的 provider 卡片上有 **Quota** 按钮，调用
`GET /admin/api/config/providers/<name>/quota`（加 `?refresh=1` 绕过一分钟缓存），
按凭据和窗口逐行显示，并给出降级原因。

```json
{
  "provider": "opencode-go",
  "supported": true,
  "probe": "opencode_go",
  "credentials": [
    {
      "credential": "e8c2967da349b907",
      "reading": {
        "status": "ok",
        "windows": [
          {"name": "5 hours", "used_percent": 0, "resets_at": "2026-10-19T16:09:07Z", "exhausted": false},
          {"name": "Month", "used_percent": 47, "resets_at": "2026-11-11T22:26:38Z", "exhausted": false}
        ]
      },
      "demotion": null
    }
  ]
}
```

`used_percent` 是**已用**比例，两家 vendor 都按这个口径上报。窗口的 status 不是
`ok`（OpenCode 的 `rate-limited`）时，即使比例低于 100 也算耗尽。

## 无 key 通道

[OpenCode Zen 公共通道](providers/opencode.md#public-free-lane)没有凭据，因此没有按 key
的额度。它返回 429 时，会给整个 provider 一个短暂冷却，而不是降级某个 key。
