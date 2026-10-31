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
Shim **自带 probe**：`quota.py` 放在 vendor 目录下（与 `llm/`、`embeddings/` 等
类型目录并列，该 vendor 的每种类型共用），一次性 lane 则与自己的 `provider.yaml`
放在一起，由 `quota.probe` 按名称选中。端点、
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

`opencode_go` 实现在 `providers/opencode/quota.py`；shim loader 对 vendor 目录执行一次，
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

## 读数如何影响路由

过去读数只在两端起作用：窗口耗尽时降级凭据（或封锁提供方），在那之前配额层对路由是不可见的。
两个都“可用”的提供方会平分流量，即使其中一个已经用掉 96%。

现在每次选择都会按剩余额度给候选加权。选择器取一个提供方全部凭据中**最小的剩余份额**，
再按它缩放配置权重：

```
weight × (0.25 + 0.75 × headroom)
```

于是未被使用的提供方保持原权重，接近耗尽的只承担约四分之一的份额，而且**任何权重都不会被
缩到零** —— 提供方是由配额状态（窗口耗尽、封锁生效）剔除的，不是由这个系数。没有任何读数
的提供方保持其配置权重：没被测量不等于已经耗尽。

两点值得知道：

- **配置的权重仍然是运维的表达**。这个比例只是缩放它，所以配置为 `weight: 3` 的提供方在
  同样余量下依然比同伴更有吸引力。
- 客户端亲和仍然先做哈希。余量只影响在多个提供方之间怎么选，不影响一段对话是否留在它缓存
  命中过的那个提供方上。

当一份**新鲜**读数显示某个提供方的每个凭据都已耗尽时，它是被**移除**而不只是降权。规则刻意很窄，
且对整个 key 环是全有或全无 —— 因为排除无法自我纠正：没有流量会发往被排除的提供方，探针从未读过
的凭据就永远读不到，提供方会被排除整整一个新鲜度窗口。降权没有这个陷阱：流量继续流动，凭据会被
读到，权重随之恢复。因此排除要求完整的证据：key 环里的每个凭据都有读数，读数足够新鲜，解出了一个
窗口，且解出的每个窗口都已耗尽。超过十五分钟的读数 —— 或者超过提供方自己的 `quota.interval`（当
后者更长时，这样慢速轮询不会让每份读数一到达就过期）—— 不能说明现在的情况；没有探针的通道也绝不
因“未测量”而被排除。

排除会随读数年龄过期，也会在厂商给出的重置时刻自动失效；它在策略统计候选之前生效，因此剩余提供方
之间是均分的，而不是由其中一个吞掉被排除者本该轮到的那几轮。当读数让某个模型一个可用提供方都不剩时，
客户端收到的是说明额度已耗尽的 `503`（厂商给出重置时间时，它还带在 `Retry-After` 里），而不是
“正在初始化”的消息 —— 后者会让客户端去找一个并不存在的启动问题。

该阶段在选择链中名为 `quota`，在 anchor 之前运行，因此它是提供信息而不是做决定 —— 策略
本身没有改变。

提供方卡片会展示路由的结论 —— 正在冷却、被排除及其原因，或是它实际服务的权重与最紧凭据的剩余
份额 —— 用的就是路由调用的同一个函数。没有用量端点的通道也会得到同样的结论，而这恰恰最要紧：
它的排除无法从任何百分比读出来。一个凭据耗尽、另一个凭据健康的提供方，最紧凭据读数是 `0%`，
但它仍按配置权重的四分之一服务 —— 卡片两者都会说清楚。

## 无 key 通道

[OpenCode Zen 公共通道](providers/opencode.md#public-free-lane)没有凭据，因此没有按 key
的额度。它返回 429 时，会给整个 provider 一个短暂冷却，而不是降级某个 key。

## 没有用量端点的通道

有些通道的额度厂商从不向应用暴露：[ChatGPT 套餐通道](providers/openai_chatgpt.md) 按
5 小时窗口与按应用的周上限计量，而这些都在账号自己的设置里。通道唯一给出的信号是一次
拒绝，并指明耗尽的窗口 —— `subscription_sharing_usage_limit_exceeded`，可能出现在响应里，
也可能在流中以 `response.failed` 出现 —— 因此没有任何可轮询的东西。

这类 shim 把探针声明为**仅 in-band**：

```yaml
quota:
  probe: openai_codex
  in_band: true
```

探针只为其 `inband()` 分类器保留，从不轮询。拒绝到达时，该凭据被降级一段很短且有界的
冷却时间 —— 60 秒 —— 而不是等到一个厂商并未给出的重置时间：与其他地方一致，读取失败
从不降级，而错误从不臆造窗口。提供方卡片会报告**没有可读的配额**并给出原因，而不是报错；
它的 **管理用量** 链接指向厂商的设置页，那是唯一能看到额度的地方。
