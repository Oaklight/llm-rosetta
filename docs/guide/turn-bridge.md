---
title: 多轮状态桥接
---

# 多轮状态桥接

!!! note "谁需要看这页"
    这页是给要把库嵌入多轮应用、或者需要扩展网关桥接行为的开发者看的。如果你只是用网关，跨轮次状态 `TurnBridge` 会自动处理，可以跳过。

网关在两种不同 API 格式之间代理时，每个 HTTP 轮次各跑一条独立的转换 pipeline，各有各的 `ConversionContext`。context 里的 provider 特有元数据在轮次结束后就没了。这页讲的是怎么把状态桥接到下一轮。

## 问题

考虑一个接受 OpenAI Chat 请求、转发到 Responses API 上游的网关：

```text
第 1 轮（响应）：
  Responses 上游 → IR → Chat 响应 → 客户端
  ↑ ConversionContext A（响应完成后销毁）

第 2 轮（请求）：
  客户端 Chat 请求 → IR → Responses 上游
  ↑ ConversionContext B（全新的，空的）
```

Responses API 的 `function_call` 项有两个身份字段：

- `id`（如 `fc_abc123`）— 输出项标识
- `call_id`（如 `call_xyz789`）— 关联键，用于匹配结果

IR 只有 `tool_call_id`（映射到 `call_id`）。`id` 在第 1 轮通过 `provider_metadata["responses_item_id"]` 存储，但 Context A 在第 2 轮已不存在。

这不是 Responses 独有的问题。每种格式都有需要跨轮次保留的 provider 特定数据：

| 格式 | 跨轮次数据 | 示例 |
|------|-----------|------|
| Responses | `responses_item_id`、`namespace`、`include` | Codex 子代理路由 |
| Chat | `reasoning_details`、`encrypted_content` | o 系列推理模型 |
| Anthropic | `citations`、`cache_control` | Claude 引用元数据 |
| Google | `thought_signature` | Gemini 2.5+/3 必需 |

## 架构：两层设计

### 第 1 层：Converter（库用户）

Converter 层在 `ConversionContext` 上提供标准 API，用于导出和导入桥接状态。Converter 不关心会话、缓存或 HTTP —— 它们只读写 context。

```python
from llm_rosetta.converters.base.context import ConversionContext

# 响应转换完成后 — 提取需要桥接的状态
ctx1 = ConversionContext()
ir_response = converter.response_from_provider(response, context=ctx1)
bridge_state = ctx1.get_bridge_state()
# bridge_state 是一个普通 dict，可安全序列化（JSON、pickle 等）

# 下次请求转换前 — 注入上一轮的状态
ctx2 = ConversionContext()
ctx2.set_bridge_state(bridge_state)
provider_request = converter.request_to_provider(ir_request, context=ctx2)
```

库用户自行管理存储 —— 内存 dict、Redis、数据库，或任何适合其架构的机制。

#### 桥接状态包含什么？

每个 converter 将自己的数据存入桥接状态 dict，使用 converter 作用域的 key 以避免冲突：

```python
{
    "tool_call_metadata": {
        "call_xyz789": {
            "responses_item_id": "fc_abc123",
            "namespace": "multi_agent_v1.agent_name"
        }
    }
}
```

只有 (a) 没有 IR 等价物且 (b) 下一轮需要的数据才会被包含。`_sequence_number` 或 `current_block_index` 等临时状态不会被包含。

### 第 2 层：Gateway（自动桥接）

Gateway 层提供 `TurnBridge` —— 一个进程级缓存，自动在每次响应后提取桥接状态、在每次请求前注入。网关用户无需编写任何桥接代码。

```text
┌─────────────────────────────────────────────────────┐
│ Gateway                                             │
│                                                     │
│  ┌───────────┐    ┌────────────┐    ┌───────────┐   │
│  │ Pipeline 1 │───▶│ TurnBridge │───▶│ Pipeline 2 │  │
│  │（响应）    │    │（缓存）    │    │（请求）    │  │
│  └───────────┘    └────────────┘    └───────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

`TurnBridge` 替代了之前的 `ProviderMetadataStore`，范围更广：

| 能力 | `ProviderMetadataStore`（旧） | `TurnBridge`（新） |
|-----|------------------------------|-------------------|
| Tool call `provider_metadata` | ✅ | ✅ |
| 响应身份信息 (id, model) | ❌ | ✅ |
| 响应回显字段 | ❌ | ✅ |
| 流式事件 | ✅（仅 tool_call_start） | ✅（所有携带桥接数据的事件） |
| TTL 和淘汰策略 | ✅ | ✅ |

## 组件总览

完整系统有三组不同的组件：

```text
Converter 层（库）
├── ConversionContext        — 单次 pipeline 状态载体
│   ├── get_bridge_state()   — 导出跨轮次数据
│   └── set_bridge_state()   — 导入上一轮数据
├── ProviderPassthroughEvent — IR 不透明流式事件（无 IR 等价物）
└── ProviderPassthroughItem  — IR 不透明非流式项（无 IR 等价物）

Gateway 层
└── TurnBridge               — 自动跨轮次缓存（带 TTL）
    ├── cache_from_context()  — 响应 pipeline 后提取
    ├── inject_into_context() — 请求 pipeline 前注入
    ├── cache_from_stream_event() — 从流式事件中提取
    └── clear()
```

这些组件职责不重叠：

- **`ConversionContext`** 是载体 — 穿透单次 pipeline 传递状态
- **`ProviderPassthroughEvent/Item`** 用于 IR 中完全没有表示的数据（如 `tool_search_call`）
- **`TurnBridge`** 是持久层 — 跨 HTTP 边界存取桥接状态

## 无法桥接的数据

某些数据从根本上无法跨格式保留：

| 数据 | 原因 |
|------|------|
| `billing`、`completed_at` | 上游生成，无法合成 |
| `_sequence_number` | 每个流重置是正确行为 |
| `current_block_index` | 位置计数器，非语义 |

另外，如果客户端构建了全新的请求而没有回传完整历史，则该轮次的桥接状态不可用。这是预期行为：桥接是对客户端回传历史的补充，而非替代。
