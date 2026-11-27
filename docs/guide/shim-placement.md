---
title: Shim 归属与成熟度
---

# Shim 归属与成熟度 { #shim-placement-and-maturity }

provider shim 放在两处之一：**核心**（随 `llm-rosetta` 发行包一起发布），或者一个发布带版本扩展索引、由网关从中安装的**源仓库**（见 [Manifest 源](shims.md#manifest-sources)）。本页说明哪些 shim 放在哪里、shim 如何在两处之间迁移，以及核心与源提供同名 shim 时网关怎么做。规则来自 epic [#1660](https://github.com/Oaklight/llm-rosetta/issues/1660)。

## 归属规则 { #placement-rule }

只有同时满足以下两条，shim 才放进核心：

1. **它是模型服务方自己的 API**，而不是聚合商、转售商或镜像。
2. **它只用静态 API key 认证。** 服务方目录里没有 `auth.py`。允许有 `quota.py`，前提是它的探测只用这个 key。

其余的 shim 永久放在源仓库：

- OAuth 和订阅通道（OpenCode、OpenDesign、Codex / `openai_chatgpt`、Gemini 与 Claude 订阅）；
- 聚合商和免费端点（Kilo、OpenRouter、Groq、NVIDIA、SiliconFlow、ModelScope、免费档）；
- Antigravity。

把它们分开，可以让变化快的登录流程和服务条款风险不进入核心发布。

**符合条件是必要条件，不是充分条件。** 新 shim 或很少使用的 shim 先放在 `contrib`，成熟后再升入核心。现有的核心服务方也按这一条处理。

## 成熟度阶梯 { #maturity-ladder }

| 阶段 | 位置 | 含义 |
|------|------|------|
| `experimental` | contrib | 新加入，或上游 API 仍在变化；不承诺稳定 |
| `beta` | contrib | 有测试，并做过一次在线验证；shim 仍可能变化 |
| `stable` | contrib | 满足升入核心的条件；是核心的候选 |
| core | 核心 | 随核心发布，遵守核心的兼容性规则 |

源在索引中按服务方记录阶段（`maturity`、`since` 和最近一次 `conformance` 结果）。管理面板在每张服务方卡片上显示它（**适配模块 → 可安装**）。

## 升入核心 { #promotion }

只有以下条件**全部**满足，服务方才升入核心：

1. 符合[归属规则](#placement-rule)。
2. 在 contrib 中处于 `stable` 至少一个次版本周期。
3. 最近 30 天内，它的每种格式都通过了一次在线 `tools/conformance` 测试。
4. 最近 4 周没有 shim 引起的修复，也没有针对它的未关闭 bug。
5. contrib 中有它的注册与 transform 测试，并有文档条目。
6. 有任意网关上的真实使用证据。这一条是参考，不是必需。
7. 升入核心经过 PR 评审。

服务方进入核心后，核心会注册它的 shim 名称。运行该核心的网关不再从源加载这些名称：按 shim 判定，核心优先（见[优先级](#precedence)）。

## 降回 contrib { #demotion }

满足以下**任一**条件，核心服务方就降回 contrib：

- 服务方改了认证方式，不再是单纯的 API key。
- 一个一致性测试失败在一个次版本周期内没有修复。
- 两个次版本周期内既没有使用证据，也没有维护者。

shim 名称不变，迁移的通知保留一个次版本。

## 名称与位置无关 { #names }

shim 名称（provider 条目里的 `type: <vendor>--<format>`）在核心和源里相同。升入与降级只改变从哪里安装 shim，从不改变用户的配置。从源选中该服务方的网关在升入核心后照常工作（由核心提供该名称），在降级后也照常工作（只要 `shims.install` 选中它，就由源提供）。

## 核心与源之间的优先级 { #precedence }

当核心和源提供同一个 shim 名称时：

- **按 shim 判定，核心优先。** 网关不从源加载该名称，并在日志里写明源、服务方和被忽略的名称。该服务方的其他 shim 照常安装。所有 shim 都是核心或已安装插件名称的服务方会被跳过，其卡片显示*已提供（由核心或已安装插件）…*。
- **两个源发布同一服务方时，`shims.sources` 中排在前面的源优先**；另一份会记入日志并被忽略。

## 现有服务方的归属 { #placement-today }

2026-11-21 确定的归属：

| 位置 | 服务方 |
|------|------|
| 核心 | `openai`、`anthropic`、`google`、`deepseek`；格式参考 `cohere`、`jina`、`voyage`（embeddings 与 rerank）和 `typesafe`（decisions），保留它们是因为核心的测试与一致性测试每种格式需要一个真实服务方 |
| contrib，`stable`，可升入核心 | `minimax`、`zhipu` / `zai`、`xai`、`qwen`、`moonshot`、`volcengine` |
| contrib，永久 | `opencode`、`openai_chatgpt`（含 `openai_codex` 探测）、`antigravity`、`kilo`、`openrouter`、`groq`、`nvidia`、`siliconflow`、`modelscope`、免费档服务方、`opendesign` |
| contrib，新加入，从 `experimental` 开始 | `mistral` |
| Argonne | `alcf`、`argo`、`asksage` |

迁出核心的工作在 #1660（S7）中跟踪。服务方迁出之前，核心仍提供它，并且核心优先于源中的副本。
