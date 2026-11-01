# 一致性探针

开发者专用的探针工具。放在 `packages/` 之外，因此 wheel 不会带上它。

[English](README.md) | 中文

```bash
python -m tools.conformance run --suite drift --out /tmp/sdk-report.json
```

报告带 schema 版本（`schema_version: 1`），记录运行环境和每例一个结果，使用四种状态：

| 状态 | 含义 |
| --- | --- |
| `pass` | 探针运行了，期望成立 |
| `fail` | 探针运行了，结果错误 |
| `error` | 探针无法运行（缺模块、网络失败、超时） |
| `skip` | 此处不可比较（没有基线，或 SDK 已变） |

一例失败不影响其他例。瞬时错误可以用 `--retries` 重试。进入报告的每个字符串都经过三层脱敏：套件解析出的凭据原值（`--secret-env NAME`，可重复）、敏感映射键下的值（`x-api-key`、`authorization`、`token` 等）、以及与网关共用的凭据模式，再加上 `user:pass@` 和 `key=value` 参数。工具已知的凭据不会出现在任何位置，包括错误消息中的单纯提及。

## 套件

### `drift`

包装 [`scripts/dev/scan_sdk_types.py`](../../scripts/dev/scan_sdk_types.py)。
每个提供方一例。基线 `baselines/sdk-types.json` 中没有、而复刻类型中出现的字段导致该例失败。SDK 版本不同时跳过，因为基线只与产出它的 SDK 可比。

审查新的漂移后，刷新基线：

```bash
python -c "from tools.conformance.suites.drift import write_baseline; write_baseline()"
```

### `fields`

每个聊天格式的全部响应字段，取自提供方 SDK 类型
（`openai.types.chat.ChatCompletion`、`openai.types.responses.Response`、
`anthropic.types.Message`、`google.genai.types.GenerateContentResponse`），
Interactions API 则取自手写 schema。每个字段路径记入
`baselines/response-fields.json` 并带分类：

| 分类 | 含义 |
| --- | --- |
| `content` | 参与比较的轮次内容 |
| `envelope` | id、model、type、index 之类的元数据 |
| `terminal` | 轮次结束的原因 |
| `usage` | token 计数 |
| `provider_only` | 不参与比较的提供方元数据（必须写理由） |
| `echoed_request` | Responses 回显的请求配置（必须写理由） |
| `not_on_wire` | 仅 SDK 使用的辅助项（必须写理由） |
| `not_compared` | 工具不覆盖的联合分支（必须写理由） |

SDK 声明了、而台账没有的字段导致该例失败；台账有、而 SDK 不再声明的字段只报告，不失败。联合分支按同样方式分类，因此新的输出项或内容块可见，而无需枚举工具不覆盖的类型的每个深层字段。

```bash
python -m tools.conformance run --suite fields --out /tmp/fields-report.json
```

审查变更后，刷新台账：

```bash
python -c "from tools.conformance.suites.fields import write_baseline; write_baseline()"
```

新字段以 `unclassified` 写入，因此套件会持续失败，直到有人分类。`write_baseline(seed=True)` 改为填入启发式分类；只在首次建立台账时使用。

台账记录生成时用的 SDK 版本，但版本只作来源标记，不是判定条件：字段集无论版本都比较，因此上游新增字段是一个有名字的失败，并给出可审查的差异。SDK 未安装的格式跳过；台账缺失或改名则失败，因为那是仓库状态，不是环境属性。该套件在 CI 的 `conformance` 任务中运行，该任务安装 `[all]`。

### `request-fields`

每个聊天格式的全部请求字段，取自提供方 SDK 的请求类型
（`CompletionCreateParamsNonStreaming`、`ResponseCreateParamsNonStreaming`、
`MessageCreateParamsNonStreaming`、`GenerateContentConfig` 加 `contents[]`），
Interactions API 则取自手写 surface。每个字段路径记入
`baselines/request-fields.json`，并写明读入方向
（`request_from_provider`）如何处理它：

| 分类 | 含义 |
| --- | --- |
| `mapped` | 读入方把它存入 IR；IR 无处安放、但同格式往返必须保留的，存入 `provider_extensions` |
| `dropped` | 读入方忽略它（必须写理由） |
| `not_on_wire` | 仅 SDK 使用的辅助项，例如 `http_options`，从不发送 |

格式声明了、而台账没有记录的字段导致该例失败；台账记录了、而格式不再声明的字段只报告，不失败。新路径以 `unclassified` 写入，并持续失败，直到有人读过读入方。分类不能从类型推导，因为读入方可能直接读键，也可能通过名字列表读。

```bash
python -m tools.conformance run --suite request-fields --out /tmp/request-fields-report.json
```

审查并分类变更后，刷新台账：

```bash
python -c "from tools.conformance.suites.request_fields import write_baseline; write_baseline()"
```

台账按 `conformance` 任务解析出的 SDK 版本生成，并记入 `sdk_versions`。上游新增字段是一个有名字的失败，版本偏差由此被发现。

### `stream-events`

流没有单一可遍历的声明类型，而单位不能是一个事件：在空的 `StreamContext` 中，`finish`、`usage`、`stream_end` 不产出任何东西，因为它们需要先有开启的响应或已开始的块。因此单位是一个序列。

一条规范 IR 流（开启、推理块、带 refusal 和 citation 的文本块、工具调用及其结果、结束、用量）依次写入五个聊天方言，再读回，结果记入 `baselines/stream-events.json`：

| 键 | 含义 |
| --- | --- |
| `chunks` | 写入方产生的 wire 消息条数 |
| `recovered` | 读入方产生的 IR 事件类型，按顺序 |

任何一格变化都导致该例失败：恢复的类型消失是丢失，出现是值得记录的改进，重排同样是变化。

```bash
python -m tools.conformance run --suite stream-events --out /tmp/stream-events-report.json
```

审查变更后，刷新台账：

```bash
python -c "from tools.conformance.suites.stream_events import write_baseline; write_baseline()"
```

### `wire`

一条规范请求，依次经五个聊天格式往返（`IR → 目标 body → IR`），每个格式两例：

```bash
python -m tools.conformance run --suite wire --out /tmp/wire-report.json
```

| 例 | 请求 |
| --- | --- |
| `request/<format>` | instructions、一个用户轮次、一个 function 工具、`tool_choice: auto`、`temperature`/`top_p`/`max_tokens` |
| `sampling/<format>` | 同上，另加其余全部 `GenerationConfig` 开关：`top_k`、`stop_sequences`、`seed`、`logprobs`、`top_logprobs` |

这两类是**同一性**往返（格式 → 自身的 body → 返回），不是 25 个有序格式对；跨格式转换由转换器测试套件覆盖。每例报告往返丢失了什么，以及哪些转换警告提到了它。

只有丢失了没有任何转换警告解释的字段时，该例才**失败**。「解释」按警告的 `subject` 判断：它必须点名该字段（或承载它的容器）。文字描述只是后备，且仅用于尚无 subject 的上报者，且仅用于消失的值——*变化*的值需要有点名它的警告。这样该套件只断言它在不知道各家 surface 时能断言的那一条：调用方被告知的丢失是数据，静默的丢失是红色用例。

规范请求刻意落在共享核内，因此出现差异说明该格式没有携带它应当携带的内容，而不是请求索要了扩展。该套件离线且确定；上游 wire 行为（厂商接受并回显什么）是下一类用例，需要 epic 计划中的、带凭据的定时运行。

`baselines/wire.json` 记录每例当前丢失什么。出现的、基线中没有的丢失导致该例失败，并给出变化的路径；消失的丢失作为 `restored` 报告，不失败。审查并确认变更后：

```bash
python -c "from tools.conformance.suites.wire import write_baseline; write_baseline()"
```

新增一例是两步改动：基线中没有条目时，它报告的每个丢失都是「新」的，因此首次运行必然失败——这正是目的，因为用例必须经过审查，而不是默认可信。先运行一次，确认这些丢失就是你想要的，再重建基线。

CI 在每个触及核心的 PR 上运行该套件，并上传报告。
