# IR 设计架构讨论观察记录

本文档记录了关于 LLM Provider Converter 中间表示（IR）设计的深度讨论和关键观察。

## 讨论背景

在设计统一的中间表示（IR）以支持 OpenAI、Anthropic 和 Google GenAI 三家 provider 之间的消息转换时，我们需要决定：

1. 应该学习哪家 provider 的架构设计？
2. 是否需要独立的`tool`角色？
3. 采用扁平化还是嵌套的消息结构？

## 核心观察

### 观察 1：Anthropic 的内容块架构最优雅

**推荐理由**：

1. **极简角色系统**：只有`user`和`assistant`两种角色
2. **统一的内容块架构**：所有功能通过内容块实现
3. **高度一致性**：无论什么类型的内容，都遵循相同的结构模式

```python
# Anthropic的统一模式
{
    "role": "user" | "assistant",
    "content": [
        {"type": "text", "text": "..."},
        {"type": "image", "source": {...}},
        {"type": "tool_use", "id": "...", "name": "...", "input": {...}},
        {"type": "tool_result", "tool_use_id": "...", "content": "..."},
        {"type": "thinking", "thinking": "..."}
    ]
}
```

**优势**：

- 从 Anthropic 架构转换到其他 provider 更容易
- 内容块架构足够灵活，可以表达各种复杂场景
- 符合实际使用场景：一条消息可能包含多种类型的内容

### 观察 2：Anthropic 的 web_search 工具设计揭示了关键原则

Anthropic 对内部工具（如`web_search`）采用了**特殊的内容块类型**：

```python
# 服务器端工具调用
class ServerToolUseBlockParam(TypedDict, total=False):
    type: Required[Literal["server_tool_use"]]  # 特殊的type
    name: Required[Literal["web_search"]]
    id: Required[str]
    input: Required[Dict[str, object]]

# 服务器端工具结果
class WebSearchToolResultBlockParam(TypedDict, total=False):
    type: Required[Literal["web_search_tool_result"]]  # 特殊的type
    tool_use_id: Required[str]
    content: Required[...]
```

**关键启示**：

- 通过**不同的`type`值**区分工具来源（普通工具 vs 服务器工具）
- 即使都在`user`角色中，通过`type`字段就能清楚区分内容性质
- **不需要 tool 角色也能保持清晰性**

**工具结果和用户输入可以混合**：

```python
{
    "role": "user",
    "content": [
        {"type": "tool_result", "tool_use_id": "...", "result": "..."},
        {"type": "web_search_tool_result", "tool_use_id": "...", "content": {...}},
        {"type": "text", "text": "另外，我还想知道..."}
    ]
}
```

### 观察 3：OpenAI Responses API 的重大架构变化

OpenAI 的新 Responses API 采用了与 Anthropic 类似的设计理念，这是一个**重要的行业趋势信号**。

#### 3.1 从角色导向转向类型导向

**旧的 Chat Completions API**：

```python
# 基于角色的设计
{"role": "user", "content": "..."}
{"role": "assistant", "content": "..."}
{"role": "tool", "tool_call_id": "...", "content": "..."}  # 独立的tool角色
```

**新的 Responses API**：

```python
# 基于type的设计
ResponseInputItemParam = Union[
    EasyInputMessageParam,          # type: "message"
    ResponseFunctionToolCallParam,  # type: "function_call"
    FunctionCallOutput,             # type: "function_call_output"
    ShellCall,                      # type: "shell_call"
    McpCall,                        # type: "mcp_call"
    # ... 更多类型
]
```

**关键变化**：

- **不再有独立的`tool`角色**
- 工具调用和工具输出都是**独立的类型**，通过`type`字段区分
- 每种类型都是平等的输入项，不再嵌套在消息的角色中

#### 3.2 扁平化的输入结构

**旧 API（嵌套）**：

```python
messages = [
    {"role": "user", "content": "..."},
    {"role": "assistant", "tool_calls": [...]},  # 工具调用嵌套在assistant消息中
    {"role": "tool", "content": "..."}
]
```

**新 API（扁平）**：

```python
input_items = [
    {"type": "message", "role": "user", "content": "..."},
    {"type": "function_call", "name": "...", "arguments": "..."},  # 独立的输入项
    {"type": "function_call_output", "call_id": "...", "output": "..."}  # 独立的输入项
]
```

#### 3.3 角色的重新定义

**EasyInputMessageParam**（简化消息）：

```python
role: Required[Literal["user", "assistant", "system", "developer"]]
```

- 4 种角色，用于简单场景
- 向后兼容

**Message**（结构化消息）：

```python
role: Required[Literal["user", "system", "developer"]]
```

- **只有 3 种角色，不包括`assistant`**
- Assistant 的行为通过其他类型表示：
  - `type: "function_call"` - 工具调用
  - `type: "reasoning"` - 推理过程
  - `type: "text_response"` - 文本回复

**设计理念**：

- 角色只用于明确的"输入消息"
- Assistant 的行为被拆分为不同的类型
- 更细粒度的控制和类型安全

#### 3.4 MCP 工具的原生支持

```python
class McpCall(TypedDict, total=False):
    type: Required[Literal["mcp_call"]]
    id: Required[str]
    name: Required[str]
    server_label: Required[str]  # MCP服务器标签
    arguments: Required[str]
    approval_request_id: Optional[str]  # 批准请求
    status: Literal["in_progress", "completed", "incomplete", "calling", "failed"]
```

**关键点**：

- MCP 工具是**一等公民**，有专门的类型
- 支持批准流程
- 有独立的状态跟踪

## 关键问题的深度分析

### 问题 1：是否需要独立的`tool`角色？

#### 支持保留`tool`角色的理由

1. **语义清晰性**：角色层面一目了然
2. **可读性**：对话历史直观
3. **调试友好**：错误来源清晰
4. **OpenAI 兼容**：与旧版 OpenAI API 一致

#### 支持取消`tool`角色的理由

1. **结构简洁**：减少角色类型
2. **行业趋势**：
   - Anthropic 从未使用 tool 角色
   - Google 不使用 tool 角色
   - **OpenAI Responses API 也不使用 tool 角色**
3. **更灵活**：工具结果可以和用户输入混合
4. **可扩展性**：通过 type 细分不同类型的工具（如`server_tool_use`, `mcp_call`）

#### 实际场景对比

**方案 A：有 tool 角色**

```python
[
    {"role": "user", "content": [{"type": "text", "text": "北京天气如何？"}]},
    {"role": "assistant", "content": [{"type": "tool_use", ...}]},
    {"role": "tool", "content": [{"type": "tool_result", ...}]},  # 独立角色
    {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
]
```

**方案 B：无 tool 角色**

```python
[
    {"role": "user", "content": [{"type": "text", "text": "北京天气如何？"}]},
    {"role": "assistant", "content": [{"type": "tool_use", ...}]},
    {"role": "user", "content": [{"type": "tool_result", ...}]},  # user角色
    {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
]
```

**方案 C：扁平化（OpenAI Responses 风格）**

```python
[
    {"type": "message", "role": "user", "content": [{"type": "text", "text": "..."}]},
    {"type": "tool_call", "call_id": "...", "tool_name": "...", ...},  # 独立项
    {"type": "tool_result", "call_id": "...", "result": "..."},  # 独立项
    {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "..."}]}
]
```

### 问题 2：扁平化 vs 嵌套结构？

#### 扁平化结构（OpenAI Responses 风格）

**优点**：

- 每个项都是独立的，类型明确
- 易于添加新的项类型
- 状态跟踪更清晰
- 与 OpenAI Responses API 一致

**缺点**：

- 与 Anthropic/Google 的嵌套结构不一致
- 转换时需要拆分/合并

#### 嵌套结构（Anthropic 风格）

**优点**：

- 与 Anthropic/Google 一致
- 一条消息可以包含多种内容
- 更符合"对话"的概念

**缺点**：

- 工具调用嵌套在 assistant 消息中
- 不如扁平结构清晰

## 三家 Provider 架构对比总结

| 特性             | OpenAI Chat Completions | OpenAI Responses | Anthropic  | Google    |
| ---------------- | ----------------------- | ---------------- | ---------- | --------- |
| **角色数量**     | 6 种                    | 3-4 种           | 2 种       | 2 种      |
| **tool 角色**    | ✓                       | ✗                | ✗          | ✗         |
| **结构**         | 嵌套                    | 扁平             | 嵌套       | 嵌套      |
| **type 字段**    | 部分                    | 全面             | 全面       | 全面      |
| **工具调用位置** | 嵌套在 assistant        | 独立输入项       | 内容块     | Part 字段 |
| **MCP 支持**     | ✗                       | ✓ 原生           | ✗          | ✗         |
| **设计理念**     | 角色导向                | 类型导向         | 内容块导向 | Part 导向 |

## 初步建议

基于以上观察和分析，我们的初步建议是：

### 1. 核心架构：借鉴 Anthropic 的内容块设计

- 采用内容块架构作为核心
- 通过`type`字段区分不同类型的内容
- 保持结构的一致性和可扩展性

### 2. 角色系统：保留 3 种角色

```python
RoleType = Literal["system", "user", "assistant"]
```

- 不使用独立的`tool`角色
- 工具结果放在`user`消息的内容块中
- 与 Anthropic、Google 和 OpenAI Responses 一致

### 3. 结构选择：需要进一步讨论

两种方案都有优点：

**方案 A：嵌套结构（Anthropic 风格）**

```python
Message = {
    "role": "user" | "assistant" | "system",
    "content": List[ContentPart]
}
```

**方案 B：扁平结构（OpenAI Responses 风格）**

```python
InputItem = Union[
    Message,
    ToolCall,
    ToolResult,
    McpCall,
    ...
]
```

### 4. MCP 工具：提供一等支持

- 专门的`mcp_call`和`mcp_result`类型
- 支持批准流程
- 与 OpenAI Responses API 对齐

## 待决策的问题

1. **扁平化 vs 嵌套**：哪种结构更适合我们的场景？
2. **assistant 角色的使用**：
   - 是否只用于文本消息？
   - 工具调用是否需要角色？
3. **转换策略**：
   - 如何在扁平和嵌套之间转换？
   - 如何处理不兼容的特性？

## 下一步

1. 确定最终的结构选择（扁平 vs 嵌套）
2. 设计详细的类型定义
3. 制定转换规则
4. 编写转换示例

## 参考文档

- [comparison.md](comparison.md) - 三家 provider 的详细对比
- [anthropic.md](anthropic.md) - Anthropic 类型定义
- [openai_responses.md](openai_responses.md) - OpenAI Responses API 类型定义
- [ir_design.md](ir_design.md) - 初版 IR 设计
- [ir_design_revised.md](ir_design_revised.md) - 修订版 IR 设计
