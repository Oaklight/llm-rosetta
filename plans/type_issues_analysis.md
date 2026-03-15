# ty 类型检查问题深度分析报告

## 概述

本报告对 llm-rosetta 项目的 ty 类型检查结果进行深度分析，将 645 个诊断（644 错误 + 1 警告）按根因分类，并提出系统性的修复策略。

## 错误类型分布回顾

| 错误类型 | 数量 | 占比 |
|---------|------|------|
| `invalid-key` | 469 | 72.7% |
| `invalid-return-type` | 72 | 11.2% |
| `invalid-argument-type` | 67 | 10.4% |
| `unresolved-attribute` | 16 | 2.5% |
| `invalid-method-override` | 7 | 1.1% |
| `missing-typed-dict-key` | 6 | 0.9% |
| `invalid-assignment` | 4 | 0.6% |
| `no-matching-overload` | 2 | 0.3% |
| `invalid-type-form` | 1 | 0.2% |

---

## 根因分类

### 根因 1: IRStreamEvent 联合类型的类型收窄失败

**错误数量**: ~400+ 个（占 `invalid-key` 错误的绝大部分）

**问题描述**:

代码使用 `IRStreamEvent` 联合类型（包含 10 种事件类型），然后通过字符串比较判断具体类型，再访问该类型特有的字段。但 ty 无法通过字符串比较进行类型收窄。

**典型代码模式**:

```python
# converters/anthropic/converter.py:643
event_type = ir_event["type"]
if event_type == "stream_start":
    context.response_id = ir_event["response_id"]  # 错误：ty 认为 ir_event 仍是联合类型
    context.model = ir_event["model"]              # 错误：其他事件类型没有 model 字段
```

**ty 报错示例**:

```
error[invalid-key]: Unknown key "response_id" for TypedDict `StreamEndEvent`
error[invalid-key]: Unknown key "response_id" for TypedDict `TextDeltaEvent`
error[invalid-key]: Unknown key "model" for TypedDict `FinishEvent`
...
```

**影响范围**:

| 文件 | 错误数 |
|------|--------|
| `converters/anthropic/converter.py` | ~100 |
| `converters/google_genai/converter.py` | ~80 |
| `converters/openai_chat/converter.py` | ~80 |
| `converters/openai_responses/converter.py` | ~100 |

**根本原因**:

1. TypedDict 联合类型的类型收窄需要特殊处理
2. ty 不支持通过 `dict["type"] == "xxx"` 进行类型收窄
3. 需要使用 TypeGuard 或 TypeIs 函数

---

### 根因 2: 函数返回 dict[str, Any] 而非声明的 TypedDict

**错误数量**: 72 个 `invalid-return-type`

**问题描述**:

函数声明返回 TypedDict（如 `GenerationConfig`、`IRRequest`、`IRResponse`），但实际构建时使用 `result: Dict[str, Any] = {}`，然后逐步填充字段。ty 无法将 `dict[str, Any]` 视为 TypedDict。

**典型代码模式**:

```python
# converters/anthropic/config_ops.py:102-135
def p_generation_config_to_ir(
    provider_config: Any, **kwargs: Any
) -> GenerationConfig:
    result: Dict[str, Any] = {}  # 声明为 Dict[str, Any]
    
    if "max_tokens" in provider_config:
        result["max_tokens"] = provider_config["max_tokens"]
    # ... 更多条件填充
    
    return result  # 错误：返回 dict[str, Any] 而非 GenerationConfig
```

**ty 报错示例**:

```
error[invalid-return-type]: Return type does not match returned value
   --> src/llm_rosetta/converters/anthropic/config_ops.py:135:16
    |
135 |         return result
    |                ^^^^^^ expected `GenerationConfig`, found `dict[str, Any]`
```

**影响范围**:

| 模块 | 受影响的方法 |
|------|-------------|
| `config_ops.py` (所有 provider) | `p_generation_config_to_ir`, `p_stream_config_to_ir`, `p_reasoning_config_to_ir`, `p_cache_config_to_ir`, `p_response_format_to_ir` |
| `converter.py` (所有 provider) | `request_from_provider`, `response_from_provider` |
| `tool_ops.py` (所有 provider) | `p_tool_definition_to_ir`, `p_tool_config_to_ir`, `p_tool_choice_to_ir` |

**根本原因**:

1. 使用 `Dict[str, Any]` 作为中间变量类型
2. TypedDict 的结构化类型系统要求精确匹配
3. 条件性填充字段导致无法静态验证

---

### 根因 3: 联合类型参数传递给期望具体类型的函数

**错误数量**: 67 个 `invalid-argument-type`

**问题描述**:

代码通过 `part.get("type")` 判断类型后调用对应的转换函数，但 ty 无法收窄类型，导致传递联合类型给期望具体类型的函数。

**典型代码模式**:

```python
# converters/anthropic/converter.py:357-365
for part in message.get("content", []):
    part_type = part.get("type")
    if part_type == "text":
        anthropic_content.append(self.content_ops.ir_text_to_p(part))
        # 错误：part 仍是 TextPart | ImagePart | ToolCallPart | ...
    elif part_type == "tool_call":
        anthropic_content.append(self.tool_ops.ir_tool_call_to_p(part))
        # 错误：part 仍是联合类型
```

**ty 报错示例**:

```
error[invalid-argument-type]: Argument to function `ir_text_to_p` is incorrect
   --> src/llm_rosetta/converters/anthropic/converter.py:359:80
    |
359 |     anthropic_content.append(self.content_ops.ir_text_to_p(part))
    |                                                            ^^^^ Expected `TextPart`, found `TextPart | ImagePart | ToolCallPart | ...`
```

**影响范围**:

- 所有 `converter.py` 的 `response_to_provider` 方法
- 所有 `message_ops.py` 的消息转换方法
- 所有 `_ir_message_to_p` 和 `_p_content_part_to_ir` 方法

**根本原因**:

1. 与根因 1 相同的类型收窄问题
2. 需要使用 TypeGuard 函数进行类型收窄

---

### 根因 4: TypedDict 定义与实际使用不匹配

**错误数量**: ~30 个（分布在多种错误类型中）

#### 子问题 4a: ImagePart/FilePart/AudioPart 字段访问方式不一致

**问题描述**:

代码直接访问 `ir_image["data"]` 和 `ir_image["media_type"]`，但 TypedDict 定义使用嵌套结构 `image_data: ImageData`。

**TypedDict 定义**:

```python
# types/ir/parts.py
class ImagePart(TypedDict):
    type: Required[Literal["image"]]
    image_url: NotRequired[str]
    image_data: NotRequired[ImageData]  # 嵌套结构
    detail: NotRequired[Literal["auto", "low", "high"]]
```

**实际代码**:

```python
# converters/google_genai/content_ops.py:88-90
return {
    "inline_data": {
        "mime_type": ir_image["media_type"],  # 错误：ImagePart 没有 media_type
        "data": ir_image["data"],              # 错误：ImagePart 没有 data
    }
}
```

#### 子问题 4b: CitationPart 的字典类型定义过于严格

**问题描述**:

`CitationPart` 的 `url_citation` 字段定义为 `Dict[Literal["start_index", "end_index", "title", "url"], Any]`，但实际构建时使用普通 dict，ty 无法匹配。

**TypedDict 定义**:

```python
class CitationPart(TypedDict):
    type: Required[Literal["citation"]]
    url_citation: NotRequired[
        Dict[Literal["start_index", "end_index", "title", "url"], Any]
    ]
```

**实际代码**:

```python
# converters/openai_chat/content_ops.py:277-282
return CitationPart(
    type="citation",
    url_citation={
        "start_index": provider_citation.get("start_index", 0),
        "end_index": provider_citation.get("end_index", 0),
        "title": provider_citation.get("title", ""),
        "url": provider_citation.get("url", ""),
    },  # 错误：dict[str, Any] 不匹配 Dict[Literal[...], Any]
)
```

#### 子问题 4c: AudioPart 返回值缺少必需字段

**问题描述**:

代码返回的 AudioPart 缺少必需的 `audio_id` 字段。

```python
# converters/google_genai/content_ops.py:231-235
return {
    "type": "audio",
    "url": None,                              # 错误：AudioPart 没有 url 字段
    "media_type": inline_data["mime_type"],   # 错误：AudioPart 没有 media_type 字段
}  # 错误：缺少必需的 audio_id 字段
```

---

### 根因 5: 方法重写签名不兼容

**错误数量**: 7 个 `invalid-method-override`

**问题描述**:

子类方法参数名与父类不同，违反 Liskov 替换原则。

**父类定义**:

```python
# converters/base/converter.py:169-173
@abstractmethod
def stream_response_from_provider(
    self,
    chunk: Dict[str, Any],  # 参数名为 chunk
    context: Optional[StreamContext] = None,
) -> List[IRStreamEvent]:
```

**子类实现**:

```python
# converters/anthropic/converter.py:435-439
def stream_response_from_provider(
    self,
    event: Dict[str, Any],  # 参数名为 event，与父类不同
    context: Optional[StreamContext] = None,
) -> List[IRStreamEvent]:
```

**影响范围**:

- `AnthropicConverter.stream_response_from_provider`
- `AnthropicConverter.stream_response_to_provider`
- `GoogleGenAIConverter.stream_response_to_provider`
- 其他 converter 的类似方法

---

### 根因 6: TypedDict 缺少必需键

**错误数量**: 6 个 `missing-typed-dict-key`

**问题描述**:

使用 `**kwargs` 解包创建 TypedDict 时，ty 无法验证所有必需键都存在。

**典型代码**:

```python
# converters/google_genai/tool_ops.py:246
return ToolCallPart(**tool_call_kwargs)
# 错误：可能缺少 tool_call_id, tool_input, tool_name, type
```

---

### 根因 7: 类型表达式格式错误

**错误数量**: 1 个 `invalid-type-form`

**问题描述**:

```python
T = TypeVar("T", bound=TypedDict)  # TypedDict 不能用作类型边界
```

**位置**: `types/ir/helpers.py`

---

### 根因 8: 其他小问题

#### 8a: 未使用的 type: ignore 注释

**错误数量**: 1 个警告

#### 8b: dict.get() 重载匹配问题

**错误数量**: 2 个 `no-matching-overload`

```python
# converters/anthropic/converter.py:290
reason_map.get(stop_reason_val, "stop")
# 错误：stop_reason_val 类型不匹配
```

#### 8c: 字典赋值类型不兼容

**错误数量**: 4 个 `invalid-assignment`

```python
# converters/anthropic/converter.py:724
result["index"] = context.current_block_index
# 错误：期望 str | dict[str, str | Unknown]，得到 int
```

---

## 修复优先级

### P0: 可能导致运行时错误的类型问题

| 问题 | 数量 | 说明 |
|------|------|------|
| 根因 4c: AudioPart 缺少必需字段 | 2 | 可能导致运行时 KeyError |
| 根因 6: TypedDict 缺少必需键 | 6 | 可能导致运行时 KeyError |

### P1: 影响 API 正确性的类型问题

| 问题 | 数量 | 说明 |
|------|------|------|
| 根因 4a: 字段访问方式不一致 | ~10 | TypedDict 定义与实际使用不匹配 |
| 根因 4b: CitationPart 类型定义过严 | ~5 | 类型定义需要调整 |
| 根因 5: 方法重写签名不兼容 | 7 | 违反 LSP，可能导致多态调用问题 |
| 根因 7: TypeVar bound 错误 | 1 | 类型系统基础问题 |

### P2: 类型系统无法正确推断但运行时正确的问题

| 问题 | 数量 | 说明 |
|------|------|------|
| 根因 1: 联合类型收窄失败 | ~400 | 需要添加 TypeGuard |
| 根因 2: 返回 dict 而非 TypedDict | 72 | 需要重构构建方式 |
| 根因 3: 联合类型参数传递 | 67 | 需要添加 TypeGuard |

### P3: 代码风格/最佳实践问题

| 问题 | 数量 | 说明 |
|------|------|------|
| 根因 8a: 未使用的 type: ignore | 1 | 清理即可 |
| 根因 8b/8c: 其他小问题 | 6 | 局部修复 |

---

## 修复策略

### 策略 1: 为 IRStreamEvent 添加 TypeGuard 函数（解决根因 1、3）

**目标**: 解决 ~470 个错误

**实现方案**:

在 `types/ir/type_guards.py` 中添加事件类型守卫函数：

```python
from typing import TypeGuard

def is_stream_start_event(event: IRStreamEvent) -> TypeGuard[StreamStartEvent]:
    return event.get("type") == "stream_start"

def is_text_delta_event(event: IRStreamEvent) -> TypeGuard[TextDeltaEvent]:
    return event.get("type") == "text_delta"

def is_tool_call_start_event(event: IRStreamEvent) -> TypeGuard[ToolCallStartEvent]:
    return event.get("type") == "tool_call_start"

# ... 为每种事件类型添加
```

**使用方式**:

```python
# 修改前
event_type = ir_event["type"]
if event_type == "stream_start":
    context.response_id = ir_event["response_id"]

# 修改后
if is_stream_start_event(ir_event):
    context.response_id = ir_event["response_id"]  # 现在 ty 知道这是 StreamStartEvent
```

### 策略 2: 为 ContentPart 添加 TypeGuard 函数（解决根因 3）

**目标**: 解决 ~50 个错误

**实现方案**:

```python
def is_text_part(part: ContentPart) -> TypeGuard[TextPart]:
    return isinstance(part, dict) and part.get("type") == "text"

def is_image_part(part: ContentPart) -> TypeGuard[ImagePart]:
    return isinstance(part, dict) and part.get("type") == "image"

def is_tool_call_part(part: ContentPart) -> TypeGuard[ToolCallPart]:
    return isinstance(part, dict) and part.get("type") == "tool_call"

# ... 为每种内容类型添加
```

### 策略 3: 重构 TypedDict 构建方式（解决根因 2）

**目标**: 解决 72 个错误

**方案 A: 使用 TypedDict 构造函数**

```python
# 修改前
def p_generation_config_to_ir(...) -> GenerationConfig:
    result: Dict[str, Any] = {}
    if "max_tokens" in provider_config:
        result["max_tokens"] = provider_config["max_tokens"]
    return result

# 修改后
def p_generation_config_to_ir(...) -> GenerationConfig:
    return GenerationConfig(
        max_tokens=provider_config.get("max_tokens"),
        temperature=provider_config.get("temperature"),
        # ... 显式列出所有字段
    )
```

**方案 B: 使用 cast（不推荐，但快速）**

```python
from typing import cast

def p_generation_config_to_ir(...) -> GenerationConfig:
    result: Dict[str, Any] = {}
    # ... 填充逻辑
    return cast(GenerationConfig, result)
```

**方案 C: 使用 TypedDict 的 total=False**

如果大部分字段是可选的，可以考虑将 TypedDict 定义为 `total=False`。

### 策略 4: 修复 TypedDict 定义（解决根因 4）

**4a: 统一 ImagePart/FilePart/AudioPart 的字段访问方式**

选项 1: 修改 TypedDict 定义，添加顶层字段

```python
class ImagePart(TypedDict):
    type: Required[Literal["image"]]
    image_url: NotRequired[str]
    image_data: NotRequired[ImageData]
    # 添加顶层字段作为便捷访问
    data: NotRequired[str]
    media_type: NotRequired[str]
```

选项 2: 修改代码，使用嵌套访问

```python
# 修改前
ir_image["data"]

# 修改后
ir_image.get("image_data", {}).get("data")
```

**4b: 修复 CitationPart 类型定义**

```python
# 修改前
url_citation: NotRequired[Dict[Literal["start_index", "end_index", "title", "url"], Any]]

# 修改后 - 使用更宽松的类型
url_citation: NotRequired[Dict[str, Any]]

# 或者定义专门的 TypedDict
class UrlCitation(TypedDict):
    start_index: int
    end_index: int
    title: str
    url: str

class CitationPart(TypedDict):
    type: Required[Literal["citation"]]
    url_citation: NotRequired[UrlCitation]
```

**4c: 修复 AudioPart 返回值**

```python
# 修改代码，确保返回正确的字段
return AudioPart(
    type="audio",
    audio_id=generate_audio_id(),  # 添加必需字段
)
```

### 策略 5: 统一方法签名（解决根因 5）

**实现方案**:

将所有子类的参数名统一为父类定义的名称：

```python
# 父类
def stream_response_from_provider(
    self,
    chunk: Dict[str, Any],
    context: Optional[StreamContext] = None,
) -> List[IRStreamEvent]:

# 子类 - 使用相同的参数名
def stream_response_from_provider(
    self,
    chunk: Dict[str, Any],  # 改为 chunk，而非 event
    context: Optional[StreamContext] = None,
) -> List[IRStreamEvent]:
```

### 策略 6: 修复 TypedDict 构造（解决根因 6）

**实现方案**:

避免使用 `**kwargs` 解包，显式构造 TypedDict：

```python
# 修改前
return ToolCallPart(**tool_call_kwargs)

# 修改后
return ToolCallPart(
    type="tool_call",
    tool_call_id=tool_call_kwargs["tool_call_id"],
    tool_name=tool_call_kwargs["tool_name"],
    tool_input=tool_call_kwargs["tool_input"],
)
```

### 策略 7: 修复 TypeVar bound（解决根因 7）

**实现方案**:

```python
# 修改前
T = TypeVar("T", bound=TypedDict)

# 修改后 - 使用 Mapping 或具体的基类
from typing import Mapping
T = TypeVar("T", bound=Mapping[str, Any])
```

---

## 建议的修复顺序

### 阶段 1: 基础设施（预计影响 ~500 个错误）

1. **添加 TypeGuard 函数** - 在 `types/ir/type_guards.py` 中添加所有需要的类型守卫
2. **修复 TypedDict 定义** - 调整 `ImagePart`、`FilePart`、`AudioPart`、`CitationPart` 的定义

### 阶段 2: 核心转换器（预计影响 ~100 个错误）

1. **重构 config_ops.py** - 使用 TypedDict 构造函数或 cast
2. **统一方法签名** - 修复所有 `invalid-method-override` 错误

### 阶段 3: 应用 TypeGuard（预计影响 ~400 个错误）

1. **修改 converter.py** - 在流式事件处理中使用 TypeGuard
2. **修改 message_ops.py** - 在内容部分处理中使用 TypeGuard

### 阶段 4: 清理（预计影响 ~10 个错误）

1. **修复小问题** - `no-matching-overload`、`invalid-assignment` 等
2. **移除未使用的 type: ignore**

---

## 风险评估

### 低风险修复

- 添加 TypeGuard 函数（不改变运行时行为）
- 统一方法签名（参数名变化不影响调用）
- 移除未使用的 type: ignore

### 中风险修复

- 修改 TypedDict 定义（可能影响其他使用该类型的代码）
- 使用 cast（隐藏潜在的类型问题）

### 高风险修复

- 重构 TypedDict 构建方式（需要大量代码修改）
- 修改字段访问方式（可能引入运行时错误）

---

## 总结

llm-rosetta 的 645 个类型错误主要源于以下几个核心问题：

1. **联合类型收窄** - ty 无法通过字符串比较收窄 TypedDict 联合类型（~470 个错误）
2. **TypedDict 构建方式** - 使用 `Dict[str, Any]` 构建然后返回（72 个错误）
3. **TypedDict 定义不一致** - 定义与实际使用不匹配（~30 个错误）

通过添加 TypeGuard 函数和调整 TypedDict 定义，可以解决绝大部分问题。建议按照上述阶段顺序进行修复，优先处理基础设施问题，然后逐步应用到各个模块。
