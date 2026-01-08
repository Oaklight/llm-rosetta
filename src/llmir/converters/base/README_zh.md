# Base Converter Module - 轻量化模块架构

[English Version](./README_en.md) | [中文版](./README_zh.md)

## 概述

Base Converter Module 提供了转换器的抽象基类和轻量化模块架构，采用**组合优于继承**的设计模式，为所有 provider 转换器提供统一且灵活的接口规范。

## 文件结构

```
src/llmir/converters/base/
├── __init__.py          # 模块导出
├── converter.py         # BaseConverter 主转换器抽象基类
├── atomic_ops.py        # BaseAtomicOps 原子级转换操作抽象基类
├── complex_ops.py       # BaseComplexOps 复杂转换操作抽象基类
├── README_en.md         # 英语文档
└── README_zh.md         # 中文文档
```

## 模块说明

### `converter.py` - BaseConverter 主转换器抽象基类（轻量化）

- **职责**: 定义转换器的核心接口和通用逻辑
- **主要功能**:
  - `to_provider()`: 智能转换接口（IR → Provider）
  - `from_provider()`: 智能转换接口（Provider → IR）
  - `validate_ir_input()`: IR 输入验证
  - `atomic_ops_class`: 类属性，指定使用的原子操作类
  - `complex_ops_class`: 类属性，指定使用的复杂操作类

**重要变更**: 已移除所有原子级和复杂级的抽象方法，改用组合模式通过类属性指定 ops 类。

### `atomic_ops.py` - BaseAtomicOps 原子级转换操作抽象基类

- **职责**: 定义基础内容类型转换的统一接口
- **包含的抽象方法**:
  - 文本转换 (`ir_text_to_p()`, `p_text_to_ir()`)
  - 图像转换 (`ir_image_to_p()`, `p_image_to_ir()`)
  - 文件转换 (`ir_file_to_p()`, `p_file_to_ir()`)
  - 工具调用转换 (`ir_tool_call_to_p()`, `p_tool_call_to_ir()`)
  - 工具结果转换 (`ir_tool_result_to_p()`, `p_tool_result_to_ir()`)
  - 工具定义转换 (`ir_tool_to_p()`, `p_tool_to_ir()`)
  - 工具选择转换 (`ir_tool_choice_to_p()`, `p_tool_choice_to_ir()`)
  - 内容部分转换 (`p_content_part_to_ir()`)

### `complex_ops.py` - BaseComplexOps 复杂转换操作抽象基类

- **职责**: 定义消息、请求、响应级别转换的统一接口
- **包含的抽象方法**:
  - 消息级转换 (`ir_message_to_p()`, `p_message_to_ir()`)
  - 内容部分转换 (`ir_content_part_to_p()`)
  - 请求级转换 (`ir_request_to_p()`, `p_request_to_ir()`)
  - 响应级转换 (`ir_response_to_p()`, `p_response_to_ir()`)
  - 辅助方法 (`p_user_message_to_ir()`, `p_assistant_message_to_ir()`)

## 设计优势

1. **轻量化设计**: BaseConverter 从 298 行减少到约 100 行（减少 66%）
2. **组合优于继承**: 通过类属性指定 ops 类，而非强制实现抽象方法
3. **减少样板代码**: 每个 converter 减少约 60-80 行委托代码
4. **清晰的职责分离**: 原子操作、复杂操作、主转换器各司其职
5. **强类型约束**: 通过抽象基类确保所有实现的一致性
6. **可扩展性**: 新增 provider 时只需实现 ops 类并设置类属性
7. **可维护性**: 模块化设计便于理解、测试和维护
8. **灵活性**: 可以轻松切换或组合不同的 ops 实现

## 实现指南

### 创建新的 Provider 转换器

1. **创建 provider 目录**:

   ```
   src/llmir/converters/your_provider/
   ├── __init__.py
   ├── converter.py
   ├── atomic_ops.py
   └── complex_ops.py
   ```

2. **实现原子级操作**:

   ```python
   from ..base import BaseAtomicOps

   class YourProviderAtomicOps(BaseAtomicOps):
       @staticmethod
       def ir_text_to_p(text_part, **kwargs):
           # 实现文本转换逻辑
           pass

       # 实现其他抽象方法...
   ```

3. **实现复杂操作**:

   ```python
   from ..base import BaseComplexOps

   class YourProviderComplexOps(BaseComplexOps):
       @staticmethod
       def ir_message_to_p(message, ir_input, **kwargs):
           # 实现消息转换逻辑
           pass

       # 实现其他抽象方法...
   ```

4. **实现主转换器（轻量化方式）**:

   ```python
   from ..base import BaseConverter
   from .atomic_ops import YourProviderAtomicOps
   from .complex_ops import YourProviderComplexOps

   class YourProviderConverter(BaseConverter):
       # 设置ops类属性
       atomic_ops_class = YourProviderAtomicOps
       complex_ops_class = YourProviderComplexOps

       def to_provider(self, ir_data, **kwargs):
           # 直接调用ops类的静态方法
           converted, warnings = self.complex_ops_class.ir_message_to_p(message, **kwargs)
           # ... 实现转换逻辑
           pass

       def from_provider(self, provider_data, **kwargs):
           # 直接调用ops类的静态方法
           ir_message = self.complex_ops_class.p_message_to_ir(provider_message)
           # ... 实现转换逻辑
           pass
   ```

   **注意**: 不再需要实现大量委托方法，直接使用`self.atomic_ops_class`和`self.complex_ops_class`调用 ops 方法即可。

## 现有实现

### 已重构（轻量化架构）

- ✅ **OpenAI Chat Converter**: 已采用轻量化架构
  - [`OpenAIChatConverter`](../openai_chat/converter.py): 主转换器（236 行，减少 28%）
  - [`OpenAIChatAtomicOps`](../openai_chat/atomic_ops.py): 原子级操作
  - [`OpenAIChatComplexOps`](../openai_chat/complex_ops.py): 复杂操作

### 待重构（传统架构）

- ⏳ **Anthropic Converter**: 使用传统 monolithic 架构
- ⏳ **Google Converter**: 使用传统 monolithic 架构
- ⏳ **OpenAI Responses Converter**: 使用传统 monolithic 架构

**注意**: 传统架构的 converter 仍然可以正常工作，可按需逐步迁移到轻量化架构。

## 双向转换支持

轻量化架构完全支持双向转换：

1. **IR → Provider**: `ir_*_to_p()` 系列方法
2. **Provider → IR**: `p_*_to_ir()` 系列方法

这使得 LLMIR 可以用于构建 AI API 桥接服务，在不同格式之间进行双向转换。

## 重构成果

通过采用轻量化架构，我们实现了：

- **代码量减少**: BaseConverter 减少 66%，OpenAIChatConverter 减少 28%
- **更清晰的架构**: 组合优于继承，职责更明确
- **100%向后兼容**: 所有 137 个测试通过
- **更易维护**: 减少样板代码，提高代码质量

详细的重构信息请参考项目文档。
