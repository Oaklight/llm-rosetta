# 用户指南

本指南提供了 LLMIR 的详细使用说明和最佳实践。

## 核心概念

### 中间表示 (IR)

LLMIR 使用统一的中间表示来处理不同提供商的消息格式。这种设计允许：

- 无损转换：保持消息的完整性
- 扩展性：轻松添加新的提供商支持
- 一致性：统一的数据结构和处理逻辑

### 转换器

转换器是 LLMIR 的核心组件，负责在特定提供商格式和中间表示之间进行转换：

- `to_ir()`: 将提供商格式转换为中间表示
- `from_ir()`: 将中间表示转换为提供商格式

## 主要功能

### [转换器](converters/)

了解如何使用不同的转换器：

- [OpenAI Chat](converters/openai-chat.md) - OpenAI Chat Completions API
- [OpenAI Responses](converters/openai-responses.md) - OpenAI Responses API
- [Anthropic](converters/anthropic.md) - Claude 消息格式
- [Google](converters/google.md) - Gemini/PaLM 消息格式

### [中间表示](intermediate-representation.md)

深入了解 LLMIR 的中间表示设计和数据结构。

### [自动检测](auto-detection.md)

学习如何使用自动检测功能来识别消息格式。

## 最佳实践

### 性能优化

- 重用转换器实例以避免重复初始化
- 对于大量消息，考虑批量处理
- 使用适当的配置选项来平衡性能和功能

### 错误处理

- 始终处理转换可能出现的异常
- 使用验证功能确保输入数据的正确性
- 记录转换过程中的警告和错误

### 类型安全

- 使用类型注解来提高代码质量
- 利用 IDE 的类型检查功能
- 遵循 LLMIR 的类型定义

## 常见用例

1. **多提供商支持**: 在应用中同时支持多个 LLM 提供商
2. **格式标准化**: 将不同格式的消息统一为标准格式
3. **数据迁移**: 在不同提供商之间迁移对话数据
4. **测试和开发**: 使用统一格式进行测试和开发

## 下一步

- [查看具体的转换器文档](converters/)
- [了解 API 参考](../api/)
- [查看实际示例](../examples/)