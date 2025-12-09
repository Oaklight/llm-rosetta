# Provider Messages Typing Schemas 分析计划

## 项目目标

创建一组中间表示（Intermediate Representation, IR），用于在不同 LLM provider（OpenAI、Anthropic、Google GenAI）之间转换消息格式。

## 分析范围

### 目标类型定义

1. **OpenAI**: `ChatCompletionMessageParam`
2. **Anthropic**: `MessageParam`
3. **Google GenAI**: `ContentListUnionDict`

## 分析方法

### 第一阶段：类型定义提取

对每个 provider，我们将：

1. **定位源代码文件**

   - 在 conda 环境 `l_t_c` 中查找已安装包的位置
   - 定位类型定义文件（通常在 `types.py` 或 `_types.py` 中）

2. **提取完整类型结构**

   - 主类型定义
   - 所有相关的子类型和 Union 类型
   - TypedDict 定义
   - 字面量类型（Literal）

3. **记录关键信息**
   - 字段名称和类型注解
   - 必需字段 vs 可选字段（Required vs NotRequired）
   - 字段的 docstrings 和注释
   - 类型约束和验证规则

### 第二阶段：结构化文档

为每个 provider 创建独立的 Markdown 文档，包含：

1. **类型层次结构**

   - 使用 Mermaid 图表展示类型继承和组合关系
   - 标注 Union 类型的各个分支

2. **详细字段说明**

   - 表格形式列出所有字段
   - 字段类型、是否必需、默认值、说明

3. **代码示例**

   - 展示实际的类型定义代码
   - 提供使用示例

4. **特殊注意事项**
   - 各 provider 的独特特性
   - 版本差异
   - 已知限制

### 第三阶段：对比分析

创建对比文档，识别：

1. **共同特性**

   - 所有 provider 都支持的消息类型（如 text、image）
   - 通用的字段（如 role、content）
   - 相似的结构模式

2. **差异点**

   - 独有的消息类型（如 tool_use、function_call）
   - 字段命名差异
   - 结构组织差异
   - 类型表达方式差异

3. **转换挑战**
   - 不可直接映射的特性
   - 需要特殊处理的情况
   - 信息丢失的风险点

### 第四阶段：IR 设计

基于对比分析，设计中间表示：

1. **核心原则**

   - 保留所有 provider 的关键信息
   - 支持双向转换
   - 类型安全
   - 易于扩展

2. **设计考虑**

   - 使用 TypedDict 还是 dataclass
   - 如何处理 provider 特有特性
   - 版本兼容性策略
   - 验证和错误处理

3. **转换策略**
   - IR → Provider 的映射规则
   - Provider → IR 的解析规则
   - 边界情况处理
   - 降级策略（当目标 provider 不支持某特性时）

## 文档结构

```
docs/provider_messages_typing_schemas/
├── README.md                    # 本文件：总体规划
├── openai.md                    # OpenAI类型定义详解
├── anthropic.md                 # Anthropic类型定义详解
├── google.md                    # Google GenAI类型定义详解
├── comparison.md                # 三家对比分析
└── ir_design.md                 # 中间表示设计文档
```

## 预期产出

1. **完整的类型定义文档**（3 个 provider 各一份）
2. **对比分析报告**
3. **IR 设计规范**
4. **转换策略文档**
5. **实现建议和最佳实践**

## 下一步行动

按照 todo list 的顺序，首先从 OpenAI 开始：

1. 定位类型定义文件
2. 提取完整结构
3. 创建文档

然后依次处理 Anthropic 和 Google，最后进行对比和 IR 设计。
