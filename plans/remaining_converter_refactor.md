# 剩余 Converter 重构计划

## 背景

LLM-Rosetta 项目正在将所有 converter 从旧的单文件架构重构为 Bottom-Up Ops Pattern。

### 完成状态

所有 4 个 converter 均已完成 Bottom-Up Ops Pattern 重构。

| Converter | PR | 状态 |
|-----------|-----|------|
| OpenAI Chat | PR #16 | ✅ 已完成 |
| Anthropic | PR #22 | ✅ 已完成 |
| Google GenAI | PR #23 | ✅ 已完成 |
| OpenAI Responses | PR #24 | ✅ 已完成 |

## 重构模式（参考已完成的实现）

每个 converter 拆分为 5 个文件：

```
src/llm-rosetta/converters/{provider}/
├── __init__.py          # 导出所有类
├── content_ops.py       # 继承 BaseContentOps
├── tool_ops.py          # 继承 BaseToolOps
├── message_ops.py       # 继承 BaseMessageOps（组合 content_ops + tool_ops）
├── config_ops.py        # 继承 BaseConfigOps
└── converter.py         # 继承 BaseConverter（组合 4 个 Ops + 流式方法）
```

测试结构：

```
tests/converters/{provider}/
├── __init__.py
├── test_content_ops.py
├── test_tool_ops.py
├── test_message_ops.py
├── test_config_ops.py
└── test_converter.py

tests/integration/
├── test_{provider}_sdk_e2e.py
└── test_{provider}_rest_e2e.py
```

## 已完成的 Subtask

### Subtask 1: Google GenAI Converter 重构 ✅

- PR #23 已合并
- 目录重命名 `google/` → `google_genai/` 完成
- 4 个 Ops 文件 + converter.py 重写完成
- 分层测试完成

### Subtask 2: OpenAI Responses Converter 重构 ✅

- PR #24 已合并
- 4 个 Ops 文件 + converter.py 重写完成
- 分层测试完成

### Subtask 3: 清理工作 ✅

- 删除 `src/llm-rosetta/utils/` 目录和 `tests/utils/` 目录
- 删除 `src/llm-rosetta/types/providers/` 空目录
- 更新 `plans/architecture.md`（移除 DEPRECATED 标记，更新重构状态表）
- 更新 `src/llm-rosetta/converters/__init__.py`（恢复导出所有 converter）
- 更新 `src/llm-rosetta/__init__.py`（恢复导出所有 converter）

## 关键参考文件

- **架构设计**：`plans/architecture.md`
- **Base ABC**：`src/llm-rosetta/converters/base/`
- **OpenAI Chat 参考实现**：`src/llm-rosetta/converters/openai_chat/`
- **Anthropic 参考实现**：`src/llm-rosetta/converters/anthropic/`
- **IR 类型**：`src/llm-rosetta/types/ir/`
- **流式事件类型**：`src/llm-rosetta/types/ir/stream.py`