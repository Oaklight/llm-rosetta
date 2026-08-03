---
title: 贡献指南
---

# 为 LLM-Rosetta 做贡献

## 开始

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feature/my-change`）
3. 安装开发依赖：`pip install -e ".[all]"`
4. 设置 pre-commit hooks：`pre-commit install`
5. 进行修改
6. 运行 `make lint`（或在 commit 时由 pre-commit 自动检查）
7. 运行 `make test` 确保没有破坏现有功能
8. Commit 并 push
9. 提交 Pull Request

## 分支命名

使用描述性前缀：

- `feature/xxx` — 新功能
- `fix/xxx` — Bug 修复
- `refactor/xxx` — 代码重构
- `docs/xxx` — 文档更新
- `test/xxx` — 测试增加或修改

## Commit 消息

保持 commit 消息简洁，聚焦于 *为什么* 而非 *做了什么*。每个 commit 只包含一个逻辑变更。

## Pull Request

- 保持 PR 聚焦——每个 PR 只包含一个功能或修复
- 简要描述改了什么以及为什么
- 明确标注任何破坏性变更
- 提交前确保 `make lint` 通过
- Merge 策略：rebase（可使用 `scripts/merge-pr.sh` 进行本地 rebase 合并）

## 转换器架构

LLM-Rosetta 使用轴辐式（hub-and-spoke）架构，每个转换器负责特定 API 格式与共享 IR 之间的双向转换。

### 结构

每个转换器位于 `src/llm_rosetta/converters/<name>/` 下，继承 `BaseConverter`。基类使用**组合模式**——子类通过类属性声明各关注点的 ops 类：

```
converters/<name>/
├── converter.py      # 主转换器类（继承 BaseConverter）
├── content_ops.py    # 内容块转换（文本、图片、refusal 等）
├── message_ops.py    # 消息级转换（角色、多轮对话）
├── tool_ops.py       # 工具定义、工具调用、工具结果
├── config_ops.py     # 请求配置（temperature、top_p、流式选项）
└── _constants.py     # 格式特定的常量
```

为已有格式添加新功能（如新的内容类型、新的字段）时，应在该转换器**对应的 ops 模块**中实现，而非写在独立的临时代码中。尽可能复用 `converters/base/` 中的共享逻辑——基础模块（`content.py`、`messages.py`、`tools.py`、`reasoning.py`、`schema.py` 等）提供了所有转换器共享的通用构建块。

### 添加新的转换器

要添加对新 API 标准的支持：

1. 在 `src/llm_rosetta/converters/<name>/` 下创建转换器目录
2. 继承 `BaseConverter` 并实现所有抽象方法
3. 按照上述模式创建 ops 类
4. 在 `src/llm_rosetta/shims/providers/<name>/` 下添加 shim
5. 在 `tests/converters/` 下添加测试
6. 提交 PR

可参考现有转换器。

### Round-Trip 兼容性

所有转换路径必须保持 **round-trip 兼容性**。每个变更都必须针对以下场景进行测试：

- **A → IR → A**（同格式 round-trip）：转换为 IR 再转回相同格式，必须产生有效且语义等价的结果。不允许静默丢弃字段。
- **A → IR → B**（跨格式）：从一种格式转换为另一种格式，必须为目标格式产生有效输出，即使源格式存在没有直接对应的字段。
- **A → IR → B → IR → A**（完整 round-trip）：经过两次转换再转回的消息必须保持可用。这是网关的实际执行路径——请求在入站时转换，响应在出站时转换。

添加或修改转换器逻辑时，至少要编写覆盖前两个场景的测试。网关的跨格式路由依赖于所有转换器在 IR 语义上的一致性——一个转换器的 IR 输出出错，会导致其他所有转换器的级联失败。

## AI 辅助贡献

欢迎使用 AI 工具（如 Claude、Cursor、Copilot）辅助开发，但请注意：

- **Commit 中不添加 AI co-author 标记。** 不要在 git commit 消息中为 AI 工具添加 `Co-authored-by` 行，保持 git 历史干净可读。
- **在 PR 描述中说明。** 如果 AI 工具在你的贡献中有较大参与，在 PR 描述中简要说明（如 "AI was used to assist with implementation"）。
- **你对代码负全责。** 贡献者对提交的所有 AI 生成的代码负全部责任——需审查、测试、理解后再提交。

## 代码风格

- Python 代码遵循 `ruff` 默认规则
- Docstring 使用 Google 风格
- 注释和 docstring 使用英文
- 鼓励使用 type hints
- 不要编辑 `src/llm_rosetta/_vendor/` 下的文件——这些文件由外部管理

## 许可证

参与贡献即表示你同意你的贡献将按 [MIT 许可证](https://github.com/Oaklight/llm-rosetta/blob/master/LICENSE) 授权。
