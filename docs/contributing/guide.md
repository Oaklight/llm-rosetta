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

## 添加新的转换器

LLM-Rosetta 使用轴辐式（hub-and-spoke）架构。要添加对新 API 标准的支持：

1. 在 `src/llm_rosetta/converters/<name>/` 下创建转换器目录
2. 实现与 IR 之间的双向转换（请求/响应）
3. 如有需要，在 `src/llm_rosetta/shims/builtins.py` 中添加 shim
4. 在 `tests/converters/` 下添加测试
5. 提交 PR

可参考 `src/llm_rosetta/converters/` 下的现有转换器。

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
