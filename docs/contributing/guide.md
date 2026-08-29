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
- 简要说明改了什么、为什么要改
- 有破坏性变更的话明确标出来
- 提交前跑一遍 `make lint`
- Merge 策略：rebase（可使用 `scripts/merge-pr.sh` 进行本地 rebase 合并）

## AI 辅助贡献

欢迎使用 AI 工具（如 Claude、Cursor、Copilot）辅助开发，但请注意：

- **Commit 中不添加 AI co-author 标记。** 不要在 git commit 消息中为 AI 工具添加 `Co-authored-by` 行，保持 git 历史干净可读。
- **在 PR 描述中说明。** 如果 AI 工具在你的贡献中有较大参与，在 PR 描述中简要说明（如 "AI was used to assist with implementation"）。
- **你对代码负全责。** 贡献者对提交的所有 AI 生成的代码负全部责任——需审查、测试、理解后再提交。

## 许可证

参与贡献即表示你同意你的贡献将按 [MIT 许可证](https://github.com/Oaklight/llm-rosetta/blob/master/LICENSE) 授权。

## 下一步

- [风格指南](style-guide.md) — 代码风格、命名、docstring 和工具链
- [架构指南](architecture.md) — 转换器结构、ops 模块和 round-trip 兼容性
