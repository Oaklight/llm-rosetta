---
title: 风格指南
---

# 风格指南

## Python 代码

- 代码遵循 `ruff` 默认规则——运行 `make lint` 或在 commit 时由 pre-commit 自动检查
- 鼓励在所有公开 API 上使用 type hints
- 不要编辑 `src/llm_rosetta/_vendor/` 下的文件——这些文件由外部管理

## Docstring

- 使用 [Google 风格](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstring
- 所有公开函数和类应包含 docstring
- Docstring 和注释使用英文

## 命名约定

详细的变量、函数、类和文件命名规则参见[命名约定](naming-conventions.md)。

## 工具链

| 工具 | 用途 | 配置 |
|------|------|------|
| `ruff` | Lint + 格式化 | `pyproject.toml` |
| `ty` | 类型检查 | `pyproject.toml` |
| `pre-commit` | Git hooks（commit 时运行 ruff、ty） | `.pre-commit-config.yaml` |
| `complexipy` | 复杂度分析 | `pyproject.toml` |

### Pre-commit 工作流

Pre-commit hooks 在 `git commit` 时自动运行。如果 hook 修改了文件（如 `ruff format`），需重新 `git add` 后再次 commit。如果 hook 报错（如 `ty check`），需手动修复后重试。

!!! note
    项目使用 `language: system` hooks，依赖当前 shell `PATH` 中的工具。Commit 前请确保已激活项目对应的 conda 环境。
