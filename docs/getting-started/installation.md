---
title: 安装
---

# 安装

## 基本安装

```bash
pip install llm-rosetta
```

核心库仅有最少依赖（`typing_extensions>=4.0.0`）。

## 提供商 SDK（可选）

如需直接调用提供商 API，请安装相应的 SDK：

```bash
# OpenAI
pip install openai

# Anthropic
pip install anthropic

# Google GenAI
pip install google-genai

# 所有提供商
pip install llm-rosetta[openai,anthropic,google]
```

!!! note

    提供商 SDK 仅在直接调用 API 时需要。LLM-Rosetta 的转换函数使用纯字典，不依赖 SDK。

## 开发安装

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
pip install -e ".[dev]"
```
