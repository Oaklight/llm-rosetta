---
title: 安装
---

# 安装

## 库

```bash
pip install llm-rosetta
```

核心库仅有最少依赖（`typing_extensions>=4.0.0`）。

### 提供商 SDK（可选）

如需直接调用提供商 API，请安装相应的 SDK：

```bash
# 单独安装
pip install openai
pip install anthropic
pip install google-genai

# 一次安装所有
pip install "llm-rosetta[openai,anthropic,google]"
```

!!! note

    提供商 SDK 仅在直接调用 API 时需要。LLM-Rosetta 的转换函数使用纯字典，不依赖 SDK。

## 网关

```bash
pip install "llm-rosetta[gateway]"
```

网关**无外部运行时依赖** — 使用内嵌的仅标准库 HTTP 服务器和客户端模块。

### Docker

DockerHub 上提供预构建镜像：

```bash
docker pull oaklight/llm-rosetta-gateway:latest
docker run -p 8765:8765 -v /path/to/config:/config oaklight/llm-rosetta-gateway
```

或使用 Docker Compose — 参见仓库中的 `docker/docker-compose.yaml`。

## 开发安装

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
pip install -e ".[all]"
```
