---
title: 安装
---

# 安装

!!! info "环境要求"
    需要 Python **≥ 3.10**。核心库依赖极少（`typing_extensions>=4.0.0`）。

## 库

```bash
pip install llm-rosetta
```

### Provider SDK（可选）

如需直接调用 API，可安装对应 Provider 的 SDK：

```bash
# 单独安装
pip install openai
pip install anthropic
pip install google-genai

# 一次性安装全部
pip install "llm-rosetta[openai,anthropic,google]"
```

!!! note

    Provider SDK 仅在直接调用 API 时需要。LLM-Rosetta 的格式转换功能使用纯字典操作，不依赖 SDK。

## 网关

### pip

```bash
pip install "llm-rosetta[gateway]"
```

网关**零外部运行时依赖** — 使用内置的纯标准库 HTTP 服务器和客户端模块。

### 独立二进制文件

预编译的单文件可执行程序可从 [GitHub Releases](https://github.com/Oaklight/llm-rosetta/releases) 下载，无需 Python 运行时。

| 平台 | 文件名 |
|------|--------|
| Linux x86_64 (glibc) | `llm-rosetta-gateway-<ver>-linux-x86_64` |
| Linux x86_64 (musl) | `llm-rosetta-gateway-<ver>-linux-x86_64-musl` |
| Linux arm64 (glibc) | `llm-rosetta-gateway-<ver>-linux-arm64` |
| Linux arm64 (musl) | `llm-rosetta-gateway-<ver>-linux-arm64-musl` |
| macOS arm64 | `llm-rosetta-gateway-<ver>-macos-arm64` |
| Windows x86_64 | `llm-rosetta-gateway-<ver>-windows-x86_64.exe` |

```bash
# 下载后运行（Linux/macOS）
chmod +x llm-rosetta-gateway-*
./llm-rosetta-gateway-<ver>-linux-x86_64 --help
```

!!! tip

    Alpine 环境和 Docker 请使用 **musl** 版本。Ubuntu、Debian 等大多数 Linux 发行版请使用 **glibc** 版本。

### Docker

[DockerHub](https://hub.docker.com/r/oaklight/llm-rosetta-gateway) 提供三种镜像变体：

| 标签 | 基础镜像 | 大小 | 用途 |
|------|---------|------|------|
| `:<ver>` / `latest` | Alpine + 二进制 | ~21 MB | 默认，最小 |
| `:<ver>-glibc` | busybox:glibc + 二进制 | ~25 MB | 仅 glibc 环境 |
| `:<ver>-python` | python:alpine + pip | ~80 MB | 需要 pip 扩展 |

```bash
# 默认（Alpine，推荐）
docker pull oaklight/llm-rosetta-gateway:latest

# 挂载配置目录运行
docker run -p 8765:8765 -v /path/to/config:/config oaklight/llm-rosetta-gateway

# 自定义 UID/GID 映射
docker run --user $(id -u):$(id -g) -p 8765:8765 -v /path/to/config:/config oaklight/llm-rosetta-gateway
```

也可使用 Docker Compose — 参见仓库中的 `docker/docker-compose.yaml`。

## 开发

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
pip install -e ".[all]"
```
