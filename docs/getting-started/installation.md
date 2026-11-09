---
title: 安装
---

# 安装

!!! info "环境要求"
    需要 Python **≥ 3.10**。Python 3.11+ 上没有第三方运行时依赖；Python 3.10 上也只需要 `typing_extensions`，即标准库 `typing` 的向后移植包（backport）。

## 库

```bash
pip install llm-rosetta
```

## 网关

### pip

```bash
pip install llm-rosetta-gateway
```

网关作为独立发行版发布，并依赖一个兼容的 `llm-rosetta` 核心版本。
`pip install "llm-rosetta[gateway]"` 仍可作为过渡别名使用，会同时安装两者。

网关**零外部运行时依赖** — 使用内置的纯标准库 HTTP 服务器和客户端模块。

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

#### 从源码构建 { #building-from-source }

上面的镜像安装的是已发布的 wheel。若要从工作树直接运行网关，`docker/docker-compose.source.yaml` 会构建 `docker/Dockerfile.source`，直接安装 `packages/llm-rosetta` 与 `packages/llm-rosetta-gateway` 两个发行版：

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
mkdir -p config   # 容器以 uid 1000 运行，无法自行创建该目录
docker compose -f docker/docker-compose.source.yaml up -d --build
```

不使用 Compose：

```bash
docker build -f docker/Dockerfile.source -t llm-rosetta-gateway:src .
mkdir -p config
docker run -p 8765:8765 -v "$PWD/config:/config" llm-rosetta-gateway:src
```

源码版挂载仓库根目录的 `config/`，`PUID`、`PGID` 与 `environment:` 的用法与镜像版一致；两个文件是两个入口 —— `docker/docker-compose.yaml` 运行已发布的镜像，`docker/docker-compose.source.yaml` 从工作树构建 —— 且都发布 8765 端口，一次只跑一个。首次启动前先创建 `config/`：缺失的 bind mount 目录会被 compose 以 `root` 创建，而网关以 uid 1000 运行，首次写入 `config.jsonc` 会失败。一个由你（或你传入的 `PUID`）拥有空目录即可。

发布路径不变：`make build-docker` 仍构建 `docker/Dockerfile`，从 `dist/` 或 PyPI 安装 wheel。

## 开发

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
pip install -e "packages/llm-rosetta[conformance]" -e packages/llm-rosetta-gateway
```
