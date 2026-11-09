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

也可使用 Docker Compose — 参见仓库中的 `docker/docker-compose.yaml`。该文件把 `./config` 挂进容器（即 `docker/config`），首次启动前先创建它并交给容器运行的 uid：

```bash
mkdir -p docker/config && sudo chown "${PUID:-1000}:${PGID:-1000}" docker/config
docker compose -f docker/docker-compose.yaml up -d
```

#### 从源码构建 { #building-from-source }

上面的镜像安装的是已发布的 wheel。若要从工作树直接运行网关，`docker/docker-compose.source.yaml` 会构建 `docker/Dockerfile.source`，直接安装 `packages/llm-rosetta` 与 `packages/llm-rosetta-gateway` 两个发行版：

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
mkdir -p config   # 容器以 ${PUID:-1000} 运行，无法自行创建该目录
docker compose -f docker/docker-compose.source.yaml up -d --build
```

不使用 Compose：

```bash
docker build -f docker/Dockerfile.source -t llm-rosetta-gateway:src .
mkdir -p config
docker run --user "$(id -u):$(id -g)" -p 8765:8765 -v "$PWD/config:/config" llm-rosetta-gateway:src
```

源码版挂载仓库根目录的 `config/`。两个 compose 文件都通过 Compose 的 `user:` 指令以 `${PUID:-1000}:${PGID:-1000}` 运行容器：镜像声明的是 `USER appuser`（uid 1000），`user:` 会取代它，除此之外没有任何东西读取这对变量。挂载的 `config/` 必须对同一个 uid 可写 —— 首次启动前先创建，用户 id 不是 1000 时把它一起传进去：

```bash
PUID=$(id -u) PGID=$(id -g) docker compose -f docker/docker-compose.source.yaml up -d --build
```

缺失的 bind mount 目录会被 compose 以 `root` 创建，而首次运行要在其中写入 `config.jsonc`，所以目录属主不对会让容器在启动时停下：entrypoint 报两条 HOME 错误之一 —— 无法创建目录时为 `error: cannot create HOME=… as uid …`，目录已存在但该 uid 不可写时为 `error: HOME=… exists but is not writable by uid …` —— 并退出，而不是让网关在更晚处失败。不经 Compose 的 `docker run` 形式不读取 `PUID`，要像上面的例子那样用 `--user`。

非 1000 的 uid 在镜像里没有 passwd 条目，Docker 会把 `HOME` 留成 `/`，进程无法写入。源码版 compose 文件为这种情况传入 `HOME=/config/home`，由 entrypoint 创建；镜像版不传，交给 entrypoint 处理，因此旧镜像保持原有的 `HOME`。默认 uid 在两种情况下都仍是 `/home/appuser`。也就是说，`PUID`/`PGID` 取非 1000 需要一个包含这个 entrypoint 的镜像 —— 即本次改动之后发布的镜像；用旧 tag 配非 1000 的 uid，`HOME` 仍然是 `/`。Sign in with ChatGPT 的记录基于 `Path.home()`（`~/.config/llm-rosetta/siwc/…`），因此会落在 `config/` 下 —— 这也是该目录必须属于运行时 uid 的第二个原因。shim 归档缓存不受影响：它锚定在配置目录（`config/shims-cache`），与 `HOME` 是否可写无关。

两个文件是两个入口 —— `docker/docker-compose.yaml` 运行已发布的镜像，`docker/docker-compose.source.yaml` 从工作树构建 —— 且都发布 8765 端口，一次只跑一个。

发布路径不变：`make build-docker` 仍构建 `docker/Dockerfile`，从 `dist/` 或 PyPI 安装 wheel。

## 开发

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
pip install -e "packages/llm-rosetta[conformance]" -e packages/llm-rosetta-gateway
```
