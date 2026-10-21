---
title: 升级
---

# 升级网关

网关是独立的发行包 `llm-rosetta-gateway`，依赖转换核心 `llm-rosetta`。两者从同一个仓库构建、共用同一个版本号，因此"哪些版本配套"只有一个答案：相同版本号。

## 版本策略

- **1.0 之前保持同步版本号。** 发布 `v0.15.0` 即同时发布 `llm-rosetta 0.15.0` 与 `llm-rosetta-gateway 0.15.0`；发布流程会拒绝两个 `__version__` 与 tag 不一致的情况。
- **同 minor 兼容。** 网关声明 `llm-rosetta>=0.14.0.dev0,<0.15`：核心的 patch 版本无需新网关即可被接受；核心升 minor 则需要配套的网关发布。
- **dev 构建**（`0.14.0.devN`）发布到 TestPyPI 与 Docker 的 `-dev` 标签，用于验证发布线，不用于生产。

## 用 Docker 升级

镜像内两个包版本相同，因此升级镜像就是升级整套——不需要第二步。

```bash
docker pull oaklight/llm-rosetta-gateway:0.15.0-python
docker compose pull && docker compose up -d
```

- 生产环境请固定到精确 tag；`:latest-python` 会跟随最新发布，适合测试机，生产上容易出乎意料。
- 生产环境不要使用 `-dev` 标签：它跟随开发分支，而不是某个发布版本。

## 用 pip 升级

```bash
pip install -U "llm-rosetta[gateway]"    # 网关 + 其兼容的核心
pip install -U llm-rosetta-gateway       # 只升网关；核心只在区间要求时才动
pip install -U llm-rosetta               # 只升核心（库用户）
```

`llm-rosetta[gateway]` 是一个发行周期的元包别名：它解析出网关包，再由网关拉取它测试过的核心版本。因此"升级整套"的惯用命令是第一条。

如果当初只安装了网关，单独升级它就够了：pip 会解析出网关区间允许的最新核心。单独升级核心在网关区间接受时是安全的；若不被接受，pip 会报冲突，而不会配出从未测试过的组合。

## 确认实际运行的版本

```bash
llm-rosetta-gateway --version
# llm-rosetta-gateway 0.15.0 (llm-rosetta 0.15.0)
```

```bash
curl -s http://localhost:8765/health | jq .versions
# {"gateway": "0.15.0", "core": "0.15.0"}
```

启动 banner 会打印同一对版本号；管理面板的 **设置** 中显示网关版本，两者不一致时会追加 `· core v…`。

## 回滚

- **pip：** 显式安装一对版本，避免解析器留下"新核心 + 旧网关"：

  ```bash
  pip install "llm-rosetta-gateway==0.14.0" "llm-rosetta==0.14.0"
  ```

- **Docker：** 把镜像 tag 改回去即可；镜像同时携带两个包，回滚是原子的。

配置文件不随包版本管理：网关在加载时会把旧配置迁移到当前结构（并在旁边保留 `.pre-migration.bak`）。如果要回滚到迁移之前的构建，请先确认升级时留下的备份。
