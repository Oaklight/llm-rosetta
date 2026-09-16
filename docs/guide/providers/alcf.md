# ALCF

ALCF（阿贡领先计算设施）推理服务通过三个 shim 端点提供支持，每个计算集群对应一个端点。所有端点均使用兼容 OpenAI Chat Completions 的 API。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 | 硬件 |
|:---|:---|:---|:---|:---|
| `alcf--metis` | `openai_chat` | `https://inference-api.alcf.anl.gov/resource_server/metis/api/v1` | `ALCF_API_KEY` | SambaNova SN40L |
| `alcf--minerva` | `openai_chat` | `https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1` | `ALCF_API_KEY` | NVIDIA B200 |
| `alcf--sophia` | `openai_chat` | `https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1` | `ALCF_API_KEY` | vLLM on A100 |

## 推理支持

Sophia 集群上在 [ALCF 模型列表](https://docs.alcf.anl.gov/services/inference-endpoints/#available-models)中标注 **R** 的模型支持推理（如 `openai/gpt-oss-120b`、`openai/gpt-oss-20b`、`google/gemma-4-31B-it`、`google/gemma-4-E4B-it`）。

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `reasoning_effort` |
| Effort 范围 | 完整 IR 阶梯（`minimal` → `max`） |

Metis 和 Minerva 目前没有支持推理的模型。

## 能力标志

| 标志 | `alcf--sophia` | `alcf--minerva` | `alcf--metis` |
|:---|:---:|:---:|:---:|
| 自定义工具 | ✅ | ✅ | — |

!!! note "Metis 工具调用"
    SambaNova 存在[已知的工具调用清理问题](https://docs.alcf.anl.gov/services/inference-endpoints/#metis-tool-calling)，可能导致不完整的函数调用。因此 Metis 不声明工具支持。

## 转换规则

三个集群共享一组通用的转换规则：

| 类型 | 转换 | 用途 |
|:---|:---|:---|
| Post-IR | `strip_fields("logprobs", "top_logprobs")` | 移除不支持的请求字段 |
| Post-IR | `downgrade developer → system` | 将 developer 角色转换为 system 角色 |
| Post-IR | `default null content → ""` | 防止空内容错误 |
| Post-IR | `default_tool_description()` | 填充空的工具描述以避免上游错误 |
| IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头以提升缓存稳定性 |

!!! note "Metis 额外转换"
    Metis 集群额外剥离请求中的 `parallel_tool_calls`，因为 SambaNova 后端不支持并行工具调用。

!!! note "Minerva 响应转换"
    Minerva 在响应上应用 `rewrite_harmony_tool_calls()`，修复 `inkling-bf16` 偶尔将工具调用作为纯文本而非结构化 `tool_calls` 字段输出的问题。

三个集群均导出 `model_list_transform`，用于标准化模型列表的响应格式。

## 认证

ALCF 使用 Globus OAuth 进行认证。访问令牌的有效期较短（约 48 小时），需要定期刷新。刷新令牌本身在 **30 天**后因 ALCF 的会话策略过期 — 届时需要交互式重新登录。

项目包含 `scripts/alcf-token.py`，一个零依赖（仅标准库）的辅助脚本，处理完整的登录和刷新生命周期。

**独立下载** — 如果没有完整仓库（如纯 Docker 部署）：

```bash
curl -fsSL -o alcf-token.py https://raw.githubusercontent.com/Oaklight/llm-rosetta/master/scripts/alcf-token.py
chmod +x alcf-token.py
```

### 初始登录

在有浏览器的机器上运行（或将 URL 复制到有浏览器的机器上）：

```bash
python3 alcf-token.py --login
```

这会打开一个 Globus 授权 URL，提示你粘贴授权码，并将令牌保存到 `~/.globus/app/<client_id>/inference_app/tokens.json`。

### 令牌管理

```bash
# 打印新的访问令牌（过期时自动刷新）：
python3 scripts/alcf-token.py

# 检查令牌状态和剩余有效期：
python3 scripts/alcf-token.py --status

# 验证令牌对某个集群是否可用：
python3 scripts/alcf-token.py --probe sophia

# 强制刷新（即使未过期）：
python3 scripts/alcf-token.py --force
```

### 多用户令牌池

多个 ALCF 用户可以汇集令牌用于轮转：

```bash
# 每个用户登录到自己的文件：
python3 scripts/alcf-token.py --login --token-file /shared/alcf-tokens/alice.json
python3 scripts/alcf-token.py --login --token-file /shared/alcf-tokens/bob.json

# 网关读取所有令牌（逗号分隔输出）：
python3 scripts/alcf-token.py --tokens-dir /shared/alcf-tokens/
```

### 30 天会话过期

!!! warning "会话策略限制"
    ALCF 的 Globus 会话策略强制执行 30 天的生命周期。会话过期后，刷新令牌停止工作，网关将进入 401 重试循环。**需要交互式重新登录** — 由于 ALCF 要求基于浏览器的机构 SSO 认证，此过程无法自动化。

    请关注管理面板中令牌状态的 `consecutive_failures` 是否持续上升，出现时重新运行 `--login`。

    跟踪：[#687](https://github.com/Oaklight/llm-rosetta/issues/687)

## 网关配置

=== "静态 API Key"

    手动获取令牌并设置为环境变量：

    ```bash
    export ALCF_API_KEY=$(python3 scripts/alcf-token.py)
    ```

    ```jsonc
    {
      "providers": {
        "alcf-sophia": {
          "shim": "alcf--sophia",
          "api_key": "${ALCF_API_KEY}"
        }
      }
    }
    ```

    令牌过期时需要重启网关。

=== "Token 命令（推荐）"

    让网关自动刷新令牌：

    ```jsonc
    {
      "providers": {
        "alcf-sophia": {
          "shim": "alcf--sophia",
          "token_command": ["python3", "scripts/alcf-token.py"],
          "token_refresh_interval": 1800
        }
      }
    }
    ```

    网关在启动时运行命令，然后每隔 `token_refresh_interval` 秒重新运行。收到上游 401 响应时，会立即触发非周期性刷新。

### Docker 部署

在 Docker 中运行网关时，需要将令牌文件和刷新脚本挂载到容器中：

```yaml
# docker-compose.yaml
services:
  llm-rosetta-gateway:
    image: oaklight/llm-rosetta-gateway
    volumes:
      - ./config:/config
      - ~/.globus:/home/appuser/.globus              # Globus 令牌文件（需读写权限以刷新）
      - ./alcf-token.py:/scripts/alcf-token.py:ro    # 刷新脚本（独立下载）
```

操作步骤：

1. **下载脚本**：`curl -fsSL -o alcf-token.py https://raw.githubusercontent.com/Oaklight/llm-rosetta/master/scripts/alcf-token.py`
2. **在宿主机上登录**（仅需一次）：`python3 alcf-token.py --login`
3. 使用上述 volume 挂载**启动容器**
4. 通过管理面板或 `config.jsonc` **添加提供方**：
    ```jsonc
    "token_command": ["python3", "/scripts/alcf-token.py"]
    ```

容器内的 `appuser`（uid 1000）需要对 `~/.globus` 目录有读写权限，以便就地更新刷新令牌。如果宿主机 uid 不同，请在 compose 文件中设置 `PUID`/`PGID`，或执行 `chown 1000:1000 .globus` 目录。

参见 [`docker/docker-compose.yaml`](https://github.com/Oaklight/llm-rosetta/blob/master/docker/docker-compose.yaml) 获取完整示例。

## 备注

- 每个集群运行不同的硬件，可能托管不同的模型。使用 shim 配置中的 `models_path` 查询每个集群上的可用模型。
