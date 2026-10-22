# ALCF

ALCF（阿贡领先计算设施）推理服务通过三个 shim 端点提供支持，每个计算集群对应一个端点。所有端点均使用兼容 OpenAI Chat Completions 的 API。

!!! warning "以独立包形式发布"
    自 0.14.0 起，三个 ALCF shim 不再内置在 LLM-Rosetta 中，而是由
    [llm-rosetta-shim-alcf](https://github.com/Oaklight/llm-rosetta-shim-alcf)
    包提供。通过 `shims.presets`（或等价的 `shims.remotes` 条目）声明后，下面的名称即可继续使用。
    **Settings → Shim Sources** 提供同一条目的一键预设。

    ```jsonc
    "shims": { "presets": ["alcf"] }
    ```

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 | 硬件 |
|:---|:---|:---|:---|:---|
| `alcf_metis--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/metis/api/v1` | `ALCF_API_KEY` | SambaNova SN40L |
| `alcf_minerva--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1` | `ALCF_API_KEY` | NVIDIA B200 |
| `alcf_sophia--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1` | `ALCF_API_KEY` | vLLM on A100 |

三个集群共用一个管理面板顶层：每个都声明 `group: alcf`，因此提供方类型选择器只显示一个
`alcf` 分组，其条目读作 `metis · chat_completions`、`minerva · chat_completions` 和
`sophia · chat_completions`。旧的部署优先拼写（`alcf--metis`、`alcf--minerva`、
`alcf--sophia`）不会由独立包注册；请使用上面的规范名称。

## 推理支持

Sophia 集群上在 [ALCF 模型列表](https://docs.alcf.anl.gov/services/inference-endpoints/#available-models)中标注 **R** 的模型支持推理（如 `openai/gpt-oss-120b`、`openai/gpt-oss-20b`、`google/gemma-4-31B-it`、`google/gemma-4-E4B-it`）。

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `reasoning_effort` |
| Effort 范围 | 完整 IR 阶梯（`minimal` → `max`） |

Metis 和 Minerva 目前没有支持推理的模型。

## 能力标志

| 标志 | `alcf_sophia--chat_completions` | `alcf_minerva--chat_completions` | `alcf_metis--chat_completions` |
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

Globus 交换随 **shim 包一起发布**：`llm-rosetta-shim-alcf` 内含这个零依赖（仅标准库）的辅助脚本，模块路径为 `llm_rosetta_shim_alcf.providers.alcf.token`，每个 `alcf_<cluster>--chat_completions` shim 都把它声明为自己的 `connection.token_command`。因此无需下载脚本，配置里也不必写 `token_command` —— 网关会在启动时、按 shim 声明的间隔、以及收到上游 401 之后运行它。

### 初始登录

在有浏览器的机器上运行（或将 URL 复制到有浏览器的机器上）：

```bash
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --login
```

这会打开一个 Globus 授权 URL，提示你粘贴授权码，并将令牌保存到 `~/.globus/app/<client_id>/inference_app/tokens.json`。

### 令牌管理

```bash
# 打印新的访问令牌（过期时自动刷新）：
python3 -m llm_rosetta_shim_alcf.providers.alcf.token

# 检查令牌状态和剩余有效期：
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --status

# 验证令牌对某个集群是否可用：
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --probe sophia

# 强制刷新（即使未过期）：
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --force
```

### 多用户令牌池

多个 ALCF 用户可以汇集令牌用于轮转。网关运行的是 shim 声明的命令（不带参数），因此池化需要一处覆盖 —— 把 `token_command` 指向 `--tokens-dir`：

```bash
# 每个用户登录到自己的文件：
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --login --token-file /shared/alcf-tokens/alice.json
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --login --token-file /shared/alcf-tokens/bob.json

# 手动查看所有令牌（逗号分隔输出）：
python3 -m llm_rosetta_shim_alcf.providers.alcf.token --tokens-dir /shared/alcf-tokens/
```

```jsonc
{
  "providers": {
    "alcf-sophia": {
      "shim": "alcf_sophia--chat_completions",
      "token_command": [
        "python3", "-m", "llm_rosetta_shim_alcf.providers.alcf.token",
        "--tokens-dir", "/shared/alcf-tokens/"
      ]
    }
  }
}
```

命令会为所有文件输出一个逗号分隔的字符串，网关把它当作 key ring。

### 30 天会话过期

!!! warning "会话策略限制"
    ALCF 的 Globus 会话策略强制执行 30 天的生命周期。会话过期后，刷新令牌停止工作，网关将进入 401 重试循环。**需要交互式重新登录** — 由于 ALCF 要求基于浏览器的机构 SSO 认证，此过程无法自动化。

    请关注管理面板中令牌状态的 `consecutive_failures` 是否持续上升，出现时重新运行 `--login`。

    跟踪：[#687](https://github.com/Oaklight/llm-rosetta/issues/687)

## 网关配置

=== "静态 API Key"

    手动获取令牌并设置为环境变量：

    ```bash
    export ALCF_API_KEY=$(python3 -m llm_rosetta_shim_alcf.providers.alcf.token)
    ```

    ```jsonc
    {
      "providers": {
        "alcf-sophia": {
          "shim": "alcf_sophia--chat_completions",
          "api_key": "${ALCF_API_KEY}"
        }
      }
    }
    ```

    令牌过期时需要重启网关。

=== "随包发布的交换（推荐）"

    无需接线 —— shim 已经声明了交换，提供方只需要写 shim：

    ```jsonc
    {
      "providers": {
        "alcf-sophia": {
          "shim": "alcf_sophia--chat_completions"
        }
      }
    }
    ```

    网关在启动时运行命令，按 shim 声明的间隔重新运行，并在收到上游 401 响应时立即触发一次非周期性刷新。只想改间隔而不替换命令时，在提供方上设置 `token_refresh_interval` 即可。

=== "覆盖命令"

    提供方自己声明的 `token_command` 会替换随包发布的那个 —— 即上面的多用户池化，或你自己的脚本：

    ```jsonc
    {
      "providers": {
        "alcf-sophia": {
          "shim": "alcf_sophia--chat_completions",
          "token_command": ["python3", "/opt/alcf/token-from-vault.py"],
          "token_refresh_interval": 1800
        }
      }
    }
    ```

### Docker 部署

刷新脚本随包发布，镜像已安装该包，因此只需要挂载令牌文件：

```yaml
# docker-compose.yaml
services:
  llm-rosetta-gateway:
    image: oaklight/llm-rosetta-gateway
    volumes:
      - ./config:/config
      - ~/.globus:/home/appuser/.globus              # Globus 令牌文件（需读写权限以刷新）
```

操作步骤：

1. **在宿主机上登录**（仅需一次）：`python3 -m llm_rosetta_shim_alcf.providers.alcf.token --login`
2. 使用上述 volume 挂载**启动容器**
3. 通过管理面板或 `config.jsonc` **添加提供方** —— 不需要 `token_command`

容器内的 `appuser`（uid 1000）需要对 `~/.globus` 目录有读写权限，以便就地更新刷新令牌。如果宿主机 uid 不同，请在 compose 文件中设置 `PUID`/`PGID`，或执行 `chown 1000:1000 .globus` 目录。

参见 [`docker/docker-compose.yaml`](https://github.com/Oaklight/llm-rosetta/blob/master/docker/docker-compose.yaml) 获取完整示例。

## 备注

- 每个集群运行不同的硬件，可能托管不同的模型。使用 shim 配置中的 `models_path` 查询每个集群上的可用模型。
