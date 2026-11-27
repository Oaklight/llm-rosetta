# ALCF

ALCF（阿贡领先计算设施）推理服务通过三个 shim 端点提供支持，每个计算集群对应一个端点。所有端点均使用兼容 OpenAI Chat Completions 的 API。

!!! warning "以独立包形式发布"
    自 0.14.0 起，三个 ALCF shim 不再内置在 LLM-Rosetta 中，而是由
    [llm-rosetta-ext-argonne](https://github.com/Oaklight/llm-rosetta-ext-argonne)
    的 `alcf` 扩展提供。通过 `shims.install` 声明后，下面的名称即可继续使用。
    **网关 → Shims → 可安装** 提供同一条目的一键安装。

    ```jsonc
    "shims": { "install": ["argonne:alcf"] }
    ```

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 | 硬件 |
|:---|:---|:---|:---|:---|
| `alcf_metis--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/metis/api/v1` | （Globus 登录） | SambaNova SN40L |
| `alcf_minerva--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1` | （Globus 登录） | NVIDIA B200 |
| `alcf_sophia--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1` | （Globus 登录） | vLLM on A100 |

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

ALCF 使用 Globus OAuth。推理服务接受的是由 Globus 授权生成的访问令牌，授权要由人到 `auth.globus.org` 登录一次才能获得；随后返回的刷新令牌会在之后每次生成令牌时被消耗，无需浏览器。因为 ALCF 的会话策略，刷新令牌在 **30 天**后过期；之后需要交互式重新登录。

交换逻辑**随扩展一起发布**：`alcf` 扩展在服务方清单旁带有 `auth.py`，因此三个集群 shim 的认证方式相同。提供方就是它的 shim；不需要编写 `token_command`，也不需要导出 `ALCF_API_KEY`。网关会在启动时、按 shim 的间隔以及上游返回 401 之后向该交换请求凭据。

### 登录

在有浏览器的机器上运行登录，再把记录交给网关（[`credential <provider>`](../../gateway/cli.md#credential-provider)）。命令会打印一个 Globus 授权 URL，并接收你粘贴回来的授权码：

```bash
llm-rosetta gateway credential alcf-sophia --out record.json
```

在管理面板中，提供方认证页上的 **安装记录** 接收这份记录。提供方在还没有记录时也可以先保存。

授权保存在厂商自己的工具存放它的位置，即 `~/.globus/app/<client_id>/inference_app/tokens.json`，并以同样的布局写回。因此已经用 `alcf-ai` 登录过的环境无需再次登录，挂载了 `~/.globus` 的容器也只需配置一次。

### 30 天会话过期

!!! warning "会话策略限制"
    ALCF 的 Globus 会话策略强制 30 天有效期。会话过期后，刷新令牌失效，网关会进入 401 重试循环。**需要交互式重新登录**——这无法自动化，因为 ALCF 要求基于浏览器的机构 SSO。

    请留意管理面板令牌状态中 `consecutive_failures` 不断升高，出现时请重新登录。

    跟踪议题：[#687](https://github.com/Oaklight/llm-rosetta/issues/687)

## 网关配置

```jsonc
{
  "providers": {
    "alcf-sophia": {
      "shim": "alcf_sophia--chat_completions"
    }
  }
}
```

要修改刷新间隔，在提供方上设置 `token_refresh_interval`。要汇集多个 ALCF 账号，请为每个账号声明一个提供方：网关会给每个提供方各自的状态目录。

### Docker 部署

交换逻辑在扩展里，网关会从 `argonne` 源安装该扩展，因此只需要挂载令牌文件：

```yaml
# docker-compose.yaml
services:
  llm-rosetta-gateway:
    image: oaklight/llm-rosetta-gateway
    volumes:
      - ./config:/config
      - ~/.globus:/home/appuser/.globus              # Globus 令牌文件（需读写权限以刷新）
```

镜像把已安装的扩展放在 `/config/extensions`（`LLM_ROSETTA_EXTENSIONS_DIR`），因此扩展不会随容器消失。容器以 `${PUID:-1000}` 运行（compose 的 `user:` 指令，默认即镜像里的 `appuser`，uid 1000），挂载的 `~/.globus` 必须可被该 uid 写入，刷新令牌才能原地更新。如果你的宿主 uid 不同，请在 compose 文件里设置 `PUID`/`PGID`，并让 `.globus` 目录属于同一个 uid。

完整示例见 [`docker/docker-compose.yaml`](https://github.com/Oaklight/llm-rosetta/blob/master/docker/docker-compose.yaml)。

## 备注

- 每个集群运行不同的硬件，可能托管不同的模型。使用 shim 配置中的 `models_path` 查询每个集群上的可用模型。
