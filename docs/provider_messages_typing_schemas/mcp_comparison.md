# 三家 Provider 的 MCP 支持对比分析

本文档详细对比 OpenAI、Anthropic 和 Google 三家 provider 对 Model Context Protocol (MCP)的支持情况。

## 概述

**重要发现**：三家 provider 都已经支持 MCP，但实现方式各不相同，反映了不同的设计哲学。

## OpenAI Responses API 的 MCP 实现

### 最完整的原生支持

OpenAI Responses API 提供了**最完整的 MCP 支持**，包括四种 MCP 相关类型：

#### 1. McpCall - MCP 工具调用

```python
class McpCall(TypedDict, total=False):
    type: Required[Literal["mcp_call"]]
    id: Required[str]
    name: Required[str]
    server_label: Required[str]  # MCP服务器标签
    arguments: Required[str]
    approval_request_id: Optional[str]  # 批准请求ID
    output: Optional[str]  # 工具输出
    error: Optional[str]  # 错误信息
    status: Literal["in_progress", "completed", "incomplete", "calling", "failed"]
```

#### 2. McpListTools - 列出可用工具

```python
class McpListTools(TypedDict, total=False):
    type: Required[Literal["mcp_list_tools"]]
    id: Required[str]
    server_label: Required[str]
    output: Optional[str]  # 工具列表JSON
    error: Optional[str]
    status: Literal["in_progress", "completed", "incomplete", "calling", "failed"]
```

#### 3. McpApprovalRequest - 批准请求

```python
class McpApprovalRequest(TypedDict, total=False):
    type: Required[Literal["mcp_approval_request"]]
    id: Required[str]
    call: Required[McpCall]  # 需要批准的调用
    status: Literal["in_progress", "completed", "incomplete"]
```

#### 4. McpApprovalResponse - 批准响应

```python
class McpApprovalResponse(TypedDict, total=False):
    type: Required[Literal["mcp_approval_response"]]
    id: Required[str]
    approval_request_id: Required[str]
    approved: Required[bool]  # 是否批准
    status: Literal["in_progress", "completed", "incomplete"]
```

### 完整的工作流

```python
# 1. 列出可用工具
{
    "type": "mcp_list_tools",
    "id": "list_123",
    "server_label": "calculator-server",
    "status": "calling"
}

# 2. 调用工具
{
    "type": "mcp_call",
    "id": "call_456",
    "server_label": "calculator-server",
    "name": "calc_evaluate",
    "arguments": '{"expression": "26 * 9 / 5 + 32"}',
    "status": "calling"
}

# 3. 批准请求（如需要）
{
    "type": "mcp_approval_request",
    "id": "approval_789",
    "call": {...},
    "status": "in_progress"
}

# 4. 批准响应
{
    "type": "mcp_approval_response",
    "id": "response_012",
    "approval_request_id": "approval_789",
    "approved": true,
    "status": "completed"
}

# 5. 工具调用结果
{
    "type": "mcp_call",
    "id": "call_456",
    "output": "78.8",
    "status": "completed"
}
```

### 关键特性

- **MCP 工具是一等公民**：有专门的类型系统
- **完整的批准流程**：支持请求和响应
- **工具发现**：支持列出可用工具
- **独立的状态跟踪**：每个操作都有详细状态
- **错误处理**：内置错误字段
- **输出管理**：工具调用和列表都有输出字段

## Anthropic 的 MCP 实现

### 通过内容块实现

Anthropic 将 MCP 工具集成到其内容块架构中，保持了设计的一致性。

#### 1. MCPToolUseBlockParam - MCP 工具调用

```python
class MCPToolUseBlockParam(TypedDict, total=False):
    type: Required[Literal["mcp_tool_use"]]
    id: Required[str]
    server_name: Required[str]  # 注意：叫server_name而非server_label
    name: Required[str]
    input: Required[Dict[str, object]]
    cache_control: Optional[CacheControlEphemeralParam]
```

#### 2. MCPToolResultBlockParam - MCP 工具结果

```python
class MCPToolResultBlock(BaseModel):
    type: Literal["mcp_tool_result"]
    tool_use_id: str
    content: Union[str, List[TextBlock]]
    is_error: bool
```

### 使用示例

```python
# 在API请求中配置MCP服务器
mcp_servers = [
    {
        "name": "calculator-server",
        "type": "url",
        "url": "https://calculator-mcp-server.example.com",
        "authorization_token": "auth_token_123",
        "tool_configuration": {
            "allowed_tools": ["calc_evaluate", "calc_help"],
            "enabled": True
        }
    }
]

# Assistant发起MCP工具调用
{
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Let me calculate that for you."
        },
        {
            "type": "mcp_tool_use",
            "id": "mcptool_456",
            "server_name": "calculator-server",
            "name": "calc_evaluate",
            "input": {"expression": "26 * 9 / 5 + 32"}
        }
    ]
}

# MCP工具结果（由系统自动处理）
{
    "type": "mcp_tool_result",
    "tool_use_id": "mcptool_456",
    "content": "78.8",
    "is_error": False
}
```

### 配置方式

通过 API 请求的`mcp_servers`参数配置：

```python
class RequestMCPServerURLDefinitionParam(TypedDict, total=False):
    name: Required[str]
    type: Required[Literal["url"]]
    url: Required[str]
    authorization_token: Optional[str]
    tool_configuration: Optional[RequestMCPServerToolConfigurationParam]
```

### 关键特性

- **内容块架构**：MCP 工具遵循内容块设计
- **系统自动处理**：工具结果由系统自动处理，不需要用户提供
- **工具配置**：支持工具白名单和启用/禁用
- **缓存支持**：MCP 工具调用支持缓存控制
- **使用`server_name`**：而非 OpenAI 的`server_label`

### 关键差异

- 没有显式的批准流程
- 没有工具列表功能
- 工具结果自动处理

## Google 的 MCP 实现

### 通过适配器模式实现

Google 采用了完全不同的方法，通过适配器将 MCP 工具转换为 Gemini 原生工具。

#### 适配器架构

```python
class McpToGenAiToolAdapter:
    """将MCP工具转换为Gemini可用的工具"""

    def __init__(
        self,
        session: "mcp.ClientSession",
        list_tools_result: "mcp_types.ListToolsResult",
    ) -> None:
        self._mcp_session = session
        self._list_tools_result = list_tools_result

    async def call_tool(
        self, function_call: FunctionCall
    ) -> "mcp_types.CallToolResult":
        """调用MCP服务器上的函数"""
        name = function_call.name if function_call.name else ""
        arguments = dict(function_call.args) if function_call.args else {}

        return await self._mcp_session.call_tool(
            name=name,
            arguments=arguments,
        )

    @property
    def tools(self) -> list[Tool]:
        """返回Google GenAI工具列表"""
        return mcp_to_gemini_tools(self._list_tools_result.tools)
```

#### 工具转换

```python
def mcp_to_gemini_tool(tool: McpTool) -> types.Tool:
    """将MCP工具转换为Google GenAI工具"""
    return types.Tool(
        function_declarations=[{
            "name": tool.name,
            "description": tool.description,
            "parameters": types.Schema.from_json_schema(
                json_schema=types.JSONSchema(
                    **_filter_to_supported_schema(tool.inputSchema)
                )
            ),
        }]
    )

def mcp_to_gemini_tools(tools: list[McpTool]) -> list[types.Tool]:
    """将MCP工具列表转换为Google GenAI工具列表"""
    return [mcp_to_gemini_tool(tool) for tool in tools]
```

### 使用示例

```python
import asyncio
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai

client = genai.Client()

# 创建stdio连接的服务器参数
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@philschmid/weather-mcp"],
    env=None,
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化客户端和服务器之间的连接
            await session.initialize()

            # 向模型发送带有MCP函数声明的请求
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents="What is the weather in London?",
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],  # 直接传递会话
                ),
            )
            print(response.text)

asyncio.run(run())
```

### 内部工作原理

1. **工具转换**：MCP 工具被转换为 Gemini 的`FunctionDeclaration`
2. **自动函数调用**：默认情况下 SDK 自动处理函数调用
3. **会话管理**：SDK 管理 MCP 会话的生命周期
4. **错误处理**：SDK 提供错误处理机制

### 关键特性

- **透明转换**：MCP 工具被转换为普通的`function_call`
- **自动调用**：默认自动处理工具调用
- **会话管理**：SDK 管理 MCP 会话生命周期
- **异步操作**：完全异步的 API
- **无特殊类型**：在消息层面没有 MCP 特定类型

## MCP 实现对比总结

| 特性           | OpenAI Responses | Anthropic     | Google                    |
| -------------- | ---------------- | ------------- | ------------------------- |
| **实现方式**   | 专门的类型       | 内容块        | 适配器转换                |
| **服务器标识** | `server_label`   | `server_name` | 会话对象                  |
| **工具列表**   | ✓ `McpListTools` | ✗             | ✓ 通过会话                |
| **批准流程**   | ✓ 显式支持       | ✗             | ✗                         |
| **工具结果**   | 在`McpCall`中    | 独立内容块    | 转换为`function_response` |
| **配置方式**   | 输入项           | API 参数      | 会话对象                  |
| **状态跟踪**   | ✓ 详细状态       | ✗             | ✗                         |
| **错误处理**   | ✓ 内置字段       | ✓ `is_error`  | ✓ SDK 处理                |
| **自动调用**   | 可选             | 自动          | 默认自动                  |
| **类型可见性** | 高（专门类型）   | 中（内容块）  | 低（透明转换）            |
| **缓存支持**   | ✗                | ✓             | ✗                         |
| **异步要求**   | ✗                | ✗             | ✓ 必须                    |

## 关键洞察

### 1. OpenAI 最显式

- MCP 是一等公民，有完整的类型系统
- 支持完整的工作流（发现、调用、批准）
- 适合需要精细控制的场景
- 状态跟踪最详细

### 2. Anthropic 最一致

- MCP 工具遵循内容块架构
- 与普通工具调用保持一致
- 系统自动处理结果
- 支持缓存控制

### 3. Google 最透明

- MCP 工具被转换为普通工具
- 用户无需关心 MCP 细节
- 适合简单集成场景
- 完全异步的 API

## 对 IR 设计的启示

### 方案 A：学习 OpenAI - 显式类型

```python
IRInputItem = Union[
    IRMessage,
    IRToolCall,
    IRToolResult,
    IRMcpCall,              # 专门的MCP调用类型
    IRMcpListTools,         # MCP工具列表
    IRMcpApprovalRequest,   # 批准请求
    IRMcpApprovalResponse,  # 批准响应
]
```

**优点**：

- 完整的 MCP 工作流支持
- 精细控制
- 与 OpenAI Responses API 一致

**缺点**：

- 类型较多，复杂度高
- 与 Anthropic/Google 不一致

### 方案 B：学习 Anthropic - 统一内容块

```python
ContentPart = Union[
    TextPart,
    ToolUsePart,
    ToolResultPart,
    McpToolUsePart,      # MCP工具调用（内容块）
    McpToolResultPart,   # MCP工具结果（内容块）
]
```

**优点**：

- 与普通工具一致
- 结构简洁
- 与 Anthropic 一致

**缺点**：

- 缺少批准流程
- 缺少工具列表功能

### 方案 C：学习 Google - 透明转换

```python
# MCP工具被转换为普通工具调用
ToolCall = {
    "type": "tool_call",
    "tool_name": "calc_evaluate",
    "tool_input": {...},
    "mcp_server": "calculator-server"  # 可选字段标识MCP
}
```

**优点**：

- 最简单
- 用户无需关心 MCP 细节
- 易于实现

**缺点**：

- 失去 MCP 特有功能（批准、工具列表）
- 不够显式

### 推荐方案：混合方案

**基础层面**：采用方案 B（内容块）

```python
ContentPart = Union[
    TextPart,
    ToolUsePart,
    ToolResultPart,
    McpToolUsePart,      # 包含server_name字段
    McpToolResultPart,
]
```

**通过 metadata 支持高级功能**：

```python
class McpToolUsePart(TypedDict):
    type: Required[Literal["mcp_tool_use"]]
    tool_call_id: Required[str]
    server_name: Required[str]  # MCP服务器名称
    tool_name: Required[str]
    tool_input: Required[Dict[str, Any]]
    metadata: NotRequired[McpMetadata]  # 支持批准等高级功能

class McpMetadata(TypedDict, total=False):
    approval_required: bool
    approval_request_id: Optional[str]
    list_tools: bool  # 是否为工具列表请求
```

**提供简化 API**：

```python
# 对于简单场景，可以自动转换
def create_tool_call(name: str, input: dict, mcp_server: Optional[str] = None):
    if mcp_server:
        return McpToolUsePart(
            type="mcp_tool_use",
            server_name=mcp_server,
            tool_name=name,
            tool_input=input
        )
    else:
        return ToolUsePart(
            type="tool_use",
            tool_name=name,
            tool_input=input
        )
```

## 结论

三家 provider 的 MCP 实现各有特色，反映了不同的设计哲学：

1. **OpenAI**：显式、完整、可控
2. **Anthropic**：一致、简洁、自动
3. **Google**：透明、简单、异步

我们的 IR 设计应该：

- 采用 Anthropic 的内容块架构作为基础
- 通过 metadata 支持 OpenAI 的高级功能
- 提供 Google 风格的简化 API
- 确保三家之间的转换都是可行的
