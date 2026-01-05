我在看 comparison.md 和 ir_design.md

### 内容类型支持部分

音频输入和输出我们可以暂时按下不做，这个目前我也没需求，但是我们要准备好后续万一要做的时候有足够的兼容性

文档和文件名字虽然不同，但是不是其实是同一个东西？

搜索结果的话我们也可以暂时不做，或者将其变为工具调用的结果形式呈现

### 思考过程

openai 的兼容第三方很多在 message 的 content 之外添加了 reasoning_content/reasoning_details 字段，这部分作为 openai 类型组的思考过程的表示

另外我们这里其实说的 openai 是 openai-chatcompletions，他们还有更新的 responses api，我们可以之后做兼容，目前先不管

### 工具调用

这部分在我看来其实三家基本一致的，就是呈现的方式不同。内里的东西基本一致。自定义工具我们可以之后考虑做兼容，目前来看用的人很少。服务器工具部分，每家其实都有一些，我们自己内部也做了相应的一些工具。但是实际上我们可以都将其认为是 function 调用。有一个需要注意的就是 mcp 工具部分，这部分我没看你提到，你可能需要再找一下。

### 特殊功能部分

这个我们其实都不用管，可以直接忽略

### 引用系统

这个暂时不管

### 类型系统设计哲学

对于已经过时的类型，比如 openai 的 function 相关类型，我们直接忽略，直接用 tool 代替，遇到就直接换成 tool

#### 工具调用部分

google 虽然没有 ID 信息，但是他们实际上的做法是根据顺序来匹配的，意味着模型返回什么顺序的 tool call，用户回馈回去的东西的顺序就得对上，他们会默认 tool call 结果就是和原来的需求顺序一致的

工具部分涉及到的 tool choice 这些我们似乎文档中没处理，你也得看一下。mcp 是一种工具类型，你得看看他们三家代码

openai: ChatCompletionToolChoiceOptionParam
anthropic: ToolChoiceParam
google-genai: ToolConfig

google.md 中角色系统什么时候多了一个 function？

他们只有 user/model 两个角色

工具选择他们有四种模式，ANY+allowed_function_names 进行限制的情况下会进行强制选择的第四种模式

### system prompt

google: GenerateContentConfig 中设置 system_instruction
example:

```
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Why is the sky blue?",
    config=GenerateContentConfig(
        system_instruction=[
            "You're a language translator.",
            "Your mission is to translate text in English to French.",
        ]
    ),
)
print(response.text)
# Example response:
# Pourquoi le ciel est-il bleu ?
```

anthropic: set when sending off request
example:

```
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a Python expert."
        },
        {
            "type": "text",
            "text": "Always include type hints and docstrings."
        }
    ],
    messages=[
        {"role": "user", "content": "Help me with Python code"}
    ]
)
```
