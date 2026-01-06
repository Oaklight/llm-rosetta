"""
OpenAI Chat Converter Integration Test

真实API测试文件 - 测试OpenAI Chat Completions API的完整转换流程
Real API test file - test complete conversion flow for OpenAI Chat Completions API

测试内容 / Test Coverage:
1. 基础文本消息转换 (Basic text message conversion)
2. 多模态内容转换 - 图片/文件 (Multimodal content - image/file)
3. 工具调用转换 (Tool call conversion)
4. 工具结果转换 (Tool result conversion)
5. 多轮对话流程 (Multi-turn conversation)
6. 流式响应转换 (Streaming response conversion)
"""

import base64
import json
import os
import sys
from typing import Generator, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import dotenv
from openai import OpenAI

from examples.tools import available_tools, tools_spec
from llmir.converters.openai_chat_converter import OpenAIChatConverter
from llmir.types.ir import (
    ImagePart,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    create_tool_result_message,
    extract_text_content,
    extract_tool_calls,
)

# 加载环境变量
dotenv.load_dotenv()


# ============================================================================
# Client and Converter Initialization
# ============================================================================

openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError(
        "请设置 OPENAI_API_KEY 环境变量 / Please set OPENAI_API_KEY environment variable"
    )

openai_client = OpenAI(
    api_key=openai_api_key,
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)
openai_converter = OpenAIChatConverter()


# ============================================================================
# Helper Functions
# ============================================================================


def display_message(message: Message, label: str = "Message") -> None:
    """显示消息内容 / Display message content"""
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    text = extract_text_content(message)
    if text:
        print(f"Text: {text}")
    tool_calls = extract_tool_calls(message)
    for tc in tool_calls:
        print(f"Tool Call: {tc['tool_name']}({json.dumps(tc['tool_input'], indent=2)})")


def execute_tool_call(tool_call: ToolCallPart) -> str:
    """执行工具调用并返回结果 / Execute tool call and return result"""
    function_name = tool_call["tool_name"]
    function_args = tool_call["tool_input"]

    print(f"\n--- Executing tool: {function_name}({json.dumps(function_args)}) ---")

    if function_name not in available_tools:
        return json.dumps({"error": f"Unknown tool: {function_name}"})

    function_to_call = available_tools[function_name]
    function_response = function_to_call(**function_args)

    print(f"Tool Result: {function_response}")
    return function_response


def print_section(title: str) -> None:
    """打印章节标题 / Print section title"""
    print(f"\n\n{'#' * 80}")
    print(f"# {title}")
    print(f"{'#' * 80}\n")


def print_test_result(test_name: str, success: bool, error: str = None) -> None:
    """打印测试结果 / Print test result"""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status}: {test_name}")
    if error:
        print(f"  Error: {error}")


# ============================================================================
# Test Cases
# ============================================================================


def test_basic_text_message():
    """测试1: 基础文本消息转换 / Test 1: Basic text message conversion"""
    print_section("测试1: 基础文本消息转换 / Test 1: Basic Text Message Conversion")

    try:
        # IR格式消息
        ir_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is 2+2?"}],
            },
        ]

        print("IR Input:")
        print(json.dumps(ir_messages, indent=2, ensure_ascii=False))

        # IR → OpenAI
        openai_payload, warnings = openai_converter.to_provider(ir_messages)
        print(f"\nWarnings: {warnings}")
        print("\nOpenAI Payload:")
        print(json.dumps(openai_payload, indent=2, ensure_ascii=False))

        # 调用API
        response = openai_client.chat.completions.create(
            model=openai_model,
            **openai_payload,
        )
        print(f"\nAPI Response (usage): {response.usage}")

        # OpenAI → IR
        ir_from_openai = openai_converter.from_provider(
            response.choices[0].message.model_dump()
        )
        print("\nIR Output:")
        for msg in ir_from_openai:
            print(json.dumps(dict(msg), indent=2, ensure_ascii=False))

        display_message(ir_from_openai[-1], "Assistant Response")

        print_test_result("基础文本消息转换", True)
        return True

    except Exception as e:
        print_test_result("基础文本消息转换", False, str(e))
        return False


def test_image_message():
    """测试2: 图片消息转换 / Test 2: Image message conversion"""
    print_section("测试2: 图片消息转换 / Test 2: Image Message Conversion")

    try:
        # 使用示例图片URL (OpenAI logo)
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/1200px-ChatGPT_logo.svg.png"

        ir_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {
                        "type": "image",
                        "image_url": image_url,
                    },
                ],
            },
        ]

        print("IR Input (with image):")
        print(json.dumps(ir_messages, indent=2, ensure_ascii=False))

        # IR → OpenAI
        openai_payload, warnings = openai_converter.to_provider(ir_messages)
        print(f"\nWarnings: {warnings}")

        # 调用API
        response = openai_client.chat.completions.create(
            model=openai_model,
            **openai_payload,
        )

        # OpenAI → IR
        ir_from_openai = openai_converter.from_provider(
            response.choices[0].message.model_dump()
        )
        display_message(ir_from_openai[-1], "Image Description Response")

        print_test_result("图片消息转换", True)
        return True

    except Exception as e:
        print_test_result("图片消息转换", False, str(e))
        return False


def test_base64_image():
    """测试3: Base64编码图片转换 / Test 3: Base64 encoded image conversion"""
    print_section("测试3: Base64编码图片转换 / Test 3: Base64 Encoded Image Conversion")

    try:
        # 创建一个小的红色正方形图片 (1x1 pixel, red)
        red_pixel_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

        ir_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image?"},
                    {
                        "type": "image",
                        "image_data": {
                            "data": red_pixel_base64,
                            "media_type": "image/png",
                        },
                    },
                ],
            },
        ]

        print("IR Input (with base64 image):")
        print(json.dumps(ir_messages, indent=2, ensure_ascii=False))

        # IR → OpenAI
        openai_payload, warnings = openai_converter.to_provider(ir_messages)
        print(f"\nWarnings: {warnings}")
        print("\nOpenAI Payload content:")
        print(json.dumps(openai_payload["messages"][0], indent=2, ensure_ascii=False))

        # 调用API
        response = openai_client.chat.completions.create(
            model=openai_model,
            **openai_payload,
        )

        # OpenAI → IR
        ir_from_openai = openai_converter.from_provider(
            response.choices[0].message.model_dump()
        )
        display_message(ir_from_openai[-1], "Color Description Response")

        print_test_result("Base64图片转换", True)
        return True

    except Exception as e:
        print_test_result("Base64图片转换", False, str(e))
        return False


def test_tool_call_single_turn():
    """测试4: 单轮工具调用 (不执行工具) / Test 4: Single turn tool call (without executing tool)"""
    print_section("测试4: 单轮工具调用 / Test 4: Single Turn Tool Call")

    try:
        # IR格式消息 - 询问天气
        ir_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's the weather in San Francisco?"}
                ],
            },
        ]

        print("IR Input:")
        print(json.dumps(ir_messages, indent=2, ensure_ascii=False))

        # IR → OpenAI (with tools)
        openai_payload, warnings = openai_converter.to_provider(
            ir_messages,
            tools=tools_spec,
            tool_choice={"mode": "auto"},
        )
        print(f"\nWarnings: {warnings}")
        print("\nOpenAI Tools:")
        print(json.dumps(openai_payload.get("tools"), indent=2, ensure_ascii=False))

        # 调用API
        response = openai_client.chat.completions.create(
            model=openai_model,
            **openai_payload,
        )

        print(f"\nAPI Response finish_reason: {response.choices[0].finish_reason}")

        # OpenAI → IR
        ir_from_openai = openai_converter.from_provider(
            response.choices[0].message.model_dump()
        )

        # 检查是否请求了工具调用
        if ir_from_openai:
            display_message(ir_from_openai[-1], "Assistant Response")
            tool_calls = extract_tool_calls(ir_from_openai[-1])
            if tool_calls:
                print(f"\n✓ Tool call requested: {len(tool_calls)} call(s)")
                for tc in tool_calls:
                    print(f"  - {tc['tool_name']}({json.dumps(tc['tool_input'])})")
                print_test_result("单轮工具调用", True)
                return tool_calls
            else:
                print("✗ No tool call requested")
                print_test_result("单轮工具调用", False, "No tool call requested")
                return None
        else:
            print("✗ No response message")
            print_test_result("单轮工具调用", False, "No response message")
            return None

    except Exception as e:
        print_test_result("单轮工具调用", False, str(e))
        return None


def test_tool_call_multi_turn():
    """测试5: 多轮工具调用 (完整流程) / Test 5: Multi-turn tool call (complete flow)"""
    print_section(
        "测试5: 多轮工具调用 - 完整流程 / Test 5: Multi-Turn Tool Call - Complete Flow"
    )

    try:
        ir_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's the weather in San Francisco?"}
                ],
            },
        ]

        print("=== Round 1: Initial Request ===")

        # Round 1: IR → OpenAI
        openai_payload, warnings = openai_converter.to_provider(
            ir_messages,
            tools=tools_spec,
            tool_choice={"mode": "auto"},
        )

        # API调用
        response = openai_client.chat.completions.create(
            model=openai_model,
            **openai_payload,
        )

        # OpenAI → IR
        ir_from_openai = openai_converter.from_provider(
            response.choices[0].message.model_dump()
        )
        ir_messages.extend(ir_from_openai)

        # 显示响应
        display_message(ir_messages[-1], "Round 1 - Assistant Response")

        # 检查工具调用
        tool_calls = extract_tool_calls(ir_messages[-1])
        if not tool_calls:
            print("✗ No tool call requested")
            return False

        # 执行工具调用
        tool_call = tool_calls[0]
        function_response = execute_tool_call(tool_call)

        # 创建工具结果消息
        tool_result_message = create_tool_result_message(
            tool_call["tool_call_id"], function_response
        )
        ir_messages.append(tool_result_message)

        print(f"\nTool Result Message: {tool_result_message}")

        # Round 2: Send tool result and get final response
        print("\n=== Round 2: Tool Result ===")

        openai_payload, warnings = openai_converter.to_provider(
            ir_messages,
            tools=tools_spec,
        )

        response = openai_client.chat.completions.create(
            model=openai_model,
            **openai_payload,
        )

        ir_from_openai = openai_converter.from_provider(
            response.choices[0].message.model_dump()
        )
        ir_messages.extend(ir_from_openai)

        display_message(ir_messages[-1], "Round 2 - Final Response")

        print_test_result("多轮工具调用", True)
        return True

    except Exception as e:
        print_test_result("多轮工具调用", False, str(e))
        return False


def test_multiple_tool_calls():
    """测试6: 多个工具调用 / Test 6: Multiple tool calls"""
    print_section("测试6: 多个工具调用 / Test 6: Multiple Tool Calls")

    try:
        # 创建一个需要多个工具调用的查询
        ir_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Compare the weather in San Francisco and Tokyo, and find a flight from SF to Tokyo.",
                    }
                ],
            },
        ]

        print("IR Input:")
        print(json.dumps(ir_messages, indent=2, ensure_ascii=False))

        # IR → OpenAI (include both weather and flight tools)
        openai_payload, warnings = openai_converter.to_provider(
            ir_messages,
            tools=tools_spec,
            tool_choice={"mode": "auto"},
        )

        # API调用
        response = openai_client.chat.completions.create(
            model=openai_model,
            **openai_payload,
        )

        # OpenAI → IR
        ir_from_openai = openai_converter.from_provider(
            response.choices[0].message.model_dump()
        )
        ir_messages.extend(ir_from_openai)

        display_message(ir_messages[-1], "Tool Request Response")

        tool_calls = extract_tool_calls(ir_messages[-1])
        print(f"\n✓ Tool calls requested: {len(tool_calls)} call(s)")

        # 执行所有工具调用
        for i, tool_call in enumerate(tool_calls):
            function_response = execute_tool_call(tool_call)
            tool_result_message = create_tool_result_message(
                tool_call["tool_call_id"], function_response
            )
            ir_messages.append(tool_result_message)

        # 发送所有工具结果
        openai_payload, warnings = openai_converter.to_provider(
            ir_messages,
            tools=tools_spec,
        )

        response = openai_client.chat.completions.create(
            model=openai_model,
            **openai_payload,
        )

        ir_from_openai = openai_converter.from_provider(
            response.choices[0].message.model_dump()
        )
        ir_messages.extend(ir_from_openai)

        display_message(ir_messages[-1], "Final Response with All Tool Results")

        print_test_result("多个工具调用", True)
        return True

    except Exception as e:
        print_test_result("多个工具调用", False, str(e))
        return False


def test_streaming_response():
    """测试7: 流式响应转换 / Test 7: Streaming response conversion"""
    print_section("测试7: 流式响应 / Test 7: Streaming Response")

    try:
        ir_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Count from 1 to 5."}],
            },
        ]

        print("IR Input:")
        print(json.dumps(ir_messages, indent=2, ensure_ascii=False))

        # IR → OpenAI
        openai_payload, warnings = openai_converter.to_provider(ir_messages)

        print("\n=== Streaming Response ===")

        # 流式API调用
        stream: Generator = openai_client.chat.completions.create(
            model=openai_model,
            **openai_payload,
            stream=True,
        )

        collected_content = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                collected_content.append(content)

        print("\n")

        # 收集完整响应并转换为IR
        full_response = "".join(collected_content)

        # 模拟非流式响应的格式
        simulated_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": full_response,
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        ir_from_openai = openai_converter.from_provider(simulated_response)
        display_message(ir_from_openai[-1], "Streaming Response (IR)")

        print_test_result("流式响应", True)
        return True

    except Exception as e:
        print_test_result("流式响应", False, str(e))
        return False


def test_conversation_history():
    """测试8: 对话历史管理 / Test 8: Conversation history management"""
    print_section("测试8: 对话历史管理 / Test 8: Conversation History Management")

    try:
        ir_messages = []

        # 添加多轮对话
        conversation = [
            ("user", "Hi, my name is Alice."),
            ("assistant", "Hello Alice! Nice to meet you."),
            ("user", "What's my name?"),
            ("assistant", "Your name is Alice!"),
            ("user", "What can you do?"),
        ]

        print("=== Building Conversation History ===")

        for role, content in conversation:
            ir_messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                }
            )

            print(f"\n{role.upper()}: {content}")

            # 每隔一条消息调用API测试
            if role == "assistant":
                # IR → OpenAI
                openai_payload, warnings = openai_converter.to_provider(ir_messages)

                # API调用
                response = openai_client.chat.completions.create(
                    model=openai_model,
                    **openai_payload,
                )

                # OpenAI → IR
                ir_from_openai = openai_converter.from_provider(
                    response.choices[0].message.model_dump()
                )
                ir_messages.extend(ir_from_openai)

        print("\n=== Final Conversation ===")
        print(f"Total messages: {len(ir_messages)}")

        # 显示最后一条assistant消息
        for msg in reversed(ir_messages):
            if msg.get("role") == "assistant":
                display_message(msg, "Last Assistant Message")
                break

        print_test_result("对话历史管理", True)
        return True

    except Exception as e:
        print_test_result("对话历史管理", False, str(e))
        return False


def test_tool_choice_options():
    """测试9: 工具选择选项 / Test 9: Tool choice options"""
    print_section("测试9: 工具选择选项 / Test 9: Tool Choice Options")

    results = []

    # Test auto mode
    print("\n=== Tool Choice: auto ===")
    try:
        ir_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What's the weather?"}],
            },
        ]

        openai_payload, warnings = openai_converter.to_provider(
            ir_messages,
            tools=tools_spec,
            tool_choice={"mode": "auto"},
        )

        response = openai_client.chat.completions.create(
            model=openai_model,
            **openai_payload,
        )

        tool_calls = extract_tool_calls(
            openai_converter.from_provider(response.choices[0].message.model_dump())[-1]
        )

        if tool_calls:
            print(f"✓ Tool called: {tool_calls[0]['tool_name']}")
            results.append(True)
        else:
            print("✗ No tool called (may be expected if model chose not to use tools)")
            results.append(True)  # Not necessarily a failure

    except Exception as e:
        print(f"✗ Error: {e}")
        results.append(False)

    # Test none mode
    print("\n=== Tool Choice: none ===")
    try:
        ir_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What's 2+2?"}],
            },
        ]

        openai_payload, warnings = openai_converter.to_provider(
            ir_messages,
            tools=tools_spec,
            tool_choice={"mode": "none"},
        )

        response = openai_client.chat.completions.create(
            model=openai_model,
            **openai_payload,
        )

        ir_from_openai = openai_converter.from_provider(
            response.choices[0].message.model_dump()
        )
        display_message(ir_from_openai[-1], "Response with tool_choice='none'")

        tool_calls = extract_tool_calls(ir_from_openai[-1])
        if not tool_calls:
            print("✓ No tool called (as expected)")
            results.append(True)
        else:
            print("✗ Tool was called unexpectedly")
            results.append(False)

    except Exception as e:
        print(f"✗ Error: {e}")
        results.append(False)

    return all(results)


def test_file_upload():
    """测试10: 文件上传转换 / Test 10: File upload conversion"""
    print_section("测试10: 文件上传 / Test 10: File Upload")

    try:
        # 注意: OpenAI Chat API 目前不支持直接的文件上传
        # 测试converter是否能正确处理文件格式
        # Note: OpenAI Chat API doesn't support direct file upload currently
        # Test if converter handles file format correctly

        # IR格式包含文件引用
        ir_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Process this file:",
                    },
                    {
                        "type": "file",
                        "file_url": "https://example.com/document.pdf",
                        "file_name": "document.pdf",
                    },
                ],
            },
        ]

        print("IR Input (with file reference):")
        print(json.dumps(ir_messages, indent=2, ensure_ascii=False))

        # IR → OpenAI
        openai_payload, warnings = openai_converter.to_provider(ir_messages)
        print(f"\nWarnings: {warnings}")
        print("\nOpenAI Payload:")
        print(json.dumps(openai_payload, indent=2, ensure_ascii=False))

        print_test_result("文件上传转换", True)
        return True

    except Exception as e:
        print_test_result("文件上传转换", False, str(e))
        return False


# ============================================================================
# Main Test Runner
# ============================================================================


def run_all_tests():
    """运行所有测试 / Run all tests"""
    print("\n" + "=" * 80)
    print("OpenAI Chat Converter Integration Tests")
    print("=" * 80)
    print(f"\nModel: {openai_model}")
    print(f"Base URL: {openai_client.base_url}")

    tests = [
        ("Basic Text Message", test_basic_text_message),
        ("Image Message", test_image_message),
        ("Base64 Image", test_base64_image),
        ("Single Turn Tool Call", test_tool_call_single_turn),
        ("Multi-Turn Tool Call", test_tool_call_multi_turn),
        ("Multiple Tool Calls", test_multiple_tool_calls),
        ("Streaming Response", test_streaming_response),
        ("Conversation History", test_conversation_history),
        ("Tool Choice Options", test_tool_choice_options),
        ("File Upload", test_file_upload),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed + failed} | Passed: {passed} | Failed: {failed}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
