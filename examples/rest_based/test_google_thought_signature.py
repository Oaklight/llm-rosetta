#!/usr/bin/env python3
"""
测试Google Gemini API的thought signature处理

验证：
1. Thought signature从响应中正确提取
2. Thought signature在多轮对话中正确保留和返回
3. 并行function calls时只有第一个有signature
"""

import os

import requests
from dotenv import load_dotenv

from llmir.converters.google_genai import GoogleConverter

# 加载环境变量
load_dotenv()

# API配置
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("请设置GOOGLE_API_KEY环境变量")

MODEL = "gemini-2.5-flash"
BASE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
)

# 工具定义
weather_tool = {
    "type": "function",
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            }
        },
        "required": ["location"],
    },
}

converter = GoogleConverter()


def call_api(contents, tools=None):
    """调用Google API"""
    payload = {"contents": contents}
    if tools:
        payload["tools"] = tools
        payload["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}

    response = requests.post(
        BASE_URL,
        headers={"Content-Type": "application/json"},
        params={"key": API_KEY},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def test_single_function_call():
    """测试单个function call的thought signature"""
    print("=" * 80)
    print("测试1: 单个Function Call的Thought Signature")
    print("=" * 80)

    # 第一轮：用户请求
    ir_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What's the weather in Paris?"}],
        }
    ]

    google_payload, _ = converter.to_provider(ir_messages, tools=[weather_tool])
    print("\n--- 第一轮请求 ---")
    print(f"Contents: {len(google_payload['contents'])}")

    # 调用API
    response = call_api(google_payload["contents"], google_payload.get("tools"))
    print("\n--- 第一轮响应 ---")
    print(f"Candidates: {len(response['candidates'])}")

    # 转换为IR
    ir_response = converter.from_provider(response)
    print("\n--- IR响应 ---")
    print(f"Messages: {len(ir_response)}")

    if ir_response:
        msg = ir_response[0]
        print(f"Role: {msg['role']}")
        print(f"Content parts: {len(msg['content'])}")

        tool_call_found = False
        tool_call_id = None

        for part in msg["content"]:
            print(f"\nPart type: {part.get('type')}")
            if part["type"] == "tool_call":
                tool_call_found = True
                tool_call_id = part.get("tool_call_id")
                print(f"Tool call: {part['tool_name']}")
                print(f"Tool call ID: {tool_call_id}")
                print(f"Has provider_metadata: {'provider_metadata' in part}")
                if "provider_metadata" in part:
                    metadata = part["provider_metadata"]
                    print(f"Metadata keys: {list(metadata.keys())}")
                    if "google" in metadata:
                        print(
                            f"Google metadata keys: {list(metadata['google'].keys())}"
                        )
                        if "thought_signature" in metadata["google"]:
                            sig = metadata["google"]["thought_signature"]
                            print(f"Thought signature length: {len(sig)}")
                            print(f"Thought signature preview: {sig[:50]}...")
            elif part["type"] == "text":
                print(f"Text: {part.get('text', 'N/A')[:100]}")

        if not tool_call_found:
            print("\n⚠ No tool call found in response, skipping second round")
            print("✓ 测试1完成（部分）\n")
            return

    # 第二轮：返回tool result
    ir_messages.extend(ir_response)
    ir_messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_call_id": tool_call_id,
                    "result": '{"temperature": "15C", "condition": "sunny"}',
                }
            ],
        }
    )

    google_payload, _ = converter.to_provider(ir_messages, tools=[weather_tool])
    print("\n--- 第二轮请求 ---")
    print(f"Contents: {len(google_payload['contents'])}")

    # 检查thought signature是否被保留
    model_content = google_payload["contents"][1]  # 第二个content是model的响应
    print(f"\nModel content parts: {len(model_content['parts'])}")
    for i, part in enumerate(model_content["parts"]):
        print(f"\nPart {i + 1}:")
        print(f"  Keys: {list(part.keys())}")
        if "thoughtSignature" in part:
            sig = part["thoughtSignature"]
            print(f"  ✓ Thought signature preserved (length: {len(sig)})")
        else:
            print("  ✗ No thought signature")

    # 调用API获取最终响应
    response = call_api(google_payload["contents"], google_payload.get("tools"))
    print("\n--- 第二轮响应 ---")
    final_ir = converter.from_provider(response)
    if final_ir:
        print(f"Final text: {final_ir[0]['content'][0].get('text', 'N/A')}")

    print("\n✓ 测试1完成\n")


def test_parallel_function_calls():
    """测试并行function calls的thought signature"""
    print("=" * 80)
    print("测试2: 并行Function Calls的Thought Signature")
    print("=" * 80)

    # 请求检查两个城市的天气
    ir_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's the weather in Paris and London?"}
            ],
        }
    ]

    google_payload, _ = converter.to_provider(ir_messages, tools=[weather_tool])
    print("\n--- 请求 ---")
    print(f"Contents: {len(google_payload['contents'])}")

    # 调用API
    response = call_api(google_payload["contents"], google_payload.get("tools"))
    print("\n--- 响应 ---")

    # 检查原始响应
    if response.get("candidates"):
        parts = response["candidates"][0]["content"]["parts"]
        print(f"Response parts: {len(parts)}")
        for i, part in enumerate(parts):
            print(f"\nPart {i + 1}:")
            if "functionCall" in part:
                print(f"  Function: {part['functionCall']['name']}")
                print(f"  Has thoughtSignature: {'thoughtSignature' in part}")
                if "thoughtSignature" in part:
                    sig = part["thoughtSignature"]
                    print(f"  Signature length: {len(sig)}")

    # 转换为IR
    ir_response = converter.from_provider(response)
    print("\n--- IR响应 ---")
    print(f"Messages: {len(ir_response)}")

    if ir_response:
        msg = ir_response[0]
        tool_calls = [p for p in msg["content"] if p["type"] == "tool_call"]
        print(f"Tool calls: {len(tool_calls)}")

        for i, tc in enumerate(tool_calls):
            print(f"\nTool call {i + 1}: {tc['tool_name']}")
            has_metadata = "provider_metadata" in tc
            print(f"  Has provider_metadata: {has_metadata}")
            if has_metadata and "google" in tc["provider_metadata"]:
                has_sig = "thought_signature" in tc["provider_metadata"]["google"]
                print(f"  Has thought_signature: {has_sig}")
                if has_sig:
                    sig = tc["provider_metadata"]["google"]["thought_signature"]
                    print(f"  Signature length: {len(sig)}")

    print("\n✓ 测试2完成\n")


if __name__ == "__main__":
    test_single_function_call()
    test_parallel_function_calls()
    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)
