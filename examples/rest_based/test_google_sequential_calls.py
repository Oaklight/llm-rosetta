#!/usr/bin/env python3
"""
测试Google Gemini API的顺序function calling和thought signature处理

这个测试会触发多个城市的天气查询，可能导致：
1. 并行调用（3个城市同时查询）
2. 顺序调用（先查一个，再查另一个，最后查第三个）

验证每个步骤的thought signature都被正确保留和返回。
"""

import json
import os

import requests
from dotenv import load_dotenv

from llmir.converters.google_converter import GoogleConverter

# 加载环境变量
load_dotenv()

# API配置
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("请设置GOOGLE_API_KEY环境变量")

MODEL = os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash")
BASE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
)

print(f"Using model: {MODEL}")
print(f"API endpoint: {BASE_URL}")
print()

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
                "description": "The city name, e.g. San Francisco, CA",
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


def mock_weather_api(location):
    """模拟天气API"""
    weather_data = {
        "Tokyo": {"temperature": "18", "condition": "cloudy", "unit": "celsius"},
        "Shanghai": {"temperature": "22", "condition": "rainy", "unit": "celsius"},
        "Paris": {"temperature": "12", "condition": "sunny", "unit": "celsius"},
    }
    return weather_data.get(
        location, {"temperature": "20", "condition": "unknown", "unit": "celsius"}
    )


def test_multi_city_weather():
    """测试多城市天气查询，可能触发顺序或并行调用"""
    print("=" * 80)
    print("测试: 多城市天气查询（Tokyo, Shanghai, Paris）")
    print("=" * 80)

    # 初始化对话
    ir_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's the weather like in Tokyo, Shanghai, and Paris? Please check each city.",
                }
            ],
        }
    ]

    step = 1
    max_steps = 10  # 防止无限循环

    while step <= max_steps:
        print(f"\n{'=' * 80}")
        print(f"Step {step}")
        print(f"{'=' * 80}")

        # 转换为Google格式
        google_payload, _ = converter.to_provider(ir_messages, tools=[weather_tool])
        print("\n--- 请求 ---")
        print(f"Contents: {len(google_payload['contents'])}")

        # 调用API
        response = call_api(google_payload["contents"], google_payload.get("tools"))
        print("\n--- 响应 ---")

        # 打印原始响应以便调试
        print("\n--- 原始响应JSON ---")
        print(json.dumps(response, indent=2)[:1000] + "...")

        # 检查原始响应
        if response.get("candidates"):
            candidate = response["candidates"][0]
            parts = candidate["content"]["parts"]
            print(f"Response parts: {len(parts)}")

            # 分析每个part
            has_function_call = False
            for i, part in enumerate(parts):
                print(f"\nPart {i + 1}:")
                if "functionCall" in part:
                    has_function_call = True
                    func_call = part["functionCall"]
                    print(f"  Function: {func_call['name']}")
                    print(f"  Args: {func_call.get('args', {})}")
                    print(f"  Has thoughtSignature: {'thoughtSignature' in part}")
                    if "thoughtSignature" in part:
                        sig = part["thoughtSignature"]
                        print(f"  Signature length: {len(sig)}")
                elif "text" in part:
                    text = part["text"]
                    print(f"  Text: {text[:100]}...")
                    print(f"  Has thoughtSignature: {'thoughtSignature' in part}")

            # 如果没有function call，说明对话结束
            if not has_function_call:
                print("\n✓ 对话完成，没有更多function calls")
                break

        # 转换为IR
        ir_response = converter.from_provider(response)
        print("\n--- IR响应 ---")
        print(f"Messages: {len(ir_response)}")

        if ir_response:
            msg = ir_response[0]
            tool_calls = [p for p in msg["content"] if p["type"] == "tool_call"]
            text_parts = [p for p in msg["content"] if p["type"] == "text"]

            if tool_calls:
                print(f"Tool calls: {len(tool_calls)}")
                for i, tc in enumerate(tool_calls):
                    print(f"\n  Tool call {i + 1}:")
                    print(f"    Name: {tc['tool_name']}")
                    print(f"    Args: {tc['tool_input']}")
                    print(f"    ID: {tc['tool_call_id']}")
                    has_metadata = "provider_metadata" in tc
                    print(f"    Has provider_metadata: {has_metadata}")
                    if has_metadata and "google" in tc["provider_metadata"]:
                        has_sig = (
                            "thought_signature" in tc["provider_metadata"]["google"]
                        )
                        print(f"    Has thought_signature: {has_sig}")
                        if has_sig:
                            sig = tc["provider_metadata"]["google"]["thought_signature"]
                            print(f"    Signature length: {len(sig)}")

                # 添加assistant的响应到历史
                ir_messages.extend(ir_response)

                # 执行所有tool calls并添加结果
                print("\n--- 执行Tool Calls ---")
                tool_results = []
                for tc in tool_calls:
                    location = tc["tool_input"].get("location", "Unknown")
                    result = mock_weather_api(location)
                    print(f"  {location}: {result}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_call_id": tc["tool_call_id"],
                            "result": json.dumps(result),
                        }
                    )

                # 添加tool results到历史
                ir_messages.append({"role": "user", "content": tool_results})

            elif text_parts:
                print(f"Text response: {text_parts[0].get('text', 'N/A')[:200]}")
                break

        step += 1

    if step > max_steps:
        print(f"\n⚠ 达到最大步骤数 {max_steps}")

    print(f"\n{'=' * 80}")
    print(f"总步骤数: {step}")
    print(f"总消息数: {len(ir_messages)}")
    print(f"{'=' * 80}")

    # 验证thought signatures在整个对话中都被保留
    print("\n--- 验证Thought Signatures ---")
    google_payload, _ = converter.to_provider(ir_messages, tools=[weather_tool])

    signature_count = 0
    for i, content in enumerate(google_payload["contents"]):
        if content.get("role") == "model":
            for j, part in enumerate(content.get("parts", [])):
                if "thoughtSignature" in part:
                    signature_count += 1
                    print(f"Content {i + 1}, Part {j + 1}: ✓ Signature preserved")

    print(f"\n总共保留的signatures: {signature_count}")
    print("\n✓ 测试完成！")


if __name__ == "__main__":
    test_multi_city_weather()
