#!/usr/bin/env python3
"""
测试Google Gemini API的顺序function calling - 使用tools.py中的工具

场景：用户想去天气最好的城市旅行
1. 先查询多个城市的天气
2. 根据天气结果决定目的地
3. 查询航班信息

这应该触发真正的顺序调用，因为第3步依赖第1-2步的结果。
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from examples.tools import available_tools, tools_spec
from llm_provider_converter.converters.google_converter import GoogleConverter

# 加载环境变量
load_dotenv()

# API配置
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("请设置GOOGLE_API_KEY环境变量")

MODEL = os.environ.get("GOOGLE_MODEL", "gemini-3-flash-preview")
BASE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
)

print(f"Using model: {MODEL}")
print(f"API endpoint: {BASE_URL}")
print()

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


def execute_tool_call(tool_name, tool_args):
    """执行工具调用"""
    if tool_name in available_tools:
        func = available_tools[tool_name]
        try:
            result = func(**tool_args)
            return result, False
        except Exception as e:
            return str(e), True
    else:
        return f"Unknown tool: {tool_name}", True


def test_sequential_travel_planning():
    """测试旅行规划场景 - 应该触发顺序调用"""
    print("=" * 80)
    print("测试场景：智能旅行规划")
    print("=" * 80)
    print()
    print("用户请求：我想从San Francisco出发去旅行，请帮我查一下Tokyo和Paris")
    print("         的天气，然后帮我预订到天气更温暖的那个城市的航班。")
    print()

    # 初始化对话
    ir_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I want to travel from San Francisco. Please check the weather "
                        "in Tokyo and Paris, then book a flight from San Francisco to "
                        "whichever city has warmer weather."
                    ),
                }
            ],
        }
    ]

    step = 1
    max_steps = 10
    all_signatures = []

    while step <= max_steps:
        print(f"\n{'=' * 80}")
        print(f"Step {step}")
        print(f"{'=' * 80}")

        # 转换为Google格式
        google_payload, _ = converter.to_provider(ir_messages, tools=tools_spec)
        print("\n--- 请求 ---")
        print(f"Contents: {len(google_payload['contents'])}")

        # 调用API
        response = call_api(google_payload["contents"], google_payload.get("tools"))

        # 检查原始响应
        if response.get("candidates"):
            candidate = response["candidates"][0]
            parts = candidate["content"]["parts"]
            print("\n--- 响应 ---")
            print(f"Response parts: {len(parts)}")

            # 分析每个part
            has_function_call = False
            for i, part in enumerate(parts):
                print(f"\nPart {i + 1}:")
                if "functionCall" in part:
                    has_function_call = True
                    func_call = part["functionCall"]
                    print(f"  Function: {func_call['name']}")
                    print(f"  Args: {json.dumps(func_call.get('args', {}), indent=4)}")
                    has_sig = "thoughtSignature" in part
                    print(f"  Has thoughtSignature: {has_sig}")
                    if has_sig:
                        sig = part["thoughtSignature"]
                        print(f"  Signature length: {len(sig)}")
                        all_signatures.append((step, i + 1, len(sig)))
                elif "text" in part:
                    text = part["text"]
                    print(f"  Text: {text[:150]}...")
                    has_sig = "thoughtSignature" in part
                    print(f"  Has thoughtSignature: {has_sig}")
                    if has_sig:
                        sig = part["thoughtSignature"]
                        print(f"  Signature length: {len(sig)}")
                        all_signatures.append((step, i + 1, len(sig)))

            # 如果没有function call，说明对话结束
            if not has_function_call:
                print("\n✓ 对话完成，没有更多function calls")
                break

        # 转换为IR
        ir_response = converter.from_provider(response)
        print("\n--- IR响应 ---")
        print(f"Messages: {len(ir_response)}")

        # 打印完整的IR结构
        print("\n--- 完整IR结构 ---")
        print(json.dumps(ir_response, indent=2, ensure_ascii=False))

        if ir_response:
            msg = ir_response[0]
            tool_calls = [p for p in msg["content"] if p["type"] == "tool_call"]
            text_parts = [p for p in msg["content"] if p["type"] == "text"]

            if tool_calls:
                print("\n--- Tool Calls分析 ---")
                print(f"Tool calls: {len(tool_calls)}")
                for i, tc in enumerate(tool_calls):
                    print(f"\n  Tool call {i + 1}:")
                    print(f"    Name: {tc['tool_name']}")
                    print(f"    Args: {json.dumps(tc['tool_input'], indent=6)}")
                    print(f"    ID: {tc['tool_call_id']}")
                    has_metadata = "provider_metadata" in tc
                    print(f"    Has provider_metadata: {has_metadata}")
                    if has_metadata and "google" in tc["provider_metadata"]:
                        has_sig = (
                            "thought_signature" in tc["provider_metadata"]["google"]
                        )
                        print(f"    Has thought_signature: {has_sig}")
                        if has_sig:
                            sig_preview = tc["provider_metadata"]["google"][
                                "thought_signature"
                            ][:50]
                            print(f"    Signature preview: {sig_preview}...")

                # 添加assistant的响应到历史
                ir_messages.extend(ir_response)

                # 执行所有tool calls并添加结果
                print("\n--- 执行Tool Calls ---")
                tool_results = []
                for tc in tool_calls:
                    tool_name = tc["tool_name"]
                    tool_args = tc["tool_input"]
                    result, is_error = execute_tool_call(tool_name, tool_args)
                    print(f"  {tool_name}({json.dumps(tool_args)})")
                    print(f"    → {result}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_call_id": tc["tool_call_id"],
                            "result": result,
                            "is_error": is_error,
                        }
                    )

                # 添加tool results到历史
                ir_messages.append({"role": "user", "content": tool_results})

            elif text_parts:
                print(f"Text response: {text_parts[0].get('text', 'N/A')}")
                break

        step += 1

    if step > max_steps:
        print(f"\n⚠ 达到最大步骤数 {max_steps}")

    print(f"\n{'=' * 80}")
    print(f"总步骤数: {step}")
    print(f"总消息数: {len(ir_messages)}")
    print(f"{'=' * 80}")

    # 总结thought signatures
    print("\n--- Thought Signatures总结 ---")
    if all_signatures:
        print(f"总共出现的signatures: {len(all_signatures)}")
        for step_num, part_num, sig_len in all_signatures:
            print(f"  Step {step_num}, Part {part_num}: {sig_len} bytes")
    else:
        print("没有发现thought signatures")

    # 验证signatures在对话历史中都被保留
    print("\n--- 验证Signatures保留 ---")
    google_payload, _ = converter.to_provider(ir_messages, tools=tools_spec)

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
    test_sequential_travel_planning()
