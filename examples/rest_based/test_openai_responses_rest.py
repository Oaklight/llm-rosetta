#!/usr/bin/env python3
"""
测试OpenAI Responses API的REST调用

使用/v1/responses endpoint和OpenAIResponsesConverter
参考：https://platform.openai.com/docs/api-reference/responses
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from examples.tools import available_tools, tools_spec
from llmir.converters.openai_responses_converter import (
    OpenAIResponsesConverter,
)
from src.llmir.types.ir import Message

# 加载环境变量
load_dotenv()

# API配置
API_KEY = os.environ.get("OPENAI_RESPONSES_API_KEY")
BASE_URL = os.environ.get("OPENAI_RESPONSES_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("OPENAI_RESPONSES_MODEL", "o4-mini")

if not API_KEY:
    raise ValueError("请设置OPENAI_RESPONSES_API_KEY环境变量")

print(f"Using model: {MODEL}")
print(f"API endpoint: {BASE_URL}/responses")
print()

converter = OpenAIResponsesConverter()


def call_api(input_items, tools=None):
    """调用OpenAI Responses API
    
    参考文档示例：
    curl https://api.openai.com/v1/responses \\
      -H "Authorization: Bearer $OPENAI_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "o4-mini",
        "input": [
          {
            "type": "message",
            "role": "user",
            "content": "Hello!"
          }
        ]
      }'
    """
    url = f"{BASE_URL}/responses"

    payload = {
        "model": MODEL,
        "input": input_items,
    }

    if tools:
        payload["tools"] = tools

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def execute_tool_call(tool_name, tool_args):
    """执行工具调用"""
    if tool_name in available_tools:
        func = available_tools[tool_name]
        try:
            # 处理JSON字符串参数
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)
            result = func(**tool_args)
            return result, False
        except Exception as e:
            return str(e), True
    else:
        return f"Unknown tool: {tool_name}", True


def test_simple_weather():
    """测试简单的天气查询"""
    print("=" * 80)
    print("测试1：简单天气查询")
    print("=" * 80)

    # 初始化对话
    ir_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What's the weather like in Tokyo?"}],
        }
    ]

    step = 1
    max_steps = 5

    while step <= max_steps:
        print(f"\n{'=' * 80}")
        print(f"Step {step}")
        print(f"{'=' * 80}")

        # 转换为OpenAI Responses格式
        responses_payload, _ = converter.to_provider(ir_messages, tools=tools_spec)
        print("\n--- 请求 ---")
        print(f"Input items: {len(responses_payload['input'])}")
        if "tools" in responses_payload:
            print(f"Tools: {len(responses_payload['tools'])}")

        # 打印请求payload示例
        print("\n--- Request Payload (first 500 chars) ---")
        print(json.dumps(responses_payload, indent=2)[:500] + "...")

        # 调用API
        response = call_api(responses_payload["input"], responses_payload.get("tools"))

        print("\n--- 原始响应 ---")
        print(f"Response keys: {list(response.keys())}")
        if "output" in response:
            print(f"Output items: {len(response['output'])}")
            # 打印output的前几个item类型
            for i, item in enumerate(response["output"][:3]):
                print(f"  Item {i + 1}: type={item.get('type')}")

        # 转换为IR
        ir_response = converter.from_provider(response)
        print("\n--- IR响应 ---")
        print(f"Messages: {len(ir_response)}")

        # 打印完整IR
        print("\n--- 完整IR结构 ---")
        print(json.dumps(ir_response, indent=2, ensure_ascii=False))

        if ir_response:
            msg = ir_response[0]
            tool_calls = [p for p in msg["content"] if p["type"] == "tool_call"]
            text_parts = [p for p in msg["content"] if p["type"] == "text"]
            reasoning_parts = [p for p in msg["content"] if p["type"] == "reasoning"]

            if reasoning_parts:
                print("\n--- Reasoning Parts ---")
                for i, rp in enumerate(reasoning_parts):
                    reasoning_text = rp.get("reasoning", "")
                    print(f"Reasoning {i + 1}: {reasoning_text[:200]}...")

            if tool_calls:
                print("\n--- Tool Calls ---")
                for i, tc in enumerate(tool_calls):
                    print(f"\nTool call {i + 1}:")
                    print(f"  Name: {tc['tool_name']}")
                    print(f"  Args: {json.dumps(tc['tool_input'], indent=4)}")
                    print(f"  ID: {tc['tool_call_id']}")

                # 添加assistant响应到历史
                ir_messages.extend(ir_response)

                # 执行工具调用
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
                ir_messages.append(Message(role="user", content=tool_results))

            elif text_parts:
                print("\n--- 最终回答 ---")
                for tp in text_parts:
                    print(tp.get("text", ""))
                break

        step += 1

    print(f"\n{'=' * 80}")
    print(f"总步骤数: {step}")
    print(f"总消息数: {len(ir_messages)}")
    print(f"{'=' * 80}")
    print("\n✓ 测试1完成！\n\n")


def test_travel_planning():
    """测试复杂的旅行规划（顺序调用）"""
    print("=" * 80)
    print("测试2：智能旅行规划（顺序工具调用）")
    print("=" * 80)

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
    all_reasoning = []

    while step <= max_steps:
        print(f"\n{'=' * 80}")
        print(f"Step {step}")
        print(f"{'=' * 80}")

        # 转换为OpenAI Responses格式
        responses_payload, _ = converter.to_provider(ir_messages, tools=tools_spec)
        print("\n--- 请求 ---")
        print(f"Input items: {len(responses_payload['input'])}")

        # 调用API
        response = call_api(responses_payload["input"], responses_payload.get("tools"))

        print("\n--- 响应摘要 ---")
        if "output" in response:
            print(f"Output items: {len(response['output'])}")
            item_types = [item.get("type") for item in response["output"]]
            print(f"Item types: {item_types}")

        # 转换为IR
        ir_response = converter.from_provider(response)

        if ir_response:
            msg = ir_response[0]
            tool_calls = [p for p in msg["content"] if p["type"] == "tool_call"]
            text_parts = [p for p in msg["content"] if p["type"] == "text"]
            reasoning_parts = [p for p in msg["content"] if p["type"] == "reasoning"]

            if reasoning_parts:
                for rp in reasoning_parts:
                    reasoning_text = rp.get("reasoning", "")
                    all_reasoning.append((step, reasoning_text))
                    print(f"\n💭 Reasoning: {reasoning_text[:150]}...")

            if tool_calls:
                print(f"\n🔧 Tool calls: {len(tool_calls)}")
                for tc in tool_calls:
                    print(f"  - {tc['tool_name']}({json.dumps(tc['tool_input'])})")

                # 执行工具
                ir_messages.extend(ir_response)
                tool_results = []
                for tc in tool_calls:
                    result, is_error = execute_tool_call(
                        tc["tool_name"], tc["tool_input"]
                    )
                    print(f"    → {result}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_call_id": tc["tool_call_id"],
                            "result": result,
                            "is_error": is_error,
                        }
                    )
                ir_messages.append(Message(role="user", content=tool_results))

            elif text_parts:
                print("\n📝 最终回答:")
                for tp in text_parts:
                    print(f"  {tp.get('text', '')}")
                break

        step += 1

    print(f"\n{'=' * 80}")
    print(f"总步骤数: {step}")
    print(f"总消息数: {len(ir_messages)}")
    if all_reasoning:
        print(f"总推理次数: {len(all_reasoning)}")
        print("\n推理摘要:")
        for step_num, reasoning in all_reasoning:
            print(f"  Step {step_num}: {reasoning[:100]}...")
    print(f"{'=' * 80}")
    print("\n✓ 测试2完成！")


if __name__ == "__main__":
    test_simple_weather()
    test_travel_planning()
