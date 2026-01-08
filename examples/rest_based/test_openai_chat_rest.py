#!/usr/bin/env python
"""
REST API test for OpenAI Chat Completions
使用原始HTTP请求测试，避免SDK包装
"""

import json
import os

import requests
from dotenv import load_dotenv

from llmir.converters.openai_chat import (
    OpenAIChatConverter,
)
from llmir.types.ir import extract_text_content, extract_tool_calls

# Load environment variables
load_dotenv()

# Tool definition
tools_spec = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]


# Mock tool function
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Mock weather function"""
    return json.dumps({"location": location, "temperature": "22", "unit": unit})


def main():
    print("=" * 80)
    print("OpenAI Chat Completions REST API Test")
    print("=" * 80)

    # API configuration
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    print(f"\nAPI Base URL: {base_url}")
    print(f"Model: {model}")

    # Initialize converter
    converter = OpenAIChatConverter()

    # Create IR message
    ir_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's the weather like in San Francisco?"}
            ],
        }
    ]

    print(f"\nUser: {ir_messages[0]['content'][0]['text']}")

    # Convert to OpenAI format
    print("\n--- Converting IR to OpenAI Chat format ---")
    payload, warnings = converter.to_provider(
        ir_messages, tools=tools_spec, tool_choice={"mode": "auto"}
    )

    if warnings:
        print(f"Warnings: {warnings}")

    print(f"Payload keys: {list(payload.keys())}")
    print(f"Messages: {len(payload['messages'])}")
    print(f"Tools: {len(payload.get('tools', []))}")

    # Prepare REST API request
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    request_body = {
        "model": model,
        "messages": payload["messages"],
        "tools": payload.get("tools"),
        "tool_choice": payload.get("tool_choice"),
    }

    print("\n--- Request Body ---")
    print(json.dumps(request_body, indent=2, ensure_ascii=False))

    # Call REST API
    print("\n--- Calling REST API ---")
    try:
        response = requests.post(url, headers=headers, json=request_body, timeout=30)
        response.raise_for_status()

        response_data = response.json()
        print(f"Response status: {response.status_code}")
        print(f"Response keys: {list(response_data.keys())}")

        # Print raw response
        print("\n--- Raw Response ---")
        print(json.dumps(response_data, indent=2, ensure_ascii=False))

        # Convert back to IR
        print("\n--- Converting Response to IR ---")
        ir_from_response = converter.from_provider(response_data)

        print(f"IR messages: {len(ir_from_response)}")
        for i, msg in enumerate(ir_from_response):
            print(f"\nMessage {i + 1}:")
            print(f"  Role: {msg['role']}")
            print(f"  Content parts: {len(msg['content'])}")

            # Display text
            text = extract_text_content(msg)
            if text:
                print(f"  Text: {text[:100]}...")

            # Display tool calls
            tool_calls = extract_tool_calls(msg)
            if tool_calls:
                print(f"  Tool calls: {len(tool_calls)}")
                for tc in tool_calls:
                    print(f"    - {tc['tool_name']}({json.dumps(tc['tool_input'])})")

        # Execute tool calls if any
        if ir_from_response and len(ir_from_response) > 0:
            last_message = ir_from_response[-1]
            tool_calls = extract_tool_calls(last_message)

            if tool_calls:
                print("\n--- Executing Tool Calls ---")
                # Add response to message history
                ir_messages.extend(ir_from_response)

                # Execute each tool call
                for tc in tool_calls:
                    function_name = tc["tool_name"]
                    function_args = tc["tool_input"]

                    print(f"\nExecuting: {function_name}({json.dumps(function_args)})")

                    # Execute the tool
                    function_response = get_current_weather(**function_args)
                    print(f"Result: {function_response}")

                    # Add tool result to message history
                    tool_result_msg = {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_call_id": tc["tool_call_id"],
                                "result": function_response,
                            }
                        ],
                    }
                    ir_messages.append(tool_result_msg)

                # Get final response with tool results
                print("\n--- Getting Final Response ---")
                final_payload, _ = converter.to_provider(
                    ir_messages, tools=tools_spec, tool_choice={"mode": "auto"}
                )

                print(f"\nFinal payload messages: {len(final_payload['messages'])}")

                final_request_body = {
                    "model": model,
                    "messages": final_payload["messages"],
                    "tools": final_payload.get("tools"),
                    "tool_choice": final_payload.get("tool_choice"),
                }

                print("\n--- Final Request Body ---")
                print(json.dumps(final_request_body, indent=2, ensure_ascii=False))

                final_response = requests.post(
                    url, headers=headers, json=final_request_body, timeout=30
                )
                final_response.raise_for_status()

                final_response_data = final_response.json()
                print(f"\nFinal response status: {final_response.status_code}")

                # Print final raw response
                print("\n--- Final Raw Response ---")
                print(json.dumps(final_response_data, indent=2, ensure_ascii=False))

                # Convert final response
                ir_final = converter.from_provider(final_response_data)
                ir_messages.extend(ir_final)

                print(f"\nFinal IR messages: {len(ir_final)}")
                for i, msg in enumerate(ir_final):
                    print(f"\nFinal Message {i + 1}:")
                    print(f"  Role: {msg['role']}")
                    text = extract_text_content(msg)
                    if text:
                        print(f"  Text: {text}")

        print("\n✓ Test completed successfully!")
        print(f"\nTotal conversation messages: {len(ir_messages)}")

    except requests.exceptions.RequestException as e:
        print(f"\n✗ HTTP Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
