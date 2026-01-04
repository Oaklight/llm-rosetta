#!/usr/bin/env python
"""
Simple test script for OpenAI Responses API only
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from llm_provider_converter.converters.openai_responses_converter import (
    OpenAIResponsesConverter,
)
from llm_provider_converter.types.ir import extract_text_content, extract_tool_calls

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
    return json.dumps({"location": location, "temperature": "15", "unit": unit})


def main():
    print("=" * 80)
    print("OpenAI Responses API Test")
    print("=" * 80)

    # Initialize client
    model = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4o")

    client = OpenAI(
        api_key=os.getenv("OPENAI_RESPONSES_API_KEY"),
        base_url=os.getenv("OPENAI_RESPONSES_BASE_URL"),
    )
    converter = OpenAIResponsesConverter()

    # Create IR message
    ir_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What's the weather like in Tokyo?"}],
        }
    ]

    print(f"\nModel: {model}")
    print(f"User: {ir_messages[0]['content'][0]['text']}")

    # Convert to Responses API format
    print("\n--- Converting IR to Responses API format ---")
    payload, warnings = converter.to_provider(
        ir_messages, tools=tools_spec, tool_choice={"mode": "auto"}
    )

    if warnings:
        print(f"Warnings: {warnings}")

    print(f"Payload keys: {list(payload.keys())}")
    print(f"Input items: {len(payload['input'])}")
    print(f"Tools: {len(payload.get('tools', []))}")

    # Debug: print the actual tools structure
    print("\nTools structure:")
    print(json.dumps(payload.get("tools", []), indent=2))

    # Call API
    print("\n--- Calling Responses API ---")
    try:
        response = client.responses.create(
            model=model,
            input=payload["input"],
            tools=payload.get("tools"),
            tool_choice=payload.get("tool_choice"),
        )

        print(f"Response type: {type(response)}")
        response_dict = response.model_dump()
        print(f"Response keys: {list(response_dict.keys())}")

        # Debug: print the output structure
        print(f"\nResponse output type: {type(response_dict.get('output'))}")
        if response_dict.get("output"):
            print(f"Output length: {len(response_dict['output'])}")
            print(
                f"Output sample: {json.dumps(response_dict['output'][:2] if isinstance(response_dict['output'], list) else response_dict['output'], indent=2)}"
            )

        # Convert back to IR
        print("\n--- Converting Response to IR ---")
        ir_from_response = converter.from_provider(response_dict)

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

                # Debug: print the final payload
                print(f"\nFinal payload input items: {len(final_payload['input'])}")
                print("Final payload structure:")
                for i, item in enumerate(final_payload["input"][-5:]):  # Last 5 items
                    print(
                        f"  Item {i}: type={item.get('type')}, keys={list(item.keys())[:5]}"
                    )

                final_response = client.responses.create(
                    model=model,
                    input=final_payload["input"],
                    tools=final_payload.get("tools"),
                    tool_choice=final_payload.get("tool_choice"),
                )

                print(f"Final response status: {final_response.status}")

                # Convert final response
                ir_final = converter.from_provider(final_response.model_dump())
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

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
