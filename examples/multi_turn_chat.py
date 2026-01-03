import json
import os

import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from tools import available_tools, tools_spec

from llm_provider_converter.converters.anthropic_converter import AnthropicConverter
from llm_provider_converter.converters.openai_chat_converter import OpenAIChatConverter
from llm_provider_converter.types.ir import (
    Message,
    ToolCallPart,
    ToolResultPart,
    is_tool_call_part,
)

# Load environment variables from .env file
load_dotenv()

# --- Client and Converter Initialization ---
openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
openai_converter = OpenAIChatConverter()

anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-4-sonnet")
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
anthropic_converter = AnthropicConverter()


def main():
    """
    This example demonstrates a multi-provider conversation flow:
    1.  Start a conversation with OpenAI.
    2.  OpenAI decides to use a tool.
    3.  Execute the tool call locally.
    4.  Append the tool result to the conversation history (in IR format).
    5.  Switch to Anthropic, converting the entire history to its format.
    6.  Get the final response from Anthropic.
    """
    # 1. Initial user message in IR format
    ir_messages: list[Message] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's the weather like in San Francisco?"}
            ],
        }
    ]
    print("--- 1. Initial IR Messages ---")
    print(json.dumps(ir_messages, indent=2))

    # 2. Convert to OpenAI format and make the first call
    print("\n--- 2. Converting to OpenAI and calling API ---")
    openai_payload, _ = openai_converter.to_provider(
        ir_messages, tools=tools_spec, tool_choice={"mode": "auto"}
    )
    openai_response = openai_client.chat.completions.create(
        model=openai_model, **openai_payload
    )

    # Append OpenAI's response (as IR) to history
    ir_from_openai = openai_converter.from_provider(
        openai_response.choices[0].message.model_dump()
    )
    ir_messages.extend(ir_from_openai)
    print("\n--- 3. IR History after OpenAI Tool Call ---")
    print(json.dumps(ir_messages, indent=2))

    # 4. Execute the tool call
    print("\n--- 4. Executing Tool Call ---")
    last_message = ir_messages[-1]
    tool_call_part = next(
        (part for part in last_message["content"] if is_tool_call_part(part)),
        None,
    )

    if not tool_call_part:
        print("OpenAI did not request a tool call. Exiting.")
        return

    # The `tool_call_part` dictionary itself contains the necessary information.
    function_name = tool_call_part["tool_name"]
    function_args = tool_call_part["tool_input"]

    print(f"Executing: {function_name}({function_args})")
    function_to_call = available_tools[function_name]
    function_response = function_to_call(**function_args)

    # Create a tool result message in IR format
    # NOTE: For Anthropic, tool results must be sent in a `user` role message
    # that immediately follows the `assistant` message containing the tool call.
    tool_result_message: Message = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": tool_call_part["tool_call_id"],
                "result": function_response,
            }
        ],
    }
    ir_messages.append(tool_result_message)
    print("\n--- 5. IR History after adding Tool Result ---")
    print(json.dumps(ir_messages, indent=2))

    # 6. Switch to Anthropic: Convert the entire history and call API
    print("\n--- 6. Converting to Anthropic and calling API ---")
    anthropic_payload, _ = anthropic_converter.to_provider(
        ir_messages, tools=tools_spec
    )

    # Anthropic's API has a different structure
    anthropic_response = anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=1024,
        messages=anthropic_payload["messages"],
        tools=anthropic_payload.get("tools"),
    )

    # 7. Get final response and convert back to IR
    print("\n--- 7. Raw Anthropic Response ---")
    anthropic_response_dump = anthropic_response.model_dump()
    print(json.dumps(anthropic_response_dump, indent=2))

    ir_from_anthropic = anthropic_converter.from_provider(anthropic_response_dump)
    ir_messages.extend(ir_from_anthropic)
    print("\n--- 8. Final IR History from Anthropic ---")
    print(json.dumps(ir_messages, indent=2))

    final_text = ""
    if ir_messages and ir_messages[-1]["content"]:
        final_text = "".join(
            part.get("text", "")
            for part in ir_messages[-1]["content"]
            if part.get("type") == "text"
        )
    print(f"\n--- Final Answer from Anthropic ---\n{final_text}")


if __name__ == "__main__":
    main()
