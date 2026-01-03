# NOTE: This example requires optional dependencies. Please install them first:
# pip install -e ".[openai,anthropic,google]"

import base64
import json
import os

import anthropic
import google.genai as genai
from dotenv import load_dotenv
from openai import OpenAI
from tools import available_tools, tools_spec

from llm_provider_converter.converters.anthropic_converter import AnthropicConverter
from llm_provider_converter.converters.google_converter import GoogleConverter
from llm_provider_converter.converters.openai_chat_converter import OpenAIChatConverter
from llm_provider_converter.types.ir import (
    Message,
    is_tool_call_part,
)

# Load environment variables from .env file
load_dotenv()

# --- Client and Converter Initialization ---
openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai_converter = OpenAIChatConverter()

anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
anthropic_converter = AnthropicConverter()

google_api_key = os.getenv("GOOGLE_API_KEY")
google_client = None
google_config = None
if google_api_key:
    google_client = genai.Client(api_key=google_api_key)
google_model_name = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash-latest")
google_converter = GoogleConverter()


def json_serializable_default(o):
    """Custom JSON serializer for objects that are not serializable by default."""
    if isinstance(o, bytes):
        return base64.b64encode(o).decode("utf-8")
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def build_google_config(google_payload):
    """Extract tools and tool_config from google_payload into a config dict."""
    config = {}
    if "tools" in google_payload and google_payload["tools"]:
        config["tools"] = google_payload["tools"]
    if "tool_config" in google_payload and google_payload["tool_config"]:
        config["tool_config"] = google_payload["tool_config"]
    return config if config else None


def main():
    """
    This example demonstrates a multi-provider conversation flow:
    1.  Start a conversation with OpenAI for an initial tool call.
    2.  Switch to Anthropic to get a summary.
    3.  Ask a follow-up question and switch to Google GenAI for the final tool call.
    """
    # Pre-build Google config once (tools and tool_config are static for this example)
    global google_config
    if google_client and google_config is None:
        # Convert empty messages just to get the tools configuration
        temp_payload, _ = google_converter.to_provider([], tools=tools_spec)
        google_config = build_google_config(temp_payload)

    # 1. Initial user message in IR format
    ir_messages: list[Message] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's the weather like in San Francisco?"}
            ],
        }
    ]
    print("--- Starting conversation ---")
    print(f"User: {ir_messages[0]['content'][0]['text']}")

    # 2. Convert to OpenAI format and make the first call
    print("\n>>> Calling OpenAI to handle the tool call...")
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

    # Print the assistant's text response, if any
    assistant_response_text = "".join(
        part.get("text", "")
        for part in ir_messages[-1]["content"]
        if part.get("type") == "text"
    )
    if assistant_response_text:
        print(f"Assistant: {assistant_response_text}")

    # 4. Execute the tool call
    last_message = ir_messages[-1]
    tool_call_part = next(
        (part for part in last_message["content"] if is_tool_call_part(part)),
        None,
    )

    if not tool_call_part:
        print("OpenAI did not request a tool call. Exiting.")
        return

    function_name = tool_call_part["tool_name"]
    function_args = tool_call_part["tool_input"]

    print(f"--- Executing tool: {function_name}({json.dumps(function_args)}) ---")
    function_to_call = available_tools[function_name]
    function_response = function_to_call(**function_args)

    # Create a tool result message in IR format
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
    print(f"Tool Result: {function_response}")

    # 6. Switch to Anthropic: Convert the entire history and call API
    print("\n>>> Calling Anthropic for a summary...")
    anthropic_payload, _ = anthropic_converter.to_provider(
        ir_messages, tools=tools_spec
    )

    anthropic_response = anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=1024,
        messages=anthropic_payload["messages"],
        tools=anthropic_payload.get("tools"),
    )

    # Get final response and convert back to IR
    ir_from_anthropic = anthropic_converter.from_provider(
        anthropic_response.model_dump()
    )
    ir_messages.extend(ir_from_anthropic)

    final_text = "".join(
        part.get("text", "")
        for part in ir_messages[-1]["content"]
        if part.get("type") == "text"
    )
    print(f"Assistant: {final_text}")

    # --- 9. Final Iteration with Google GenAI ---
    new_user_message_text = "That's great, now what about Paris?"
    print(f"\nUser: {new_user_message_text}")
    new_user_message: Message = {
        "role": "user",
        "content": [{"type": "text", "text": new_user_message_text}],
    }
    ir_messages.append(new_user_message)

    print("\n>>> Calling Google GenAI for the final answer...")

    if not google_client:
        print("Google API key not configured. Skipping Google GenAI call.")
        return

    google_payload, _ = google_converter.to_provider(ir_messages, tools=tools_spec)
    google_response = google_client.models.generate_content(
        model=google_model_name,
        contents=google_payload["contents"],
        config=google_config,
    )

    ir_from_google = google_converter.from_provider(google_response.model_dump())
    ir_messages.extend(ir_from_google)

    # Display the response - either text or tool calls
    last_msg = ir_messages[-1]
    final_google_text = "".join(
        part.get("text", "")
        for part in last_msg["content"]
        if part.get("type") == "text"
    )

    # Check for tool calls
    tool_calls = [part for part in last_msg["content"] if is_tool_call_part(part)]

    if final_google_text:
        print(f"Assistant: {final_google_text}")

    if tool_calls:
        for tc in tool_calls:
            print(
                f"Assistant: [Requesting tool call: {tc['tool_name']}({json.dumps(tc['tool_input'])})]"
            )

        # Execute the tool call immediately
        tool_call_part = tool_calls[0]  # Use the first tool call
        function_name = tool_call_part["tool_name"]
        function_args = tool_call_part["tool_input"]

        print(f"--- Executing tool: {function_name}({json.dumps(function_args)}) ---")
        function_to_call = available_tools[function_name]
        function_response = function_to_call(**function_args)

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
        print(f"Tool Result: {function_response}")

        # Call Google again with the tool result
        print("\n>>> Calling Google GenAI with the tool result...")
        google_payload, _ = google_converter.to_provider(ir_messages, tools=tools_spec)

        final_google_response = google_client.models.generate_content(
            model=google_model_name,
            contents=google_payload["contents"],
            config=google_config,
        )

        ir_from_final_google = google_converter.from_provider(
            final_google_response.model_dump()
        )
        ir_messages.extend(ir_from_final_google)

        final_summary_text = "".join(
            part.get("text", "")
            for part in ir_messages[-1]["content"]
            if part.get("type") == "text"
        )
        print(f"Assistant: {final_summary_text}")

    print("\n--- Conversation finished ---")


if __name__ == "__main__":
    main()
