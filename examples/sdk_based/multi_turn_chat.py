# NOTE: This example requires optional dependencies. Please install them first:
# pip install -e ".[openai,anthropic,google]"

import base64
import json
import os
import sys
from typing import List

import anthropic
import google.genai as genai
from dotenv import load_dotenv
from openai import OpenAI

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from examples.tools import available_tools, tools_spec
from llmir.converters.anthropic import AnthropicConverter
from llmir.converters.google import GoogleConverter
from llmir.converters.openai_chat import OpenAIChatConverter
from llmir.converters.openai_responses import (
    OpenAIResponsesConverter,
)
from llmir.types.ir import (
    Message,
    ToolCallPart,
    create_tool_result_message,
    extract_text_content,
    extract_tool_calls,
)

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Client and Converter Initialization
# ============================================================================

openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)
openai_converter = OpenAIChatConverter()

# Separate client and model for Responses API
openai_responses_api_key = os.getenv("OPENAI_RESPONSES_API_KEY")
openai_responses_model = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4o")
openai_responses_client = None
if openai_responses_api_key:
    openai_responses_client = OpenAI(
        api_key=openai_responses_api_key,
        base_url=os.getenv("OPENAI_RESPONSES_BASE_URL", "https://api.openai.com/v1"),
    )
openai_responses_converter = OpenAIResponsesConverter()

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


# ============================================================================
# Helper Functions
# ============================================================================


def json_serializable_default(o):
    """Custom JSON serializer for objects that are not serializable by default."""
    if isinstance(o, bytes):
        return base64.b64encode(o).decode("utf-8")
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def display_assistant_response(message: Message) -> None:
    """Display assistant's response including text and tool calls.

    Args:
        message: The assistant's message in IR format
    """
    # Display text content
    text_content = extract_text_content(message)
    if text_content:
        print(f"Assistant: {text_content}")

    # Display tool call requests
    tool_calls = extract_tool_calls(message)
    for tc in tool_calls:
        print(
            f"Assistant: [Requesting tool call: {tc['tool_name']}({json.dumps(tc['tool_input'])})]"
        )


def execute_tool_calls(
    tool_calls: List[ToolCallPart], ir_messages: List[Message]
) -> None:
    """Execute all tool calls and append results to message history.

    Args:
        tool_calls: List of tool calls to execute
        ir_messages: Message history to append results to
    """
    for tool_call in tool_calls:
        function_name = tool_call["tool_name"]
        function_args = tool_call["tool_input"]

        print(f"--- Executing tool: {function_name}({json.dumps(function_args)}) ---")

        # Execute the tool
        function_to_call = available_tools[function_name]
        function_response = function_to_call(**function_args)

        # Create and append tool result message
        tool_result_message = create_tool_result_message(
            tool_call["tool_call_id"], function_response
        )
        ir_messages.append(tool_result_message)
        print(f"Tool Result: {function_response}")


# ============================================================================
# Main Conversation Flow
# ============================================================================


def main():
    """
    This example demonstrates a multi-provider conversation flow.
    Each provider handles one complete cycle: question → tool call → tool result → final answer

    1. OpenAI Chat: San Francisco weather
    2. Anthropic: Paris weather
    3. Google GenAI: Shanghai weather
    4. OpenAI Responses: Tokyo weather
    5. OpenAI Chat: Temperature comparison (SF vs Paris)
    6. Anthropic: Temperature comparison (Paris vs Shanghai)
    7. Google GenAI: Temperature comparison (Shanghai vs SF)
    """
    # Pre-build Google config once (tools are static for this example)
    global google_config
    if google_client and google_config is None:
        google_config = google_converter.build_config(tools=tools_spec)

    # Initialize conversation history
    ir_messages: List[Message] = []

    print("=" * 80)
    print("MULTI-PROVIDER CONVERSATION DEMO")
    print("=" * 80)

    # ========================================================================
    # Round 1: OpenAI - San Francisco Weather
    # ========================================================================

    print("\n" + "=" * 80)
    print("ROUND 1: OpenAI - San Francisco Weather")
    print("=" * 80)

    # Add user question
    user_message_1 = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's the weather like in San Francisco?"}
        ],
    }
    ir_messages.append(user_message_1)
    print(f"User: {user_message_1['content'][0]['text']}")

    # --- Conversion: IR → OpenAI ---
    openai_payload, _ = openai_converter.to_provider(
        ir_messages, tools=tools_spec, tool_choice={"mode": "auto"}
    )

    # --- Request: Call OpenAI API ---
    openai_response = openai_client.chat.completions.create(
        model=openai_model, **openai_payload
    )

    # --- Conversion: OpenAI → IR ---
    ir_from_openai = openai_converter.from_provider(
        openai_response.choices[0].message.model_dump()
    )
    if isinstance(ir_from_openai, dict) and "choices" in ir_from_openai:
        ir_messages.append(ir_from_openai["choices"][0]["message"])
    else:
        ir_messages.extend(ir_from_openai)

    # --- Display Response ---
    display_assistant_response(ir_messages[-1])

    # --- Extract and Execute Tool Calls ---
    tool_calls = extract_tool_calls(ir_messages[-1])
    if not tool_calls:
        print("OpenAI did not request a tool call.")
    else:
        execute_tool_calls(tool_calls, ir_messages)

        # Get final response with tool results
        print("\n" + "-" * 80)
        print("OpenAI - Final Response with Tool Results")
        print("-" * 80)

        openai_payload, _ = openai_converter.to_provider(ir_messages, tools=tools_spec)
        openai_final_response = openai_client.chat.completions.create(
            model=openai_model, **openai_payload
        )
        ir_from_openai_final = openai_converter.from_provider(
            openai_final_response.choices[0].message.model_dump()
        )
        if isinstance(ir_from_openai_final, dict) and "choices" in ir_from_openai_final:
            ir_messages.append(ir_from_openai_final["choices"][0]["message"])
        else:
            ir_messages.extend(ir_from_openai_final)
        display_assistant_response(ir_messages[-1])

    # ========================================================================
    # Round 2: Anthropic - Paris Weather
    # ========================================================================

    print("\n" + "=" * 80)
    print("ROUND 2: Anthropic - Paris Weather")
    print("=" * 80)

    # Add user question
    user_message_2 = {
        "role": "user",
        "content": [{"type": "text", "text": "What's the weather like in Paris?"}],
    }
    ir_messages.append(user_message_2)
    print(f"User: {user_message_2['content'][0]['text']}")

    # --- Conversion: IR → Anthropic ---
    anthropic_payload, _ = anthropic_converter.to_provider(
        ir_messages, tools=tools_spec
    )

    # --- Request: Call Anthropic API ---
    anthropic_response = anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=1024,
        messages=anthropic_payload["messages"],
        tools=anthropic_payload.get("tools"),
    )

    # --- Conversion: Anthropic → IR ---
    ir_from_anthropic = anthropic_converter.from_provider(
        anthropic_response.model_dump()
    )
    if isinstance(ir_from_anthropic, dict) and "choices" in ir_from_anthropic:
        ir_messages.append(ir_from_anthropic["choices"][0]["message"])
    else:
        ir_messages.extend(ir_from_anthropic)

    # --- Display Response ---
    display_assistant_response(ir_messages[-1])

    # --- Extract and Execute Tool Calls ---
    tool_calls = extract_tool_calls(ir_messages[-1])
    if not tool_calls:
        print("Anthropic did not request a tool call.")
    else:
        execute_tool_calls(tool_calls, ir_messages)

        # Get final response with tool results
        print("\n" + "-" * 80)
        print("Anthropic - Final Response with Tool Results")
        print("-" * 80)

        anthropic_payload, _ = anthropic_converter.to_provider(
            ir_messages, tools=tools_spec
        )
        anthropic_final_response = anthropic_client.messages.create(
            model=anthropic_model,
            max_tokens=1024,
            messages=anthropic_payload["messages"],
            tools=anthropic_payload.get("tools"),
        )
        ir_from_anthropic_final = anthropic_converter.from_provider(
            anthropic_final_response.model_dump()
        )
        if (
            isinstance(ir_from_anthropic_final, dict)
            and "choices" in ir_from_anthropic_final
        ):
            ir_messages.append(ir_from_anthropic_final["choices"][0]["message"])
        else:
            ir_messages.extend(ir_from_anthropic_final)
        display_assistant_response(ir_messages[-1])

    # ========================================================================
    # Round 3: Google GenAI - Shanghai Weather
    # ========================================================================

    print("\n" + "=" * 80)
    print("ROUND 3: Google GenAI - Shanghai Weather")
    print("=" * 80)

    if not google_client:
        print("Google API key not configured. Skipping Google GenAI call.")
        return

    # Add user question
    user_message_3 = {
        "role": "user",
        "content": [{"type": "text", "text": "What's the weather like in Shanghai?"}],
    }
    ir_messages.append(user_message_3)
    print(f"User: {user_message_3['content'][0]['text']}")

    # --- Conversion: IR → Google ---
    google_payload, _ = google_converter.to_provider(ir_messages, tools=tools_spec)

    # --- Request: Call Google API ---
    google_response = google_client.models.generate_content(
        model=google_model_name,
        contents=google_payload["contents"],
        config=google_config,
    )

    # --- Conversion: Google → IR ---
    ir_from_google = google_converter.from_provider(google_response.model_dump())
    if isinstance(ir_from_google, dict) and "choices" in ir_from_google:
        ir_messages.append(ir_from_google["choices"][0]["message"])
    else:
        ir_messages.extend(ir_from_google)

    # --- Display Response ---
    display_assistant_response(ir_messages[-1])

    # --- Extract and Execute Tool Calls ---
    tool_calls = extract_tool_calls(ir_messages[-1])
    if not tool_calls:
        print("Google GenAI did not request a tool call.")
    else:
        execute_tool_calls(tool_calls, ir_messages)

        # Get final response with tool results
        print("\n" + "-" * 80)
        print("Google GenAI - Final Response with Tool Results")
        print("-" * 80)

        google_payload, _ = google_converter.to_provider(ir_messages, tools=tools_spec)
        google_final_response = google_client.models.generate_content(
            model=google_model_name,
            contents=google_payload["contents"],
            config=google_config,
        )
        ir_from_google_final = google_converter.from_provider(
            google_final_response.model_dump()
        )
        if isinstance(ir_from_google_final, dict) and "choices" in ir_from_google_final:
            ir_messages.append(ir_from_google_final["choices"][0]["message"])
        else:
            ir_messages.extend(ir_from_google_final)
        display_assistant_response(ir_messages[-1])

    # ========================================================================
    # Round 4: OpenAI Responses API - Tokyo Weather
    # ========================================================================

    print("\n" + "=" * 80)
    print("ROUND 4: OpenAI Responses API - Tokyo Weather")
    print("=" * 80)

    # Add user question
    user_message_4 = {
        "role": "user",
        "content": [{"type": "text", "text": "What's the weather like in Tokyo?"}],
    }
    ir_messages.append(user_message_4)
    print(f"User: {user_message_4['content'][0]['text']}")

    # --- Conversion: IR → OpenAI Responses ---
    responses_payload, _ = openai_responses_converter.to_provider(
        ir_messages, tools=tools_spec, tool_choice={"mode": "auto"}
    )

    if not openai_responses_client:
        print("OpenAI Responses API key not configured. Skipping Responses API call.")
        print("Set OPENAI_RESPONSES_API_KEY and OPENAI_RESPONSES_MODEL in .env file.")
    else:
        try:
            # --- Request: Call OpenAI Responses API ---
            responses_response = openai_responses_client.responses.create(
                model=openai_responses_model,
                input=responses_payload["input"],
                tools=responses_payload.get("tools"),
                tool_choice=responses_payload.get("tool_choice"),
            )
        except Exception as e:
            print(f"Error calling OpenAI Responses API: {e}")
            print("Skipping Responses API call.")
            responses_response = None

        if responses_response:
            # --- Conversion: OpenAI Responses → IR ---
            ir_from_responses = openai_responses_converter.from_provider(
                responses_response.model_dump()
            )
            if isinstance(ir_from_responses, dict) and "choices" in ir_from_responses:
                ir_messages.append(ir_from_responses["choices"][0]["message"])
            else:
                ir_messages.extend(ir_from_responses)

            # --- Display Response ---
            display_assistant_response(ir_messages[-1])

            # --- Extract and Execute Tool Calls ---
            tool_calls = extract_tool_calls(ir_messages[-1])
            if not tool_calls:
                print("OpenAI Responses API did not request a tool call.")
            else:
                execute_tool_calls(tool_calls, ir_messages)

                # Get final response with tool results
                print("\n" + "-" * 80)
                print("OpenAI Responses API - Final Response with Tool Results")
                print("-" * 80)

                # Convert back to Responses format with tool results
                responses_payload_with_results, _ = (
                    openai_responses_converter.to_provider(
                        ir_messages, tools=tools_spec
                    )
                )

                try:
                    # Call Responses API again with tool results
                    responses_final_response = openai_responses_client.responses.create(
                        model=openai_responses_model,
                        input=responses_payload_with_results["input"],
                        tools=responses_payload_with_results.get("tools"),
                    )

                    ir_from_final_responses = openai_responses_converter.from_provider(
                        responses_final_response.model_dump()
                    )
                    if (
                        isinstance(ir_from_final_responses, dict)
                        and "choices" in ir_from_final_responses
                    ):
                        ir_messages.append(
                            ir_from_final_responses["choices"][0]["message"]
                        )
                    else:
                        ir_messages.extend(ir_from_final_responses)
                    display_assistant_response(ir_messages[-1])
                except Exception as e:
                    print(f"Error in final Responses API call: {e}")

    # ========================================================================
    # Round 5: OpenAI - Temperature Difference (San Francisco vs Paris)
    # ========================================================================

    print("\n" + "=" * 80)
    print("ROUND 5: OpenAI - Temperature Comparison")
    print("=" * 80)

    # Add user question
    user_message_5 = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's the temperature difference between San Francisco and Paris?",
            }
        ],
    }
    ir_messages.append(user_message_5)
    print(f"User: {user_message_5['content'][0]['text']}")

    # --- Conversion: IR → OpenAI ---
    openai_payload, _ = openai_converter.to_provider(ir_messages, tools=tools_spec)

    # --- Request: Call OpenAI API ---
    openai_response = openai_client.chat.completions.create(
        model=openai_model, **openai_payload
    )

    # --- Conversion: OpenAI → IR ---
    ir_from_openai = openai_converter.from_provider(
        openai_response.choices[0].message.model_dump()
    )
    if isinstance(ir_from_openai, dict) and "choices" in ir_from_openai:
        ir_messages.append(ir_from_openai["choices"][0]["message"])
    else:
        ir_messages.extend(ir_from_openai)

    # --- Display Response ---
    display_assistant_response(ir_messages[-1])

    # ========================================================================
    # Round 6: Anthropic - Temperature Difference (Paris vs Shanghai)
    # ========================================================================

    print("\n" + "=" * 80)
    print("ROUND 6: Anthropic - Temperature Comparison")
    print("=" * 80)

    # Add user question
    user_message_6 = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's the temperature difference between Paris and Shanghai?",
            }
        ],
    }
    ir_messages.append(user_message_6)
    print(f"User: {user_message_6['content'][0]['text']}")

    # --- Conversion: IR → Anthropic ---
    anthropic_payload, _ = anthropic_converter.to_provider(
        ir_messages, tools=tools_spec
    )

    # --- Request: Call Anthropic API ---
    anthropic_response = anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=1024,
        messages=anthropic_payload["messages"],
        tools=anthropic_payload.get("tools"),
    )

    # --- Conversion: Anthropic → IR ---
    ir_from_anthropic = anthropic_converter.from_provider(
        anthropic_response.model_dump()
    )
    if isinstance(ir_from_anthropic, dict) and "choices" in ir_from_anthropic:
        ir_messages.append(ir_from_anthropic["choices"][0]["message"])
    else:
        ir_messages.extend(ir_from_anthropic)

    # --- Display Response ---
    display_assistant_response(ir_messages[-1])

    # ========================================================================
    # Round 7: Google GenAI - Temperature Difference (Shanghai vs San Francisco)
    # ========================================================================

    print("\n" + "=" * 80)
    print("ROUND 7: Google GenAI - Temperature Comparison")
    print("=" * 80)

    if not google_client:
        print("Google API key not configured. Skipping Google GenAI call.")
        return

    # Add user question
    user_message_7 = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's the temperature difference between Shanghai and San Francisco?",
            }
        ],
    }
    ir_messages.append(user_message_7)
    print(f"User: {user_message_7['content'][0]['text']}")

    # --- Conversion: IR → Google ---
    google_payload, _ = google_converter.to_provider(ir_messages, tools=tools_spec)

    # --- Request: Call Google API ---
    google_response = google_client.models.generate_content(
        model=google_model_name,
        contents=google_payload["contents"],
        config=google_config,
    )

    # --- Conversion: Google → IR ---
    ir_from_google = google_converter.from_provider(google_response.model_dump())
    if isinstance(ir_from_google, dict) and "choices" in ir_from_google:
        ir_messages.append(ir_from_google["choices"][0]["message"])
    else:
        ir_messages.extend(ir_from_google)

    # --- Display Response ---
    display_assistant_response(ir_messages[-1])

    # ========================================================================
    # Conversation End
    # ========================================================================

    print("\n" + "=" * 80)
    print("CONVERSATION END")
    print("=" * 80)
    print(f"\nTotal messages in history: {len(ir_messages)}")
    print("\nProviders used:")
    print("  1. OpenAI Chat Completions API")
    print("  2. Anthropic Messages API")
    print("  3. Google GenAI SDK")
    print("  4. OpenAI Responses API")


if __name__ == "__main__":
    main()
