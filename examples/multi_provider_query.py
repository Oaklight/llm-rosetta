"""
Multi-Provider Query Functions

This module provides functions for querying different LLM providers (OpenAI and Anthropic)
and converting messages between different provider formats.
"""

import os
from typing import Any, Dict, List, Optional

import anthropic
import openai
from dotenv import load_dotenv
from google import genai


def query_openai(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
) -> Any:
    """
    Send a query to OpenAI.

    Args:
        messages: List of message dictionaries in OpenAI format
        tools: Optional list of tools in OpenAI format
        model: Optional model name to use (defaults to environment variable or gpt-3.5-turbo)

    Returns:
        OpenAI response object
    """
    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    )

    # Set up request parameters
    kwargs = {
        "model": model or os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
        "messages": messages,
    }

    # Add tools if provided
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    # Send request
    response = client.chat.completions.create(**kwargs)
    return response


def query_anthropic(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
) -> Any:
    """
    Send a query to Anthropic.

    Args:
        messages: List of message dictionaries in Anthropic format
        tools: Optional list of tools in Anthropic format
        model: Optional model name to use (defaults to environment variable or claude-haiku-4)

    Returns:
        Anthropic response object
    """
    # Initialize Anthropic client
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=os.getenv("ANTHROPIC_BASE_URL")
    )

    # Extract system message if present
    system_content = None
    anthropic_messages = []

    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            anthropic_messages.append(msg)

    # Set up request parameters
    kwargs = {
        "model": model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4"),
        "max_tokens": 1024,
        "messages": anthropic_messages,
    }

    # Add system message if present
    if system_content:
        kwargs["system"] = system_content

    # Add tools if provided
    if tools:
        kwargs["tools"] = tools

    # Send request
    response = client.messages.create(**kwargs)
    return response


def query_google(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
) -> Any:
    """
    Send a query to Google Generative AI using google-genai package.

    Args:
        messages: List of message dictionaries in OpenAI format
        tools: Optional list of tools in OpenAI format
        model: Optional model name to use (defaults to environment variable or gemini-1.5-flash)

    Returns:
        Google GenAI response object
    """
    # Initialize Google GenAI client
    client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY"), http_options={"api_version": "v1beta"}
    )

    # Convert messages to Google format
    google_messages = []
    system_instruction = None

    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            google_messages.append(
                genai.types.Content(
                    role="user", parts=[genai.types.Part.from_text(msg["content"])]
                )
            )
        elif msg["role"] == "assistant":
            google_messages.append(
                genai.types.Content(
                    role="model", parts=[genai.types.Part.from_text(msg["content"])]
                )
            )

    # Set up request parameters
    model_name = model or os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

    config = genai.types.GenerateContentConfig(
        temperature=0.7, max_output_tokens=1024, system_instruction=system_instruction
    )

    # Add tools if provided
    if tools:
        # Convert OpenAI tools to Google function declarations
        google_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func_decl = genai.types.FunctionDeclaration(
                    name=tool["function"]["name"],
                    description=tool["function"]["description"],
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            k: genai.types.Schema(
                                type=genai.types.Type.STRING
                                if v.get("type") == "string"
                                else genai.types.Type.OBJECT,
                                description=v.get("description", ""),
                            )
                            for k, v in tool["function"]["parameters"][
                                "properties"
                            ].items()
                        },
                        required=tool["function"]["parameters"].get("required", []),
                    ),
                )
                google_tools.append(func_decl)

        config.tools = [genai.types.Tool(function_declarations=google_tools)]

    # Send request
    response = client.models.generate_content(
        model=model_name, contents=google_messages, config=config
    )

    return response


# def convert_messages(
#     messages: List[Dict[str, Any]], source_provider: str, target_provider: str
# ) -> List[Dict[str, Any]]:
#     """
#     Convert messages between provider formats.

#     Args:
#         messages: List of message dictionaries
#         source_provider: Source provider format ("openai" or "anthropic")
#         target_provider: Target provider format ("openai" or "anthropic")

#     Returns:
#         Converted messages in the target provider format
#     """
#     converted_messages = []

#     for msg in messages:
#         # Handle special cases for different providers
#         if source_provider == "openai" and target_provider == "anthropic":
#             if msg["role"] == "system":
#                 # System message is handled separately in Anthropic
#                 continue

#             # Handle tool calls in OpenAI messages
#             if msg["role"] == "assistant":
#                 # For tool calls, we need special handling
#                 content = msg.get("content", "")
#                 if content.isinstance(list): # nested content
#                     for i, entry in content:
#                         if entry['type']=='


#                 if 'tool_calls' in msg:
#                     for each in msg['tool_calls']:
#                         converted


#                 converted_msg = {"role": "assistant", "content": }

#                 continue

#             # Standard message conversion
#             converted_msg = {"role": msg["role"], "content": msg["content"]}
#             converted_messages.append(converted_msg)

#         elif source_provider == "anthropic" and target_provider == "openai":
#             # Convert Anthropic message to OpenAI format
#             content = msg["content"]

#             # Handle Anthropic's content blocks (list of content blocks)
#             if isinstance(content, list):
#                 text_content = ""
#                 for block in content:
#                     if isinstance(block, dict) and "text" in block:
#                         text_content += block["text"]
#                 content = text_content

#             converted_msg = {"role": msg["role"], "content": content}
#             converted_messages.append(converted_msg)

#     return converted_messages


# # Example usage (to be replaced by your own main function):
# def main():
#     # Example messages in OpenAI format
#     openai_messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What's weather like in Chicago?"},
#     ]

#     # Example tool in OpenAI format
#     calculator_tool = {
#         "type": "function",
#         "function": {
#             "name": "calculate",
#             "description": "Perform a calculation",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "expression": {
#                         "type": "string",
#                         "description": "The mathematical expression to evaluate",
#                     }
#                 },
#                 "required": ["expression"],
#             },
#         },
#     }
#     # Example tool in OpenAI format
#     get_weather_tool = {
#         "type": "function",
#         "function": {
#             "name": "get_weather",
#             "description": "Get the current weather for a specified location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state/country, e.g. 'San Francisco, CA'",
#                     },
#                     "unit": {
#                         "type": "string",
#                         "enum": ["celsius", "fahrenheit"],
#                         "description": "The temperature unit to use",
#                         "default": "celsius",
#                     },
#                 },
#                 "required": ["location"],
#             },
#         },
#     }

#     def get_weather(location, unit="celsius"):
#         """Dummy weather function that always returns 25°C"""
#         if unit == "fahrenheit":
#             return {"location": location, "temperature": 77, "unit": "°F", "condition": "sunny"}
#         else:
#             return {"location": location, "temperature": 25, "unit": "°C", "condition": "sunny"}

#     # Query both providers
#     openai_response = query_openai(openai_messages, tools=[calculator_tool, get_weather_tool])

#     # get openai response message and tool call if exist and add them to openai_messages
#     if openai_response.choices[0].message:
#         message = {"role": "assistant", "content": openai_response.choices[0].message.content}
#         # add tool calls if exist
#         if openai_response.choices[0].message.tool_calls:
#             message["tool_calls"] = [
#                 each.model_dump() for each in openai_response.choices[0].message.tool_calls
#             ]

#         openai_messages.append(message)

#     print(openai_messages)

#     # Convert messages to Anthropic format
#     anthropic_messages = convert_messages(openai_messages, "openai", "anthropic")
#     print(anthropic_messages)

#     exit()
#     # Convert tool to Anthropic format (would use the llm_provider_converter library)
#     # anthropic_tool = convert(calculator_tool, from_provider="openai", to_provider="anthropic")

#     # Query Anthropic
#     anthropic_response = query_anthropic(anthropic_messages)

#     # Print results
#     print("\nOpenAI Response:")
#     print(openai_response.choices[0].message.content)

#     print("\nAnthropic Response:")
#     for block in anthropic_response.content:
#         if hasattr(block, "text"):
#             print(block.text)


# if __name__ == "__main__":
#     load_dotenv()
#     # You will implement your own main function here
#     main()
