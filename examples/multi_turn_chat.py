import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from tools import tools_spec

from llm_provider_converter.converters.openai_chat_converter import OpenAIChatConverter
from llm_provider_converter.types.ir import Message

# Load environment variables from .env file
load_dotenv()

# --- Main Logic ---
model_name = os.getenv("MODEL", "gpt-4.1-nano")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
converter = OpenAIChatConverter()


def main():
    """
    This example demonstrates a basic, single-turn conversation.
    It shows how to:
    1. Create a message using the IR format.
    2. Convert the IR message to the OpenAI provider format.
    3. Call the OpenAI API.
    4. Convert the OpenAI response back to the IR format.
    """
    # 1. Create a message using the IR format.
    # NOTE: `Message` is a `TypedDict`, not a class. It provides static type
    # checking for dictionary structures. We create it using standard dict
    # literals `{}`, not by instantiating a class like `Message(...)`.
    ir_messages: list[Message] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's the weather like in San Francisco?"}
            ],
        }
    ]
    print("--- Initial IR Messages ---")
    print(json.dumps(ir_messages, indent=2))

    # 2. Convert the IR message to the OpenAI provider format
    # Note: The `to_provider` method returns a tuple: (provider_dict, warnings)
    provider_payload, warnings = converter.to_provider(
        ir_messages, tools=tools_spec, tool_choice={"mode": "auto"}
    )
    if warnings:
        print("\n--- Conversion Warnings ---")
        print(warnings)

    print("\n--- Converted Provider Payload (for OpenAI API) ---")
    print(json.dumps(provider_payload, indent=2))

    # 3. Call the OpenAI API
    response = client.chat.completions.create(model=model_name, **provider_payload)

    # 4. Convert the OpenAI response back to the IR format
    # The from_provider method expects a dictionary, so we convert the Pydantic
    # model returned by the OpenAI client using .model_dump().
    # It returns an IRInput object (a list of messages).
    ir_response_messages = converter.from_provider(response.choices[0].message.model_dump())
    print("\n--- Converted IR Response ---")
    print(json.dumps(ir_response_messages, indent=2))
    
    # You can now extend your IR message history with the new messages
    ir_messages.extend(ir_response_messages)


if __name__ == "__main__":
    main()
