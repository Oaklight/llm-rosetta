import base64
import json
import struct
import zlib


def _make_tiny_png() -> str:
    """Generate a minimal 8x8 red square PNG as base64."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    width, height = 8, 8
    # RGBA red pixel
    raw = b""
    for _ in range(height):
        raw += b"\x00" + b"\xff\x00\x00\xff" * width  # filter=none + RGBA

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
    png = (
        sig
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


_CHART_IMAGE_BASE64 = _make_tiny_png()


# --- Tool Definitions ---
def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location."""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
        )
    elif "paris" in location.lower() or "shanghai" in location.lower():
        return json.dumps(
            {"location": location, "temperature": "22", "unit": "celsius"}
        )
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def get_flight_info(origin: str, destination: str):
    """Get flight information between two locations."""
    return json.dumps(
        {
            "origin": origin,
            "destination": destination,
            "flight_number": "AA123",
            "departure": "2024-12-25T08:00:00Z",
        }
    )


def generate_chart(chart_type: str = "bar"):
    """Generate a chart visualization (returns multimodal content)."""
    return [
        {"type": "text", "text": f"Generated {chart_type} chart of world population:"},
        {
            "type": "image",
            "image_data": {"data": _CHART_IMAGE_BASE64, "media_type": "image/png"},
        },
    ]


available_tools = {
    "get_current_weather": get_current_weather,
    "get_flight_info": get_flight_info,
    "generate_chart": generate_chart,
}

generate_chart_spec = {
    "type": "function",
    "name": "generate_chart",
    "description": "Generate a chart or data visualization. Returns an image of the chart.",
    "parameters": {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "enum": ["bar", "line", "pie"],
                "description": "The type of chart to generate",
            },
        },
    },
}

# This now follows the IR `ToolDefinition` format
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
    },
    {
        "type": "function",
        "name": "get_flight_info",
        "description": "Get flight information between two locations",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "The origin city"},
                "destination": {
                    "type": "string",
                    "description": "The destination city",
                },
            },
            "required": ["origin", "destination"],
        },
    },
]

# tools_spec + generate_chart for multimodal integration tests
multimodal_tools_spec = tools_spec + [generate_chart_spec]
