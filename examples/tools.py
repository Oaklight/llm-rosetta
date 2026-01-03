import json


# --- Tool Definitions ---
def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location."""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
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


available_tools = {
    "get_current_weather": get_current_weather,
    "get_flight_info": get_flight_info,
}

tools_spec = [
    {
        "type": "function",
        "function": {
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
    },
    {
        "type": "function",
        "function": {
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
    },
]
