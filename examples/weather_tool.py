"""
Mock Weather Tool for LLM Provider Converter Examples

提供一个简单的天气查询工具，用于演示工具调用功能
"""

import random
from typing import Any, Dict


def get_weather(city: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    获取指定城市的天气信息（模拟数据）

    Args:
        city: 城市名称
        unit: 温度单位，"celsius" 或 "fahrenheit"

    Returns:
        包含天气信息的字典
    """
    # 模拟天气条件
    conditions = ["晴天", "多云", "小雨", "阴天", "雷阵雨"]
    condition = random.choice(conditions)

    # 模拟温度（摄氏度）
    temp_celsius = random.randint(-10, 35)

    # 转换温度单位
    if unit.lower() == "fahrenheit":
        temperature = temp_celsius * 9 / 5 + 32
        temp_unit = "°F"
    else:
        temperature = temp_celsius
        temp_unit = "°C"

    return {
        "city": city,
        "condition": condition,
        "temperature": temperature,
        "unit": temp_unit,
        "humidity": random.randint(30, 90),
        "wind_speed": random.randint(0, 20),
    }


# 工具定义（IR格式）
WEATHER_TOOL_DEFINITION = {
    "type": "function",
    "name": "get_weather",
    "description": "获取指定城市的天气信息",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称，如'北京'、'上海'等"},
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "温度单位",
                "default": "celsius",
            },
        },
        "required": ["city"],
    },
}

# OpenAI格式的工具定义
OPENAI_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如'北京'、'上海'等",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位",
                    "default": "celsius",
                },
            },
            "required": ["city"],
        },
    },
}
