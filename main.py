import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import httpx
from pydantic import BaseModel, Field, ValidationError
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver


# User preferences stored per user
USER_PREFERENCES = {
    "user_1": {"unit": "celsius", "language": "en"},
    "user_2": {"unit": "fahrenheit", "language": "es"},
}

# Weather code to description mapping (WMO codes)
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def _geocode(city: str) -> tuple[float, float, str]:
    """Convert city name to coordinates using Open-Meteo geocoding API."""
    resp = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    )
    resp.raise_for_status()
    data = resp.json()
    if "results" not in data:
        raise ValueError(f"City '{city}' not found")
    result = data["results"][0]
    return result["latitude"], result["longitude"], result.get("name", city)


def _fetch_weather(lat: float, lon: float, unit: str) -> dict:
    """Fetch current weather from Open-Meteo API."""
    temp_unit = "celsius" if unit == "celsius" else "fahrenheit"
    resp = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "temperature_unit": temp_unit,
            "wind_speed_unit": "kmh",
        },
    )
    resp.raise_for_status()
    return resp.json()["current"]


@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get live weather for a given city using Open-Meteo API."""
    user_id = runtime.config.get("configurable", {}).get("user_id", "unknown")
    prefs = USER_PREFERENCES.get(user_id, {"unit": "celsius", "language": "en"})

    try:
        lat, lon, resolved_name = _geocode(city)
        weather = _fetch_weather(lat, lon, prefs["unit"])

        temp = weather["temperature_2m"]
        humidity = weather["relative_humidity_2m"]
        wind = weather["wind_speed_10m"]
        condition = WEATHER_CODES.get(weather["weather_code"], "Unknown")
        unit_symbol = "°C" if prefs["unit"] == "celsius" else "°F"

        return (
            f"Weather in {resolved_name}: {temp}{unit_symbol}, {condition}, "
            f"Humidity: {humidity}%, Wind: {wind} km/h"
        )
    except ValueError as e:
        return str(e)
    except httpx.HTTPError:
        return f"Failed to fetch weather data for {city}. Please try again."


def main():
    agent = create_agent(
        model=ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=1024,
            timeout=30,
            max_retries=2,
        ),
        tools=[get_weather],
        checkpointer=InMemorySaver(),
        system_prompt=(
            "You are a helpful weather assistant. "
            "Before using the get_weather tool, validate that the user's input is a real city in the world. "
            "If it is not a valid city name, politely tell the user and do not call the tool. "
            "You can reference previous messages in the conversation."
        ),
    )

    config = {
        "recursion_limit": 10,
        "configurable": {"thread_id": "session_1", "user_id": "user_1"},
    }

    print("Weather Agent (type 'quit' to exit)")
    print("=" * 50)

    while True:
        raw_input = input("\nYou: ").strip()

        if raw_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": raw_input}]},
            config=config,
        )

        last_message = result["messages"][-1]
        print(f"Agent: {last_message.content}")


if __name__ == "__main__":
    main()
