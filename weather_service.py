import time

import requests

# Mapping from Open-Meteo weathercode to human-readable description
WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def get_current_weather(lat: float, lon: float) -> dict | None:
    """
    Fetch current weather from Open-Meteo for given latitude/longitude.
    Returns a dict with temperature (C), windspeed (km/h), winddirection (deg),
    weathercode (int), and time (ISO string).
    Returns None if the API is temporarily unavailable.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
    }

    last_error = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            cw = data.get("current_weather")
            if not cw:
                return None

            return {
                "temperature": cw["temperature"],
                "windspeed": cw["windspeed"],
                "winddirection": cw["winddirection"],
                "time": cw["time"],
                "weathercode": cw["weathercode"],
            }
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
            continue

    print(f"WARNING: Weather service unavailable ({last_error})")
    return None


def describe_weather(code: int) -> str:
    """Convert Open-Meteo weathercode to human-readable text."""
    return WEATHER_CODE_MAP.get(code, f"Unknown ({code})")
