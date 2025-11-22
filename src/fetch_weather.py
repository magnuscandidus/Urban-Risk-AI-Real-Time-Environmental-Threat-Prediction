import requests
import yaml
import os
from datetime import datetime


# -------------------------------------------------------
# Load YAML config
# -------------------------------------------------------
def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------
# Build Open-Meteo request URL
# -------------------------------------------------------
def build_weather_url(cfg):
    base = cfg["openmeteo"]["weather_url"]
    lat = cfg["openmeteo"]["latitude"]
    lon = cfg["openmeteo"]["longitude"]
    hourly = ",".join(cfg["openmeteo"]["hourly"])
    tz = cfg["openmeteo"]["timezone"]

    return (
        f"{base}?latitude={lat}"
        f"&longitude={lon}"
        f"&hourly={hourly}"
        f"&timezone={tz}"
    )


# -------------------------------------------------------
# Fetch weather data
# -------------------------------------------------------
def fetch_weather():
    cfg = load_config()

    url = build_weather_url(cfg)
    print(f"[INFO] Requesting: {url}")

    try:
        response = requests.get(url)

        if response.status_code != 200:
            print(f"[ERROR] Open-Meteo returned status {response.status_code}")
            print(response.text)
            return None

        return response.json()

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return None


# -------------------------------------------------------
# Save JSON to raw folder
# -------------------------------------------------------
def save_weather(data, cfg):
    raw_path = cfg["paths"]["raw_weather"]

    os.makedirs(raw_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(raw_path, f"weather_{timestamp}.json")

    with open(filename, "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, indent=2)

    print(f"[SUCCESS] Saved: {filename}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    data = fetch_weather()

    if data:
        save_weather(data, cfg)
