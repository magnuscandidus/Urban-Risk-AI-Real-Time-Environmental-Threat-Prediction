import requests
import yaml
import os
import json
from datetime import datetime


def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)


def build_air_url(cfg):
    base = cfg["air_quality"]["url"]
    lat = cfg["air_quality"]["latitude"]
    lon = cfg["air_quality"]["longitude"]

    hourly = ",".join(cfg["air_quality"]["hourly"])

    return (
        f"{base}?latitude={lat}"
        f"&longitude={lon}"
        f"&hourly={hourly}"
    )


def fetch_air_quality():
    cfg = load_config()
    url = build_air_url(cfg)

    print("[INFO] Requesting AQ:", url)
    r = requests.get(url)

    if r.status_code != 200:
        print("[ERROR] Air Quality API:", r.status_code)
        print(r.text)
        return None

    return r.json()


def save_air(data, cfg):
    path = cfg["paths"]["raw_air_quality"]
    os.makedirs(path, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(path, f"air_{ts}.json")

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print("[SUCCESS] Saved:", filename)


if __name__ == "__main__":
    cfg = load_config()
    aq = fetch_air_quality()

    if aq:
        save_air(aq, cfg)
