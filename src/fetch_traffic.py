import requests
import yaml
import os
from datetime import datetime

SETTINGS_PATH = "config/settings.yaml"

def load_config():
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fetch_traffic():
    print("=== Fetching Traffic Data ===")

    cfg = load_config()
    traffic_cfg = cfg["traffic"]

    base = traffic_cfg["source_url"]
    lat = traffic_cfg["latitude"]
    lon = traffic_cfg["longitude"]
    key = traffic_cfg["api_key"]

    url = f"{base}?point={lat},{lon}&key={key}"
    print("[INFO] Requesting:", url)

    try:
        r = requests.get(url, timeout=30)

        if r.status_code != 200:
            print("[ERROR] Traffic API returned:", r.status_code)
            print(r.text)
            return

        data = r.json()

    except Exception as e:
        print("[ERROR] Exception requesting traffic API:", e)
        return

    out_folder = cfg["paths"]["raw_traffic"]
    os.makedirs(out_folder, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(out_folder, f"traffic_{ts}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, indent=2)

    print("[SUCCESS] Saved:", file_path)


if __name__ == "__main__":
    fetch_traffic()
