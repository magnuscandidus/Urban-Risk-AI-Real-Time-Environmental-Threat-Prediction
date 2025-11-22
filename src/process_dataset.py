import os
import json
import pandas as pd


# ------------------------------------------------------------
# Load latest JSON file from a folder
# ------------------------------------------------------------
def load_latest_json(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not files:
        print(f"[WARN] No JSON files found in {folder}")
        return None

    newest = sorted(files)[-1]
    path = os.path.join(folder, newest)
    print(f"[INFO] Loading {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# WEATHER PROCESSING (Open-Meteo)
# Fixes missing precipitation + humidity key change
# ------------------------------------------------------------
def process_weather(data):
    hourly = data.get("hourly", {})

    times = hourly.get("time", [])
    n = len(times)

    # Humidity can be either relativehumidity_2m OR relative_humidity_2m
    humidity = (
        hourly.get("relative_humidity_2m") or
        hourly.get("relativehumidity_2m") or
        [None] * n
    )

    temperature = hourly.get("temperature_2m", [None] * n)
    precipitation = hourly.get("precipitation", [0.0] * n)

    # Fix mismatched lengths (rare but safe)
    humidity = humidity[:n]
    temperature = temperature[:n]
    precipitation = precipitation[:n]

    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "temperature": temperature,
        "humidity": humidity,
        "precipitation": precipitation,
    })

    return df


# ------------------------------------------------------------
# AIR QUALITY PROCESSING (Open-Meteo Air Quality)
# Extract only PM2.5 for the final dataset
# ------------------------------------------------------------
def process_air_quality(data):
    hourly = data.get("hourly", {})

    times = hourly.get("time", [])
    pm25 = hourly.get("pm2_5", [])

    if not times:
        print("[WARN] Air quality times missing")
        return pd.DataFrame(columns=["time", "pm25"])

    n = len(times)
    pm25 = pm25[:n] if pm25 else [None] * n

    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "pm25": pm25,
    })

    return df


# ------------------------------------------------------------
# TRAFFIC PROCESSING (TomTom)
# One record per fetch → duplicate for all hours
# ------------------------------------------------------------
def process_traffic(data):
    flow = data.get("flowSegmentData", {})

    current_speed = flow.get("currentSpeed")
    free_speed = flow.get("freeFlowSpeed")

    timestamp = pd.Timestamp.now().floor("H")

    df = pd.DataFrame([{
        "time": timestamp,
        "traffic_speed": current_speed,
        "free_flow_speed": free_speed,
        "congestion_index": (free_speed - current_speed)
        if free_speed is not None and current_speed is not None
        else None
    }])

    return df


# ------------------------------------------------------------
# MERGE ALL DATASETS
# ------------------------------------------------------------
def merge_datasets(weather_df, air_df, traffic_df):

    # Align air-quality to hours
    air_df["time_hour"] = air_df["time"].dt.floor("H")
    air_hour = air_df.groupby("time_hour")["pm25"].mean().reset_index()
    air_hour = air_hour.rename(columns={"time_hour": "time"})

    # Merge weather WITH air quality
    df = weather_df.merge(air_hour, on="time", how="left")

    # Traffic → duplicate same value for all hours
    traffic_row = traffic_df.iloc[0]
    df["traffic_speed"] = traffic_row["traffic_speed"]
    df["free_flow_speed"] = traffic_row["free_flow_speed"]
    df["congestion_index"] = traffic_row["congestion_index"]

    # Fill missing pm25 (sometimes missing early hours)
    df["pm25"] = df["pm25"].fillna(method="ffill")

    return df


# ------------------------------------------------------------
# MAIN PROCESSING PIPELINE
# ------------------------------------------------------------
def run_processing():
    print("=== Processing dataset ===")

    weather_raw = load_latest_json("data_raw/weather")
    air_raw = load_latest_json("data_raw/air_quality")
    traffic_raw = load_latest_json("data_raw/traffic")

    if not weather_raw or not air_raw or not traffic_raw:
        print("[ERROR] Missing input data!")
        return

    weather_df = process_weather(weather_raw)
    air_df = process_air_quality(air_raw)
    traffic_df = process_traffic(traffic_raw)

    final_df = merge_datasets(weather_df, air_df, traffic_df)

    os.makedirs("data_processed", exist_ok=True)
    out_path = "data_processed/final_dataset.csv"
    final_df.to_csv(out_path, index=False)

    print(f"[SUCCESS] Saved processed dataset: {out_path}")


# ------------------------------------------------------------
# RUN SCRIPT
# ------------------------------------------------------------
if __name__ == "__main__":
    run_processing()
