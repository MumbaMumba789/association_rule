
from pathlib import Path

import pandas as pd


# Find the monitoring log in the project folder.
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "monitoring_log.csv"


# Load the collected monitoring records.
df = pd.read_csv(LOG_FILE)

if df.empty:
    print("No monitoring records found yet.")

else:
    # Convert the recorded timestamps into datetime values.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Display basic monitoring information.
    print("\n--- MODEL API MONITORING REPORT ---")

    print(f"\nTotal requests: {len(df)}")

    print("\nRequests by model version:")
    print(df["version"].value_counts())

    print("\nRequest status:")
    print(df["status"].value_counts())

    # Calculate latency statistics using successful requests.
    successful = df[df["status"] == "success"]

    if not successful.empty:
        print("\nLatency (milliseconds):")
        print(f"Average: {successful['latency_ms'].mean():.2f} ms")
        print(f"Maximum: {successful['latency_ms'].max():.2f} ms")
        print(f"Minimum: {successful['latency_ms'].min():.2f} ms")

        # Calculate how often the model returns a noise label.
        labelled = successful.dropna(subset=["is_noise"])

        if not labelled.empty:
            noise_rate = labelled["is_noise"].astype(bool).mean() * 100
            print(f"\nNoise rate: {noise_rate:.2f}%")

    # Calculate the percentage of requests that failed.
    error_rate = (df["status"] == "error").mean() * 100
    print(f"\nError rate: {error_rate:.2f}%")
