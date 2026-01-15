# src/generate_synthetic.py
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

def generate_training_data(n_samples=20000, out_file="training_data.csv"):
    np.random.seed(42)
    data = {
        "trip_id": [f"T{i}" for i in range(n_samples)],
        "avg_speed": np.clip(np.random.normal(50, 15, n_samples), 0, 160),
        "max_speed": np.clip(np.random.normal(80, 20, n_samples), 0, 240),
        "overspeed_ratio": np.clip(np.random.beta(2,5, n_samples), 0, 1),
        "harsh_brake_count": np.random.poisson(1.8, n_samples),
        "sharp_turn_count": np.random.poisson(2.2, n_samples),
        "night_ratio": np.clip(np.random.beta(1,3, n_samples), 0, 1),
        "trip_distance_km": np.clip(np.random.gamma(2.0, 5.0, n_samples), 0.1, 500),
        "trip_duration_min": np.clip(np.random.gamma(3.0, 8.0, n_samples), 1, 1440),
    }
    df = pd.DataFrame(data)

    # Derived binary features
    df["is_night_trip"] = (df["night_ratio"] > 0.5).astype(int)
    df["has_harsh_brake"] = (df["harsh_brake_count"] > 0).astype(int)
    df["has_sharp_turn"] = (df["sharp_turn_count"] > 0).astype(int)
    df["is_overspeed"] = (df["overspeed_ratio"] > 0.15).astype(int)

    # missing count (none here, but keep parity with feature set)
    df["missing"] = 0

    # Simple rule-based risk score to create labels
    risk_score = (
        df["overspeed_ratio"] * 35 +
        df["harsh_brake_count"] * 6 +
        df["sharp_turn_count"] * 3.5 +
        df["night_ratio"] * 12
    )
    # binary target: 1 (risky) if risk_score > threshold
    df["target"] = (risk_score > 28).astype(int)

    out_path = os.path.join(OUT, out_file)
    df.to_csv(out_path, index=False)
    print("Wrote", out_path)
    return out_path

if __name__ == "__main__":
    generate_training_data()
