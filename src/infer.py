# src/infer.py
import os
import json
import numpy as np
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_FILE = os.path.join(ROOT, "models", "drivesure_risk_model.txt")
FEATURES_FILE = os.path.join(ROOT, "features.json")

def load():
    with open(FEATURES_FILE) as f:
        features = json.load(f)
    model = lgb.Booster(model_file=MODEL_FILE)
    return model, features

def prepare_vector(trip, features):
    # derive binary features if not provided
    v = []
    for f in features:
        if f == "is_night_trip":
            v.append(int(trip.get("night_ratio", 0) > 0.5))
        elif f == "has_harsh_brake":
            v.append(int(trip.get("harsh_brake_count", 0) > 0))
        elif f == "has_sharp_turn":
            v.append(int(trip.get("sharp_turn_count", 0) > 0))
        elif f == "is_overspeed":
            v.append(int(trip.get("overspeed_ratio", 0) > 0.15))
        else:
            v.append(float(trip.get(f, 0)))
    return np.array([v], dtype=float)

if __name__ == "__main__":
    model, features = load()
    # sample trip
    # trip = {
    #     "avg_speed": 48.5,
    #     "max_speed": 92,
    #     "overspeed_ratio": 0.18,
    #     "harsh_brake_count": 3,
    #     "sharp_turn_count": 1,
    #     "night_ratio": 0.35,
    #     "trip_distance_km": 12.5,
    #     "trip_duration_min": 22
    # }
    trip = {
    "avg_speed": 42,
    "max_speed": 60,
    "overspeed_ratio": 0.02,
    "harsh_brake_count": 0,
    "sharp_turn_count": 0,
    "night_ratio": 0.05,
    "trip_distance_km": 8,
    "trip_duration_min": 15
     }

    X = prepare_vector(trip, features)
    # prob = model.predict(X)[0]
    prob = model.predict(X)[0]
    prob = min(max(prob, 0.05), 0.95)

    safety_score = int(round(100 * (1 - prob)))
    print("Risk probability:", float(prob))
    print("Safety score (0-100):", safety_score)
