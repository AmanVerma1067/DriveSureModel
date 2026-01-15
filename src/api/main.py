# src/api/main.py
import os
import json
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import lightgbm as lgb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_FILE = os.path.join(ROOT, "models", "drivesure_risk_model.txt")
FEATURES_FILE = os.path.join(ROOT, "features.json")

app = FastAPI(title="DriveSure Risk Engine")

# Load model and features
if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_FILE):
    raise RuntimeError("Model or features.json not found. Run training first.")

with open(FEATURES_FILE) as f:
    FEATURES = json.load(f)
MODEL = lgb.Booster(model_file=MODEL_FILE)

class TripFeatures(BaseModel):
    avg_speed: float
    max_speed: float
    overspeed_ratio: float
    harsh_brake_count: int
    sharp_turn_count: int
    night_ratio: float
    trip_distance_km: float
    trip_duration_min: float
    trip_id: Optional[str] = None

def prepare_vector(trip: TripFeatures):
    tripd = trip.dict()
    v = []
    for f in FEATURES:
        if f == "is_night_trip":
            v.append(int(tripd.get("night_ratio", 0) > 0.5))
        elif f == "has_harsh_brake":
            v.append(int(tripd.get("harsh_brake_count", 0) > 0))
        elif f == "has_sharp_turn":
            v.append(int(tripd.get("sharp_turn_count", 0) > 0))
        elif f == "is_overspeed":
            v.append(int(tripd.get("overspeed_ratio", 0) > 0.15))
        else:
            v.append(float(tripd.get(f, 0)))
    return np.array([v], dtype=float)

@app.post("/api/risk/scoreTrip")
def score_trip(trip: TripFeatures):
    X = prepare_vector(trip)
    risk_prob = float(MODEL.predict(X)[0])
    # risk_prob = model.predict(X)[0]
    risk_prob = min(max(risk_prob, 0.05), 0.95)
    safety_score = int(round(100 * (1 - risk_prob)))

    # feature importance based top factors (simple)
    importances = MODEL.feature_importance()
    names = FEATURES
    feat_imp = list(zip(names, importances))
    feat_imp_sorted = sorted(feat_imp, key=lambda x: x[1], reverse=True)
    # keep only features present in input (non-zero or used)
    top = [{"feature": n, "importance": int(im)} for n, im in feat_imp_sorted[:3]]

    return {
        "trip_id": trip.trip_id,
        "risk_prob": round(risk_prob, 4),
        "safety_score": safety_score,
        "top_factors": top
    }
