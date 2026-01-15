# src/infer.py
import os
import json
import numpy as np
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_FILE = os.path.join(ROOT, "models", "drivesure_risk_model.txt")
FEATURES_FILE = os.path.join(ROOT, "features.json")

def load():
    """Load model and feature contract"""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}. Run train_model.py first.")
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Features not found: {FEATURES_FILE}. Run train_model.py first.")
    
    with open(FEATURES_FILE) as f:
        features = json.load(f)
    model = lgb.Booster(model_file=MODEL_FILE)
    
    print(f"✅ Loaded model from: {MODEL_FILE}")
    print(f"✅ Loaded {len(features)} features from: {FEATURES_FILE}")
    
    return model, features

def prepare_vector(trip, features):
    """
    ✅ CRITICAL: Feature engineering must match training exactly
    
    Args:
        trip: Dict with base telematics features
        features: List of feature names in correct order
    
    Returns:
        numpy array ready for model prediction
    """
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
        elif f == "missing":
            v.append(0)  # No missing values in real-time inference
        else:
            v.append(float(trip.get(f, 0)))
    
    return np.array([v], dtype=float)

def predict_safety_score(trip):
    """
    Predict safety score for a single trip
    
    Args:
        trip: Dict with trip features
    
    Returns:
        Dict with risk_prob and safety_score
    """
    model, features = load()
    X = prepare_vector(trip, features)
    
    # Get risk probability
    risk_prob = float(model.predict(X)[0])
    
    # ✅ Clip to reasonable range (avoid extreme scores)
    risk_prob = np.clip(risk_prob, 0.05, 0.95)
    
    # Convert to safety score (0-100, where 100 is safest)
    safety_score = int(round(100 * (1 - risk_prob)))
    
    return {
        "risk_prob": round(risk_prob, 4),
        "safety_score": safety_score
    }

if __name__ == "__main__":
    print("="*70)
    print("🧪 Testing DriveSure Risk Model Inference")
    print("="*70)
    
    # Test Case 1: Safe driver
    print("\n📍 Test 1: Safe Driver")
    safe_trip = {
        "avg_speed": 42,
        "max_speed": 60,
        "overspeed_ratio": 0.02,
        "harsh_brake_count": 0,
        "sharp_turn_count": 0,
        "night_ratio": 0.05,
        "trip_distance_km": 8,
        "trip_duration_min": 15
    }
    result1 = predict_safety_score(safe_trip)
    print(f"   Input: avg_speed={safe_trip['avg_speed']}, harsh_brakes={safe_trip['harsh_brake_count']}")
    print(f"   Risk Probability: {result1['risk_prob']}")
    print(f"   Safety Score: {result1['safety_score']}/100")
    
    # Test Case 2: Risky driver
    print("\n📍 Test 2: Risky Driver")
    risky_trip = {
        "avg_speed": 95,
        "max_speed": 135,
        "overspeed_ratio": 0.45,
        "harsh_brake_count": 8,
        "sharp_turn_count": 6,
        "night_ratio": 0.75,
        "trip_distance_km": 45,
        "trip_duration_min": 35
    }
    result2 = predict_safety_score(risky_trip)
    print(f"   Input: avg_speed={risky_trip['avg_speed']}, harsh_brakes={risky_trip['harsh_brake_count']}")
    print(f"   Risk Probability: {result2['risk_prob']}")
    print(f"   Safety Score: {result2['safety_score']}/100")
    
    # Test Case 3: Moderate driver
    print("\n📍 Test 3: Moderate Driver")
    moderate_trip = {
        "avg_speed": 65,
        "max_speed": 85,
        "overspeed_ratio": 0.18,
        "harsh_brake_count": 2,
        "sharp_turn_count": 3,
        "night_ratio": 0.30,
        "trip_distance_km": 22,
        "trip_duration_min": 28
    }
    result3 = predict_safety_score(moderate_trip)
    print(f"   Input: avg_speed={moderate_trip['avg_speed']}, harsh_brakes={moderate_trip['harsh_brake_count']}")
    print(f"   Risk Probability: {result3['risk_prob']}")
    print(f"   Safety Score: {result3['safety_score']}/100")
    
    print("\n" + "="*70)