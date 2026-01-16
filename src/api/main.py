import os
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import lightgbm as lgb
import numpy as np

# Path configuration
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_FILE = os.path.join(ROOT, "models", "drivesure_risk_model.txt")
FEATURES_FILE = os.path.join(ROOT, "features.json")
CONFIG_FILE = os.path.join(ROOT, "models", "model_config.json")

app = FastAPI(
    title="DriveSure Risk Scoring API",
    description="Real-time telematics risk scoring powered by LightGBM",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and features at startup
if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_FILE):
    raise RuntimeError(
        f"Model files not found!\n"
        f"Expected:\n"
        f"  - {MODEL_FILE}\n"
        f"  - {FEATURES_FILE}\n"
        f"Run 'python src/train_model.py' first."
    )

with open(FEATURES_FILE) as f:
    FEATURES = json.load(f)

MODEL = lgb.Booster(model_file=MODEL_FILE)

# Load model config if available
MODEL_CONFIG = {}
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        MODEL_CONFIG = json.load(f)

print(f"✅ Loaded DriveSure Risk Model")
print(f"   - Features: {len(FEATURES)}")
print(f"   - CV AUC: {MODEL_CONFIG.get('cv_auc', 'N/A')}")

# Pydantic models
class TripFeatures(BaseModel):
    """Input schema for trip risk scoring"""
    avg_speed: float = Field(..., ge=0, le=200, description="Average speed in km/h")
    max_speed: float = Field(..., ge=0, le=250, description="Maximum speed in km/h")
    overspeed_ratio: float = Field(..., ge=0, le=1, description="Ratio of time spent overspeeding (0-1)")
    harsh_brake_count: int = Field(..., ge=0, description="Number of harsh braking events")
    sharp_turn_count: int = Field(..., ge=0, description="Number of sharp turns")
    night_ratio: float = Field(..., ge=0, le=1, description="Ratio of trip during night time (0-1)")
    trip_distance_km: float = Field(..., ge=0, description="Total trip distance in km")
    trip_duration_min: float = Field(..., ge=0, description="Total trip duration in minutes")
    trip_id: Optional[str] = Field(None, description="Optional trip identifier")

class RiskResponse(BaseModel):
    """Output schema for risk scoring"""
    trip_id: Optional[str]
    risk_prob: float = Field(..., description="Risk probability (0-1)")
    safety_score: int = Field(..., ge=0, le=100, description="Safety score (0-100, higher is safer)")
    risk_category: str = Field(..., description="Risk category: low, medium, high, very_high")
    top_factors: List[Dict[str, Any]] = Field(..., description="Top risk factors")

def prepare_vector(trip: TripFeatures) -> np.ndarray:
    """
    ✅ CRITICAL: Feature engineering matching training exactly
    
    Args:
        trip: Pydantic model with trip features
    
    Returns:
        numpy array for model prediction
    """
    trip_dict = trip.dict()
    v = []
    
    for f in FEATURES:
        if f == "is_night_trip":
            v.append(int(trip_dict.get("night_ratio", 0) > 0.5))
        elif f == "has_harsh_brake":
            v.append(int(trip_dict.get("harsh_brake_count", 0) > 0))
        elif f == "has_sharp_turn":
            v.append(int(trip_dict.get("sharp_turn_count", 0) > 0))
        elif f == "is_overspeed":
            v.append(int(trip_dict.get("overspeed_ratio", 0) > 0.15))
        elif f == "missing":
            v.append(0)  # No missing values in real-time data
        else:
            v.append(float(trip_dict.get(f, 0)))
    
    return np.array([v], dtype=float)

def get_risk_category(risk_prob: float) -> str:
    """Categorize risk probability into buckets"""
    if risk_prob < 0.25:
        return "low"
    elif risk_prob < 0.50:
        return "medium"
    elif risk_prob < 0.75:
        return "high"
    else:
        return "very_high"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": "drivesure_risk_model",
        "version": "1.0.0",
        "features": len(FEATURES),
        "cv_performance": {
            "auc": MODEL_CONFIG.get("cv_auc"),
            "gini": MODEL_CONFIG.get("cv_gini")
        }
    }

@app.post("/api/risk/scoreTrip", response_model=RiskResponse)
async def score_trip(trip: TripFeatures):
    """
    Score a single trip for risk
    
    Args:
        trip: Trip features
    
    Returns:
        Risk probability, safety score, and top factors
    """
    try:
        # Prepare feature vector
        X = prepare_vector(trip)
        
        # Predict risk probability
        risk_prob = float(MODEL.predict(X)[0])
        
        # ✅ Clip to reasonable range
        risk_prob = np.clip(risk_prob, 0.01, 0.99)
        
        # Convert to safety score (0-100)
        safety_score = int(round(100 * (1 - risk_prob)))
        
        # Get risk category
        risk_category = get_risk_category(risk_prob)
        
        # Get feature importance (top contributing factors)
        importances = MODEL.feature_importance(importance_type='gain')
        
        # Create feature impact list (simplified for top factors)
        trip_dict = trip.dict()
        feature_impacts = []
        
        for feat, imp in zip(FEATURES, importances):
            if feat in trip_dict and imp > 0:
                # Normalize importance to 0-1 scale
                normalized_imp = imp / importances.sum()
                
                feature_impacts.append({
                    "feature": feat,
                    "importance": round(float(normalized_imp), 4),
                    "value": trip_dict.get(feat)
                })
        
        # Sort by importance and take top 5
        feature_impacts.sort(key=lambda x: x["importance"], reverse=True)
        top_factors = feature_impacts[:5]
        
        return RiskResponse(
            trip_id=trip.trip_id,
            risk_prob=round(risk_prob, 4),
            safety_score=safety_score,
            risk_category=risk_category,
            top_factors=top_factors
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/risk/batchScore")
async def batch_score(trips: List[TripFeatures]):
    """
    Score multiple trips in batch
    
    Args:
        trips: List of trip features
    
    Returns:
        List of risk scores
    """
    results = []
    for trip in trips:
        result = await score_trip(trip)
        results.append(result)
    
    return {"results": results, "count": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
