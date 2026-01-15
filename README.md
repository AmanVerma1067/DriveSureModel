# DriveSure Risk Scoring Model

Porto Seguro 2nd place solution adapted for telematics-based driver risk scoring.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python run_pipeline.py --step all
```

This will:
1. Generate 30,000 synthetic training trips
2. Train LightGBM model with cross-validation
3. Start FastAPI server at `http://localhost:8000`

### 3. Test API

Visit `http://localhost:8000/docs` for interactive API documentation.

Example request:
```bash
curl -X POST "http://localhost:8000/api/risk/scoreTrip" \
  -H "Content-Type: application/json" \
  -d '{
    "trip_id": "T123",
    "avg_speed": 65.5,
    "max_speed": 95.0,
    "overspeed_ratio": 0.18,
    "harsh_brake_count": 3,
    "sharp_turn_count": 2,
    "acceleration_events": 4,
    "night_ratio": 0.25,
    "weekend_trip": 0,
    "trip_distance_km": 25.5,
    "trip_duration_min": 35.0,
    "weather_condition": 0,
    "road_type": 1
  }'
```

## 📊 Model Performance

- Algorithm: LightGBM (2nd place Porto Seguro solution)
- Cross-validation: 5-fold stratified
- Metric: Normalized Gini coefficient
- Expected CV score: 0.28-0.29

## 🔧 Individual Steps
```bash
# Generate data only
python run_pipeline.py --step data

# Train model only
python run_pipeline.py --step train

# Start API only
python run_pipeline.py --step api
```

## 📁 Project Structure
````
drivesure-risk-model/
├── data/               # Training data
├── models/             # Trained models
├── src/                # Core ML code
├── backend/            # FastAPI application
├── notebooks/          # Jupyter notebooks
└── tests/              # Unit tests