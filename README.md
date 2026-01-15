# DriveSureModel

1. Setup
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt

2. Generate synthetic training data
   python3 src/generate_synthetic.py
   # creates data/training_data.csv

3. Train model
   python3 src/train_model.py
   # outputs models/drivesure_risk_model.txt and features.json

4. Run local test inference
   python3 src/infer.py

5. Run FastAPI server
   cd src/api
   uvicorn main:app --host 0.0.0.0 --port 8000

6. Frontend integration
   POST trip JSON to http://localhost:8000/api/risk/scoreTrip
   Frontend expects:
   {
     "trip_id": "T1",
     "avg_speed": 48.5,
     "max_speed": 92,
     "overspeed_ratio": 0.18,
     "harsh_brake_count": 3,
     "sharp_turn_count": 1,
     "night_ratio": 0.35,
     "trip_distance_km": 12.5,
     "trip_duration_min": 22
   }
