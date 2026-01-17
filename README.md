# 🚗 DriveSure Risk Scoring Model

> **Real-time telematics risk scoring API powered by LightGBM**, inspired by top Kaggle Porto Seguro solutions.

**Live API Docs:** https://drivesure-api.onrender.com/docs

---

## 🎯 Overview

DriveSure is a **Pay-As-You-Drive (PAYD)** insurance risk scoring system that evaluates real-time telematics data to estimate driver risk and generate a safety score.

### What It Does
- Accepts trip-level telematics data  
- Predicts crash/claim risk probability  
- Outputs a **0–100 safety score**, risk category, and top risk factors  
- Exposes predictions via a REST API  

### Business Value
- Fair, behavior-based insurance pricing  
- Real-time driver risk assessment  
- Actionable insights for driver coaching  
- Detection of abnormal or risky driving patterns  

---

## ✨ Key Features

| Feature | Description |
|-------|------------|
| Real-time inference | < 200 ms latency |
| Production-ready API | FastAPI + OpenAPI |
| Interpretable output | Top contributing risk factors |
| Reproducible training | Versioned model artifacts |
| Insurance-grade metrics | Gini & AUC validation |
| Synthetic data pipeline | Simulates actuarial logic |

---

## 🏗️ Architecture

Frontend (React / Mobile) │ ▼ FastAPI Backend ├─ Input validation (Pydantic) ├─ Feature engineering ├─ LightGBM inference └─ Risk scoring logic │ ▼ LightGBM Risk Model

Binary classification

13 engineered features

CV AUC ≈ 0.76


---

## 📊 Model Performance

### Cross-Validation Metrics

| Metric | Value |
|------|-------|
| AUC | 0.764 |
| Gini | 0.528 |
| Log Loss | 0.452 |

### Risk Categories

| Category | Risk Probability | Safety Score |
|---------|------------------|--------------|
| Low | 0.00 – 0.25 | 85 – 95 |
| Medium | 0.25 – 0.50 | 60 – 75 |
| High | 0.50 – 0.75 | 35 – 55 |
| Very High | 0.75 – 1.00 | 10 – 30 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip / virtualenv

### Installation

```bash
git clone https://github.com/yourusername/drivesure-risk-model.git
cd drivesure-risk-model

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


---

🧠 Model Training

Generate Synthetic Data

python3 src/generate_synthetic.py

30,000 synthetic trips

Rule-based risk labels

Realistic feature correlations


Train Model

python3 src/train_model.py

5-fold cross-validation

LightGBM gradient boosting

Model saved to models/


Offline Testing

python3 src/infer.py


---

📡 API Usage

Base URL

https://drivesure-api.onrender.com

Endpoint: Score Trip

POST /api/risk/scoreTrip

Example Request

{
  "avg_speed": 65,
  "max_speed": 90,
  "overspeed_ratio": 0.22,
  "harsh_brake_count": 3,
  "sharp_turn_count": 2,
  "night_ratio": 0.4,
  "trip_distance_km": 18.5,
  "trip_duration_min": 25,
  "trip_id": "TRIP_123"
}

Example Response

{
  "trip_id": "TRIP_123",
  "risk_prob": 0.45,
  "safety_score": 55,
  "risk_category": "medium",
  "top_factors": [
    { "feature": "overspeed_ratio", "importance": 0.28 },
    { "feature": "harsh_brake_count", "importance": 0.19 },
    { "feature": "night_ratio", "importance": 0.14 }
  ]
}


---

Health Check

GET /health

{
  "status": "healthy",
  "model_loaded": true,
  "features_count": 13
}


---

💻 Local Development

Run API Locally

cd src/api
uvicorn main:app --reload

Docs available at:
http://localhost:8000/docs


---

📁 Project Structure

drivesure-risk-model/
├── data/              # Synthetic training data
├── models/            # Trained LightGBM artifacts
├── src/
│   ├── generate_synthetic.py
│   ├── train_model.py
│   ├── infer.py
│   └── api/
│       └── main.py
├── requirements.txt
├── README.md
└── LICENSE


---

🧩 Feature Engineering

13 total features

Speed statistics

Driving behavior counts

Night driving exposure

Binary risk indicators


Top Contributors

1. Overspeed ratio


2. Harsh braking


3. Night driving


4. Average speed


5. Sharp turns




---

🛣️ Roadmap

Real telematics data integration

Driver-level risk aggregation

Rate limiting & monitoring

Model retraining automation

Premium calculation engine



---

🤝 Contributing

Pull requests and improvements are welcome.
Please open an issue for major changes.


---

📜 License

MIT License © DriveSure
