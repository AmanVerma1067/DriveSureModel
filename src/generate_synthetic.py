# src/generate_synthetic.py
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

def generate_training_data(n_samples=30000, out_file="training_data.csv"):
    """Generate synthetic telematics data with realistic distributions"""
    np.random.seed(42)
    
    # Base continuous features
    avg_speed = np.clip(np.random.normal(55, 18, n_samples), 15, 140)
    max_speed = avg_speed + np.clip(np.random.gamma(2, 10, n_samples), 5, 60)
    
    data = {
        "trip_id": [f"T{i:06d}" for i in range(n_samples)],
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "overspeed_ratio": np.clip(np.random.beta(2, 8, n_samples), 0, 1),
        "harsh_brake_count": np.random.poisson(1.2, n_samples),
        "sharp_turn_count": np.random.poisson(1.8, n_samples),
        "night_ratio": np.clip(np.random.beta(1.5, 4, n_samples), 0, 1),
        "trip_distance_km": np.clip(np.random.gamma(2.5, 6, n_samples), 0.5, 300),
        "trip_duration_min": np.clip(np.random.gamma(3.5, 8, n_samples), 2, 480),
    }
    df = pd.DataFrame(data)
    
    # Add correlation for realism
    # High speed → more harsh braking
    high_speed_mask = df["avg_speed"] > 75
    df.loc[high_speed_mask, "harsh_brake_count"] += np.random.poisson(1, high_speed_mask.sum())
    
    # Night driving → slightly higher overspeed ratio
    night_mask = df["night_ratio"] > 0.6
    df.loc[night_mask, "overspeed_ratio"] += np.clip(np.random.uniform(0, 0.15, night_mask.sum()), 0, 0.3)
    df["overspeed_ratio"] = df["overspeed_ratio"].clip(0, 1)
    
    # Derived binary features (MUST match training order)
    df["is_night_trip"] = (df["night_ratio"] > 0.5).astype(int)
    df["has_harsh_brake"] = (df["harsh_brake_count"] > 0).astype(int)
    df["has_sharp_turn"] = (df["sharp_turn_count"] > 0).astype(int)
    df["is_overspeed"] = (df["overspeed_ratio"] > 0.15).astype(int)
    df["missing"] = 0  # No missing values in synthetic data
    
    # ✅ IMPROVED: Multi-factor risk score with realistic weights
    risk_score = (
        df["overspeed_ratio"] * 40 +                          # High weight
        df["harsh_brake_count"] * 8 +                         # Medium-high
        df["sharp_turn_count"] * 5 +                          # Medium
        df["night_ratio"] * 15 +                              # Medium-high
        (df["avg_speed"] - 60).clip(0, 100) * 0.4 +          # Speed penalty
        df["is_night_trip"] * 8 +                             # Binary night penalty
        (df["max_speed"] - 100).clip(0, 100) * 0.3            # Max speed penalty
    )
    
    # Add noise for realism
    risk_score += np.random.normal(0, 8, n_samples)
    
    # Create balanced target (40% risky, 60% safe)
    threshold = np.percentile(risk_score, 60)
    df["target"] = (risk_score > threshold).astype(int)
    
    # Store raw risk score for analysis
    df["risk_score_raw"] = risk_score
    
    out_path = os.path.join(OUT, out_file)
    df.to_csv(out_path, index=False)
    
    print(f"✅ Generated {n_samples} training samples")
    print(f"   - File: {out_path}")
    print(f"   - Risky trips: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    print(f"   - Safe trips: {(1-df['target']).sum()} ({(1-df['target']).mean()*100:.1f}%)")
    print(f"\n📊 Feature Statistics:")
    print(df[['avg_speed', 'overspeed_ratio', 'harsh_brake_count', 'night_ratio']].describe())
    
    return out_path

if __name__ == "__main__":
    generate_training_data()