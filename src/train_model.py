# src/train_model.py
import os
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data", "training_data.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
FEATURES_FILE = os.path.join(ROOT, "features.json")
MODEL_FILE = os.path.join(MODEL_DIR, "drivesure_risk_model.txt")

# Feature order contract (must match infer + FastAPI)
FEATURES = [
    "avg_speed",
    "max_speed",
    "overspeed_ratio",
    "harsh_brake_count",
    "sharp_turn_count",
    "night_ratio",
    "trip_distance_km",
    "trip_duration_min",
    "is_night_trip",
    "has_harsh_brake",
    "has_sharp_turn",
    "is_overspeed",
    "missing"
]

def train():
    print("Loading", DATA)
    df = pd.read_csv(DATA)
    y = df["target"].values
    X = df[FEATURES].values

    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.6,
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "seed": 42,
    }

    NFOLDS = 5
    kf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(df))
    models = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        print("Fold", fold+1)
        dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx])
        dval = lgb.Dataset(X[val_idx], label=y[val_idx])
        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100)
            ]
        )

        # bst = lgb.train(
        #     params,
        #     dtrain,
        #     num_boost_round=1000,
        #     valid_sets=[dval],
        #     early_stopping_rounds=50,
        #     verbose_eval=100
        # )
        oof[val_idx] = bst.predict(X[val_idx], num_iteration=bst.best_iteration)
        models.append(bst)

    # Save a single model (the last fold) — good enough for inference/demo
    print("Saving model to", MODEL_FILE)
    models[-1].save_model(MODEL_FILE)

    # Save features list
    with open(FEATURES_FILE, "w") as f:
        json.dump(FEATURES, f)
    print("Saved features contract to", FEATURES_FILE)

    # Basic CV score (AUC-ish proxy; prints mean of predictions for rough check)
    print("OOF mean risk (approx):", oof.mean())

if __name__ == "__main__":
    train()
