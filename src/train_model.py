# src/train_model.py
import os
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data", "training_data.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
FEATURES_FILE = os.path.join(ROOT, "features.json")
MODEL_FILE = os.path.join(MODEL_DIR, "drivesure_risk_model.txt")

# ✅ CRITICAL: Feature order must match inference exactly
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

def Gini(y_true, y_pred):
    """Gini coefficient for model evaluation"""
    n_samples = y_true.shape[0]
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]
    
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)
    
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    return G_pred / G_true

def train():
    print("="*70)
    print("🚀 DriveSure Risk Model Training")
    print("="*70)
    
    print(f"\n📂 Loading training data from: {DATA}")
    df = pd.read_csv(DATA)
    
    # Verify all features exist
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    y = df["target"].values
    X = df[FEATURES].values
    
    print(f"\n📊 Dataset Summary:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Risky trips (target=1): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   - Safe trips (target=0): {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
    print(f"   - Features: {len(FEATURES)}")
    
    # ✅ IMPROVED: Better hyperparameters
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "min_data_in_leaf": 50,
        "min_sum_hessian_in_leaf": 5.0,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": 42,
        "metric": "binary_logloss"
    }
    
    NFOLDS = 5
    kf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(df))
    models = []
    fold_scores = []
    
    print(f"\n🔄 Starting {NFOLDS}-Fold Cross-Validation...")
    print("-"*70)
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n📍 Fold {fold+1}/{NFOLDS}")
        
        X_train, X_val = X[tr_idx], X[val_idx]
        y_train, y_val = y[tr_idx], y[val_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(50)
            ]
        )
        
        # Predict on validation set
        val_preds = bst.predict(X_val, num_iteration=bst.best_iteration)
        oof[val_idx] = val_preds
        
        # Calculate metrics
        auc = roc_auc_score(y_val, val_preds)
        gini = Gini(y_val, val_preds)
        
        fold_scores.append({'auc': auc, 'gini': gini})
        
        print(f"   ✅ Fold {fold+1} Results:")
        print(f"      - AUC:  {auc:.6f}")
        print(f"      - Gini: {gini:.6f}")
        print(f"      - Best iteration: {bst.best_iteration}")
        
        models.append(bst)
    
    # Overall CV performance
    overall_auc = roc_auc_score(y, oof)
    overall_gini = Gini(y, oof)
    
    print("\n" + "="*70)
    print("🏆 Cross-Validation Results")
    print("="*70)
    print(f"Overall AUC:  {overall_auc:.6f}")
    print(f"Overall Gini: {overall_gini:.6f}")
    print(f"\nFold-wise AUC:  {[f'{s['auc']:.6f}' for s in fold_scores]}")
    print(f"Fold-wise Gini: {[f'{s['gini']:.6f}' for s in fold_scores]}")
    
    # Save the best model (last fold for simplicity, or choose best AUC fold)
    best_model = models[-1]
    
    print(f"\n💾 Saving model to: {MODEL_FILE}")
    best_model.save_model(MODEL_FILE)
    
    # Save feature contract
    with open(FEATURES_FILE, "w") as f:
        json.dump(FEATURES, f, indent=2)
    print(f"✅ Saved feature contract to: {FEATURES_FILE}")
    
    # Save model config
    config = {
        "features": FEATURES,
        "cv_auc": overall_auc,
        "cv_gini": overall_gini,
        "fold_scores": fold_scores,
        "params": params,
        "best_iteration": best_model.best_iteration
    }
    config_path = os.path.join(MODEL_DIR, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved model config to: {config_path}")
    
    # Feature importance
    print("\n📊 Top 10 Feature Importances:")
    importance = best_model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(FEATURES, importance), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(feat_imp[:10], 1):
        print(f"   {i:2d}. {feat:25s} {imp:8.0f}")
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)

if __name__ == "__main__":
    train()