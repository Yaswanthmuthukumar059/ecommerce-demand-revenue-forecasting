import os
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_model(csv_path: str, models_dir: str = "models"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded data shape: {df.shape}")

    if "Units_Sold" not in df.columns:
        raise ValueError("Dataset must contain 'Units_Sold' column as target.")

    y = df["Units_Sold"]
    X = df.drop(columns=["Units_Sold"])

    # One-hot encode categoricals
    X_enc = pd.get_dummies(X, drop_first=True)
    feature_names = X_enc.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=42),
            {"n_estimators": [50, 100]},
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [50, 100]},
        ),
    }

    best_model = None
    best_name = None
    best_score = float("inf")
    best_metrics = {}

    for name, (est, param_grid) in models.items():
        print(f"[INFO] Training {name} with GridSearch...")
        grid = GridSearchCV(est, param_grid, cv=3, n_jobs=1, verbose=0)
        grid.fit(X_train, y_train)

        cand = grid.best_estimator_
        preds = cand.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5   # RMSE
        r2 = r2_score(y_test, preds)
        print(f"[RESULT] {name} -> RMSE: {rmse:.3f}, R2: {r2:.3f}")

        if rmse < best_score:
            best_score = rmse
            best_model = cand
            best_name = name
            best_metrics = {"rmse": float(rmse), "r2": float(r2)}

    if best_model is None:
        raise RuntimeError("No model trained successfully.")

    print(f"[BEST] Model: {best_name}, metrics: {best_metrics}")

    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    columns_path = os.path.join(models_dir, "columns.json")
    meta_path = os.path.join(models_dir, "meta.json")

    dump(best_model, model_path)
    dump(scaler, scaler_path)
    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Feature importance if available
    top_features = []
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        idx = np.argsort(imp)[-10:]  # top 10
        for i in idx:
            top_features.append(
                {"feature": feature_names[i], "importance": float(imp[i])}
            )

    meta = {
        "model_name": best_name,
        "metrics": best_metrics,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_names)),
        "target": "Units_Sold",
        "top_features": top_features,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVE] Saved model artifacts to '{models_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path", help="Path to training CSV (must contain Units_Sold column)"
    )
    args = parser.parse_args()
    train_model(args.csv_path)
