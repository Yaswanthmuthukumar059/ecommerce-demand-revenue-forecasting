import os
import json
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "columns.json")
META_PATH = os.path.join(MODELS_DIR, "meta.json")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run train.py first to create models/best_model.pkl")

model = load(MODEL_PATH)
scaler = load(SCALER_PATH)
with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
    feature_names = json.load(f)

meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

app = FastAPI(title="E-Commerce Demand & Revenue API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local UI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    data: Dict[str, Any]  # one row of features


@app.post("/predict")
def predict(req: PredictRequest):
    """Predict Units_Sold and Revenue for a single input row."""
    import numpy as np

    df = pd.DataFrame([req.data])

    # one-hot encode and align columns
    df_enc = pd.get_dummies(df)
    df_enc = df_enc.reindex(columns=feature_names, fill_value=0)

    X_scaled = scaler.transform(df_enc)
    units_pred = float(model.predict(X_scaled)[0])

    # Revenue = predicted_units * Price (if Price provided)
    price = req.data.get("Price", 0)
    try:
        price = float(price)
    except Exception:
        price = 0.0

    revenue_pred = units_pred * price

    return {
        "predicted_units": units_pred,
        "predicted_revenue": revenue_pred,
    }


@app.get("/model-info")
def model_info():
    """Return basic model metrics and top features."""
    return meta
