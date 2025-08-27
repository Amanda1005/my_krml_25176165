from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.special import expit

app = FastAPI()

# Load the model
models = {
    "logistic": joblib.load("logit_model.pkl"),
    "random_forest": joblib.load("rf_model.pkl"),
    "xgboost": joblib.load("xgb_model.pkl"),
    "sgd": joblib.load("sgd_model.pkl"),
    "poly_logit": joblib.load("poly_logit_pipeline.pkl")
}

# Define input format
class PlayerFeatures(BaseModel):
    pts: float
    Min_per: float
    obpm: float
    TS_per: float
    bpm: float
    usg: float
    eFG: float
    mp: float
    drtg: float
    adrtg: float

@app.post("/predict/{model_name}")
def predict(model_name: str, features: PlayerFeatures):
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found.")

    model = models[model_name]

    # Prepare features
    input_data = np.array([[
        features.pts, features.Min_per, features.obpm, features.TS_per,
        features.bpm, features.usg, features.eFG, features.mp,
        features.drtg, features.adrtg
    ]])

    # Special handling of the case where SGD does not have predict_proba
    try:
        prob = model.predict_proba(input_data)[0][1]
    except AttributeError:
        decision = model.decision_function(input_data)[0]
        prob = float(expit(decision))

    return {
        "model": model_name,
        "draft_probability": round(prob, 4)
    }
