# NBA Draft Prediction API

This FastAPI web service allows real-time predictions of NBA draft probability based on trained machine learning models.

## How to Run the API

1. Open terminal and go to the `api/` directory:

```bash
cd api

2. Start the FastAPI server with uvicorn:
uvicorn main:app --reload

3. Visit the interactive API docs at:
http://127.0.0.1:8000/docs

Input Format

Send a POST request to /predict/{model_name} with a JSON body containing the following 13 features:

{
  "pts": 25.4,
  "Min_per": 30.1,
  "obpm": 4.2,
  "TS_per": 0.582,
  "bpm": 5.1,
  "usg": 25.3,
  "eFG": 0.541,
  "mp": 780,
  "drtg": 101.3,
  "adrtg": 107.5,
  "pts_per_min": 0.845,
  "ts_usg": 14.72,
  "eff_combo": 12.56
}

## Available Models

## You can replace {model_name} in the URL with any of the following:
logit – Baseline Logistic Regression
rf – Random Forest
xgboost – XGBoost Classifier
sgd – SGDClassifier
poly_logit – Polynomial Logistic Regression

# Example request to use XGBoost model:
POST http://127.0.0.1:8000/predict/xgboost
Content-Type: application/json

Expected Response
{
  "model": "xgboost",
  "draft_probability": 0.8732
}

## Author

Student ID: [25176165]

Group: [30]