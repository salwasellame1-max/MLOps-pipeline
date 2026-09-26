"""The API: exposes the churn model over HTTP.

Run from the project root:
    python -m uvicorn src.api:app --reload
Then open http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException

from src import config
from src.predict import predict
from src.schemas import Customer, PredictionResponse

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
)


@app.get("/health")
def health():
    """Simple check: is the API alive and can it reach the model file?"""
    return {"status": "ok", "model_path": str(config.MODEL_PATH)}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: Customer):
    """Predict churn probability for one customer."""
    try:
        return predict(customer.model_dump())
    except FileNotFoundError as e:
        # The model file doesn't exist yet (train.py was never run)
        raise HTTPException(status_code=503, detail=str(e))