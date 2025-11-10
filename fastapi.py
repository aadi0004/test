"""
app.py
-------
Deploy your trained ML model using FastAPI.
It loads the saved best_model.joblib and provides a /predict endpoint.

Run command:
    uvicorn app:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize FastAPI app
app = FastAPI(
    title="AI/ML Model Deployment API",
    description="Predict outcomes using your trained ML model",
    version="1.0"
)

# Load your trained model
try:
    model = joblib.load("best_model.joblib")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    raise e

# Define the request body using Pydantic for data validation
class PredictionRequest(BaseModel):
    # Define example input features — update these to match your dataset
    feature1: float
    feature2: float
    feature3: str
    feature4: int

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "🚀 ML Model API is running. Use /predict to get results."}

# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([request.dict()])

        # Predict using the loaded model
        prediction = model.predict(data)
        result = int(prediction[0]) if hasattr(prediction, "__len__") else prediction

        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# uvicorn app:app --reload