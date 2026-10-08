from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "iris_model.joblib"

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]
target_names = saved_data["target_names"]

app = FastAPI(title="Iris Flower Prediction API")


class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in centimetres")
    sepal_width: float = Field(..., gt=0, description="Sepal width in centimetres")
    petal_length: float = Field(..., gt=0, description="Petal length in centimetres")
    petal_width: float = Field(..., gt=0, description="Petal width in centimetres")


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Iris prediction API is running"}


@app.post("/predict")
def predict(input_data: IrisInput) -> dict[str, object]:
    features = np.array([[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width,
    ]])

    prediction_number = int(model.predict(features)[0])
    prediction = target_names[prediction_number]

    return {
        "prediction": prediction,
        "class_number": prediction_number,
    }
