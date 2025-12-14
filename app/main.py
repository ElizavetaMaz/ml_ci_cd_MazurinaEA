
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
import joblib

app = FastAPI()
VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
# Загружаем модель (предварительно сохрани её через joblib.dump)
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
model = joblib.load(MODEL_PATH)

# Схема входных данных
class PredictRequest(BaseModel):
    x: list[float]

@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}

@app.post("/predict")
def predict(request: PredictRequest):
    X = np.array(request.x).reshape(1, -1)
    y_pred = model.predict(X)[0]

    return {
        "status": "ok",
        "prediction": int(y_pred),
        "version": VERSION
    }
