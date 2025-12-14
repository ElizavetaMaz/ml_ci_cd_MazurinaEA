from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
import joblib
import logging
from prometheus_client import Counter

# Настройка логов
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

requests_total = Counter("requests_total", "Total number of requests", ["endpoint"])

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
    requests_total.labels(endpoint="/health").inc()
    logger.info("Зашли в Health")
    return {"status": "ok", "version": VERSION}

@app.post("/predict")
def predict(request: PredictRequest):
    requests_total.labels(endpoint="/predict").inc()
    logger.info(f"Начинаю предикт для интпута: {request.x}")
    X = np.array(request.x).reshape(1, -1)
    y_pred = model.predict(X)[0]
    logger.info(f"Получен предикт: {y_pred}")

    return {
        "status": "ok",
        "prediction": int(y_pred),
        "version": VERSION
    }

