"""
Modulo 3: API de inferencia con FastAPI.

Endpoints:
- GET  /health  -> estado del servicio y si el modelo esta cargado
- POST /predict -> recibe los datos de un coche y devuelve el precio estimado
- GET  /docs    -> Swagger (lo genera FastAPI automaticamente)

El modelo es models/pipeline.joblib: preprocesado + modelo en un solo artefacto,
asi que el cliente solo manda los datos "en crudo".
"""
import os
import json
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "pipeline.joblib"))
METRICS_PATH = os.path.join(os.path.dirname(MODEL_PATH), "metrics.json")

# Marcas vistas en entrenamiento (si llega otra, devolvemos 422)
BRANDS = Literal[
    "Ambassador", "Audi", "Bmw", "Chevrolet", "Daewoo", "Datsun", "Fiat",
    "Force", "Ford", "Honda", "Hyundai", "Isuzu", "Jaguar", "Jeep", "Kia",
    "Land", "Mahindra", "Maruti", "Mercedes-Benz", "Mg", "Mitsubishi",
    "Nissan", "Opelcorsa", "Renault", "Skoda", "Tata", "Toyota",
    "Volkswagen", "Volvo",
]


class CarInput(BaseModel):
    """Datos de entrada de un coche."""
    brand: BRANDS = Field(..., description="Marca del coche", examples=["Maruti"])
    year: int = Field(..., ge=1990, le=2020, description="Año de matriculación (rango visto en entrenamiento)", examples=[2015])
    km_driven: int = Field(..., ge=500, le=300000, description="Kilómetros recorridos", examples=[50000])
    fuel: Literal["Petrol", "Diesel", "CNG", "LPG", "Electric"] = Field(..., examples=["Petrol"])
    seller_type: Literal["Individual", "Dealer", "Trustmark Dealer"] = Field(..., examples=["Individual"])
    transmission: Literal["Manual", "Automatic"] = Field(..., examples=["Manual"])
    owner: Literal[
        "First Owner", "Second Owner", "Third Owner",
        "Fourth & Above Owner", "Test Drive Car",
    ] = Field(..., examples=["First Owner"])


class PredictionOutput(BaseModel):
    """Respuesta de /predict."""
    predicted_price: float = Field(..., description="Precio estimado en rupias indias (INR)")
    currency: str = "INR"


app = FastAPI(
    title="CarDekho Price API",
    description="Predice el precio de reventa de un coche usado (dataset CarDekho).",
    version="1.0.0",
)

# Cargamos el modelo una sola vez al arrancar
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/version")
def version():
    """Version de la API y del modelo servido (con sus metricas de test)."""
    return {"api_version": app.version, "model": metrics}


@app.post("/predict", response_model=PredictionOutput)
def predict(car: CarInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Ejecuta primero el entrenamiento.")
    try:
        df = pd.DataFrame([car.model_dump()])
        price = float(model.predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")
    # Un precio negativo no tiene sentido, lo acotamos al minimo visto (20.000)
    price = max(price, 20000.0)
    return PredictionOutput(predicted_price=round(price, 2))
