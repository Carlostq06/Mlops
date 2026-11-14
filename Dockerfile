FROM python:3.13-slim

WORKDIR /app

# Instalamos solo lo necesario para servir la API
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Codigo de la API y el artefacto del modelo (preprocesado + modelo)
COPY src/api.py src/api.py
COPY models/pipeline.joblib models/pipeline.joblib

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
