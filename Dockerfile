# --- Etapa 1: instalar dependencias (con pip y cache de build) ---
FROM python:3.13-slim AS builder

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# --- Etapa 2: imagen final, solo lo necesario para ejecutar ---
FROM python:3.13-slim

WORKDIR /app

# Copiamos solo los paquetes ya instalados en la etapa 1
COPY --from=builder /install /usr/local

# Codigo de la API, el modelo (preprocesado + modelo) y sus metricas
COPY src/api.py src/api.py
COPY models/pipeline.joblib models/metrics.json models/

# Seguridad: no ejecutar como root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
