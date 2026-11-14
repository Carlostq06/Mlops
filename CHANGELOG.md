# Changelog

## v1.0.0 (2026-11-14)

Primera versión completa del proyecto.

### M1: Datos
- Descarga automática del dataset CarDekho (`src/download_data.py`).
- Auditoría inicial impresa y EDA visual completa (`src/eda_audit.py`, notebook 01).
- 5 decisiones de preparación justificadas con gráficas, y splits train/val/test 70/15/15 (`src/data_prep.py`).

### M2: Modelo
- Comparación del baseline dummy con Ridge, Random Forest y Gradient Boosting. Gana Gradient Boosting (−49,8% de MAE frente al dummy).
- Test: MAE 150.807 INR, R² 0,71. Métricas guardadas en `models/metrics.json`.
- Seeds fijas y dependencias fijadas con `==`.

### M3: Despliegue
- API FastAPI con `/health`, `/predict`, `/version` y Swagger en `/docs`.
- Validación de entrada con Pydantic (422) y errores controlados (503/500).
- Dockerfile multi-stage ejecutado con usuario no-root.

### M4: App
- Mini-app Streamlit que consume la API, con estados de espera y de error.

### M5: Automatización
- `run_pipeline.py`: todo el procesado en un solo comando.
- CI: flake8 + pytest en cada push y PR.
- Reentrenamiento semanal con puerta de métrica.
- Detección de drift (KS + chi²).
- CD con tags: imagen en GHCR y GitHub Release.
