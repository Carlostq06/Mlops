# Proyecto Final MLOps - Predicción de Precios de Vehículos de Ocasión (CarDekho)

**Autor:** Carlos  
**Módulo:** Data Science & MLOps - Práctica Final  
**Curso:** 2026 - Immune Technology Institute  

---

## 1. Justificación del Proyecto y Elección del Dataset

* **Dataset:** CarDekho Used Car Dataset (`CAR DETAILS FROM CAR DEKHO.csv`)
* **Fuente:** Kaggle ([Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)).
* **Problema de Negocio:** Estimación del precio de reventa de vehículos de segunda mano en función de sus características técnicas y de uso (antigüedad, kilometraje, combustible, transmisión, propietario).
* **Tipo de Tarea:** Regresión supervisada.
* **Variable Target:** `selling_price` (precio de venta en rupias / moneda local).
* **Métrica Principal Elegida:** **MAE (Mean Absolute Error)** y **RMSE (Root Mean Squared Error)**:
  - El MAE es directamente interpretable para el usuario final (error promedio en unidades monetarias).
  - El RMSE penaliza desviaciones grandes en vehículos de gama alta.

---

## 2. Estructura del Repositorio

```text
mlp/
├── .github/workflows/ci.yml       # Integración Continua (Linter + Smoke Tests)
├── data/
│   ├── raw/                       # Datos crudos descargados
│   └── processed/                 # Datos limpios y particionados (train, val, test)
├── notebooks/
│   ├── 01_eda_and_audit.ipynb     # Auditoría inicial y EDA completa
│   └── 02_model_evaluation.ipynb  # Entrenamiento, validación y comparativa
├── src/
│   ├── download_data.py           # Descarga automatizada del dataset
│   ├── data_prep.py               # Pipeline de limpieza y 5 decisiones justificadas
│   ├── train.py                   # Entrenamiento y exportación del pipeline unificado
│   ├── api.py                     # API de inferencia en FastAPI (/predict, /health)
│   └── app.py                     # Frontend de usuario en Streamlit
├── tests/
│   └── test_api.py                # Smoke tests y validaciones de la API
├── models/
│   └── pipeline.joblib            # Artefacto único (preprocesado + modelo)
├── Dockerfile                     # Imagen Docker de la API
├── requirements.txt               # Dependencias fijadas
└── README.md                      # Memoria técnica completa del proyecto
```

---

## 3. Guía de Ejecución Rápida (Demo en 2 Minutos)

### Requisitos previos
- Python 3.10+
- Entorno virtual o dependencias instaladas:
```bash
pip install -r requirements.txt
```

### Ejecutar el pipeline completo en 1 solo comando:
```bash
python src/train.py
```

### Arrancar la API de inferencia (FastAPI):
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```
- Documentación Swagger interactiva: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Arrancar la Mini-App (Streamlit):
```bash
streamlit run src/app.py
```
- Acceso en navegador: `http://localhost:8501`
