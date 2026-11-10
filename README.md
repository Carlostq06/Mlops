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

---

## 4. Módulo 1: Auditoría y Decisiones de Preparación de Datos

### Auditoría Inicial (Datos Crudos):
* **Registros:** 4,340 filas × 8 columnas.
* **Valores nulos:** 0 nulos detectados.
* **Duplicados:** 763 registros repetidos (17.58%).
* **Target (`selling_price`):** Min: 20,000 | Mediana: 350,000 | Media: 504,127 | Max: 8,900,000. Alta asimetría positiva hacia la derecha.

### Las 5 Decisiones de Preparación Justificadas (M1.4):
1. **Decisión 1 (Drop - Eliminación de Duplicados):**
   - *Motivo:* 763 registros eran réplicas exactas. Mantenerlos provocaría sobreajuste y fuga de datos (data leakage) entre train y test.
   - *Resultado:* Reducción de 4,340 a 3,577 registros únicos.
   - *Gráfico:* `reports/figures/decision_1_drop_duplicates.png`.
2. **Decisión 2 (Crear - Extracción de la Marca `brand`):**
   - *Motivo:* El campo `name` contenía 1,491 nombres de modelos largos con excesiva cardinalidad. Extrayendo la primera palabra obtenemos 29 marcas principales que preservan la señal del segmento y reputación del fabricante.
   - *Resultado:* Nueva columna `brand` categórica (Top 5: Maruti, Hyundai, Mahindra, Tata, Ford).
   - *Gráfico:* `reports/figures/decision_2_create_brand.png`.
3. **Decisión 3 (Crear/Transformar - Antigüedad `vehicle_age`):**
   - *Motivo:* El año calendario `year` es menos informativo que la antigüedad del coche en años (`2020 - year`), la cual correlaciona directamente con la depreciación acumulada del vehículo.
   - *Resultado:* Variable continua `vehicle_age` entre 0 y 28 años.
   - *Gráfico:* `reports/figures/decision_3_vehicle_age.png`.
4. **Decisión 4 (Filtro/Drop - Tratamiento de Outliers en `km_driven`):**
   - *Motivo:* Existen vehículos con kilometrajes poco creíbles (>800,000 km) o cercanos a cero (1 km). Se filtran valores atípicos superiores al percentil 99.5 (~250,720 km) e inferiores a 500 km.
   - *Resultado:* 22 registros anómalos eliminados, estabilizando la distribución a 3,555 instancias fiables.
   - *Gráfico:* `reports/figures/decision_4_filter_outliers.png`.
5. **Decisión 5 (Recodificar - Orden de `owner` y Binaria para `transmission`):**
   - *Motivo:* `owner` es de naturaleza ordinal por número de propietarios anteriores. Se recodifica numéricamente de 0 a 4 (0: Test Drive Car, 1: First Owner, 2: Second Owner, 3: Third Owner, 4: Fourth+ Owner). `transmission` se codifica a binario (0: Manual, 1: Automatic).
   - *Resultado:* Variables numéricas listas para regresores manteniendo coherencia semántica.
   - *Gráfico:* `reports/figures/decision_5_recodification.png`.

