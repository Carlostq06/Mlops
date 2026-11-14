"""
Modulo 4: Mini-app para el usuario final (Streamlit).

El usuario rellena un formulario y la app llama a la API de M3 (/predict).
La app NO carga el modelo: solo consume la API.
"""
import json
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
BRAND_MODELS_PATH = os.path.join("models", "brand_models.json")

DEFAULT_BRAND_MODELS = {
    "Maruti": ["Swift", "Alto", "Wagon", "Dzire", "Baleno"],
    "Hyundai": ["i20", "i10", "Grand", "Verna", "Creta"],
    "Honda": ["City", "Amaze", "Brio", "Jazz"],
    "Toyota": ["Innova", "Fortuner", "Etios", "Corolla"],
    "Mahindra": ["Scorpio", "XUV500", "Bolero"],
    "Tata": ["Indica", "Nexon", "Tiago", "Safari"],
    "Ford": ["Figo", "Ecosport", "Endeavour"],
}

if os.path.exists(BRAND_MODELS_PATH):
    with open(BRAND_MODELS_PATH, encoding="utf-8") as f:
        BRAND_MODELS = json.load(f)
else:
    BRAND_MODELS = DEFAULT_BRAND_MODELS

BRANDS = sorted(BRAND_MODELS.keys())

st.set_page_config(page_title="¿Cuánto vale mi coche?", page_icon="🚗")
st.title("🚗 ¿Cuánto vale mi coche usado?")
st.write("Rellena los datos de tu coche y te damos un precio estimado de reventa.")

# Estado de la API (el usuario siempre ve si el sistema está disponible)
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("model_loaded"):
        st.success("Servicio de predicción disponible ✅")
    else:
        st.warning("La API responde pero el modelo no está cargado.")
except requests.exceptions.RequestException:
    st.error(f"No se puede conectar con la API en {API_URL}. ¿Está arrancada?")

col1, col2 = st.columns(2)
with col1:
    default_brand_idx = BRANDS.index("Maruti") if "Maruti" in BRANDS else 0
    brand = st.selectbox("Marca", BRANDS, index=default_brand_idx)
    available_models = BRAND_MODELS.get(brand, ["Otro"])
    default_model_idx = available_models.index("Swift") if "Swift" in available_models else 0
    car_model = st.selectbox("Modelo", available_models, index=default_model_idx)
    year = st.slider("Año de matriculación", 1992, 2025, 2015)
    km_driven = st.number_input("Kilómetros recorridos", min_value=500, max_value=300000, value=50000, step=1000)
with col2:
    fuel = st.selectbox("Combustible", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = st.radio("Transmisión", ["Manual", "Automatic"])
    seller_type = st.selectbox("Tipo de vendedor", ["Individual", "Dealer", "Trustmark Dealer"])
    owner = st.selectbox(
        "Propietarios anteriores",
        ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
    )

submitted = st.button("Calcular precio", type="primary")

if submitted:
    payload = {
        "brand": brand,
        "car_model": car_model,
        "year": int(year),
        "km_driven": int(km_driven),
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }
    with st.spinner("Calculando precio..."):
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        except requests.exceptions.RequestException:
            st.error("No se pudo contactar con la API. Inténtalo de nuevo en unos segundos.")
            st.stop()

    if r.status_code == 200:
        data = r.json()
        price = data["predicted_price"]
        price_inr = data.get("predicted_price_inr")
        st.metric("Precio estimado", f"{price:,.0f} €")
        if year > 2020:
            st.info(
                "Nota: el dataset de entrenamiento llega hasta 2020; para años 2021–2025 "
                "el modelo toma como referencia el valor de un coche de 2020."
            )
        extra_inr = f" (~₹ {price_inr:,.0f})" if price_inr is not None else ""
        st.caption(
            f"Estimación del modelo (Gradient Boosting){extra_inr}. "
            "Error medio en test: unos 1.560 € (~₹140.000), tómalo como orientación, no como precio exacto."
        )
    elif r.status_code == 422:
        st.error("Algún dato no es válido. Revisa el formulario.")
        st.json(r.json())
    else:
        st.error(f"Error del servicio ({r.status_code}): {r.text}")
