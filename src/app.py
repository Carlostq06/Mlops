"""
Modulo 4: Mini-app para el usuario final (Streamlit).

El usuario rellena un formulario y la app llama a la API de M3 (/predict).
La app NO carga el modelo: solo consume la API.
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

BRANDS = [
    "Ambassador", "Audi", "Bmw", "Chevrolet", "Daewoo", "Datsun", "Fiat",
    "Force", "Ford", "Honda", "Hyundai", "Isuzu", "Jaguar", "Jeep", "Kia",
    "Land", "Mahindra", "Maruti", "Mercedes-Benz", "Mg", "Mitsubishi",
    "Nissan", "Opelcorsa", "Renault", "Skoda", "Tata", "Toyota",
    "Volkswagen", "Volvo",
]

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

with st.form("car_form"):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Marca", BRANDS, index=BRANDS.index("Maruti"))
        year = st.slider("Año de matriculación", 1992, 2020, 2015)
        km_driven = st.number_input("Kilómetros recorridos", min_value=500, max_value=300000, value=50000, step=1000)
        fuel = st.selectbox("Combustible", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    with col2:
        transmission = st.radio("Transmisión", ["Manual", "Automatic"])
        seller_type = st.selectbox("Tipo de vendedor", ["Individual", "Dealer", "Trustmark Dealer"])
        owner = st.selectbox(
            "Propietarios anteriores",
            ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
        )
    submitted = st.form_submit_button("Calcular precio")

if submitted:
    payload = {
        "brand": brand,
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
        price = r.json()["predicted_price"]
        st.metric("Precio estimado", f"₹ {price:,.0f}")
        st.caption(
            "Estimación del modelo (Gradient Boosting). Error medio en test: "
            "unos ₹150.000, tómalo como orientación, no como precio exacto."
        )
    elif r.status_code == 422:
        st.error("Algún dato no es válido. Revisa el formulario.")
        st.json(r.json())
    else:
        st.error(f"Error del servicio ({r.status_code}): {r.text}")
