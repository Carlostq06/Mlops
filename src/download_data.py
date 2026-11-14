"""
Script para descargar el dataset CarDekho desde el repositorio original de Kaggle/GitHub.
"""
import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/chandanverma07/DataSets/master/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv"
OUTPUT_PATH = os.path.join("data", "raw", "car_details.csv")


def download_dataset():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"Descargando dataset desde {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, OUTPUT_PATH)
    file_size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"Dataset descargado exitosamente en: {OUTPUT_PATH} ({file_size_kb:.2f} KB)")


if __name__ == "__main__":
    download_dataset()
