"""
Módulo 1: Auditoría inicial y Análisis Exploratorio de Datos (EDA).
Genera métricas impresas de auditoría y figuras visuales completas.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo visual
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
RAW_DATA_PATH = os.path.join("data", "raw", "car_details.csv")

def run_audit(df: pd.DataFrame):
    print("=" * 60)
    print("AUDITORÍA INICIAL DEL DATASET (REQUISITO M1.2)")
    print("=" * 60)
    print(f"1. Dimensiones:")
    print(f"   - Filas:    {df.shape[0]}")
    print(f"   - Columnas: {df.shape[1]}")
    print(f"\n2. Tipos de datos por columna:")
    for col, dtype in df.dtypes.items():
        print(f"   - {col:15s}: {dtype}")
    
    print(f"\n3. Valores Nulos:")
    null_counts = df.isnull().sum()
    for col, nulls in null_counts.items():
        print(f"   - {col:15s}: {nulls} ({(nulls / len(df)) * 100:.2f}%)")
    print(f"   Total nulos en el dataset: {null_counts.sum()}")
    
    dup_count = df.duplicated().sum()
    print(f"\n4. Registros Duplicados:")
    print(f"   - Cantidad: {dup_count} ({(dup_count / len(df)) * 100:.2f}%)")
    
    print(f"\n5. Estadísticas de la Variable Target ('selling_price'):")
    target = df["selling_price"]
    print(f"   - Mínimo:       {target.min():,.0f}")
    print(f"   - Percentil 25: {target.quantile(0.25):,.0f}")
    print(f"   - Mediana:      {target.median():,.0f}")
    print(f"   - Media:        {target.mean():,.0f}")
    print(f"   - Percentil 75: {target.quantile(0.75):,.0f}")
    print(f"   - Máximo:       {target.max():,.0f}")
    print(f"   - Desv. Típica: {target.std():,.0f}")
    print("=" * 60)

def generate_eda_plots(df: pd.DataFrame):
    print("\nGenerando visualizaciones EDA completas...")

    # 1. Univariante numérica: Distribución del Target (selling_price) y km_driven
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["selling_price"], kde=True, ax=axes[0], color="royalblue", bins=40)
    axes[0].set_title("Distribución de selling_price (Target)")
    axes[0].set_xlabel("Precio de Venta")
    axes[0].set_ylabel("Frecuencia")

    sns.histplot(df["km_driven"], kde=True, ax=axes[1], color="teal", bins=40)
    axes[1].set_title("Distribución de km_driven")
    axes[1].set_xlabel("Kilómetros Recorridos")
    axes[1].set_ylabel("Frecuencia")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "01_eda_univariate_numeric.png"))
    plt.close()

    # 2. Univariante categórica: Frecuencia de fuel, seller_type, transmission, owner
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.countplot(data=df, x="fuel", ax=axes[0, 0], palette="Blues_r", hue="fuel", legend=False)
    axes[0, 0].set_title("Distribución por Tipo de Combustible (fuel)")
    
    sns.countplot(data=df, x="seller_type", ax=axes[0, 1], palette="Greens_r", hue="seller_type", legend=False)
    axes[0, 1].set_title("Distribución por Tipo de Vendedor (seller_type)")
    
    sns.countplot(data=df, x="transmission", ax=axes[1, 0], palette="Oranges_r", hue="transmission", legend=False)
    axes[1, 0].set_title("Distribución por Transmisión")
    
    sns.countplot(data=df, x="owner", ax=axes[1, 1], palette="Purples_r", hue="owner", legend=False)
    axes[1, 1].set_title("Distribución por Propietario (owner)")
    axes[1, 1].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "02_eda_univariate_categorical.png"))
    plt.close()

    # 3. Bivariante contra el Target: selling_price vs transmission, fuel, year
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(data=df, x="transmission", y="selling_price", ax=axes[0], palette="Set2", hue="transmission", legend=False)
    axes[0].set_title("Precio vs Transmisión")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("selling_price (Escala Log)")

    sns.boxplot(data=df, x="fuel", y="selling_price", ax=axes[1], palette="Set2", hue="fuel", legend=False)
    axes[1].set_title("Precio vs Combustible")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("selling_price (Escala Log)")

    sns.lineplot(data=df, x="year", y="selling_price", ax=axes[2], color="crimson")
    axes[2].set_title("Evolución de Precio medio vs Año")
    axes[2].set_ylabel("Precio medio de venta")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "03_eda_bivariate_target.png"))
    plt.close()

    # 4. Outliers: Boxplots de variables numéricas clave
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(y=df["selling_price"], ax=axes[0], color="lightcoral")
    axes[0].set_title("Detección de Outliers en selling_price")
    axes[0].set_ylabel("Precio")

    sns.boxplot(y=df["km_driven"], ax=axes[1], color="lightblue")
    axes[1].set_title("Detección de Outliers en km_driven")
    axes[1].set_ylabel("Kilómetros")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "04_eda_outliers.png"))
    plt.close()

    # 5. Correlaciones: Mapa de calor entre variables numéricas
    num_df = df[["year", "km_driven", "selling_price"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = num_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Matriz de Correlación Numérica")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "05_eda_correlations.png"))
    plt.close()

    print(f"Todas las visualizaciones EDA guardadas en: {FIGURES_DIR}")

if __name__ == "__main__":
    df_raw = pd.read_csv(RAW_DATA_PATH)
    run_audit(df_raw)
    generate_eda_plots(df_raw)
