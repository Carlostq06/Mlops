"""
Módulo 1 - Paso 2: Preparación de Datos con 5 Decisiones Justificadas.
Genera datos limpios en data/processed/ y las figuras comparativas correspondientes.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")
FIGURES_DIR = os.path.join("reports", "figures")
RAW_PATH = os.path.join("data", "raw", "car_details.csv")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def prepare_data():
    print("=" * 65)
    print("EJECUTANDO LAS 5 DECISIONES DE PREPARACIÓN DE DATOS (M1.4)")
    print("=" * 65)
    df_raw = pd.read_csv(RAW_PATH)
    n_initial = len(df_raw)
    print(f"Dataset inicial: {n_initial} filas, {df_raw.shape[1]} columnas.")

    # -------------------------------------------------------------
    # DECISIÓN 1: Drop de duplicados exactos
    # -------------------------------------------------------------
    df_step1 = df_raw.drop_duplicates().copy()
    n_duplicates_dropped = n_initial - len(df_step1)
    print("\n[Decisión 1 - Drop Duplicados]")
    print(f" -> Filas eliminadas: {n_duplicates_dropped} duplicados.")
    print(f" -> Filas resultantes: {len(df_step1)}")

    # Visual Decisión 1
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Crudo (con duplicados)", "Limpio (sin duplicados)"], [n_initial, len(df_step1)], color=["#e74c3c", "#2ecc71"])
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_ylabel("Número de Registros")
    ax.set_title("Decisión 1: Eliminación de 763 Registros Duplicados")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "decision_1_drop_duplicates.png"))
    plt.close()

    # -------------------------------------------------------------
    # DECISIÓN 2: Crear variable 'brand' extrayendo la marca de 'name'
    # -------------------------------------------------------------
    df_step2 = df_step1.copy()
    df_step2["brand"] = df_step2["name"].apply(lambda x: str(x).split(" ")[0].strip().title())
    print("\n[Decisión 2 - Crear Variable 'brand']")
    print(f" -> De {df_raw['name'].nunique()} nombres únicos a {df_step2['brand'].nunique()} marcas.")
    top_brands = df_step2["brand"].value_counts().head(10)
    print(" -> Top 5 marcas más frecuentes:\n", top_brands.head(5).to_dict())

    # Visual Decisión 2
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_brands.values, y=top_brands.index, palette="viridis", hue=top_brands.index, legend=False, ax=ax)
    ax.set_title("Decisión 2: Top 10 Marcas Extraídas a partir del Nombre")
    ax.set_xlabel("Frecuencia")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "decision_2_create_brand.png"))
    plt.close()

    # -------------------------------------------------------------
    # DECISIÓN 3: Crear variable 'vehicle_age' a partir de 'year'
    # -------------------------------------------------------------
    REFERENCE_YEAR = 2020  # Último año presente en el dataset
    df_step3 = df_step2.copy()
    df_step3["vehicle_age"] = REFERENCE_YEAR - df_step3["year"]
    print("\n[Decisión 3 - Crear Variable 'vehicle_age']")
    print(f" -> vehicle_age calculada como ({REFERENCE_YEAR} - year).")
    print(f" -> Rango de antigüedad: de {df_step3['vehicle_age'].min()} a {df_step3['vehicle_age'].max()} años.")

    # Visual Decisión 3
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df_step3, x="vehicle_age", y="selling_price", alpha=0.5, color="darkblue", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Decisión 3: Depreciación de selling_price según vehicle_age")
    ax.set_ylabel("selling_price (Escala Log)")
    ax.set_xlabel("Antigüedad del Vehículo (Años)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "decision_3_vehicle_age.png"))
    plt.close()

    # -------------------------------------------------------------
    # DECISIÓN 4: Tratamiento de Outliers en 'km_driven'
    # -------------------------------------------------------------
    # Acotamos al percentil 99.5 para eliminar lecturas anómalas superiores a ~300.000 km
    km_cap = df_step3["km_driven"].quantile(0.995)
    df_step4 = df_step3[(df_step3["km_driven"] >= 500) & (df_step3["km_driven"] <= km_cap)].copy()
    n_outliers_dropped = len(df_step3) - len(df_step4)
    print("\n[Decisión 4 - Filtro de Outliers en km_driven]")
    print(f" -> Umbral percentil 99.5: {km_cap:,.0f} km.")
    print(f" -> Registros anómalos eliminados: {n_outliers_dropped}")
    print(f" -> Filas resultantes: {len(df_step4)}")

    # Visual Decisión 4
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(y=df_step3["km_driven"], ax=axes[0], color="tomato")
    axes[0].set_title("km_driven ANTES del filtrado (max: 806,599)")
    sns.boxplot(y=df_step4["km_driven"], ax=axes[1], color="mediumseagreen")
    axes[1].set_title(f"km_driven DESPUÉS del filtrado (max: {df_step4['km_driven'].max():,.0f})")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "decision_4_filter_outliers.png"))
    plt.close()

    # -------------------------------------------------------------
    # DECISIÓN 5: Recodificación de 'owner' y 'transmission'
    # -------------------------------------------------------------
    df_step5 = df_step4.copy()
    owner_map = {
        "Test Drive Car": 0,
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4
    }
    df_step5["owner_num"] = df_step5["owner"].map(owner_map).fillna(1).astype(int)
    df_step5["transmission_num"] = (df_step5["transmission"] == "Automatic").astype(int)
    print("\n[Decisión 5 - Recodificación Ordinal y Binaria]")
    print(" -> 'owner' mapeado ordenadamente a 'owner_num' (0 a 4).")
    print(" -> 'transmission' mapeado a binario 'transmission_num' (Manual: 0, Automatic: 1).")

    # Visual Decisión 5
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df_step5, x="owner_num", y="selling_price", ax=axes[0], palette="Blues", hue="owner_num", legend=False)
    axes[0].set_yscale("log")
    axes[0].set_title("selling_price vs owner_num (0=Test Drive, 4=4th+)")
    sns.boxplot(data=df_step5, x="transmission_num", y="selling_price", ax=axes[1], palette="Purples", hue="transmission_num", legend=False)
    axes[1].set_yscale("log")
    axes[1].set_title("selling_price vs transmission_num (0=Manual, 1=Auto)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "decision_5_recodification.png"))
    plt.close()

    # -------------------------------------------------------------
    # Exportación final limpia y Split Train / Val / Test (M2.3 anticipado)
    # -------------------------------------------------------------
    cleaned_path = os.path.join(PROCESSED_DIR, "car_details_cleaned.csv")
    df_step5.to_csv(cleaned_path, index=False)
    print(f"\nDataset limpio exportado a: {cleaned_path} ({len(df_step5)} filas)")

    # Split 70% train, 15% val, 15% test con seed fija
    train_val_df, test_df = train_test_split(df_step5, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=42)  # 0.1765 de 0.85 es ~0.15

    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print(f"Particiones guardadas en {PROCESSED_DIR}:")
    print(f" - Train: {len(train_df)} filas ({len(train_df)/len(df_step5)*100:.1f}%)")
    print(f" - Val:   {len(val_df)} filas ({len(val_df)/len(df_step5)*100:.1f}%)")
    print(f" - Test:  {len(test_df)} filas ({len(test_df)/len(df_step5)*100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    prepare_data()
