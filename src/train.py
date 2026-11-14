"""
Módulo 2: Entrenamiento, Validación y Exportación del Pipeline del Modelo.
Cumple con todos los requisitos mínimos de M2:
- Problema -> Tarea -> Métrica explicada
- Baseline Dummy explícito superado
- Split train/val/test sin fugas
- Visualizaciones de validación (residuales, comparativa, reales vs predichos)
- Reproducibilidad con seeds fijas y exportación de artefacto único en models/pipeline.joblib
"""
import os
import json
from datetime import datetime, timezone
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Configuración visual
sns.set_theme(style="whitegrid")
FIGURES_DIR = os.path.join("reports", "figures")
MODELS_DIR = "models"
PROCESSED_DIR = os.path.join("data", "processed")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42


def load_data():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    features = ["year", "km_driven", "fuel", "seller_type", "transmission", "owner", "brand"]
    target = "selling_price"

    return (
        train_df[features], train_df[target],
        val_df[features], val_df[target],
        test_df[features], test_df[target]
    )


def build_preprocessor():
    cat_cols = ["brand", "fuel", "seller_type", "transmission", "owner"]
    num_cols = ["year", "km_driven"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )


def train_and_evaluate():
    print("=" * 70)
    print("MÓDULO 2: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS (REQUISITOS M2)")
    print("=" * 70)
    print("1. Justificación Métrica:")
    print("   - Tarea: Regresión supervisada de precio de venta (selling_price).")
    print("   - MAE: Métrica principal, error en dinero real directamente interpretable.")
    print("   - RMSE: Penaliza errores grandes en vehículos caros.")
    print("   - R²: Coeficiente de determinación (% varianza explicada).")
    print("-" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"Dimensiones particiones: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    preprocessor = build_preprocessor()

    # Definir baseline dummy y modelos candidatos
    models = {
        "Dummy (Media)": DummyRegressor(strategy="mean"),
        "Ridge Regressor": Pipeline([
            ("prep", preprocessor),
            ("model", Ridge(alpha=1.0, random_state=SEED))
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", preprocessor),
            ("model", GradientBoostingRegressor(n_estimators=100, random_state=SEED))
        ])
    }

    results = []

    print("\n2. Evaluación en Conjunto de Validación (Decisiones sin mirar Test):")
    print(f"{'Modelo':22s} | {'Val MAE':>12s} | {'Val RMSE':>12s} | {'Val R²':>10s}")
    print("-" * 65)

    for name, model in models.items():
        if name == "Dummy (Media)":
            # Dummy fit directo
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_preds)
        rmse = root_mean_squared_error(y_val, val_preds)
        r2 = r2_score(y_val, val_preds)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "model_obj": model,
            "val_preds": val_preds
        })
        print(f"{name:22s} | {mae:12,.0f} | {rmse:12,.0f} | {r2:10.4f}")

    # Gráfico 1: Comparativa de Modelos
    names = [r["name"] for r in results]
    maes = [r["mae"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, maes, color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"])
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_ylabel("MAE (Menor es mejor)")
    ax.set_title("Comparativa de Error MAE en Validación: Dummy vs Modelos")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"))
    plt.close()

    # Seleccionar modelo campeón (menor MAE en validación)
    candidate_results = [r for r in results if r["name"] != "Dummy (Media)"]
    best_candidate = min(candidate_results, key=lambda x: x["mae"])
    champion_pipeline = best_candidate["model_obj"]
    print("\n" + "=" * 70)
    print(f"MODELO CAMPEÓN SELECCIONADO: {best_candidate['name']}")
    dummy_mae = results[0]["mae"]
    mejora = (1 - best_candidate["mae"] / dummy_mae) * 100
    print(f" -> Supera al Dummy explícitamente: MAE bajó de {dummy_mae:,.0f} a {best_candidate['mae']:,.0f} (-{mejora:.1f}%)")
    print("=" * 70)

    # 3. Evaluación final en Test Set (Sellado hasta este momento)
    test_preds = champion_pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)

    print(f"\n3. Resultados Finales en TEST SET ({best_candidate['name']}):")
    print(f"   - Test MAE:  {test_mae:,.0f} rupias")
    print(f"   - Test RMSE: {test_rmse:,.0f} rupias")
    print(f"   - Test R²:   {test_r2:.4f}")

    # 4. Generación de Gráficos de Resultados de Validación
    residuals = y_test - test_preds

    # Gráfico 2: Residuales vs Predichos
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(test_preds, residuals, alpha=0.5, color="royalblue")
    ax.axhline(0, color="crimson", linestyle="--")
    ax.set_xlabel("Valores Predichos")
    ax.set_ylabel("Residuales (Real - Predicho)")
    ax.set_title(f"Gráfico de Residuales - Test Set ({best_candidate['name']})")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "residuals_plot.png"))
    plt.close()

    # Gráfico 3: Predicciones vs Valores Reales
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, test_preds, alpha=0.5, color="darkgreen")
    diag_line = np.linspace(min(y_test.min(), test_preds.min()), max(y_test.max(), test_preds.max()), 100)
    ax.plot(diag_line, diag_line, color="red", linestyle="--", label="Predicción Perfecta (1:1)")
    ax.set_xlabel("Valor Real (selling_price)")
    ax.set_ylabel("Valor Predicho")
    ax.set_title(f"Valores Reales vs Predicciones ({best_candidate['name']})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "predictions_vs_actual.png"))
    plt.close()

    # Gráfico 4: Distribución de Residuales
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, ax=ax, color="purple", bins=35)
    ax.set_xlabel("Error residual (Real - Predicho)")
    ax.set_title("Distribución de Errores Residuales (Centrada en 0)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "residuals_distribution.png"))
    plt.close()

    # 5. Exportar Pipeline completo en un único artefacto (M3.2 requisito)
    model_artifact_path = os.path.join(MODELS_DIR, "pipeline.joblib")
    joblib.dump(champion_pipeline, model_artifact_path)
    print("\n5. Artefacto único exportado con éxito:")
    print(f"   -> {model_artifact_path} ({os.path.getsize(model_artifact_path)/1024:.1f} KB)")

    # 6. Guardar metricas junto al modelo (las usa /version y la puerta del reentrenamiento)
    metrics = {
        "model_name": best_candidate["name"],
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_mae": round(test_mae, 2),
        "test_rmse": round(test_rmse, 2),
        "test_r2": round(test_r2, 4),
        "dummy_val_mae": round(results[0]["mae"], 2),
        "n_train": len(X_train),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"   -> {os.path.join(MODELS_DIR, 'metrics.json')}")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
