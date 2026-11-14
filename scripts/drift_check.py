"""
Deteccion de drift simple (sin librerias extra, solo scipy).

    python scripts/drift_check.py                         # train.csv (referencia) vs test.csv
    python scripts/drift_check.py ref.csv actual.csv      # p. ej. datos de entrenamiento vs un lote nuevo
    python scripts/drift_check.py --simulate              # simula drift para ver que se detecta

Para cada variable de entrada del modelo:
  - Numericas (year, km_driven): test de Kolmogorov-Smirnov.
  - Categoricas (fuel, seller_type, ...): test chi-cuadrado sobre las frecuencias.
Si p-valor < 0.05 se marca drift en esa variable.
Guarda el informe en reports/drift_report.json y sale con codigo 1 si hay drift.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

NUM_COLS = ["year", "km_driven"]
CAT_COLS = ["fuel", "seller_type", "transmission", "owner"]
ALPHA = 0.05
REPORT_PATH = os.path.join("reports", "drift_report.json")


def simulate_drift(df):
    """Coches con mas km y mas diesel, para comprobar que el detector salta."""
    df = df.copy()
    rng = np.random.default_rng(42)
    df["km_driven"] = (df["km_driven"] * 1.5).astype(int)
    mask = rng.random(len(df)) < 0.3
    df.loc[mask, "fuel"] = "Diesel"
    return df


def check_drift(ref, cur):
    report = {}
    for col in NUM_COLS:
        p = ks_2samp(ref[col], cur[col]).pvalue
        report[col] = {"test": "ks", "p_value": round(float(p), 4), "drift": bool(p < ALPHA)}
    for col in CAT_COLS:
        table = pd.concat([ref[col].value_counts(), cur[col].value_counts()], axis=1).fillna(0)
        p = chi2_contingency(table.values.T)[1]
        report[col] = {"test": "chi2", "p_value": round(float(p), 4), "drift": bool(p < ALPHA)}
    return report


def main(args):
    simulate = "--simulate" in args
    files = [a for a in args if not a.startswith("--")]
    ref_path = files[0] if len(files) > 0 else os.path.join("data", "processed", "train.csv")
    cur_path = files[1] if len(files) > 1 else os.path.join("data", "processed", "test.csv")

    ref = pd.read_csv(ref_path)
    cur = pd.read_csv(cur_path)
    if simulate:
        cur = simulate_drift(cur)

    print(f"Referencia: {ref_path} ({len(ref)} filas)")
    print(f"Actual:     {cur_path} ({len(cur)} filas){' [DRIFT SIMULADO]' if simulate else ''}")
    report = check_drift(ref, cur)
    for col, r in report.items():
        print(f"  {col:13s} {r['test']:5s} p={r['p_value']:.4f}  {'DRIFT' if r['drift'] else 'ok'}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({"reference": ref_path, "current": cur_path, "simulated": simulate, "columns": report}, f, indent=2)
    print(f"Informe guardado en {REPORT_PATH}")

    any_drift = any(r["drift"] for r in report.values())
    print("RESULTADO:", "DRIFT DETECTADO" if any_drift else "sin drift")
    return 1 if any_drift else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
