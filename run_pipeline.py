"""
Modulo 5: procesado completo en un solo comando.

    python run_pipeline.py

Ejecuta en orden: descarga -> auditoria/EDA -> preparacion (5 decisiones + splits) -> entrenamiento.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")  # sin ventanas, para que funcione tambien en CI/servidores

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import download_data  # noqa: E402
import eda_audit  # noqa: E402
import data_prep  # noqa: E402
import train  # noqa: E402

if __name__ == "__main__":
    print(">>> 1/4 Descargando dataset")
    download_data.download_dataset()

    print("\n>>> 2/4 Auditoria y EDA")
    import pandas as pd
    df = pd.read_csv(eda_audit.RAW_DATA_PATH)
    eda_audit.run_audit(df)
    eda_audit.generate_eda_plots(df)

    print("\n>>> 3/4 Preparacion de datos")
    data_prep.prepare_data()

    print("\n>>> 4/4 Entrenamiento y evaluacion")
    train.train_and_evaluate()

    print("\nPipeline completado. Modelo en models/pipeline.joblib")
