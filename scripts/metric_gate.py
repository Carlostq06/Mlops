"""
Puerta de metrica (gate) para el reentrenamiento automatico.

    python scripts/metric_gate.py <metrics_actual.json> <metrics_nuevo.json>

Reglas para aceptar (promover) el modelo nuevo:
  1. Debe ganar al baseline dummy (test_mae < dummy_val_mae).
  2. Debe mejorar al modelo actual (test_mae nuevo < test_mae actual).

Escribe "promote=true/false" en $GITHUB_OUTPUT para que el workflow decida si hace commit.
Sale con codigo 1 solo si el modelo nuevo ni siquiera gana al dummy (algo va mal).
"""
import json
import os
import sys


def main(old_path, new_path):
    with open(old_path, encoding="utf-8") as f:
        old = json.load(f)
    with open(new_path, encoding="utf-8") as f:
        new = json.load(f)

    print(f"MAE modelo actual: {old['test_mae']:,.0f}")
    print(f"MAE modelo nuevo:  {new['test_mae']:,.0f}")
    print(f"MAE dummy (val):   {new['dummy_val_mae']:,.0f}")

    beats_dummy = new["test_mae"] < new["dummy_val_mae"]
    improves = new["test_mae"] < old["test_mae"]
    promote = beats_dummy and improves

    if not beats_dummy:
        print("RECHAZADO: el modelo nuevo no gana al baseline dummy.")
    elif not improves:
        print("RECHAZADO: el modelo nuevo no mejora al actual. Se mantiene el modelo en produccion.")
    else:
        print("PROMOVIDO: el modelo nuevo mejora al actual.")

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"promote={'true' if promote else 'false'}\n")

    return 0 if beats_dummy else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
