"""Selección de predictores con el EDA de entrenamiento (lab005, sección 4.3).

Solo usa la partición de entrenamiento. Mide la asociación de cada descriptor
automático de base_adap con el objetivo y la redundancia entre descriptores.
"""

from pathlib import Path

import pandas as pd

TARGET = "ref_fraccion_area"
CANDIDATES = [
    "compacidad",
    "relacion_aspecto",
    "fraccion_area",
    "simetria_izq_der",
    "energia_wavelet_d1",
]
OUT = Path("results/modelado_proyecto")
OUT.mkdir(parents=True, exist_ok=True)

t = pd.read_csv("data/processed/features.csv")
train = t[
    (t["config_id"] == "base_adap")
    & (t["eligible_for_model"] == 1)
    & (t["split"] == "train")
]
print(f"Entrenamiento: {len(train)} imágenes")

asociacion = pd.DataFrame(
    {
        "pearson": train[CANDIDATES].corrwith(train[TARGET]),
        "spearman": train[CANDIDATES].corrwith(train[TARGET], method="spearman"),
    }
).round(3)
for origen, g in train.groupby("dataset_id"):
    asociacion[f"spearman_{origen}"] = (
        g[CANDIDATES].corrwith(g[TARGET], method="spearman").round(3)
    )
asociacion = asociacion.sort_values("spearman", key=abs, ascending=False)
asociacion.to_csv(OUT / "asociacion_con_objetivo_train.csv")
print(f"\n=== Asociación con {TARGET} (train) ===")
print(asociacion.to_string())

redundancia = train[CANDIDATES].corr(method="spearman").round(3)
redundancia.to_csv(OUT / "redundancia_predictores_train.csv")
print("\n=== Redundancia entre descriptores (Spearman, train) ===")
print(redundancia.to_string())

print(f"\n=== {TARGET} por origen (train) ===")
print(train.groupby("dataset_id")[TARGET].describe().round(4).to_string())
