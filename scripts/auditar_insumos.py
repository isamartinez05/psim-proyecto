"""Auditoría de insumos del EDA antes del modelado (lab005, sección 4.1)."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = Path("data/processed/features.csv")
SPLITS = Path("data/metadata/splits.csv")
STRATEGY = "base_adap"
PREDICTORS = [
    "compacidad",
    "relacion_aspecto",
    "fraccion_area",
    "simetria_izq_der",
    "energia_wavelet_d1",
]
REFERENCES = [f"ref_{p}" for p in PREDICTORS]

features = pd.read_csv(FEATURES)
splits = pd.read_csv(SPLITS)

print("SHA-256 features.csv:", hashlib.sha256(FEATURES.read_bytes()).hexdigest())
print(
    "source_sha256 nulos:", features["source_sha256"].isna().sum(), "de", len(features)
)

print("\n=== Claves ===")
print(
    "sample_id nulos / duplicados:",
    features["sample_id"].isna().sum(),
    features["sample_id"].duplicated().sum(),
)
print(
    "image_id x config_id duplicados:",
    features.duplicated(["image_id", "config_id"]).sum(),
)
print("image_id duplicados en splits:", splits["image_id"].duplicated().sum())
print("Imágenes:", features["image_id"].nunique())
print(features["config_id"].value_counts().to_string())

print("\n=== Partición: tabla vs contrato ===")
check = features[["image_id", "split"]].merge(
    splits[["image_id", "split"]],
    on="image_id",
    how="left",
    validate="many_to_one",
    suffixes=("_tabla", "_contrato"),
)
print("Sin asignación en contrato:", check["split_contrato"].isna().sum())
print(
    "Discrepancias tabla/contrato:",
    (check["split_tabla"] != check["split_contrato"]).sum(),
)
for g in ("subject_id", "grupo_id"):
    print(
        f"{g} que cruzan particiones:", (splits.groupby(g)["split"].nunique() > 1).sum()
    )

print("\n=== Calidad ===")
print(
    pd.crosstab(features["quality_status"], features["eligible_for_model"]).to_string()
)

sub = features[
    (features["config_id"] == STRATEGY) & (features["eligible_for_model"] == 1)
].copy()
sub = sub.merge(
    splits[["image_id", "grupo_id"]], on="image_id", how="left", validate="one_to_one"
)
print(f"\n=== {STRATEGY} elegibles: {len(sub)} filas ===")
print(sub["split"].value_counts().to_string())
for c in ("image_id", "subject_id", "grupo_id"):
    print(f"{c} repetidos:", sub[c].duplicated().sum())

cols = PREDICTORS + REFERENCES + ["edad", "target"]
train = sub[sub["split"] == "train"]
print(
    pd.DataFrame(
        {
            "faltantes": sub[cols].isna().sum(),
            "infinitos": np.isinf(sub[cols].astype(float)).sum(),
            "unicos_train": train[cols].nunique(),
        }
    ).to_string()
)

# Prueba cerrada: solo se describen train y validation.
dev = sub[sub["split"].isin(["train", "validation"])]
print("\n=== target por partición y origen (desarrollo) ===")
print(
    pd.crosstab(
        [dev["split"], dev["dataset_id"]], dev["target"], margins=True
    ).to_string()
)
print("\n=== Referencias en train ===")
print(train[REFERENCES].describe().T.round(4).to_string())
print("\n=== Correlación automático vs referencia (train) ===")
for p in PREDICTORS:
    print(f"  {p}: r = {train[p].corr(train[f'ref_{p}']):.3f}")
print("\nFilas en test:", (sub["split"] == "test").sum())
