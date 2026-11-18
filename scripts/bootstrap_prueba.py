"""Intervalo bootstrap del error en prueba del modelo congelado (lab005).

Remuestrea pacientes de la prueba con reemplazo, con el modelo y sus
predicciones fijas, y calcula el RMSE del modelo, el de la media de
entrenamiento y la reduccion relativa. El intervalo es condicional al modelo
ajustado: describe la incertidumbre por la muestra de prueba, no la del ajuste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from psim import RAIZ
from psim.modeling import leer_contrato, preparar

OUT = RAIZ / "results" / "modelado_proyecto"
ORIGEN = {"NLM-ChinaCXRSet": "Shenzhen", "NLM-MontgomeryCXRSet": "Montgomery"}
REPLICAS = 2000
SEMILLA = 2026
NIVEL = 0.95


def rmse(errores: np.ndarray) -> np.ndarray:
    """RMSE sobre el ultimo eje."""
    return np.sqrt(np.mean(errores**2, axis=-1))


def intervalo(valores: np.ndarray) -> tuple[float, float]:
    """Intervalo percentil al nivel fijado."""
    cola = (1 - NIVEL) / 2
    inferior, superior = np.quantile(valores, [cola, 1 - cola])
    return float(inferior), float(superior)


def main() -> None:
    """Bootstrap por pacientes, global y por origen."""
    contrato = leer_contrato()
    objetivo = contrato["objetivo"]
    splits = pd.read_csv(RAIZ / "data" / "metadata" / "splits.csv")
    tabla_train = pd.read_csv(RAIZ / contrato["tablas"]["train"])
    train, _ = preparar(tabla_train, splits, contrato, "train")
    media_train = float(train[objetivo].mean())
    pred = pd.read_csv(OUT / "test_predictions.csv")

    rng = np.random.default_rng(SEMILLA)
    grupos = [("Todos", pred)] + [
        (ORIGEN[o], g) for o, g in pred.groupby("dataset_id", sort=True)
    ]
    filas = []
    for nombre, g in grupos:
        e_modelo = g["residuo"].to_numpy()
        e_media = (g[objetivo] - media_train).to_numpy()
        indices = rng.integers(0, len(g), size=(REPLICAS, len(g)))
        boot_modelo = rmse(e_modelo[indices])
        boot_media = rmse(e_media[indices])
        medidas = {
            "rmse_modelo": (rmse(e_modelo), boot_modelo),
            "rmse_media_train": (rmse(e_media), boot_media),
            "reduccion_relativa": (
                1 - rmse(e_modelo) / rmse(e_media),
                1 - boot_modelo / boot_media,
            ),
        }
        for medida, (puntual, replicas) in medidas.items():
            inferior, superior = intervalo(replicas)
            filas.append(
                {
                    "grupo": nombre,
                    "n_pacientes": len(g),
                    "medida": medida,
                    "estimacion": float(puntual),
                    "ic_inf": inferior,
                    "ic_sup": superior,
                }
            )
    r = pd.DataFrame(filas).round(4)
    r.to_csv(OUT / "test_bootstrap.csv", index=False)
    print(f"Bootstrap por pacientes: {REPLICAS} replicas, IC {int(NIVEL * 100)} %")
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
