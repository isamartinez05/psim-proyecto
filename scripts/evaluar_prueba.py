"""Evaluación única en prueba del modelo congelado del proyecto (lab005).

Reconstruye el ajuste desde train con el protocolo congelado, verifica que
coincide con los coeficientes registrados y evalúa la partición de prueba.
Si la evaluación ya existe, solo comprueba que la reproducción coincide.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from psim.modeling import (
    coeficientes,
    leer_contrato,
    metricas_regresion,
    pipeline_continuo,
    preparar,
    verificar_grupos,
)

OUT = Path("results/modelado_proyecto")
FIG = Path("figures/modelado_proyecto")
UNIDAD = "fracción, adim."
METRICAS = ["MAE", "RMSE", "R2"]


def sha256(ruta: Path) -> str:
    """Hash SHA-256 de un archivo."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def evaluar(
    modelos: dict[str, Any], test: pd.DataFrame, objetivo: str, predictores: list[str]
) -> pd.DataFrame:
    """Métricas en prueba, global y por origen, para modelos ya ajustados."""
    filas: list[dict[str, Any]] = []
    for nombre, modelo in modelos.items():
        pred = pd.Series(modelo.predict(test[predictores]), index=test.index)
        filas.append(
            {
                "modelo": nombre,
                "subgrupo": "todos",
                "n": len(test),
                **metricas_regresion(test[objetivo], pred),
            }
        )
        for origen, g in test.groupby("dataset_id"):
            filas.append(
                {
                    "modelo": nombre,
                    "subgrupo": origen,
                    "n": len(g),
                    **metricas_regresion(g[objetivo], pred.loc[g.index]),
                }
            )
    return pd.DataFrame(filas)


def graficar(test: pd.DataFrame, objetivo: str, titulo: str) -> None:
    """Referencia-predicción y residuos-predicción en prueba."""
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
    for origen, g in test.groupby("dataset_id"):
        ejes[0].scatter(g[objetivo], g["prediccion"], alpha=0.7, label=origen)
        ejes[1].scatter(g["prediccion"], g["residuo"], alpha=0.7, label=origen)
    limites = [
        min(test[objetivo].min(), test["prediccion"].min()),
        max(test[objetivo].max(), test["prediccion"].max()),
    ]
    ejes[0].plot(limites, limites, "k--", linewidth=1)
    ejes[0].set(xlabel=f"Referencia ({UNIDAD})", ylabel=f"Predicción ({UNIDAD})")
    ejes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    ejes[1].set(xlabel=f"Predicción ({UNIDAD})", ylabel=f"Residuo ({UNIDAD})")
    ejes[0].legend(fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(FIG / "prueba_prediccion_residuos.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Evalúa una sola vez el modelo congelado en la partición de prueba."""
    plt.switch_backend("Agg")
    protocolo = json.loads((OUT / "frozen_protocol.json").read_text(encoding="utf-8"))
    contrato = leer_contrato()
    if contrato != protocolo["contrato"]:
        raise ValueError("El contrato cambió después de congelar el protocolo.")
    objetivo, predictores = contrato["objetivo"], contrato["predictores"]
    rutas = {k: Path(v) for k, v in contrato["tablas"].items()}
    if sha256(rutas["train"]) != contrato["sha256_train"]:
        raise ValueError("features.csv cambió después de congelar el protocolo.")
    if sha256(rutas["validation"]) != protocolo["sha256_validation"]:
        raise ValueError("features_validation.csv cambió después de congelar.")

    splits = pd.read_csv("data/metadata/splits.csv")
    verificar_grupos(splits)
    train, _ = preparar(pd.read_csv(rutas["train"]), splits, contrato, "train")
    test, excluidos = preparar(pd.read_csv(rutas["test"]), splits, contrato, "test")
    if set(train["grupo_id"]) & set(test["grupo_id"]):
        raise ValueError("Un grupo aparece en train y en prueba.")

    elegido = protocolo["modelo_elegido"]
    x_tr, y_tr = train[predictores], train[objetivo]
    modelo = pipeline_continuo(protocolo["grado"]).fit(x_tr, y_tr)
    reconstruido = coeficientes(modelo, predictores)["coeficiente"].to_numpy()
    congelado = np.array([c["coeficiente"] for c in protocolo["coeficientes"]])
    if not np.allclose(reconstruido, congelado, rtol=1e-6, atol=1e-8):
        raise ValueError("El ajuste reconstruido no coincide con el congelado.")

    modelos: dict[str, Any] = {
        elegido: modelo,
        "referencia_media": DummyRegressor(strategy="mean").fit(x_tr, y_tr),
        "referencia_mediana": DummyRegressor(strategy="median").fit(x_tr, y_tr),
    }
    resultado = evaluar(modelos, test, objetivo, predictores)
    print(f"Prueba: {len(test)} grupos | modelo congelado: {elegido}")
    print(resultado.round(4).to_string(index=False))

    destino = OUT / "test_metrics.csv"
    if destino.exists():
        previo = pd.read_csv(destino)
        iguales = np.allclose(
            previo[METRICAS].to_numpy(),
            resultado[METRICAS].to_numpy(),
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True,
        )
        if not iguales:
            raise ValueError("La reproducción no coincide con la evaluación previa.")
        print("\nPrueba ya evaluada: la reproducción coincide.")
        return

    pred = modelo.predict(test[predictores])
    test = test.assign(prediccion=pred, residuo=test[objetivo] - pred)
    resultado.to_csv(destino, index=False)
    test.to_csv(OUT / "test_predictions.csv", index=False)
    excluidos.to_csv(OUT / "test_excluded_missing_target.csv", index=False)
    graficar(test, objetivo, f"Prueba, n={len(test)} grupos, {elegido}")
    evaluacion = {
        "modelo": elegido,
        "n_test_grupos": len(test),
        "sha256_test": sha256(rutas["test"]),
        "coeficientes_verificados": True,
    }
    (OUT / "test_evaluation.json").write_text(
        json.dumps(evaluacion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nResiduo por origen:")
    print(
        test.groupby("dataset_id")["residuo"]
        .agg(["count", "mean", "std"])
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
