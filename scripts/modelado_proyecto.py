"""Modelado del proyecto (lab005): ajuste en train y selección en validación.

No lee la partición de prueba. Ajusta regresión múltiple de grado 1 y 2, la
compara con referencias constantes, produce diagnósticos y congela el
protocolo en results/modelado_proyecto/frozen_protocol.json.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor

from psim.modeling import (
    coeficientes,
    leer_contrato,
    metricas_regresion,
    pipeline_continuo,
    preparar,
    rango_diseno,
    verificar_grupos,
)

OUT = Path("results/modelado_proyecto")
FIG = Path("figures/modelado_proyecto")
CANDIDATOS = {"grado_1": 1, "grado_2": 2}
UNIDAD = "fracción, adim."


def sha256(ruta: Path) -> str:
    """Hash SHA-256 de un archivo."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def comparar(
    modelos: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    objetivo: str,
    predictores: list[str],
) -> pd.DataFrame:
    """Ajusta en train y calcula métricas en validación, global y por origen."""
    filas: list[dict[str, Any]] = []
    for nombre, modelo in modelos.items():
        modelo.fit(train[predictores], train[objetivo])
        pred = pd.Series(modelo.predict(val[predictores]), index=val.index)
        filas.append(
            {
                "modelo": nombre,
                "subgrupo": "todos",
                "n": len(val),
                **metricas_regresion(val[objetivo], pred),
            }
        )
        for origen, g in val.groupby("dataset_id"):
            filas.append(
                {
                    "modelo": nombre,
                    "subgrupo": origen,
                    "n": len(g),
                    **metricas_regresion(g[objetivo], pred.loc[g.index]),
                }
            )
    return pd.DataFrame(filas)


def graficar(
    val: pd.DataFrame, objetivo: str, predictores: list[str], titulo: str
) -> None:
    """Referencia-predicción, residuos-predicción y residuos frente a predictores."""
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
    for origen, g in val.groupby("dataset_id"):
        ejes[0].scatter(g[objetivo], g["prediccion"], alpha=0.7, label=origen)
        ejes[1].scatter(g["prediccion"], g["residuo"], alpha=0.7, label=origen)
    limites = [
        min(val[objetivo].min(), val["prediccion"].min()),
        max(val[objetivo].max(), val["prediccion"].max()),
    ]
    ejes[0].plot(limites, limites, "k--", linewidth=1)
    ejes[0].set(xlabel=f"Referencia ({UNIDAD})", ylabel=f"Predicción ({UNIDAD})")
    ejes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    ejes[1].set(xlabel=f"Predicción ({UNIDAD})", ylabel=f"Residuo ({UNIDAD})")
    ejes[0].legend(fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(FIG / "validacion_prediccion_residuos.png", dpi=160)
    plt.close(fig)

    fig, ejes = plt.subplots(1, len(predictores), figsize=(4 * len(predictores), 3.8))
    for eje, p in zip(np.atleast_1d(ejes), predictores, strict=True):
        for origen, g in val.groupby("dataset_id"):
            eje.scatter(g[p], g["residuo"], alpha=0.7, label=origen)
        eje.axhline(0, color="black", linestyle="--", linewidth=1)
        eje.set(xlabel=p, ylabel=f"Residuo ({UNIDAD})")
    np.atleast_1d(ejes)[0].legend(fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(FIG / "validacion_residuos_predictores.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Ejecuta la fase de desarrollo sin tocar la partición de prueba."""
    plt.switch_backend("Agg")
    contrato = leer_contrato()
    objetivo, predictores = contrato["objetivo"], contrato["predictores"]
    ruta_train = Path(contrato["tablas"]["train"])
    ruta_val = Path(contrato["tablas"]["validation"])
    if sha256(ruta_train) != contrato["sha256_train"]:
        raise ValueError("features.csv no coincide con el hash del contrato.")

    splits = pd.read_csv("data/metadata/splits.csv")
    verificar_grupos(splits)
    train, excl_tr = preparar(pd.read_csv(ruta_train), splits, contrato, "train")
    val, excl_va = preparar(pd.read_csv(ruta_val), splits, contrato, "validation")
    if set(train["grupo_id"]) & set(val["grupo_id"]):
        raise ValueError("Un grupo aparece en train y en validación.")
    if train[predictores].isna().all().any():
        raise ValueError("Un predictor no tiene datos en entrenamiento.")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pd.concat([excl_tr, excl_va]).to_csv(
        OUT / "excluded_missing_target.csv", index=False
    )
    print(f"train: {len(train)} grupos | validación: {len(val)} grupos")

    modelos: dict[str, Any] = {n: pipeline_continuo(g) for n, g in CANDIDATOS.items()}
    modelos["referencia_media"] = DummyRegressor(strategy="mean")
    modelos["referencia_mediana"] = DummyRegressor(strategy="median")
    comparacion = comparar(modelos, train, val, objetivo, predictores)
    comparacion.to_csv(OUT / "validation_metrics.csv", index=False)
    print(comparacion.round(4).to_string(index=False))

    rmse = comparacion[comparacion["subgrupo"] == "todos"].set_index("modelo")["RMSE"]
    elegido = min(CANDIDATOS, key=lambda m: rmse[m])
    mejor = modelos[elegido]
    pred = mejor.predict(val[predictores])
    val = val.assign(prediccion=pred, residuo=val[objetivo] - pred)
    val.to_csv(OUT / "validation_predictions.csv", index=False)
    graficar(val, objetivo, predictores, f"Validación, n={len(val)} grupos, {elegido}")
    for nombre in CANDIDATOS:
        coeficientes(modelos[nombre], predictores).to_csv(
            OUT / f"coefficients_{nombre}.csv", index=False
        )

    diseno = {n: rango_diseno(modelos[n], train[predictores]) for n in CANDIDATOS}
    protocolo = {
        "contrato": contrato,
        "modelo_elegido": elegido,
        "grado": CANDIDATOS[elegido],
        "criterio": "menor RMSE de validación entre grado_1 y grado_2",
        "rmse_validacion": {m: float(rmse[m]) for m in modelos},
        "coeficientes": coeficientes(mejor, predictores).to_dict(orient="records"),
        "diseno_train": diseno,
        "n_train_grupos": len(train),
        "n_validacion_grupos": len(val),
        "sha256_validation": sha256(ruta_val),
        "poblacion_ajuste": "train",
        "poblacion_seleccion": "validation",
        "prueba_extraida": Path(contrato["tablas"]["test"]).exists(),
        "versiones": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    (OUT / "frozen_protocol.json").write_text(
        json.dumps(protocolo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nModelo elegido: {elegido} (RMSE validación {rmse[elegido]:.4f})")
    for nombre, d in diseno.items():
        print(
            f"  {nombre}: rango {d['rango']}/{d['columnas']}, "
            f"condición {d['condicion']:.1f}"
        )


if __name__ == "__main__":
    main()
