"""Modelado supervisado del lab005: regresión múltiple y extensión polinómica.

Las transformaciones (imputación, escalamiento y expansión) se ajustan dentro
de cada pipeline con los datos de entrenamiento que recibe, nunca con
validación ni prueba.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

CONTRATO = Path("config/modelado_lab005.json")
PREFIJOS_PROHIBIDOS = ("ref_", "err_")
COLUMNAS_PROHIBIDAS = frozenset(
    {
        "dice",
        "jaccard",
        "target",
        "sample_id",
        "image_id",
        "subject_id",
        "grupo_id",
        "split",
        "dataset_id",
        "eligible_for_model",
    }
)


def leer_contrato(ruta: Path = CONTRATO) -> dict[str, Any]:
    """Devuelve el contrato de modelado del lab005."""
    return json.loads(ruta.read_text(encoding="utf-8"))


def validar_predictores(predictores: list[str], objetivo: str) -> None:
    """Falla si el objetivo, un identificador o una columna de referencia es predictor."""
    prohibidos = [
        p
        for p in predictores
        if p == objetivo
        or p in COLUMNAS_PROHIBIDAS
        or p.startswith(PREFIJOS_PROHIBIDOS)
    ]
    if prohibidos:
        raise ValueError(f"Predictores no permitidos: {prohibidos}")


def verificar_grupos(splits: pd.DataFrame, grupo: str = "grupo_id") -> None:
    """Falla si un grupo falta o aparece en más de una partición."""
    if splits[grupo].isna().any():
        raise ValueError("Hay imágenes sin grupo.")
    n_particiones = splits.groupby(grupo)["split"].nunique()
    cruzan = list(n_particiones[n_particiones > 1].index)
    if cruzan:
        raise ValueError(f"Grupos que cruzan particiones: {cruzan}")


def preparar(
    tabla: pd.DataFrame,
    splits: pd.DataFrame,
    contrato: dict[str, Any],
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filtra estrategia y elegibilidad y deja una fila por grupo independiente.

    Devuelve los datos agregados y los grupos excluidos por objetivo faltante.
    El objetivo nunca se imputa.
    """
    objetivo = contrato["objetivo"]
    predictores = contrato["predictores"]
    validar_predictores(predictores, objetivo)
    t = tabla[
        (tabla["config_id"] == contrato["estrategia"])
        & (tabla["eligible_for_model"] == 1)
        & (tabla["split"] == split)
    ]
    if t["image_id"].duplicated().any():
        raise ValueError("Imágenes repetidas para la estrategia.")
    t = t.merge(
        splits[["image_id", "grupo_id", "split"]],
        on="image_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_contrato"),
    )
    if (t["split"] != t["split_contrato"]).any():
        raise ValueError("La tabla y splits.csv no coinciden en la partición.")
    columnas = [*predictores, objetivo]
    valores = t[columnas].apply(pd.to_numeric, errors="raise")
    if np.isinf(valores.to_numpy(dtype=float)).any():
        raise ValueError("Valores infinitos: investigar la extracción.")
    t[columnas] = valores
    agregado = (
        t.groupby("grupo_id")
        .agg(
            dataset_id=("dataset_id", "first"),
            n_imagenes=("image_id", "size"),
            **{c: (c, "mean") for c in columnas},
        )
        .reset_index()
    )
    agregado["split"] = split
    faltante = agregado[objetivo].isna()
    excluidos = agregado.loc[faltante, ["grupo_id", "split"]].assign(
        razon="objetivo faltante"
    )
    return agregado.loc[~faltante].reset_index(drop=True), excluidos


def pipeline_continuo(grado: int) -> Pipeline:
    """Imputación por mediana, escalamiento y expansión polinómica del grado dado."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=grado, include_bias=False)),
            ("model", LinearRegression()),
        ]
    )


def metricas_regresion(y: pd.Series, prediccion: pd.Series) -> dict[str, float]:
    """MAE, RMSE y R2; R2 queda indefinido si la referencia es constante."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(prediccion, dtype=float)
    r2 = float(r2_score(y_arr, p_arr)) if np.var(y_arr) > 0 else float("nan")
    return {
        "MAE": float(mean_absolute_error(y_arr, p_arr)),
        "RMSE": float(np.sqrt(mean_squared_error(y_arr, p_arr))),
        "R2": r2,
    }


def coeficientes(modelo: Pipeline, predictores: list[str]) -> pd.DataFrame:
    """Coeficientes del ajuste en entrenamiento, con el intercepto primero."""
    terminos = modelo.named_steps["poly"].get_feature_names_out(predictores)
    lineal = modelo.named_steps["model"]
    return pd.DataFrame(
        {
            "termino": ["intercepto", *terminos],
            "coeficiente": [float(lineal.intercept_), *map(float, lineal.coef_)],
        }
    )


def rango_diseno(modelo: Pipeline, x: pd.DataFrame) -> dict[str, float]:
    """Rango, columnas y número de condición de la matriz de diseño con intercepto."""
    z = modelo[:-1].transform(x)
    diseno = np.column_stack([np.ones(len(z)), z])
    return {
        "rango": int(np.linalg.matrix_rank(diseno)),
        "columnas": int(diseno.shape[1]),
        "condicion": float(np.linalg.cond(diseno)),
    }
