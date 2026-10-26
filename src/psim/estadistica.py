"""Estadistica del analisis entre grupos (OE5).

Mann-Whitney U bilateral, delta de Cliff con intervalo bootstrap por paciente
y correccion de Holm por comparaciones multiples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import mannwhitneyu, rankdata


def _limpio(valores: ArrayLike) -> np.ndarray:
    """Convierte a flotante y descarta valores no finitos."""
    a = np.asarray(valores, dtype=float)
    return a[np.isfinite(a)]


def delta_cliff(x: ArrayLike, y: ArrayLike) -> float:
    """Delta de Cliff, P(X > Y) - P(X < Y); positivo si x tiende a ser mayor."""
    x, y = _limpio(x), _limpio(y)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    rangos = rankdata(np.concatenate([x, y]))
    u = rangos[: len(x)].sum() - len(x) * (len(x) + 1) / 2
    return float(2 * u / (len(x) * len(y)) - 1)


def intervalo_delta(
    x: ArrayLike,
    y: ArrayLike,
    replicas: int = 2000,
    nivel: float = 0.95,
    semilla: int = 2026,
) -> tuple[float, float]:
    """Intervalo bootstrap percentil del delta de Cliff, remuestreando cada grupo."""
    x, y = _limpio(x), _limpio(y)
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(semilla)
    bx = rng.choice(x, size=(replicas, len(x)), replace=True)
    by = rng.choice(y, size=(replicas, len(y)), replace=True)
    rangos = rankdata(np.concatenate([bx, by], axis=1), axis=1)
    u = rangos[:, : len(x)].sum(axis=1) - len(x) * (len(x) + 1) / 2
    deltas = 2 * u / (len(x) * len(y)) - 1
    cola = (1 - nivel) / 2
    inferior, superior = np.quantile(deltas, [cola, 1 - cola])
    return float(inferior), float(superior)


def holm(pvalores: ArrayLike) -> np.ndarray:
    """Valores p ajustados por Holm; los NaN se conservan y no cuentan en m."""
    p = np.asarray(pvalores, dtype=float)
    ajustado = np.full(p.shape, np.nan)
    validos = np.flatnonzero(np.isfinite(p))
    m = len(validos)
    if m == 0:
        return ajustado
    orden = validos[np.argsort(p[validos], kind="stable")]
    escalado = (m - np.arange(m)) * p[orden]
    ajustado[orden] = np.minimum(np.maximum.accumulate(escalado), 1.0)
    return ajustado


def comparar_grupos(
    datos: pd.DataFrame,
    descriptores: list[str],
    grupo: str = "target",
    estrato: str = "dataset_id",
    replicas: int = 2000,
    semilla: int = 2026,
    alfa: float = 0.05,
) -> pd.DataFrame:
    """Compara el grupo 1 contra el 0 por descriptor, dentro de cada estrato.

    Devuelve tamanos, medianas, U, valor p, delta de Cliff con su intervalo y el
    valor p ajustado por Holm sobre todas las pruebas de la tabla.
    """
    filas: list[dict[str, Any]] = []
    indice = 0
    for nombre, g in datos.groupby(estrato, sort=True):
        for d in descriptores:
            x = _limpio(g.loc[g[grupo] == 1, d])
            y = _limpio(g.loc[g[grupo] == 0, d])
            fila: dict[str, Any] = {
                "estrato": nombre,
                "descriptor": d,
                "n_con": len(x),
                "n_sin": len(y),
                "mediana_con": float(np.median(x)) if len(x) else float("nan"),
                "mediana_sin": float(np.median(y)) if len(y) else float("nan"),
                "U": float("nan"),
                "p": float("nan"),
                "delta": float("nan"),
                "ic_inf": float("nan"),
                "ic_sup": float("nan"),
            }
            if len(x) >= 2 and len(y) >= 2:
                prueba = mannwhitneyu(x, y, alternative="two-sided")
                fila["U"] = float(prueba.statistic)
                fila["p"] = float(prueba.pvalue)
                fila["delta"] = delta_cliff(x, y)
                fila["ic_inf"], fila["ic_sup"] = intervalo_delta(
                    x, y, replicas=replicas, semilla=semilla + indice
                )
            filas.append(fila)
            indice += 1
    resultado = pd.DataFrame(filas)
    resultado["p_holm"] = holm(resultado["p"])
    resultado["significativo"] = resultado["p_holm"] < alfa
    return resultado
