"""Histogramas: distribucion, funcion acumulada, especificacion y ecualizacion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance
from skimage import exposure

Imagen = NDArray[np.floating]
Mascara = NDArray[np.bool_]


def histograma(valores: NDArray, bins: int = 256) -> tuple[NDArray, NDArray]:
    """Centros de clase y densidad en [0, 1]; la densidad integra 1."""
    densidad, bordes = np.histogram(valores, bins=bins, range=(0.0, 1.0), density=True)
    return (bordes[:-1] + bordes[1:]) / 2, densidad


def cdf(valores: NDArray, bins: int = 256) -> tuple[NDArray, NDArray]:
    """Funcion de distribucion acumulada empirica en [0, 1]."""
    conteo, bordes = np.histogram(valores, bins=bins, range=(0.0, 1.0))
    acumulada = np.cumsum(conteo) / max(conteo.sum(), 1)
    return bordes[1:], acumulada


def especificar(imagen: Imagen, region: Mascara, referencia: NDArray) -> Imagen:
    """Ajusta la distribucion de la region a la de `referencia` por cuantiles.

    Es monotona: conserva el orden de las intensidades. Fuera de la region la
    imagen no cambia.
    """
    salida = imagen.copy()
    origen = imagen[region]
    n = origen.size
    if n == 0:
        return salida
    orden = np.argsort(origen, kind="stable")
    cuantiles = (np.arange(n) + 0.5) / n
    ref = np.sort(np.asarray(referencia, dtype=float))
    posiciones = (np.arange(ref.size) + 0.5) / ref.size
    nuevos = np.empty(n)
    nuevos[orden] = np.interp(cuantiles, posiciones, ref)
    salida[region] = nuevos
    return salida


def ecualizar_global(imagen: Imagen) -> Imagen:
    """Ecualizacion global del histograma."""
    return exposure.equalize_hist(imagen)


def clahe(imagen: Imagen, limite: float = 0.01) -> Imagen:
    """Ecualizacion adaptativa limitada por contraste."""
    return exposure.equalize_adapthist(imagen, clip_limit=limite)


def distancia(a: NDArray, b: NDArray) -> float:
    """Distancia de Wasserstein entre dos distribuciones de intensidad."""
    return float(wasserstein_distance(np.ravel(a), np.ravel(b)))
