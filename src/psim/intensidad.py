"""Transformaciones puntuales de intensidad, diferencia entre imagenes y
requantizacion. Trabajan sobre imagenes en flotante en [0, 1].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Imagen = NDArray[np.floating]
Mascara = NDArray[np.bool_]


def normalizar_rango(
    imagen: Imagen, region: Mascara, bajo: float = 1.0, alto: float = 99.0
) -> Imagen:
    """Lleva los percentiles `bajo` y `alto` de la region a 0 y 1, con recorte."""
    p_bajo, p_alto = np.percentile(imagen[region], [bajo, alto])
    if p_alto <= p_bajo:
        return np.zeros_like(imagen)
    return np.clip((imagen - p_bajo) / (p_alto - p_bajo), 0.0, 1.0)


def gamma(imagen: Imagen, exponente: float) -> Imagen:
    """Correccion gamma: menor que 1 aclara, mayor que 1 oscurece."""
    return np.clip(imagen, 0.0, 1.0) ** exponente


def logaritmica(imagen: Imagen) -> Imagen:
    """Transformacion logaritmica normalizada: 0 va a 0 y 1 va a 1."""
    return np.log1p(np.clip(imagen, 0.0, 1.0)) / np.log(2.0)


def diferencia(a: Imagen, b: Imagen) -> Imagen:
    """Mapa de diferencia absoluta entre dos imagenes del mismo tamano."""
    return np.abs(a - b)


def requantizar(imagen: Imagen, bits: int) -> Imagen:
    """Reduce la imagen a 2**bits niveles uniformes en [0, 1]."""
    niveles = 2**bits - 1
    return np.round(np.clip(imagen, 0.0, 1.0) * niveles) / niveles
