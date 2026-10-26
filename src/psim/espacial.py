"""Operaciones de vecindad: filtros de ruido, operadores de borde y realce.

Trabaja sobre imagenes en flotante en [0, 1]. Incluye las metricas de calidad
de OE2: contraste RMS, ruido estimado y conservacion de bordes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage import filters, restoration, util

Imagen = NDArray[np.floating]
Mascara = NDArray[np.bool_]

KERNEL_EMBOSS = np.array([[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])


def a_flotante(imagen: NDArray[np.generic]) -> Imagen:
    """Convierte a flotante en [0, 1] segun el tipo de dato."""
    return util.img_as_float(imagen)


def filtro_media(imagen: Imagen, tamano: int = 5) -> Imagen:
    """Promedio en una ventana cuadrada de lado `tamano`."""
    return ndimage.uniform_filter(imagen, size=tamano, mode="reflect")


def filtro_mediana(imagen: Imagen, tamano: int = 5) -> Imagen:
    """Mediana en una ventana cuadrada de lado `tamano`."""
    return ndimage.median_filter(imagen, size=tamano, mode="reflect")


def filtro_gaussiano(imagen: Imagen, sigma: float = 1.5) -> Imagen:
    """Suavizado gaussiano de desviacion `sigma` en pixeles."""
    return filters.gaussian(imagen, sigma=sigma, preserve_range=True)


def bordes(imagen: Imagen) -> dict[str, Imagen]:
    """Magnitud de borde con Sobel, Prewitt, Roberts y Laplaciano absoluto."""
    return {
        "Sobel": filters.sobel(imagen),
        "Prewitt": filters.prewitt(imagen),
        "Roberts": filters.roberts(imagen),
        "Laplaciano": np.abs(filters.laplace(imagen, ksize=3)),
    }


def unsharp(imagen: Imagen, radio: float = 2.0, cantidad: float = 1.0) -> Imagen:
    """Realce por mascara de desenfoque, recortado a [0, 1]."""
    realzada = filters.unsharp_mask(
        imagen, radius=radio, amount=cantidad, preserve_range=True
    )
    return np.clip(realzada, 0.0, 1.0)


def emboss(imagen: Imagen) -> Imagen:
    """Operador direccional de relieve, reescalado a [0, 1]."""
    r = ndimage.convolve(imagen, KERNEL_EMBOSS, mode="reflect")
    rango = r.max() - r.min()
    return (r - r.min()) / rango if rango > 0 else np.zeros_like(r)


def contraste_rms(imagen: Imagen, region: Mascara) -> float:
    """Desviacion estandar de la intensidad dentro de la region."""
    return float(np.std(imagen[region]))


def ruido_estimado(imagen: Imagen) -> float:
    """Desviacion del ruido estimada por wavelets (mediana absoluta en la escala fina)."""
    return float(restoration.estimate_sigma(imagen))


def conservacion_bordes(original: Imagen, procesada: Imagen, region: Mascara) -> float:
    """Correlacion de Pearson entre las magnitudes de Sobel dentro de la region."""
    a = filters.sobel(original)[region]
    b = filters.sobel(procesada)[region]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
