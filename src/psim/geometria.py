"""Geometria de regiones y morfologia: conectividad, etiquetado, transformada
de distancia, borde, morfologia binaria y top-hat / black-hat en grises.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

Mascara = NDArray[np.bool_]
Imagen = NDArray[np.floating]


def disco(radio: int) -> Mascara:
    """Elemento estructurante circular de radio `radio`."""
    yy, xx = np.ogrid[-radio : radio + 1, -radio : radio + 1]
    return (yy**2 + xx**2) <= radio**2


def etiquetar(mascara: Mascara, conectividad: int = 8) -> tuple[NDArray[np.int32], int]:
    """Etiqueta regiones conexas con vecindad 4 u 8."""
    if conectividad not in (4, 8):
        raise ValueError("La conectividad debe ser 4 u 8.")
    estructura = ndimage.generate_binary_structure(2, 1 if conectividad == 4 else 2)
    etiquetas, n = ndimage.label(mascara, structure=estructura)
    return etiquetas, int(n)


def transformada_distancia(mascara: Mascara) -> Imagen:
    """Distancia euclidiana de cada pixel de la region al fondo mas cercano."""
    return ndimage.distance_transform_edt(mascara)


def borde(mascara: Mascara) -> Mascara:
    """Borde interior: pixeles de la region que se pierden al erosionar con 3x3."""
    interior = ndimage.binary_erosion(mascara, structure=np.ones((3, 3), dtype=bool))
    return mascara & ~interior


def erosion(mascara: Mascara, radio: int) -> Mascara:
    """Erosion binaria con un disco."""
    return ndimage.binary_erosion(mascara, structure=disco(radio))


def dilatacion(mascara: Mascara, radio: int) -> Mascara:
    """Dilatacion binaria con un disco."""
    return ndimage.binary_dilation(mascara, structure=disco(radio))


def apertura(mascara: Mascara, radio: int) -> Mascara:
    """Erosion seguida de dilatacion: elimina objetos menores que el disco."""
    return ndimage.binary_opening(mascara, structure=disco(radio))


def cierre(mascara: Mascara, radio: int) -> Mascara:
    """Dilatacion seguida de erosion: rellena huecos menores que el disco."""
    return ndimage.binary_closing(mascara, structure=disco(radio))


def top_hat(imagen: Imagen, radio: int) -> Imagen:
    """Imagen menos su apertura en grises: resalta detalles claros pequenos."""
    return imagen - ndimage.grey_opening(imagen, footprint=disco(radio))


def black_hat(imagen: Imagen, radio: int) -> Imagen:
    """Cierre en grises menos la imagen: resalta detalles oscuros pequenos."""
    return ndimage.grey_closing(imagen, footprint=disco(radio)) - imagen
