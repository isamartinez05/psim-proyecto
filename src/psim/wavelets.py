"""Wavelets y multirresolucion: DWT 2D por niveles, energia por subbanda,
reconstruccion por niveles y wavelet continua sobre un perfil 1D.
"""

from __future__ import annotations

import numpy as np
import pywt
from numpy.typing import NDArray

Imagen = NDArray[np.floating]
Mascara = NDArray[np.bool_]

ORIENTACIONES = ("H", "V", "D")


def descomponer(imagen: Imagen, wavelet: str = "db2", niveles: int = 3) -> list:
    """DWT 2D en modo periodizacion: [A_n, (H_n, V_n, D_n), ..., (H_1, V_1, D_1)]."""
    return pywt.wavedec2(imagen, wavelet, level=niveles, mode="periodization")


def nombres_subbandas(niveles: int = 3) -> list[str]:
    """Nombres en orden de la descomposicion: A3, H3, V3, D3, ..., H1, V1, D1."""
    nombres = [f"A{niveles}"]
    for nivel in range(niveles, 0, -1):
        nombres += [f"{o}{nivel}" for o in ORIENTACIONES]
    return nombres


def energia_subbandas(
    imagen: Imagen, wavelet: str = "db2", niveles: int = 3
) -> dict[str, float]:
    """Fraccion de la energia total en cada subbanda, sin la media; suma 1."""
    coeficientes = descomponer(imagen - imagen.mean(), wavelet, niveles)
    energias = [float(np.sum(coeficientes[0] ** 2))]
    for detalle in coeficientes[1:]:
        energias += [float(np.sum(c**2)) for c in detalle]
    total = sum(energias)
    if total == 0:
        return dict.fromkeys(nombres_subbandas(niveles), float("nan"))
    return {
        nombre: e / total
        for nombre, e in zip(nombres_subbandas(niveles), energias, strict=True)
    }


def energia_por_nivel(energias: dict[str, float], niveles: int = 3) -> dict[str, float]:
    """Suma H, V y D de cada nivel: A_n y detalle de cada escala."""
    resumen = {f"A{niveles}": energias[f"A{niveles}"]}
    for nivel in range(niveles, 0, -1):
        resumen[f"nivel_{nivel}"] = sum(energias[f"{o}{nivel}"] for o in ORIENTACIONES)
    return resumen


def reconstruir_sin_detalles(
    imagen: Imagen, quitar_hasta: int, wavelet: str = "db2", niveles: int = 3
) -> Imagen:
    """Reconstruye anulando los detalles de los niveles 1 a `quitar_hasta`."""
    coeficientes = descomponer(imagen, wavelet, niveles)
    for indice in range(1, len(coeficientes)):
        nivel = niveles - indice + 1
        if nivel <= quitar_hasta:
            coeficientes[indice] = tuple(np.zeros_like(c) for c in coeficientes[indice])
    reconstruida = pywt.waverec2(coeficientes, wavelet, mode="periodization")
    return reconstruida[: imagen.shape[0], : imagen.shape[1]]


def perfil_horizontal(imagen: Imagen, campo: Mascara) -> tuple[int, Imagen]:
    """Fila que pasa por el centroide del campo, limitada a su extension horizontal."""
    filas, _ = np.nonzero(campo)
    fila = int(filas.mean())
    columnas = np.flatnonzero(campo[fila])
    return fila, imagen[fila, columnas.min() : columnas.max() + 1]


def escalograma(
    perfil: Imagen, escalas: NDArray[np.floating], wavelet: str = "mexh"
) -> Imagen:
    """Magnitud de la transformada wavelet continua: una fila por escala."""
    coeficientes, _ = pywt.cwt(perfil - perfil.mean(), escalas, wavelet)
    return np.abs(coeficientes)
