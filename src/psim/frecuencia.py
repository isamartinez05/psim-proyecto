"""Dominio de frecuencia: espectro, perfil radial de potencia y filtros.

Las frecuencias se expresan en ciclos por pixel (Nyquist = 0.5). Los filtros
se construyen sobre la rejilla sin centrar de la FFT, de modo que se aplican
multiplicando directamente la transformada.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Imagen = NDArray[np.floating]
Mascara = NDArray[np.bool_]


def radio(forma: tuple[int, int]) -> Imagen:
    """Frecuencia radial de cada coeficiente de la FFT, en ciclos por pixel."""
    fy = np.fft.fftfreq(forma[0])
    fx = np.fft.fftfreq(forma[1])
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    return np.sqrt(xx**2 + yy**2)


def espectro(imagen: Imagen) -> Imagen:
    """Logaritmo de la magnitud de la FFT, centrado, sin la componente media."""
    f = np.fft.fft2(imagen - imagen.mean())
    return np.log1p(np.abs(np.fft.fftshift(f)))


def pasabajas_gaussiano(forma: tuple[int, int], corte: float) -> Imagen:
    """H = exp(-r^2 / (2 corte^2))."""
    r = radio(forma)
    return np.exp(-(r**2) / (2 * corte**2))


def pasabajas_butterworth(
    forma: tuple[int, int], corte: float, orden: int = 2
) -> Imagen:
    """H = 1 / (1 + (r / corte)^(2n)); vale 0.5 en r = corte."""
    r = radio(forma)
    return 1.0 / (1.0 + (r / corte) ** (2 * orden))


def pasaaltas_butterworth(
    forma: tuple[int, int], corte: float, orden: int = 2
) -> Imagen:
    """Complemento del pasa-bajas Butterworth; anula la componente continua."""
    return 1.0 - pasabajas_butterworth(forma, corte, orden)


def pasabanda_butterworth(
    forma: tuple[int, int], bajo: float, alto: float, orden: int = 2
) -> Imagen:
    """Producto de un pasa-altas en `bajo` y un pasa-bajas en `alto`."""
    return pasaaltas_butterworth(forma, bajo, orden) * pasabajas_butterworth(
        forma, alto, orden
    )


def filtrar(imagen: Imagen, h: Imagen) -> Imagen:
    """Aplica la funcion de transferencia `h` en el dominio de frecuencia."""
    return np.real(np.fft.ifft2(np.fft.fft2(imagen) * h))


def recorte_central(imagen: Imagen, campo: Mascara, lado: int = 192) -> Imagen:
    """Recorte cuadrado centrado en el centroide del campo irradiado."""
    filas, columnas = np.nonzero(campo)
    cy, cx = int(filas.mean()), int(columnas.mean())
    mitad = lado // 2
    y0 = int(np.clip(cy - mitad, 0, imagen.shape[0] - lado))
    x0 = int(np.clip(cx - mitad, 0, imagen.shape[1] - lado))
    return imagen[y0 : y0 + lado, x0 : x0 + lado]


def con_ventana(imagen: Imagen) -> Imagen:
    """Resta la media y aplica una ventana de Hann para reducir la fuga espectral."""
    ventana = np.outer(np.hanning(imagen.shape[0]), np.hanning(imagen.shape[1]))
    return (imagen - imagen.mean()) * ventana


def perfil_radial(imagen: Imagen, bins: int = 64) -> tuple[Imagen, Imagen]:
    """Potencia media por anillo de frecuencia, entre 0 y Nyquist."""
    potencia = np.abs(np.fft.fft2(imagen - imagen.mean())) ** 2
    r = radio(imagen.shape).ravel()
    bordes = np.linspace(0.0, 0.5, bins + 1)
    indice = np.digitize(r, bordes) - 1
    valido = (indice >= 0) & (indice < bins)
    suma = np.bincount(indice[valido], weights=potencia.ravel()[valido], minlength=bins)
    cuenta = np.bincount(indice[valido], minlength=bins)
    media = np.divide(suma, cuenta, out=np.zeros(bins), where=cuenta > 0)
    return (bordes[:-1] + bordes[1:]) / 2, media


def fraccion_alta(imagen: Imagen, corte: float = 0.25) -> float:
    """Fraccion de la potencia (sin la componente media) por encima de `corte`."""
    potencia = np.abs(np.fft.fft2(imagen - imagen.mean())) ** 2
    total = potencia.sum()
    if total == 0:
        return float("nan")
    return float(potencia[radio(imagen.shape) > corte].sum() / total)
