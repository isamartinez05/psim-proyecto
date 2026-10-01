"""Descriptores morfológicos del campo pulmonar.

Los cinco descriptores son adimensionales a propósito. El conjunto Shenzhen no
documenta espaciado de píxel, de modo que ninguna medida expresada en
milímetros sería comparable entre los dos orígenes. Las razones y fracciones
sí lo son.

=========================  ================================================
Descriptor                 Definición
=========================  ================================================
``compacidad``             4·pi·area / perimetro², uno para un círculo
``relacion_aspecto``       alto / ancho de la caja envolvente
``fraccion_area``          area pulmonar / area del campo irradiado
``simetria_izq_der``       |izq - der| / (izq + der), cero si son iguales
``energia_wavelet_d1``     energía de los detalles de primer nivel
=========================  ================================================

El denominador de ``fraccion_area`` es el campo irradiado y no el campo de
imagen. Montgomery entrega siempre 4020x4892 píxeles con independencia de la
colimación, de modo que dividir por el área de la imagen mediría cuánto abrió
el colimador el técnico en lugar de cuánto pulmón tiene el paciente.

Todos los descriptores se calculan igual sobre la máscara obtenida y sobre la
de referencia, de modo que el error relativo entre ambos valores sea
interpretable. Ninguna función de este módulo recibe la etiqueta de
tuberculosis: las etiquetas se incorporan al final mediante una unión por
clave.
"""

from __future__ import annotations

import numpy as np
import pywt
from numpy.typing import NDArray
from scipy import ndimage

# Familia y nivel de la descomposición wavelet. Se declaran porque la energía
# de los detalles depende de ambos y no es comparable entre familias.
WAVELET = "db2"
NIVEL_WAVELET = 1

DESCRIPTORES = (
    "compacidad",
    "relacion_aspecto",
    "fraccion_area",
    "simetria_izq_der",
    "energia_wavelet_d1",
)


def perimetro(mascara: NDArray[np.bool_]) -> float:
    """Longitud del contorno, contando transiciones entre dentro y fuera.

    Se mide por diferencia con la máscara erosionada en lugar de trazar el
    contorno: el resultado es equivalente para el cálculo de compacidad y no
    depende de cómo se ordenen los píxeles del borde.
    """
    mascara = np.asarray(mascara, dtype=bool)
    if not mascara.any():
        return 0.0
    estructura = ndimage.generate_binary_structure(2, 1)
    interior = ndimage.binary_erosion(mascara, structure=estructura, border_value=0)
    return float((mascara & ~interior).sum())


def compacidad(mascara: NDArray[np.bool_]) -> float:
    """Qué tan circular es la región. Uno para un círculo perfecto.

    Sobre una máscara discreta el valor no coincide con el teórico: el
    perímetro contado como banda de píxeles del borde subestima la longitud
    real del contorno en las diagonales, de modo que un disco da alrededor de
    1,24 en lugar de 1. Lo que importa aquí no es el valor absoluto sino su
    estabilidad entre estrategias, y el sesgo es el mismo para la máscara
    obtenida y para la de referencia, de modo que el error relativo entre
    ambas sí es interpretable.
    """
    area = float(np.asarray(mascara, dtype=bool).sum())
    p = perimetro(mascara)
    if p == 0:
        return float("nan")
    return 4.0 * np.pi * area / (p * p)


def relacion_aspecto(mascara: NDArray[np.bool_]) -> float:
    """Alto entre ancho de la caja que envuelve la región."""
    mascara = np.asarray(mascara, dtype=bool)
    if not mascara.any():
        return float("nan")
    filas = np.flatnonzero(mascara.any(axis=1))
    columnas = np.flatnonzero(mascara.any(axis=0))
    alto = float(filas[-1] - filas[0] + 1)
    ancho = float(columnas[-1] - columnas[0] + 1)
    if ancho == 0:
        return float("nan")
    return alto / ancho


def fraccion_area(mascara: NDArray[np.bool_], campo: NDArray[np.bool_]) -> float:
    """Área pulmonar como fracción del campo irradiado."""
    denominador = float(np.asarray(campo, dtype=bool).sum())
    if denominador == 0:
        return float("nan")
    return float(np.asarray(mascara, dtype=bool).sum()) / denominador


def simetria_izq_der(
    izquierdo: NDArray[np.bool_],
    derecho: NDArray[np.bool_],
) -> float:
    """Diferencia relativa de área entre los dos campos pulmonares.

    Cero indica campos de igual área. El valor no distingue qué lado es mayor,
    porque la pregunta del proyecto es si los campos difieren y no en qué
    dirección.
    """
    a = float(np.asarray(izquierdo, dtype=bool).sum())
    b = float(np.asarray(derecho, dtype=bool).sum())
    if a + b == 0:
        return float("nan")
    return abs(a - b) / (a + b)


def separar_por_centroide(
    mascara: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Divide una máscara en campo izquierdo y derecho del paciente.

    Si hay dos componentes conexas se usan directamente. Si hay una sola,
    porque los campos se tocan, se parte por la mitad de su extensión
    horizontal. La convención radiológica sitúa el pulmón izquierdo del
    paciente a la derecha de la imagen.
    """
    mascara = np.asarray(mascara, dtype=bool)
    vacia = np.zeros(mascara.shape, dtype=bool)
    if not mascara.any():
        return vacia, vacia

    etiquetas, n = ndimage.label(mascara)
    if n >= 2:
        tamanos = ndimage.sum_labels(mascara, etiquetas, index=range(1, n + 1))
        dos = [int(i) + 1 for i in np.argsort(tamanos)[::-1][:2]]
        centros = ndimage.center_of_mass(mascara, etiquetas, index=dos)
        por_x = sorted(zip(dos, centros, strict=True), key=lambda t: t[1][1])
        return etiquetas == por_x[1][0], etiquetas == por_x[0][0]

    columnas = np.flatnonzero(mascara.any(axis=0))
    corte = int((columnas[0] + columnas[-1]) / 2)
    izquierdo = mascara.copy()
    izquierdo[:, :corte] = False
    derecho = mascara.copy()
    derecho[:, corte:] = False
    return izquierdo, derecho


def energia_wavelet(
    gris: NDArray[np.generic],
    mascara: NDArray[np.bool_],
    wavelet: str = WAVELET,
) -> float:
    """Energía de los coeficientes de detalle de primer nivel en la región.

    La imagen se anula fuera de la máscara antes de descomponer. Eso introduce
    un borde artificial en el contorno de la región, que afecta a los
    coeficientes de detalle; la energía se normaliza por el número de
    coeficientes dentro de la región para que el valor no dependa del tamaño
    del pulmón, pero el efecto de borde se declara como limitación.
    """
    mascara = np.asarray(mascara, dtype=bool)
    if not mascara.any():
        return float("nan")

    valores = np.asarray(gris, dtype=np.float64)
    lo, hi = float(valores.min()), float(valores.max())
    if hi > lo:
        valores = (valores - lo) / (hi - lo)
    dentro = np.where(mascara, valores, 0.0)

    _, (ch, cv, cd) = pywt.dwt2(dentro, wavelet)
    detalle = ch**2 + cv**2 + cd**2

    reducida = mascara[::2, ::2]
    reducida = reducida[: detalle.shape[0], : detalle.shape[1]]
    if reducida.shape != detalle.shape:
        relleno = np.zeros(detalle.shape, dtype=bool)
        relleno[: reducida.shape[0], : reducida.shape[1]] = reducida
        reducida = relleno

    n = int(reducida.sum())
    if n == 0:
        return float("nan")
    return float(detalle[reducida].sum() / n)


def extraer(
    gris: NDArray[np.generic],
    mascara: NDArray[np.bool_],
    campo: NDArray[np.bool_],
    izquierdo: NDArray[np.bool_] | None = None,
    derecho: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Calcula los cinco descriptores sobre una máscara.

    Si no se entregan los campos por separado se derivan de la máscara. La
    función no recibe la etiqueta de tuberculosis ni ninguna información que
    no estuviera disponible al procesar una imagen nueva.
    """
    mascara = np.asarray(mascara, dtype=bool)
    if izquierdo is None or derecho is None:
        izquierdo, derecho = separar_por_centroide(mascara)

    return {
        "compacidad": compacidad(mascara),
        "relacion_aspecto": relacion_aspecto(mascara),
        "fraccion_area": fraccion_area(mascara, campo),
        "simetria_izq_der": simetria_izq_der(izquierdo, derecho),
        "energia_wavelet_d1": energia_wavelet(gris, mascara),
    }


def error_relativo(obtenido: float, referencia: float) -> float:
    """Error relativo de un descriptor frente a su valor de referencia."""
    if not np.isfinite(obtenido) or not np.isfinite(referencia) or referencia == 0:
        return float("nan")
    return abs(obtenido - referencia) / abs(referencia)
