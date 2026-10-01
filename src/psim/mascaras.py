"""Lectura unificada de las máscaras de referencia.

Los dos conjuntos entregan la referencia en formatos distintos, y ninguna de
las diferencias es cosmética:

===================  ==========================  ==========================
                     Montgomery                  Shenzhen
===================  ==========================  ==========================
Archivos             Dos, uno por pulmón         Uno con los dos campos
Tipo                 ``bool``                    ``uint8`` con antialiasing
Resolución           La de la imagen             512 x 512, fija
Procedencia          Supervisión de radiólogo    Instituto Politécnico Kiev
===================  ==========================  ==========================

La diferencia de resolución es la que más consecuencias tiene. Las máscaras de
Shenzhen están a 512 x 512 mientras que sus imágenes rondan los 2940 x 3000,
de modo que superponerlas exige reescalar. Se reescala la máscara a la imagen
y no al revés, para no descartar resolución de la imagen que el procesamiento
posterior sí utiliza; la interpolación es por vecino más cercano, porque
cualquier interpolación suavizada produciría valores que no son ni pulmón ni
fondo.

Ese reescalado no crea detalle que la anotación no tenga: la máscara de
Shenzhen sigue teniendo la precisión de 512 x 512 aunque se represente sobre
una rejilla mayor. La limitación se declara y no se compensa.

Las máscaras de Shenzhen contienen valores intermedios entre 0 y 255, en torno
al 0,14 % de los píxeles, propios del antialiasing del borde. Binarizar en 128
frente a cualquier otro umbral entre 1 y 255 cambia el área en menos del
0,3 %, de modo que la elección no es crítica, pero queda fijada para que el
procedimiento sea reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from psim import DATA_RAW
from psim.io import leer

# Umbral de binarización de las máscaras con antialiasing.
UMBRAL_MASCARA = 128

# Fracción del área del campo mayor por debajo de la cual una componente
# conexa se considera fragmento de anotación y no un pulmón.
FRACCION_FRAGMENTO = 0.05


@dataclass(frozen=True)
class Referencia:
    """Máscara de referencia de una imagen, con los campos separados.

    ``izquierdo`` y ``derecho`` siguen la convención radiológica: el pulmón
    izquierdo del paciente aparece a la derecha de la imagen. La separación es
    necesaria para la simetría entre campos, que no puede calcularse sobre una
    máscara con los dos pulmones unidos.
    """

    izquierdo: NDArray[np.bool_]
    derecho: NDArray[np.bool_]
    reescalada: bool
    n_componentes_original: int
    fragmentos_descartados: int

    @property
    def total(self) -> NDArray[np.bool_]:
        """Ambos campos pulmonares en una sola máscara."""
        return self.izquierdo | self.derecho

    @property
    def area(self) -> int:
        return int(self.total.sum())

    @property
    def area_izquierdo(self) -> int:
        return int(self.izquierdo.sum())

    @property
    def area_derecho(self) -> int:
        return int(self.derecho.sum())

    @property
    def vacia(self) -> bool:
        return self.area == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "mask_area_px": self.area,
            "mask_area_izq_px": self.area_izquierdo,
            "mask_area_der_px": self.area_derecho,
            "mask_reescalada": int(self.reescalada),
            "mask_n_componentes": self.n_componentes_original,
            "mask_fragmentos": self.fragmentos_descartados,
        }


def binarizar(
    mascara: NDArray[np.generic], umbral: int = UMBRAL_MASCARA
) -> NDArray[np.bool_]:
    """Convierte una máscara a booleana, sea cual sea su tipo de origen."""
    mascara = np.asarray(mascara)
    if mascara.dtype == np.bool_:
        return mascara
    if mascara.ndim == 3:
        mascara = mascara[..., 0]
    return mascara >= umbral


def reescalar(
    mascara: NDArray[np.bool_],
    forma: tuple[int, int],
) -> NDArray[np.bool_]:
    """Lleva una máscara binaria a la forma indicada por vecino más cercano.

    Se usa vecino más cercano y no una interpolación de orden superior porque
    la máscara es una etiqueta, no una medida: un valor intermedio entre
    pulmón y fondo no tiene interpretación.
    """
    if mascara.shape == forma:
        return mascara
    factor = (forma[0] / mascara.shape[0], forma[1] / mascara.shape[1])
    return ndimage.zoom(mascara.astype(np.uint8), factor, order=0).astype(bool) > 0


def separar_campos(
    mascara: NDArray[np.bool_],
    fraccion_fragmento: float = FRACCION_FRAGMENTO,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], int, int]:
    """Separa los dos campos pulmonares de una máscara conjunta.

    Las máscaras de Shenzhen presentan entre dos y diez componentes conexas.
    Las dos mayores son los campos pulmonares; las restantes son fragmentos de
    anotación, demasiado pequeños para ser un pulmón.

    La asignación a izquierdo o derecho se hace por la posición horizontal del
    centroide, siguiendo la convención radiológica: el pulmón izquierdo del
    paciente queda a la derecha de la imagen.

    Devuelve los dos campos, el número de componentes originales y cuántas se
    descartaron por fragmentarias.
    """
    etiquetas, n = ndimage.label(mascara)
    if n == 0:
        vacia = np.zeros(mascara.shape, dtype=bool)
        return vacia, vacia, 0, 0

    tamanos = ndimage.sum_labels(mascara, etiquetas, index=range(1, n + 1))
    orden = np.argsort(tamanos)[::-1]
    mayor = tamanos[orden[0]]

    conservadas = [
        int(i) + 1 for i in orden if tamanos[i] >= mayor * fraccion_fragmento
    ]
    descartadas = n - len(conservadas)

    if len(conservadas) == 1:
        # Los dos campos se tocan y forman una sola componente. Se dividen por
        # el centroide horizontal, que es la mejor aproximación disponible.
        unico = etiquetas == conservadas[0]
        columnas = np.flatnonzero(unico.any(axis=0))
        corte = int((columnas[0] + columnas[-1]) / 2)
        a_izquierda = unico.copy()
        a_izquierda[:, corte:] = False
        a_derecha = unico.copy()
        a_derecha[:, :corte] = False
        return a_derecha, a_izquierda, n, descartadas

    primeras = conservadas[:2]
    centroides = ndimage.center_of_mass(mascara, etiquetas, index=primeras)
    por_x = sorted(zip(primeras, centroides, strict=True), key=lambda t: t[1][1])

    # El primero en orden horizontal está a la izquierda de la imagen, que
    # corresponde al pulmón derecho del paciente.
    derecho = etiquetas == por_x[0][0]
    izquierdo = etiquetas == por_x[1][0]

    for etiqueta in conservadas[2:]:
        extra = etiquetas == etiqueta
        cx = ndimage.center_of_mass(mascara, etiquetas, index=[etiqueta])[0][1]
        if cx < (por_x[0][1][1] + por_x[1][1][1]) / 2:
            derecho = derecho | extra
        else:
            izquierdo = izquierdo | extra

    return izquierdo, derecho, n, descartadas


def leer_referencia(
    fila: dict[str, object], forma_imagen: tuple[int, int]
) -> Referencia:
    """Lee la máscara de referencia de una fila del manifiesto.

    Resuelve las diferencias entre los dos conjuntos y devuelve siempre el
    mismo objeto: dos campos booleanos a la resolución de la imagen.
    """
    ruta = str(fila.get("mask_path", "")).strip()
    if not ruta:
        vacia = np.zeros(forma_imagen, dtype=bool)
        return Referencia(vacia, vacia, False, 0, 0)

    es_montgomery = str(fila["dataset_id"]).endswith("MontgomeryCXRSet")

    if es_montgomery:
        # El manifiesto registra la máscara izquierda; la derecha es el mismo
        # nombre en la carpeta hermana.
        izq_path = DATA_RAW / ruta
        der_path = DATA_RAW / ruta.replace("leftMask", "rightMask")
        izquierdo = binarizar(leer(izq_path))
        derecho = binarizar(leer(der_path))
        reescalada = izquierdo.shape != forma_imagen
        if reescalada:
            izquierdo = reescalar(izquierdo, forma_imagen)
            derecho = reescalar(derecho, forma_imagen)
        n = int(izquierdo.any()) + int(derecho.any())
        return Referencia(izquierdo, derecho, reescalada, n, 0)

    conjunta = binarizar(leer(DATA_RAW / ruta))
    reescalada = conjunta.shape != forma_imagen
    if reescalada:
        conjunta = reescalar(conjunta, forma_imagen)
    izquierdo, derecho, n, descartadas = separar_campos(conjunta)
    return Referencia(izquierdo, derecho, reescalada, n, descartadas)


def ruta_mascara_derecha(mask_path: str) -> Path:
    """Ruta de la máscara derecha de Montgomery a partir de la izquierda."""
    return DATA_RAW / mask_path.replace("leftMask", "rightMask")
