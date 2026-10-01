"""Segmentación del campo pulmonar por estrategias comparables.

El proyecto compara nueve combinaciones de realce y umbralización, con la
morfología fija. Dejar la morfología constante es deliberado: si variara junto
con lo demás, una diferencia en el resultado no podría atribuirse a una causa
concreta.

================  ==========================  ==========================
``config_id``     Realce                      Umbral
================  ==========================  ==========================
``base_fijo``     ninguno                     fijo
``base_otsu``     ninguno                     Otsu
``base_adap``     ninguno                     adaptativo
``eq_fijo``       ecualización global         fijo
``eq_otsu``       ecualización global         Otsu
``eq_adap``       ecualización global         adaptativo
``clahe_fijo``    CLAHE                       fijo
``clahe_otsu``    CLAHE                       Otsu
``clahe_adap``    CLAHE                       adaptativo
================  ==========================  ==========================

Las tres estrategias ``base_*`` son la línea base contra la que se mide el
criterio de éxito del proyecto.

Todo el procesamiento ocurre dentro de la máscara de campo irradiado. El
umbral se calcula sobre los píxeles con señal, no sobre la imagen completa:
en las radiografías más colimadas de Montgomery el fondo llega al 67 % de los
píxeles y arrastraría cualquier umbral global hacia valores sin sentido
anatómico.

El pulmón es la región **oscura** dentro del tórax, de modo que la máscara
resultante es la de los píxeles por debajo del umbral, no por encima.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage import exposure, filters, transform

from psim.recorte import mascara_campo

# Lado mayor al que se reduce la imagen antes de segmentar. Las imágenes de
# Montgomery llegan a 20 megapíxeles y procesar nueve estrategias sobre las
# 486 del conjunto de entrenamiento a resolución plena sería inviable en el
# tiempo disponible. La reducción se aplica por igual a imagen y máscara de
# referencia, de modo que el Dice se calcula sobre la misma rejilla.
LADO_TRABAJO = 512

# Percentil de los píxeles con señal que define el umbral fijo. El campo
# pulmonar ocupa en torno al 25 % del campo irradiado según la referencia, de
# modo que un percentil algo mayor deja margen para la morfología posterior.
PERCENTIL_FIJO = 35

# Tamaño de bloque del umbral adaptativo, en fracción del lado de trabajo.
FRACCION_BLOQUE = 0.25

# Límite de recorte de CLAHE y tamaño de su rejilla.
CLAHE_LIMITE = 0.02
CLAHE_REJILLA = 8

# Radio de las operaciones morfológicas, en fracción del lado de trabajo.
FRACCION_APERTURA = 0.012
FRACCION_CIERRE = 0.020

# Fracción del área de la componente mayor por debajo de la cual una región se
# descarta por no ser un campo pulmonar.
FRACCION_MINIMA_COMPONENTE = 0.15

REALCES = ("base", "eq", "clahe")
UMBRALES = ("fijo", "otsu", "adap")

CONFIGURACIONES: tuple[str, ...] = tuple(f"{r}_{u}" for r in REALCES for u in UMBRALES)


@dataclass(frozen=True)
class Segmentacion:
    """Resultado de aplicar una estrategia a una imagen."""

    config_id: str
    mascara: NDArray[np.bool_]
    n_componentes: int
    umbral_usado: float


def reducir(
    arreglo: NDArray[np.generic], lado: int = LADO_TRABAJO
) -> NDArray[np.generic]:
    """Reduce un arreglo para que su lado mayor sea el indicado.

    Las máscaras booleanas se reducen por vecino más cercano, porque son
    etiquetas y no admiten valores intermedios. Las imágenes de intensidad
    usan interpolación bilineal con suavizado previo, que es lo adecuado para
    una medida continua.
    """
    arreglo = np.asarray(arreglo)
    alto, ancho = arreglo.shape[:2]
    mayor = max(alto, ancho)
    if mayor <= lado:
        return arreglo

    escala = lado / mayor
    forma = (max(1, round(alto * escala)), max(1, round(ancho * escala)))

    if arreglo.dtype == np.bool_:
        return transform.resize(
            arreglo, forma, order=0, preserve_range=True, anti_aliasing=False
        ).astype(bool)

    return transform.resize(
        arreglo, forma, order=1, preserve_range=True, anti_aliasing=True
    ).astype(arreglo.dtype)


def realzar(
    gris: NDArray[np.generic],
    campo: NDArray[np.bool_],
    metodo: str,
) -> NDArray[np.float64]:
    """Aplica el realce indicado dentro del campo irradiado.

    La ecualización y CLAHE se calculan sobre los píxeles con señal. Incluir
    el fondo en la estimación de la distribución desplazaría el resultado en
    proporción a cuánto colimó el técnico, que es precisamente la variabilidad
    que el proyecto quiere separar de la anatomía.
    """
    valores = gris.astype(np.float64)
    if not campo.any():
        return valores

    if metodo == "base":
        return valores

    if metodo == "eq":
        # Ecualización por la función de distribución acumulada de los píxeles
        # con señal, extendida después a toda la imagen.
        dentro = valores[campo]
        orden = np.argsort(dentro)
        rangos = np.empty_like(orden, dtype=np.float64)
        rangos[orden] = np.arange(dentro.size, dtype=np.float64)
        salida = np.zeros_like(valores)
        salida[campo] = rangos / max(1, dentro.size - 1)
        return salida

    if metodo == "clahe":
        normalizada = valores.copy()
        dentro = valores[campo]
        lo, hi = float(dentro.min()), float(dentro.max())
        if hi > lo:
            normalizada = (valores - lo) / (hi - lo)
        normalizada = np.clip(normalizada, 0.0, 1.0)
        realzada = exposure.equalize_adapthist(
            normalizada,
            kernel_size=max(8, min(gris.shape) // CLAHE_REJILLA),
            clip_limit=CLAHE_LIMITE,
        )
        return np.where(campo, realzada, 0.0)

    raise ValueError(f"Realce desconocido: {metodo}")


def umbralizar(
    realzada: NDArray[np.float64],
    campo: NDArray[np.bool_],
    metodo: str,
) -> tuple[NDArray[np.bool_], float]:
    """Separa el campo pulmonar del resto del tórax.

    Devuelve la máscara de los píxeles **por debajo** del umbral: el aire del
    pulmón atenúa poco y se registra oscuro, de modo que el pulmón es la
    región de menor intensidad dentro del tórax.
    """
    dentro = realzada[campo]
    if dentro.size == 0:
        return np.zeros(realzada.shape, dtype=bool), float("nan")

    if metodo == "fijo":
        umbral = float(np.percentile(dentro, PERCENTIL_FIJO))
        return (realzada <= umbral) & campo, umbral

    if metodo == "otsu":
        umbral = float(filters.threshold_otsu(dentro))
        return (realzada <= umbral) & campo, umbral

    if metodo == "adap":
        bloque = int(min(realzada.shape) * FRACCION_BLOQUE) | 1
        bloque = max(3, bloque)
        local = filters.threshold_local(realzada, block_size=bloque, method="mean")
        return (realzada <= local) & campo, float("nan")

    raise ValueError(f"Umbral desconocido: {metodo}")


def _disco(radio: int) -> NDArray[np.bool_]:
    y, x = np.ogrid[-radio : radio + 1, -radio : radio + 1]
    return (y * y + x * x) <= radio * radio


def refinar(binaria: NDArray[np.bool_]) -> tuple[NDArray[np.bool_], int]:
    """Limpia la máscara umbralizada con morfología fija.

    La secuencia es cierre, relleno de huecos, apertura y selección de las dos
    componentes mayores. Es idéntica para las nueve estrategias, de modo que
    cualquier diferencia en el resultado proviene del realce o del umbral.

    El cierre va antes que la apertura, y no al revés: el ruido del detector
    deja la región umbralizada moteada de huecos de un píxel, y una apertura
    aplicada sobre ese moteado borra la región entera en lugar de limpiar sus
    bordes. Cerrar primero consolida la región, y la apertura posterior sí
    elimina los fragmentos pequeños que buscaba eliminar.
    """
    lado = min(binaria.shape)
    r_ap = max(1, int(lado * FRACCION_APERTURA))
    r_ci = max(1, int(lado * FRACCION_CIERRE))

    limpia = ndimage.binary_closing(binaria, structure=_disco(r_ci))
    limpia = ndimage.binary_fill_holes(limpia)
    limpia = ndimage.binary_opening(limpia, structure=_disco(r_ap))

    etiquetas, n = ndimage.label(limpia)
    if n == 0:
        return limpia, 0

    tamanos = ndimage.sum_labels(limpia, etiquetas, index=range(1, n + 1))
    orden = np.argsort(tamanos)[::-1]
    mayor = tamanos[orden[0]]

    conservadas = [
        int(i) + 1
        for i in orden[:2]
        if tamanos[i] >= mayor * FRACCION_MINIMA_COMPONENTE
    ]
    final = np.isin(etiquetas, conservadas)
    return final, len(conservadas)


def segmentar(
    gris: NDArray[np.generic],
    config_id: str,
    campo: NDArray[np.bool_] | None = None,
) -> Segmentacion:
    """Aplica una estrategia completa a una imagen ya reducida."""
    if config_id not in CONFIGURACIONES:
        raise ValueError(f"Configuracion desconocida: {config_id}")

    realce, umbral = config_id.split("_")
    if campo is None:
        campo = mascara_campo(gris)

    realzada = realzar(gris, campo, realce)
    binaria, valor = umbralizar(realzada, campo, umbral)
    final, n = refinar(binaria)
    return Segmentacion(config_id, final, n, valor)


def dice(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    """Coeficiente Dice entre dos máscaras binarias."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    suma = int(a.sum()) + int(b.sum())
    if suma == 0:
        return float("nan")
    return 2.0 * float((a & b).sum()) / suma


def jaccard(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    """Índice de Jaccard entre dos máscaras binarias."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    if union == 0:
        return float("nan")
    return float((a & b).sum()) / union
