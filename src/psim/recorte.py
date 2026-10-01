"""Recorte al campo irradiado.

Las radiografías de Montgomery se entregan siempre con el mismo tamaño de
archivo, 4020x4892 píxeles, con independencia de cuánto colimó el técnico. El
área no irradiada queda registrada como negro puro y puede llegar al 67 % de
la imagen. Shenzhen, en cambio, entrega la imagen ya recortada: su mediana de
fondo negro es del 1 %, frente al 19 % de Montgomery.

Esa diferencia no es anecdótica para este proyecto:

- La fracción de área pulmonar respecto al campo de imagen mediría cuánto
  colimó el técnico, no cuánto pulmón tiene el paciente.
- Un umbral calculado sobre el histograma completo queda dominado por un modo
  en cero que no corresponde a ninguna estructura anatómica.
- El contraste medido sobre la imagen entera se diluye en proporción al
  tamaño del área no irradiada.

El recorte iguala las dos convenciones antes de cualquier medición. No elimina
información: el área descartada no contiene señal.

Las máscaras de referencia se recortan con la misma geometría que su imagen,
nunca de forma independiente, para que la correspondencia píxel a píxel se
conserve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

# Intensidad por debajo de la cual un píxel se considera área no irradiada.
# El fondo de Montgomery es negro puro; el margen admite ruido del detector
# sin llegar a descartar tejido, que en estas imágenes empieza bastante por
# encima de este valor.
UMBRAL_FONDO = 8

# Fracción mínima de píxeles con señal que debe tener una fila o columna para
# considerarla parte del campo irradiado. Un valor pequeño evita que una
# franja de ruido aislado extienda el recorte hasta el borde.
FRACCION_MINIMA = 0.02

# Margen que se conserva alrededor del campo detectado, en fracción de la
# dimensión correspondiente. Protege contra recortar el borde del tórax si la
# transición a fondo es gradual.
MARGEN = 0.01

# Si el recorte dejara menos de esta fracción de la imagen, se descarta por
# considerarse un fallo de detección y se devuelve la imagen completa.
AREA_MINIMA = 0.10

# Radio del elemento estructurante que cierra los huecos de la máscara de
# campo. Una estructura muy densa, como la columna vertebral vista de perfil,
# puede acercarse al nivel de fondo en algunos píxeles aislados; el cierre
# impide que esos píxeles abran agujeros dentro del paciente.
RADIO_CIERRE = 0


@dataclass(frozen=True)
class Ventana:
    """Rectángulo de recorte, en coordenadas de la imagen original."""

    y0: int
    y1: int
    x0: int
    x1: int
    aplicado: bool
    razon: str

    @property
    def alto(self) -> int:
        return self.y1 - self.y0

    @property
    def ancho(self) -> int:
        return self.x1 - self.x0

    def as_dict(self) -> dict[str, object]:
        return {
            "crop_y0": self.y0,
            "crop_y1": self.y1,
            "crop_x0": self.x0,
            "crop_x1": self.x1,
            "crop_alto": self.alto,
            "crop_ancho": self.ancho,
            "crop_aplicado": int(self.aplicado),
            "crop_razon": self.razon,
        }


def detectar_ventana(
    gris: NDArray[np.generic],
    umbral: int = UMBRAL_FONDO,
    fraccion_minima: float = FRACCION_MINIMA,
    margen: float = MARGEN,
    area_minima: float = AREA_MINIMA,
) -> Ventana:
    """Localiza el rectángulo que contiene el campo irradiado.

    Se decide fila por fila y columna por columna en lugar de usar el
    rectángulo mínimo que contiene todo píxel con señal: una marca aislada o
    un píxel caliente del detector extendería ese rectángulo hasta el borde y
    el recorte no haría nada.

    Devuelve siempre una ventana válida. Si la detección fallara y dejara una
    región demasiado pequeña, se devuelve la imagen completa con la razón
    registrada, en lugar de producir un recorte silenciosamente incorrecto.
    """
    gris = np.asarray(gris)
    if gris.ndim != 2:
        raise ValueError("Se requiere una imagen de un solo canal")

    alto, ancho = gris.shape
    con_senal = gris > umbral

    filas = con_senal.mean(axis=1) >= fraccion_minima
    columnas = con_senal.mean(axis=0) >= fraccion_minima

    if not filas.any() or not columnas.any():
        return Ventana(0, alto, 0, ancho, False, "sin_senal_detectada")

    idx_f = np.flatnonzero(filas)
    idx_c = np.flatnonzero(columnas)

    my = round(alto * margen)
    mx = round(ancho * margen)

    y0 = max(0, int(idx_f[0]) - my)
    y1 = min(alto, int(idx_f[-1]) + 1 + my)
    x0 = max(0, int(idx_c[0]) - mx)
    x1 = min(ancho, int(idx_c[-1]) + 1 + mx)

    if (y1 - y0) * (x1 - x0) < area_minima * alto * ancho:
        return Ventana(0, alto, 0, ancho, False, "ventana_demasiado_pequena")

    if y0 == 0 and y1 == alto and x0 == 0 and x1 == ancho:
        return Ventana(0, alto, 0, ancho, False, "sin_area_no_irradiada")

    return Ventana(y0, y1, x0, x1, True, "recortado")


def aplicar(arreglo: NDArray[np.generic], ventana: Ventana) -> NDArray[np.generic]:
    """Aplica una ventana a un arreglo 2D.

    La misma ventana se aplica a la imagen y a sus máscaras, de modo que la
    correspondencia píxel a píxel se conserva después del recorte.
    """
    arreglo = np.asarray(arreglo)
    if arreglo.ndim != 2:
        raise ValueError("Se requiere un arreglo de dos dimensiones")
    return arreglo[ventana.y0 : ventana.y1, ventana.x0 : ventana.x1]


def recortar(
    gris: NDArray[np.generic],
    *mascaras: NDArray[np.generic],
) -> tuple[NDArray[np.generic], list[NDArray[np.generic]], Ventana]:
    """Recorta una imagen y sus máscaras con una ventana común.

    La ventana se detecta sobre la imagen, nunca sobre la máscara: la máscara
    de referencia no está disponible cuando se procesa un caso nuevo, de modo
    que usarla para decidir el recorte haría el procedimiento irreproducible
    fuera del conjunto etiquetado.
    """
    ventana = detectar_ventana(gris)
    recortada = aplicar(gris, ventana)

    salida = []
    for mascara in mascaras:
        mascara = np.asarray(mascara)
        if mascara.shape != gris.shape:
            raise ValueError(
                f"La mascara {mascara.shape} no coincide con la imagen {gris.shape}"
            )
        salida.append(aplicar(mascara, ventana))

    return recortada, salida, ventana


def fraccion_fondo(gris: NDArray[np.generic], umbral: int = UMBRAL_FONDO) -> float:
    """Fracción de píxeles por debajo del umbral de fondo."""
    return float((np.asarray(gris) <= umbral).mean())

def mascara_campo(
    gris: NDArray[np.generic],
    umbral: int = UMBRAL_FONDO,
    radio_cierre: int = RADIO_CIERRE,
) -> NDArray[np.bool_]:
    """ Delimita el campo irradiado como region, no como rectangulo.

    El area irradiada de una radiografia colimada no es rectangular: la
    colimacion deja esquinas oscuras y bordes curvos, y una banda negra
    inclinada cuando el detector no quedo alineado. Recortar al rectangulo
    envolvente deja entre un 20 y un 32 por ciento de fondo dentro de la
    region conservada, de modo que la diferencia entre conjuntos persiste.

    Esta funcion devuelve la region con senal como mascara booleana. Sobre
    ella se calculan los histogramas, el contraste y el denominador de la
    fraccion de area, de forma que esas medidas dependan de la anatomia y no
    de cuanto colimo el tecnico.

    El procedimiento es: umbralizar por encima del nivel de fondo, conservar
    la componente conexa mayor, que es el paciente, y cerrar los huecos
    internos que deje una estructura muy atenuante.
    """
    gris = np.asarray(gris)
    if gris.ndim != 2:
        raise ValueError('Se requiere una imagen de un solo canal')

    con_senal = gris > umbral
    if not con_senal.any():
        return np.zeros(gris.shape, dtype=bool)

    etiquetas, n = ndimage.label(con_senal)
    if n > 1:
        # La componente mayor es el paciente. Las menores son marcas de plomo
        # aisladas sobre el fondo o ruido del detector.
        tamanos = ndimage.sum_labels(con_senal, etiquetas, index=range(1, n + 1))
        mayor = int(np.argmax(tamanos)) + 1
        con_senal = etiquetas == mayor

    # No se rellenan huecos ni se aplica cierre morfologico. Las axilas de un
    # paciente delgado atenuan tan poco que el detector las registra al nivel
    # del fondo, y quedan rodeadas por el borde brillante del campo irradiado.
    # Rellenarlas las incorporaria a la region, y entonces el denominador de
    # la fraccion de area pasaria a medir cuanto espacio dejo el colimador en
    # lugar de cuanto cuerpo tiene el paciente. En dos de las imagenes mas
    # colimadas de Montgomery ese relleno anadia hasta 15 puntos de area sin
    # senal dentro de la region.
    #
    # El parametro radio_cierre se conserva en la firma por compatibilidad y
    # para permitir activarlo en un conjunto que lo requiera, pero su valor
    # por defecto es cero.
    if radio_cierre > 0:
        con_senal = ndimage.binary_closing(con_senal, structure=_disco(radio_cierre))

    return con_senal


def _disco(radio: int) -> NDArray[np.bool_]:
    """ Elemento estructurante circular de radio dado. """
    y, x = np.ogrid[-radio : radio + 1, -radio : radio + 1]
    return (y * y + x * x) <= radio * radio


def fraccion_campo(gris: NDArray[np.generic], umbral: int = UMBRAL_FONDO) -> float:
    """ Fraccion de la imagen que ocupa el campo irradiado. """
    return float(mascara_campo(gris, umbral).mean())