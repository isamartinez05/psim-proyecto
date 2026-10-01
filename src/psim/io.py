"""Lectura y caracterización de las radiografías del proyecto.

Este módulo resuelve tres problemas que los dos conjuntos plantean de forma
distinta y que condicionan todo el procesamiento posterior:

1. Montgomery entrega PNG de 12 bits almacenados en contenedor de 16 bits.
   El ``dtype`` del archivo dice ``uint16`` y no demuestra que el sensor haya
   usado 16 bits efectivos, de modo que la profundidad real se estima del
   valor máximo observado y se registra por separado.

2. Shenzhen entrega PNG de 8 bits codificados en tres canales RGB. Se verifica
   si los tres canales son idénticos antes de descartar dos de ellos, en lugar
   de asumirlo.

3. La documentación publicada describe las imágenes de Shenzhen con la escala
   de grises invertida. La caracterización de Fase 1 comprobó que los archivos
   distribuidos no lo están: las 800 imágenes de ambos conjuntos siguen la
   convención radiográfica. La detección se conserva como control de calidad
   del manifiesto, no como paso de corrección.

Ninguna función de este módulo modifica los archivos originales: todas leen
de ``data/raw/`` y devuelven arreglos en memoria.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from skimage import io as skio

# Fracción de la imagen que se toma de cada esquina para estimar el fondo.
FRACCION_ESQUINA = 0.08

TAMANO_BLOQUE = 1 << 20


@dataclass(frozen=True)
class PropiedadesImagen:
    """Propiedades observables de un archivo de imagen, sin interpretarlas.

    Los campos reproducen lo que el archivo declara y lo que se mide sobre sus
    valores. La distinción importa: ``dtype`` es lo declarado y
    ``bits_efectivos`` es lo estimado, y pueden no coincidir.
    """

    ruta: Path
    height: int
    width: int
    channels: int
    dtype: str
    bits_efectivos: int
    valor_min: float
    valor_max: float
    media: float
    desviacion: float
    mediana: float
    no_finitos: int
    canales_identicos: bool | None
    escala_invertida: bool
    sha256: str = field(default="", repr=False)

    def as_dict(self) -> dict[str, object]:
        """Devuelve las propiedades como diccionario plano, apto para CSV."""
        d = {k: v for k, v in self.__dict__.items()}
        d["ruta"] = self.ruta.as_posix()
        return d


def sha256(ruta: Path) -> str:
    """Hash SHA-256 del archivo, leído por bloques."""
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        while bloque := f.read(TAMANO_BLOQUE):
            h.update(bloque)
    return h.hexdigest()


def bits_efectivos(valores: NDArray[np.generic]) -> int:
    """Estima cuántos bits ocupa realmente el rango de valores.

    Un PNG de 16 bits cuyo máximo es 4095 usa 12 bits efectivos. Devolver el
    tamaño del contenedor ocultaría esa diferencia, que afecta a cualquier
    normalización posterior.
    """
    if not np.issubdtype(valores.dtype, np.integer):
        return 0
    maximo = int(valores.max())
    if maximo <= 0:
        return 0
    return int(maximo).bit_length()


def canales_son_identicos(imagen: NDArray[np.generic]) -> bool:
    """Comprueba si los canales de una imagen multicanal llevan lo mismo.

    Shenzhen distribuye escala de grises replicada en tres canales RGB. Antes
    de quedarse con un solo canal hay que verificarlo: convertir a gris con
    pesos de luminancia una imagen ya gris introduce un error de redondeo
    innecesario.
    """
    if imagen.ndim != 3 or imagen.shape[2] < 2:
        return False
    primero = imagen[..., 0]
    return all(np.array_equal(primero, imagen[..., c]) for c in range(1, imagen.shape[2]))


def a_gris(imagen: NDArray[np.generic]) -> NDArray[np.generic]:
    """Reduce una imagen a un solo canal conservando el tipo original.

    Si los canales son idénticos se toma el primero, sin aritmética. Solo
    cuando difieren se aplica la conversión por luminancia, que sí altera los
    valores y por eso se registra como decisión.
    """
    imagen = np.asarray(imagen)
    if imagen.ndim == 2:
        return imagen
    if imagen.ndim != 3:
        raise ValueError(f"Se esperaba 2D o 3D, se recibió {imagen.ndim}D")
    if imagen.shape[2] == 4:
        imagen = imagen[..., :3]
    if canales_son_identicos(imagen):
        return imagen[..., 0]
    pesos = np.array([0.2125, 0.7154, 0.0721], dtype=np.float64)
    gris = (imagen[..., :3].astype(np.float64) * pesos).sum(axis=2)
    return gris.astype(imagen.dtype)


def detectar_inversion(gris: NDArray[np.generic]) -> bool:
    """Decide si la escala de grises está invertida respecto a la convención.

    En una radiografía con la convención habitual el aire que rodea al paciente
    es lo más oscuro de la imagen y ocupa las esquinas. Se mide en qué posición
    del rango de intensidades cae ese fondo: cerca del extremo inferior indica
    convención normal, cerca del superior indica escala invertida.

    El criterio se basa en la imagen misma y no en su procedencia, de modo que
    detecta también un archivo mal etiquetado dentro de un conjunto correcto.
    """
    gris = np.asarray(gris)
    if gris.ndim != 2:
        raise ValueError("Se requiere una imagen de un solo canal")

    alto, ancho = gris.shape
    valores = gris.astype(np.float64)

    # Banda horizontal a la altura de los campos pulmonares, excluyendo
    # el tercio inferior donde empiezan el diafragma y el abdomen.
    y0, y1 = int(alto * 0.22), int(alto * 0.62)

    # Franja central: mediastino, corazón y columna vertebral.
    cx0, cx1 = int(ancho * 0.42), int(ancho * 0.58)
    mediastino = valores[y0:y1, cx0:cx1]

    # Franjas laterales: campos pulmonares izquierdo y derecho.
    lx = int(ancho * 0.16)
    pulmones = np.concatenate([
        valores[y0:y1, lx:cx0].ravel(),
        valores[y0:y1, cx1:ancho - lx].ravel(),
    ])

    if pulmones.size == 0 or mediastino.size == 0:
        return False

    # En convención radiográfica el aire del pulmón atenúa poco y se registra
    # oscuro, mientras que el mediastino atenúa mucho y se registra claro. Si
    # esa relación aparece al revés, la escala está invertida.
    #
    # El criterio no usa las esquinas: una radiografía bien colimada puede no
    # tener aire fuera del paciente, y los marcadores de plomo que suelen
    # ocupar esas zonas son de las estructuras más brillantes de la imagen.
    return float(np.median(pulmones)) > float(np.median(mediastino))


def invertir(gris: NDArray[np.generic]) -> NDArray[np.generic]:
    """Invierte la escala conservando el rango del tipo de dato."""
    gris = np.asarray(gris)
    if np.issubdtype(gris.dtype, np.integer):
        info = np.iinfo(gris.dtype)
        return (info.max - gris).astype(gris.dtype)
    return (float(gris.max()) - gris).astype(gris.dtype)


def leer(ruta: Path | str) -> NDArray[np.generic]:
    """Lee un archivo de imagen sin alterar sus valores."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta}")
    return skio.imread(ruta)


def leer_gris(ruta: Path | str, corregir_inversion: bool = False) -> NDArray[np.generic]:
    """Lee una imagen y la reduce a un solo canal.

    Es la entrada estándar del resto del proyecto.

    La corrección de inversión está desactivada por defecto. La caracterización
    de Fase 1 verificó las 800 imágenes de ambos conjuntos y ninguna está
    invertida, de modo que aplicarla no corrige nada y sí arriesga dañar una
    imagen correcta si el criterio da un falso positivo. La documentación
    publicada del conjunto Shenzhen describe una escala invertida que los
    archivos distribuidos no presentan.

    El parámetro se conserva para el caso en que se incorpore un conjunto que
    sí la requiera, y la detección sigue registrándose en el manifiesto como
    control de calidad.
    """
    gris = a_gris(leer(ruta))
    if corregir_inversion and detectar_inversion(gris):
        gris = invertir(gris)
    return gris


def describir(ruta: Path | str, calcular_hash: bool = False) -> PropiedadesImagen:
    """Extrae las propiedades observables de un archivo, sin corregirlo.

    Las estadísticas se calculan sobre la imagen reducida a un canal pero
    **antes** de corregir la inversión, para que el manifiesto registre lo que
    el archivo contiene y no lo que el proyecto decidió hacer con él.
    """
    ruta = Path(ruta)
    crudo = leer(ruta)

    if crudo.ndim == 2:
        canales = 1
        identicos: bool | None = None
    elif crudo.ndim == 3:
        canales = int(crudo.shape[2])
        identicos = canales_son_identicos(crudo)
    else:
        raise ValueError(f"Dimensionalidad no soportada: {crudo.ndim}D")

    gris = a_gris(crudo)
    finitos = np.isfinite(gris) if np.issubdtype(gris.dtype, np.floating) else None
    n_no_finitos = 0 if finitos is None else int((~finitos).sum())
    valores = gris.astype(np.float64)
    if finitos is not None:
        valores = valores[finitos]

    return PropiedadesImagen(
        ruta=ruta,
        height=int(gris.shape[0]),
        width=int(gris.shape[1]),
        channels=canales,
        dtype=str(crudo.dtype),
        bits_efectivos=bits_efectivos(gris),
        valor_min=float(valores.min()) if valores.size else float("nan"),
        valor_max=float(valores.max()) if valores.size else float("nan"),
        media=float(valores.mean()) if valores.size else float("nan"),
        desviacion=float(valores.std(ddof=0)) if valores.size else float("nan"),
        mediana=float(np.median(valores)) if valores.size else float("nan"),
        no_finitos=n_no_finitos,
        canales_identicos=identicos,
        escala_invertida=detectar_inversion(gris),
        sha256=sha256(ruta) if calcular_hash else "",
    )