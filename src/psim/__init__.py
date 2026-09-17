"""Paquete del proyecto PSIM.

Estabilidad de los descriptores morfologicos del campo pulmonar frente a
estrategias de preprocesamiento y segmentacion en radiografias de torax.

Este modulo expone las rutas del proyecto como objetos ``Path`` relativos a
la raiz del repositorio, para que ningun otro modulo tenga que construirlas
a mano ni dependa del directorio desde el que se ejecute.
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

# Raiz del repositorio: src/psim/__init__.py -> src/psim -> src -> raiz
RAIZ = Path(__file__).resolve().parent.parent.parent

DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
DOCS = RAIZ / "docs"
FIGURES = RAIZ / "figures"
RESULTS = RAIZ / "results"

# Conjuntos originales. Los nombres son los de origen: no se renombra nada
# al descomprimir. Ver data/raw/README.md.
MONTGOMERY = DATA_RAW / "MontgomerySet"
SHENZHEN = DATA_RAW / "ChinaSet_AllFiles"
SHENZHEN_MASCARAS = DATA_RAW / "shcxr-lung-mask" / "mask"

ORIGENES = ("Montgomery", "Shenzhen")


def figura(fase: int, nombre: str) -> Path:
    """Ruta de una figura de la fase indicada, creando la carpeta si falta."""
    destino = FIGURES / f"fase{fase}"
    destino.mkdir(parents=True, exist_ok=True)
    return destino / nombre


def resultado(fase: int, nombre: str) -> Path:
    """Ruta de una tabla de resultados de la fase indicada."""
    destino = RESULTS / f"fase{fase}"
    destino.mkdir(parents=True, exist_ok=True)
    return destino / nombre


def derivado(*partes: str) -> Path:
    """Ruta de un dato derivado bajo data/processed/.

    Los datos originales son de solo lectura: toda transformacion se escribe
    aqui o en results/ y figures/, nunca sobre data/raw/.
    """
    destino = DATA_PROCESSED.joinpath(*partes)
    destino.parent.mkdir(parents=True, exist_ok=True)
    return destino


__all__ = [
    "DATA_PROCESSED",
    "DATA_RAW",
    "DOCS",
    "FIGURES",
    "MONTGOMERY",
    "ORIGENES",
    "RAIZ",
    "RESULTS",
    "SHENZHEN",
    "SHENZHEN_MASCARAS",
    "derivado",
    "figura",
    "resultado",
]
