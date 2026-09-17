"""Reconstruye todos los productos derivados desde data/raw/.

Este script es el protocolo de reproduccion del proyecto. Ejecuta las tres
fases en orden, regenerando todo lo que hay en data/processed/, results/ y
figures/ a partir de los datos originales sin modificar.

Nada de lo que produce este script se versiona en el repositorio salvo el
manifiesto de datos (results/fase1/manifiesto_datos.csv). Todo lo demas se
regenera ejecutando:

    uv run python scripts/reproduce.py

Requisitos previos:
    - Los datos originales deben estar en data/raw/ segun data/raw/README.md
    - La verificacion de integridad debe haber pasado:
        uv run python scripts/check_dataset.py

Uso:
    uv run python scripts/reproduce.py           # las tres fases
    uv run python scripts/reproduce.py --fase 1   # solo Fase 1
    uv run python scripts/reproduce.py --fase 2   # solo Fase 2 (requiere Fase 1)
    uv run python scripts/reproduce.py --fase 3   # solo Fase 3 (requiere Fase 2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"


def verificar_datos() -> bool:
    """Comprueba que data/raw/ contiene las carpetas esperadas."""
    carpetas = [
        RAW / "MontgomerySet" / "CXR_png",
        RAW / "ChinaSet_AllFiles" / "CXR_png",
    ]
    for c in carpetas:
        if not c.is_dir():
            print(
                f"ERROR: no se encuentra {c.relative_to(RAIZ)}",
                file=sys.stderr,
            )
            print(
                "Ejecute primero: uv run python scripts/check_dataset.py",
                file=sys.stderr,
            )
            return False
    return True


def fase_1() -> None:
    """Fase 1: caracterizacion de la adquisicion y los datos."""
    # TODO Fase 1: implementar en semanas 4-5
    # - Lectura de metadatos (src/psim/io.py)
    # - Correccion de escala invertida Shenzhen (src/psim/io.py)
    # - Conversion RGB a gris (src/psim/color.py)
    # - Normalizacion de rango dinamico (src/psim/color.py)
    # - Analisis de vecindad y conectividad (src/psim/geometry.py)
    # - Histogramas y comparacion entre origenes (src/psim/histogram.py)
    # - Productos: results/fase1/ y figures/fase1/
    raise NotImplementedError(
        "Fase 1 no implementada. Se desarrolla en semanas 4-5."
    )


def fase_2() -> None:
    """Fase 2: procesamiento, segmentacion y extraccion de caracteristicas."""
    # TODO Fase 2: implementar en semanas 6-11
    # - Transformaciones de intensidad (src/psim/color.py)
    # - Filtros espaciales y operadores de borde (src/psim/spatial.py)
    # - Ecualizacion, CLAHE y histogram matching (src/psim/histogram.py)
    # - Umbralizacion y morfologia (src/psim/segmentation.py)
    # - Dominio de frecuencia (src/psim/frequency.py)
    # - CWT sobre perfil 1D (src/psim/wavelets.py)
    # - Productos: results/fase2/ y figures/fase2/
    raise NotImplementedError(
        "Fase 2 no implementada. Se desarrolla en semanas 6-11."
    )


def fase_3() -> None:
    """Fase 3: integracion, validacion y respuesta a la pregunta."""
    # TODO Fase 3: implementar en semanas 12-15
    # - DWT 2D y energia por subbanda (src/psim/wavelets.py)
    # - Descriptores morfologicos (src/psim/features.py)
    # - Error relativo frente a referencia (src/psim/features.py)
    # - Prueba de hipotesis y analisis de sensibilidad (src/psim/stats.py)
    # - Productos: results/fase3/ y figures/fase3/
    raise NotImplementedError(
        "Fase 3 no implementada. Se desarrolla en semanas 12-15."
    )


FASES = {1: fase_1, 2: fase_2, 3: fase_3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="ejecutar solo una fase (por defecto las tres en orden)",
    )
    args = parser.parse_args()

    if not verificar_datos():
        return 2

    PROCESSED.mkdir(parents=True, exist_ok=True)

    fases_a_correr = [args.fase] if args.fase else [1, 2, 3]

    for n in fases_a_correr:
        print(f"\n{'='*60}")
        print(f"  Fase {n}")
        print(f"{'='*60}\n")
        try:
            FASES[n]()
        except NotImplementedError as e:
            print(f"  {e}", file=sys.stderr)
            if args.fase:
                return 1
            print("  (continuando con la siguiente fase)\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
