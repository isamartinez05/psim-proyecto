"""Reconstruye todos los productos derivados desde data/raw/.

Este guion es el protocolo de reproduccion del proyecto. Encadena las tres
fases en orden y regenera cuanto hay en data/metadata/, data/processed/,
results/ y figures/ a partir de los datos originales, que no se modifican.

Requisitos previos:
    Los datos originales en data/raw/ segun data/raw/README.md.

Uso:
    uv run python scripts/reproduce.py
    uv run python scripts/reproduce.py --rapido
    uv run python scripts/reproduce.py --fase 1
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time

from psim import MONTGOMERY, RAIZ, SHENZHEN, resultado


def verificar_datos() -> bool:
    """Comprueba que los datos originales esten en su sitio."""
    faltan = [
        p for p in (MONTGOMERY / "CXR_png", SHENZHEN / "CXR_png") if not p.is_dir()
    ]
    if faltan:
        for p in faltan:
            print(f"ERROR: no se encuentra {p.relative_to(RAIZ)}", file=sys.stderr)
        print("Consulte data/raw/README.md para obtener los datos.", file=sys.stderr)
        return False
    return True


def correr(modulo: str, descripcion: str) -> float:
    """Ejecuta un guion del proyecto y devuelve cuanto tardo."""
    print(f"\n--- {descripcion} ---")
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / modulo)],
        cwd=RAIZ,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{modulo} termino con codigo {r.returncode}")
    return time.time() - t0


def fase_1() -> None:
    """Inventario, auditoria de calidad y particion congelada."""
    correr("build_manifest.py", "Fase 1: manifiesto de las imagenes fuente")
    correr("inventario.py", "Fase 1: inventario y cuantificacion de calidad")
    correr("build_splits.py", "Fase 1: particion por unidad independiente")
    correr("verificar_splits.py", "Fase 1: verificacion de la particion")


def fase_2() -> None:
    """Exploracion visual y numerica sobre la particion de entrenamiento."""
    correr("eda_exploracion.py", "Fase 2: galeria, dimensiones e intensidad")


def fase_3(rapido: bool) -> None:
    """Tabla de caracteristicas, diccionario, auditoria y decisiones."""
    if not rapido:
        correr("build_features.py", "Fase 3: extraccion de descriptores")
    elif not (RAIZ / "data" / "processed" / "features.csv").is_file():
        raise RuntimeError(
            "No existe features.csv y se pidio modo rapido. "
            "Ejecute sin --rapido al menos una vez."
        )
    else:
        print("\n--- Fase 3: extraccion omitida (--rapido) ---")

    correr("build_dictionary.py", "Fase 3: diccionario y contrato de entrada")
    correr("auditoria.py", "Fase 3: auditoria de la tabla y figuras")
    correr("build_decisiones.py", "Fase 3: registro de decisiones")


def registrar_entorno() -> None:
    """Deja constancia de las versiones con que se produjo el resultado."""
    paquetes = {}
    for nombre in ("numpy", "scipy", "pandas", "matplotlib", "skimage", "pywt"):
        try:
            modulo = __import__(nombre)
            paquetes[nombre] = getattr(modulo, "__version__", "desconocida")
        except ImportError:
            paquetes[nombre] = "no instalado"

    entorno = {
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "paquetes": paquetes,
        "semilla_particion": 20262,
        "semilla_galeria": 20262,
        "lado_trabajo": 512,
    }
    destino = resultado(1, "entorno.json")
    destino.write_text(
        json.dumps(entorno, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nentorno registrado en {destino.relative_to(RAIZ)}")


def resumen_productos() -> None:
    """Lista los artefactos generados con su tamano."""
    esperados = [
        "data/metadata/manifest.csv",
        "data/metadata/splits.csv",
        "data/metadata/feature_dictionary.csv",
        "data/processed/features.csv",
        "config/model_input.json",
        "results/fase1/manifiesto_datos.csv",
        "results/fase2/dice_por_estrategia.csv",
        "results/fase3/auditoria_tabla.csv",
        "results/fase3/decisiones.csv",
    ]
    filas = []
    print("\n--- productos ---")
    for rel in esperados:
        p = RAIZ / rel
        existe = p.is_file()
        n = (
            sum(1 for _ in p.open(encoding="utf-8")) - 1
            if existe and p.suffix == ".csv"
            else ""
        )
        print(f"  {'OK   ' if existe else 'FALTA'} {rel:<42} {n}")
        filas.append({"producto": rel, "existe": int(existe), "filas": n})

    with resultado(3, "productos.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["producto", "existe", "filas"])
        w.writeheader()
        w.writerows(filas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="ejecutar solo una fase",
    )
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="reutilizar features.csv en lugar de recalcularlo",
    )
    args = parser.parse_args()

    if not verificar_datos():
        return 2

    t0 = time.time()
    fases = [args.fase] if args.fase else [1, 2, 3]

    for n in fases:
        if n == 1:
            fase_1()
        elif n == 2:
            fase_2()
        else:
            fase_3(args.rapido)

    registrar_entorno()
    resumen_productos()
    print(f"\nreconstruccion completa en {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
