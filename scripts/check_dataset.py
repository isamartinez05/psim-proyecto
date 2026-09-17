"""Verifica la copia local de los datos originales y genera un manifiesto.

Comprueba que la estructura de ``data/raw/`` coincide con la documentada en
``data/raw/README.md``, cuenta los archivos de cada carpeta, calcula el hash
SHA-256 de cada uno y escribe ``results/fase1/manifiesto_datos.csv``.

El manifiesto se versiona en el repositorio: permite comprobar que dos copias
del conjunto son idénticas sin necesidad de redistribuir las imágenes.

Uso:
    uv run python scripts/check_dataset.py
    uv run python scripts/check_dataset.py --sin-hash
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
SALIDA = RAIZ / "results" / "fase1" / "manifiesto_datos.csv"

TAMANO_BLOQUE = 1 << 20


@dataclass(frozen=True)
class Coleccion:
    """Una carpeta de datos originales con su conteo esperado."""

    clave: str
    origen: str
    ruta_relativa: str
    patron: str
    n_esperado: int
    bytes_esperados: int | None = None

    @property
    def ruta(self) -> Path:
        return RAW.joinpath(*self.ruta_relativa.split("/"))


COLECCIONES: tuple[Coleccion, ...] = (
    Coleccion(
        clave="montgomery_imagenes",
        origen="Montgomery",
        ruta_relativa="MontgomerySet/CXR_png",
        patron="*.png",
        n_esperado=138,
        bytes_esperados=614_034_765,
    ),
    Coleccion(
        clave="montgomery_mascara_izq",
        origen="Montgomery",
        ruta_relativa="MontgomerySet/ManualMask/leftMask",
        patron="*.png",
        n_esperado=138,
    ),
    Coleccion(
        clave="montgomery_mascara_der",
        origen="Montgomery",
        ruta_relativa="MontgomerySet/ManualMask/rightMask",
        patron="*.png",
        n_esperado=138,
    ),
    Coleccion(
        clave="montgomery_lecturas",
        origen="Montgomery",
        ruta_relativa="MontgomerySet/ClinicalReadings",
        patron="*.txt",
        n_esperado=138,
    ),
    Coleccion(
        clave="shenzhen_imagenes",
        origen="Shenzhen",
        ruta_relativa="ChinaSet_AllFiles/CXR_png",
        patron="*.png",
        n_esperado=662,
        bytes_esperados=3_772_099_214,
    ),
    Coleccion(
        clave="shenzhen_lecturas",
        origen="Shenzhen",
        ruta_relativa="ChinaSet_AllFiles/ClinicalReadings",
        patron="*.txt",
        n_esperado=662,
    ),
    Coleccion(
        clave="shenzhen_mascaras",
        origen="Shenzhen",
        ruta_relativa="shcxr-lung-mask/mask",
        patron="*.png",
        n_esperado=566,
    ),
)

# Nombres alternativos frecuentes, para dar un mensaje util en vez de
# "no existe" cuando alguien renombro una carpeta al descomprimir.
ALIAS_CONOCIDOS: dict[str, tuple[str, ...]] = {
    "ChinaSet_AllFiles": ("ShenzhenSet", "Shenzhen", "China", "ChinaSet"),
    "MontgomerySet": ("Montgomery", "MontgomeryCounty", "NLM-MontgomeryCXRSet"),
    "shcxr-lung-mask": ("ShenzhenMasks", "mascaras_shenzhen", "shcxr_lung_mask"),
}


def sha256(ruta: Path) -> str:
    """Hash SHA-256 del archivo, leido por bloques para no cargarlo entero."""
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        while bloque := f.read(TAMANO_BLOQUE):
            h.update(bloque)
    return h.hexdigest()


def diagnosticar_ausencia(coleccion: Coleccion) -> str:
    """Explica por que falta una carpeta, buscando renombrados frecuentes."""
    partes = coleccion.ruta_relativa.split("/")
    primera = partes[0]

    if (RAW / primera).is_dir():
        return f"existe {primera}/ pero no la subcarpeta {'/'.join(partes[1:])}"

    for alias in ALIAS_CONOCIDOS.get(primera, ()):
        if (RAW / alias).is_dir():
            return (
                f"se encontro {alias}/ en lugar de {primera}/. "
                f"Las carpetas conservan el nombre de origen: renombre a {primera}"
            )

    presentes = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    if not presentes:
        return "data/raw/ esta vacio; falta descargar los datos"
    return f"no existe {primera}/. Carpetas presentes: {', '.join(presentes)}"


def revisar(coleccion: Coleccion, con_hash: bool) -> tuple[list[dict], list[str]]:
    """Devuelve las filas del manifiesto y los problemas encontrados."""
    problemas: list[str] = []

    if not coleccion.ruta.is_dir():
        problemas.append(f"[{coleccion.clave}] {diagnosticar_ausencia(coleccion)}")
        return [], problemas

    archivos = sorted(coleccion.ruta.glob(coleccion.patron))
    n = len(archivos)

    if n != coleccion.n_esperado:
        problemas.append(
            f"[{coleccion.clave}] se esperaban {coleccion.n_esperado} archivos "
            f"{coleccion.patron} y se encontraron {n}"
        )

    filas: list[dict] = []
    total_bytes = 0
    for archivo in archivos:
        tam = archivo.stat().st_size
        total_bytes += tam
        filas.append(
            {
                "coleccion": coleccion.clave,
                "origen": coleccion.origen,
                "archivo": archivo.relative_to(RAW).as_posix(),
                "bytes": tam,
                "sha256": sha256(archivo) if con_hash else "",
            }
        )

    if (
        coleccion.bytes_esperados is not None
        and total_bytes != coleccion.bytes_esperados
    ):
        delta = total_bytes - coleccion.bytes_esperados
        problemas.append(
            f"[{coleccion.clave}] el total es {total_bytes:,} bytes y se esperaban "
            f"{coleccion.bytes_esperados:,} (diferencia de {delta:+,})"
        )

    return filas, problemas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sin-hash",
        action="store_true",
        help="omite el calculo de SHA-256 (rapido, solo verifica conteos y tamanos)",
    )
    args = parser.parse_args()
    con_hash = not args.sin_hash

    if not RAW.is_dir():
        print(f"ERROR: no existe {RAW}", file=sys.stderr)
        print("Consulte data/raw/README.md para obtener los datos.", file=sys.stderr)
        return 2

    if con_hash:
        print("Calculando hashes SHA-256. Sobre 4 GB esto toma varios minutos.\n")

    todas_las_filas: list[dict] = []
    todos_los_problemas: list[str] = []

    for coleccion in COLECCIONES:
        filas, problemas = revisar(coleccion, con_hash)
        todas_las_filas.extend(filas)
        todos_los_problemas.extend(problemas)

        n = len(filas)
        marca = "OK  " if not problemas else "FALLA"
        total = sum(f["bytes"] for f in filas)
        print(f"{marca} {coleccion.clave:26} {n:4} archivos  {total / 1024**3:6.2f} GB")

    print()

    if todos_los_problemas:
        print("Problemas encontrados:\n", file=sys.stderr)
        for p in todos_los_problemas:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nNo se escribio el manifiesto. Corrija la copia local y vuelva a "
            "ejecutar.\nConsulte data/raw/README.md.",
            file=sys.stderr,
        )
        return 1

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(
            f, fieldnames=["coleccion", "origen", "archivo", "bytes", "sha256"]
        )
        escritor.writeheader()
        escritor.writerows(todas_las_filas)

    total = sum(f["bytes"] for f in todas_las_filas)
    print(
        f"Verificacion correcta: {len(todas_las_filas)} archivos, "
        f"{total / 1024**3:.2f} GB en total."
    )
    print(f"Manifiesto escrito en {SALIDA.relative_to(RAIZ).as_posix()}")
    if not con_hash:
        print("Aviso: se omitieron los hashes; la columna sha256 quedo vacia.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
