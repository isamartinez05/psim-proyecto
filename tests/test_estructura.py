"""Pruebas de la estructura y los documentos del proyecto.

Verifican que el repositorio cumple lo que el README y la guia del curso
declaran: estructura de carpetas, columnas de los documentos de la Fase 1 y
coherencia entre las cifras del informe y los archivos que las sustentan.

Ejecutar con:
    uv run pytest
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
DOCS = RAIZ / "docs"


# --- Estructura minima exigida por la guia (seccion 14.2) -----------------

CARPETAS = [
    "data/raw",
    "data/processed",
    "docs",
    "figures/fase1",
    "figures/fase2",
    "figures/fase3",
    "results/fase1",
    "results/fase2",
    "results/fase3",
    "scripts",
    "src/psim",
    "tests",
]

ARCHIVOS = [
    ".python-version",
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "data/raw/README.md",
    "docs/formulacion_proyecto.tex",
    "docs/dataset_card.md",
    "docs/datasets_candidatos.csv",
    "docs/matriz_tecnicas.csv",
    "docs/cronograma.csv",
    "scripts/check_dataset.py",
    "scripts/reproduce.py",
    "src/psim/__init__.py",
]


@pytest.mark.parametrize("carpeta", CARPETAS)
def test_carpeta_existe(carpeta: str) -> None:
    assert (RAIZ / carpeta).is_dir(), f"falta la carpeta {carpeta}"


@pytest.mark.parametrize("archivo", ARCHIVOS)
def test_archivo_existe(archivo: str) -> None:
    ruta = RAIZ / archivo
    assert ruta.is_file(), f"falta el archivo {archivo}"
    assert ruta.stat().st_size > 0, f"{archivo} esta vacio"


# --- Ningun archivo de texto debe llevar marca de orden de bytes ----------

EXTENSIONES = {".toml", ".csv", ".md", ".py", ".tex", ".bib"}


def test_sin_bom() -> None:
    """El BOM rompe pandas.read_csv y la configuracion de pytest."""
    con_bom = []
    for ruta in RAIZ.rglob("*"):
        if not ruta.is_file() or ruta.suffix not in EXTENSIONES:
            continue
        if any(p in ruta.parts for p in (".venv", ".git", "__pycache__")):
            continue
        if ruta.read_bytes()[:3] == b"\xef\xbb\xbf":
            con_bom.append(str(ruta.relative_to(RAIZ)))
    assert not con_bom, f"archivos con BOM: {con_bom}"


# --- Documentos de la Fase 1 ---------------------------------------------

def _leer(nombre: str) -> list[dict]:
    with (DOCS / nombre).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_datasets_candidatos() -> None:
    """La guia (seccion 10.1) fija dieciseis columnas y exige tres candidatos."""
    columnas = [
        "nombre", "fuente", "url", "modalidad", "formato", "n_sujetos",
        "n_imagenes", "variable_objetivo", "metadatos", "licencia",
        "tamano_descarga", "version", "fecha_consulta", "ventajas",
        "limitaciones", "decision",
    ]
    filas = _leer("datasets_candidatos.csv")
    assert list(filas[0].keys()) == columnas
    assert len(filas) >= 3, "la guia exige al menos tres candidatos"
    assert any(f["decision"] == "seleccionado" for f in filas)
    for f in filas:
        vacios = [k for k, v in f.items() if not v or not v.strip()]
        assert not vacios, f"{f['nombre']}: campos vacios {vacios}"


def test_matriz_tecnicas() -> None:
    """La guia (seccion 12) fija once columnas y nueve bloques tecnicos."""
    columnas = [
        "bloque", "tecnica", "fase", "entrada", "parametros", "producto",
        "metrica", "decision_esperada", "archivo_codigo", "figura_o_tabla",
        "estado",
    ]
    filas = _leer("matriz_tecnicas.csv")
    assert list(filas[0].keys()) == columnas

    bloques = {
        "Representacion y digitalizacion",
        "Espacios e intensidad",
        "Operaciones de vecindad",
        "Histogramas",
        "Umbralizacion y morfologia",
        "Dominio de frecuencia",
        "Wavelets y multirresolucion",
        "Caracteristicas y estadistica",
    }
    presentes = {f["bloque"] for f in filas}
    assert bloques <= presentes, f"bloques ausentes: {bloques - presentes}"

    for f in filas:
        assert f["fase"] in {"1", "2", "3"}, f"{f['tecnica']}: fase invalida"
        assert f["metrica"].strip(), f"{f['tecnica']}: sin metrica"
        assert f["decision_esperada"].strip(), f"{f['tecnica']}: sin criterio"


def test_cronograma() -> None:
    """La guia (seccion 13.1) exige quince semanas que cierran en el Lab 03."""
    filas = _leer("cronograma.csv")
    semanas = [int(f["semana"]) for f in filas]
    assert semanas == list(range(1, 16)), "deben ser las semanas 1 a 15"
    assert {f["fase"] for f in filas} == {"1", "2", "3"}
    for f in filas:
        assert f["responsable"].strip(), f"semana {f['semana']}: sin responsable"
        assert f["evidencia"].strip(), f"semana {f['semana']}: sin evidencia"
    ultima = filas[-1]
    assert "Laboratorio 03" in ultima["actividad"]


# --- Coherencia entre el informe y los archivos que lo sustentan ----------

def test_informe_coincide_con_la_matriz() -> None:
    """El informe afirma 53 tecnicas repartidas en 14, 30 y 9 por fase."""
    filas = _leer("matriz_tecnicas.csv")
    por_fase = {f: sum(1 for x in filas if x["fase"] == f) for f in "123"}
    tex = (DOCS / "formulacion_proyecto.tex").read_text(encoding="utf-8")

    assert f"De las {len(filas)} técnicas" in tex
    esperado = f"{por_fase['1']} corresponden a la Fase 1, {por_fase['2']} a la Fase 2 y {por_fase['3']}"
    assert esperado in tex, f"el informe no coincide con la matriz: {por_fase}"


def test_scripts_son_importables() -> None:
    """Los guiones deben compilar sin errores de sintaxis."""
    import py_compile

    for nombre in ("check_dataset.py", "reproduce.py"):
        py_compile.compile(str(RAIZ / "scripts" / nombre), doraise=True)
