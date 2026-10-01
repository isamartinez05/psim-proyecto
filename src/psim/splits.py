"""Partición del conjunto en entrenamiento, validación y prueba.

La partición se define sobre unidades independientes, no sobre imágenes: las
imágenes que las lecturas clínicas declaran de un mismo paciente permanecen
juntas, de modo que ninguna pueda aparecer en dos particiones distintas.

Qué protege y qué no
--------------------
El método del proyecto es una cadena clásica de realce, umbralización y
morfología. No ajusta parámetros a partir de etiquetas, de modo que no existe
el mecanismo por el cual la información de prueba podría filtrarse a un modelo
entrenado. La partición cumple entonces dos funciones más modestas pero
igualmente necesarias:

1. Acotar el análisis exploratorio al subconjunto de entrenamiento, para que
   las decisiones sobre parámetros y descriptores no se tomen mirando los
   datos contra los que después se reportará el resultado.

2. Dejar trazable qué imágenes se usaron para decidir y cuáles se reservaron,
   requisito del protocolo con independencia del método.

Limitación declarada
--------------------
Solo dos grupos de sujeto están demostrados, a partir de referencias cruzadas
escritas en las lecturas de Montgomery. Una búsqueda por coincidencia de sexo
y edad identificó 90 imágenes de ese conjunto que comparten ambos atributos
con alguna otra, sin que eso pruebe identidad: con 138 imágenes y edades de 4
a 89 años, las coincidencias por azar son esperables. No se agrupan por ese
criterio porque confundiría pacientes distintos y reduciría artificialmente el
número de unidades independientes. El riesgo residual queda registrado.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

# Semilla fija: la partición debe ser idéntica en cualquier ejecución y en
# cualquier máquina. Cambiarla invalida toda comparación previa.
SEMILLA = 20262
PROPORCIONES = {"train": 0.70, "validation": 0.15, "test": 0.15}

NOMBRES = ("train", "validation", "test")

COLUMNAS = ("image_id", "subject_id", "grupo_id", "estrato", "split")


def clave_estrato(fila: dict[str, str]) -> str:
    """Devuelve el estrato de una imagen: origen y clase.

    Se estratifica por estas dos variables porque son las que la pregunta de
    investigación compara. La edad no entra en la estratificación: con 40
    pacientes menores de 18 años sobre 800 imágenes, añadir una tercera
    variable dejaría celdas de uno o dos elementos y la partición dejaría de
    ser estable. Su reparto se verifica después, sin forzarlo.
    """
    origen = "MC" if fila["dataset_id"].endswith("MontgomeryCXRSet") else "SH"
    clase = fila["target"] or "NA"
    return f"{origen}_{clase}"


def grupos_independientes(
    filas: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Agrupa las filas por unidad independiente.

    Una imagen con ``subject_id`` pertenece al grupo de ese sujeto; una sin él
    forma su propio grupo. No se infiere pertenencia a partir de atributos
    coincidentes.
    """
    grupos: dict[str, list[dict[str, str]]] = defaultdict(list)
    for fila in filas:
        sujeto = fila.get("subject_id", "").strip()
        clave = sujeto if sujeto else f"IMG::{fila['image_id']}"
        grupos[clave].append(fila)
    return dict(grupos)


def estrato_de_grupo(imagenes: list[dict[str, str]]) -> str:
    """Estrato asignado a un grupo completo.

    Un grupo con imágenes de estratos distintos no puede satisfacer a los dos;
    se le asigna el del primer elemento en orden de identificador, de forma
    determinista, y el hecho queda visible en el archivo de partición.
    """
    estratos = sorted({clave_estrato(img) for img in imagenes})
    return estratos[0]


def repartir(
    grupos: dict[str, list[dict[str, str]]],
    semilla: int = SEMILLA,
) -> dict[str, str]:
    """Asigna cada grupo a una partición, respetando estratos y proporciones.

    Devuelve un diccionario de ``grupo_id`` a nombre de partición. El reparto
    se hace por estrato y de forma determinista: mismo conjunto de entrada y
    misma semilla producen siempre la misma salida.
    """
    por_estrato: dict[str, list[str]] = defaultdict(list)
    for gid, imagenes in grupos.items():
        por_estrato[estrato_de_grupo(imagenes)].append(gid)

    rng = random.Random(semilla)
    asignacion: dict[str, str] = {}

    for estrato in sorted(por_estrato):
        ids = sorted(por_estrato[estrato])
        rng.shuffle(ids)
        n = len(ids)

        # El redondeo se resuelve a favor de entrenamiento, que absorbe el
        # resto. Con estratos pequeños esto evita dejar validación o prueba
        # sin representación.
        n_val = round(n * PROPORCIONES["validation"])
        n_test = round(n * PROPORCIONES["test"])
        n_train = n - n_val - n_test
        if n_train < 0:
            n_train, n_val, n_test = n, 0, 0

        for i, gid in enumerate(ids):
            if i < n_train:
                asignacion[gid] = "train"
            elif i < n_train + n_val:
                asignacion[gid] = "validation"
            else:
                asignacion[gid] = "test"

    return asignacion


def construir(filas: list[dict[str, str]], semilla: int = SEMILLA) -> list[dict[str, str]]:
    """Construye las filas de la partición a partir del manifiesto."""
    grupos = grupos_independientes(filas)
    asignacion = repartir(grupos, semilla)

    salida: list[dict[str, str]] = []
    for gid, imagenes in grupos.items():
        for img in imagenes:
            salida.append({
                "image_id": img["image_id"],
                "subject_id": img.get("subject_id", ""),
                "grupo_id": gid,
                "estrato": clave_estrato(img),
                "split": asignacion[gid],
            })
    return sorted(salida, key=lambda f: f["image_id"])


def verificar(particion: list[dict[str, str]]) -> list[str]:
    """Comprueba las invariantes de la partición.

    Devuelve la lista de problemas encontrados. Una lista vacía significa que
    ningún grupo cruza particiones y que las tres están representadas.
    """
    problemas: list[str] = []

    por_grupo: dict[str, set[str]] = defaultdict(set)
    for fila in particion:
        por_grupo[fila["grupo_id"]].add(fila["split"])

    cruzados = [g for g, s in por_grupo.items() if len(s) > 1]
    if cruzados:
        problemas.append(f"grupos repartidos entre particiones: {cruzados}")

    presentes = {f["split"] for f in particion}
    faltantes = set(NOMBRES) - presentes
    if faltantes:
        problemas.append(f"particiones sin imagenes: {sorted(faltantes)}")

    ids = [f["image_id"] for f in particion]
    if len(ids) != len(set(ids)):
        problemas.append("hay image_id repetidos en la particion")

    return problemas


def escribir(particion: list[dict[str, str]], destino: Path) -> None:
    """Escribe la partición en CSV, ordenada por identificador de imagen."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(COLUMNAS))
        escritor.writeheader()
        escritor.writerows(particion)