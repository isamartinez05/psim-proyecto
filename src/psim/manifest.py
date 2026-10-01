"""Construcción del manifiesto de datos del proyecto.

El manifiesto es la tabla de inventario que exige la Fase 1: una fila por
imagen fuente, con su procedencia, sus propiedades observables, su etiqueta de
referencia y su estado de calidad. Es el registro contra el que se reconcilian
todos los conteos posteriores.

Tres decisiones están implementadas aquí y conviene hacerlas explícitas:

1. ``subject_id`` solo se asigna cuando hay evidencia escrita de que dos
   imágenes pertenecen al mismo paciente. Las lecturas clínicas de Montgomery
   declaran dos grupos mediante referencias cruzadas. El resto queda vacío: la
   ausencia de identificador es información, y no se inventa independencia
   donde no se puede demostrar.

2. ``quality_status`` distingue entre aceptada, revisar y excluida. Las 96
   imágenes de Shenzhen sin máscara de referencia se marcan como *revisar*, no
   como *excluida*: participan en la caracterización y en la comparación de
   realce, y solo quedan fuera de la comparación contra referencia.

3. Se registran columnas que el inventario mínimo no pide pero que la
   caracterización demostró necesarias: profundidad efectiva de bits,
   orientación, resultado de la verificación de inversión, y edad y sexo
   extraídos de las lecturas clínicas.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from psim import DATA_RAW, MONTGOMERY, SHENZHEN, SHENZHEN_MASCARAS
from psim.io import describir

# Grupos de imágenes que las lecturas clínicas declaran del mismo paciente.
# La evidencia está en el texto de la lectura, no en los metadatos.
GRUPOS_SUJETO: dict[str, tuple[str, ...]] = {
    # "stable CXR since 6M; off Rx since 6M (same pt as MCUCXR_0113_1)"
    # Las edades difieren (084Y y 085Y) porque median meses entre estudios.
    "MC_SUBJ_001": ("MCUCXR_0113_1", "MCUCXR_0117_1"),
    # "NO REPORT (same pt as MCUCXR_0162_1)" para 0166.
    # 0170 se incorpora por coincidencia de sexo y edad con los anteriores y
    # porque su lectura menciona "prior views 1M and 2M ago", compatibles con
    # las otras dos. Es evidencia indirecta y queda anotada como tal.
    "MC_SUBJ_002": ("MCUCXR_0162_1", "MCUCXR_0166_1", "MCUCXR_0170_1"),
}

_SUJETO_DE_IMAGEN = {img: suj for suj, imgs in GRUPOS_SUJETO.items() for img in imgs}

_EVIDENCIA_DIRECTA = ("MCUCXR_0117_1", "MCUCXR_0166_1")

COLUMNAS = (
    "image_id",
    "subject_id",
    "subject_evidence",
    "dataset_id",
    "dataset_version",
    "site_id",
    "modality",
    "relative_path",
    "sha256",
    "height",
    "width",
    "depth",
    "channels",
    "orientacion",
    "dtype",
    "bits_efectivos",
    "intensity_unit",
    "valor_min",
    "valor_max",
    "media",
    "desviacion",
    "mediana",
    "escala_invertida",
    "spacing_y_mm",
    "spacing_x_mm",
    "spacing_z_mm",
    "edad",
    "edad_meses",
    "sexo",
    "target",
    "target_source",
    "hallazgo_texto",
    "mask_path",
    "mask_source",
    "quality_status",
    "quality_reason",
    "split",
)


@dataclass(frozen=True)
class Conjunto:
    """Un conjunto de imágenes con su procedencia y sus rutas."""

    dataset_id: str
    dataset_version: str
    site_id: str
    modality: str
    raiz: Path
    imagenes: Path
    lecturas: Path
    # Montgomery entrega las máscaras separadas por pulmón; Shenzhen, en un
    # único archivo externo. La diferencia se resuelve aquí y no en el código
    # que consume el manifiesto.
    mascaras: tuple[Path, ...]
    mask_source: str
    # Montgomery documenta espaciado isotrópico; Shenzhen no documenta ninguno.
    spacing_mm: float | None


CONJUNTOS = (
    Conjunto(
        dataset_id="NLM-MontgomeryCXRSet",
        dataset_version="NLM 2014 (archivos 2024-08-28)",
        site_id="Montgomery County, Maryland, EE. UU.",
        modality="CR",
        raiz=MONTGOMERY,
        imagenes=MONTGOMERY / "CXR_png",
        lecturas=MONTGOMERY / "ClinicalReadings",
        mascaras=(
            MONTGOMERY / "ManualMask" / "leftMask",
            MONTGOMERY / "ManualMask" / "rightMask",
        ),
        mask_source="Trazada bajo supervision de radiologo (Candemir et al. 2014)",
        spacing_mm=0.0875,
    ),
    Conjunto(
        dataset_id="NLM-ChinaCXRSet",
        dataset_version="NLM 2014 (archivos 2024-08-28)",
        site_id="Shenzhen No. 3 People's Hospital, Guangdong, China",
        modality="DR",
        raiz=SHENZHEN,
        imagenes=SHENZHEN / "CXR_png",
        lecturas=SHENZHEN / "ClinicalReadings",
        mascaras=(SHENZHEN_MASCARAS,),
        mask_source="Instituto Politecnico de Kiev (Stirenko et al. 2018), CC BY-NC-SA 4.0",
        spacing_mm=None,
    ),
)

# Montgomery usa etiquetas de estilo DICOM; Shenzhen, texto libre.
#
# Las lecturas de Shenzhen contienen variantes de tipeo del original que una
# expresión estricta no reconoce: "femal" sin la e final, "female24yrs" sin
# espacio separador, "male 42" sin unidad, y edades expresadas en meses o
# días para los pacientes pediátricos. Las expresiones siguientes las admiten
# sin alterar el archivo fuente, que permanece intacto.
_RE_SEXO_MC = re.compile(r"Sex:\s*(\S)", re.IGNORECASE)
_RE_EDAD_MC = re.compile(r"Age:\s*0*(\d+)", re.IGNORECASE)
_RE_SEXO_SH = re.compile(r"(male|femal(?:e)?)", re.IGNORECASE)
_RE_EDAD_SH = re.compile(r"(\d+)\s*(yrs?|years?|months?|days?)?", re.IGNORECASE)

# Días por unidad, para expresar en años las edades pediátricas declaradas en
# otra escala. La edad en meses se conserva aparte: redondear 64 días a cero
# años perdería la distinción entre un neonato y un niño de pocos meses, que
# en radiografía de tórax es considerable.
_DIAS_POR_UNIDAD = {"day": 1.0, "days": 1.0, "month": 30.44, "months": 30.44}


def leer_lectura(ruta: Path) -> tuple[str, str, str, str]:
    """Extrae sexo, edad en años, edad en meses y hallazgo de una lectura.

    Devuelve cadenas vacías para lo que no esté presente. No se infiere ningún
    valor ausente: un campo vacío significa que la lectura no lo declara.

    El hallazgo es la primera línea posterior a la demográfica. Once lecturas
    de Shenzhen añaden líneas de subclasificación (ATB, NATB, pleuritis) que no
    sustituyen al hallazgo principal, de modo que tomar la última línea sería
    incorrecto.
    """
    texto = ruta.read_text(encoding="utf-8", errors="replace")
    lineas = [ln.strip() for ln in texto.splitlines() if ln.strip()]
    if not lineas:
        return "", "", "", ""

    sexo = ""
    edad = ""
    edad_meses = ""
    inicio_hallazgo = 1

    m = _RE_SEXO_MC.search(texto)
    if m:
        sexo = m.group(1).upper()
        inicio_hallazgo = 2  # sexo y edad ocupan dos líneas
    else:
        m = _RE_SEXO_SH.search(lineas[0])
        if m:
            sexo = "M" if m.group(1).lower() == "male" else "F"

    m = _RE_EDAD_MC.search(texto)
    if m:
        edad = m.group(1)
        edad_meses = str(int(m.group(1)) * 12)
    else:
        m = _RE_EDAD_SH.search(lineas[0])
        if m:
            cantidad = int(m.group(1))
            unidad = (m.group(2) or "yrs").lower()
            dias = _DIAS_POR_UNIDAD.get(unidad)
            if dias is None:  # años, con o sin unidad explícita
                edad = str(cantidad)
                edad_meses = str(cantidad * 12)
            else:
                meses = cantidad * dias / 30.44
                edad = str(int(meses // 12))
                edad_meses = f"{meses:.1f}"

    hallazgo = lineas[inicio_hallazgo] if len(lineas) > inicio_hallazgo else ""
    return sexo, edad, edad_meses, hallazgo


def etiqueta_de_nombre(image_id: str) -> str:
    """Devuelve la etiqueta de tuberculosis codificada en el nombre.

    Ambos conjuntos cierran el nombre con ``_0`` para estudios sin hallazgos y
    ``_1`` para estudios con manifestaciones.
    """
    if image_id.endswith("_0"):
        return "0"
    if image_id.endswith("_1"):
        return "1"
    return ""


def coherencia_etiqueta(target: str, hallazgo: str) -> bool:
    """Comprueba que la etiqueta del nombre concuerde con la lectura.

    Un estudio etiquetado como normal cuya lectura describe hallazgos, o al
    revés, es una contradicción que el inventario debe registrar.
    """
    h = hallazgo.strip().lower()
    if not h:
        return True
    es_normal = h in ("normal", "normale")  # variante de tipeo en el original
    if target == "0":
        return es_normal
    if target == "1":
        return not es_normal
    return True


def construir(calcular_hash: bool = True) -> list[dict[str, object]]:
    """Recorre ambos conjuntos y devuelve las filas del manifiesto."""
    filas: list[dict[str, object]] = []

    for conj in CONJUNTOS:
        if not conj.imagenes.is_dir():
            raise FileNotFoundError(
                f"No existe {conj.imagenes}. Consulte data/raw/README.md."
            )

        for ruta in sorted(conj.imagenes.glob("*.png")):
            image_id = ruta.stem
            props = describir(ruta, calcular_hash=calcular_hash)

            sexo, edad, edad_meses, hallazgo = "", "", "", ""
            lectura = conj.lecturas / f"{image_id}.txt"
            if lectura.is_file():
                sexo, edad, edad_meses, hallazgo = leer_lectura(lectura)

            mascara = ""
            for carpeta in conj.mascaras:
                candidata = carpeta / f"{image_id}.png"
                if candidata.is_file():
                    mascara = candidata.relative_to(DATA_RAW).as_posix()
                    break

            target = etiqueta_de_nombre(image_id)

            razones: list[str] = []
            if not mascara:
                razones.append("sin_mascara_referencia")
            if not lectura.is_file():
                razones.append("sin_lectura_clinica")
            if not coherencia_etiqueta(target, hallazgo):
                razones.append("etiqueta_incoherente_con_lectura")
            if props.escala_invertida:
                razones.append("verificacion_inversion_positiva")
            if props.no_finitos:
                razones.append("valores_no_finitos")

            sujeto = _SUJETO_DE_IMAGEN.get(image_id, "")
            evidencia = ""
            if sujeto:
                evidencia = (
                    "referencia_cruzada_en_lectura"
                    if image_id in _EVIDENCIA_DIRECTA
                    else "agrupado_por_sexo_edad_y_vistas_previas"
                )

            filas.append(
                {
                    "image_id": image_id,
                    "subject_id": sujeto,
                    "subject_evidence": evidencia,
                    "dataset_id": conj.dataset_id,
                    "dataset_version": conj.dataset_version,
                    "site_id": conj.site_id,
                    "modality": conj.modality,
                    "relative_path": ruta.relative_to(DATA_RAW).as_posix(),
                    "sha256": props.sha256,
                    "height": props.height,
                    "width": props.width,
                    "depth": 1,
                    "channels": props.channels,
                    "orientacion": "vertical"
                    if props.height >= props.width
                    else "horizontal",
                    "dtype": props.dtype,
                    "bits_efectivos": props.bits_efectivos,
                    "intensity_unit": "stored",
                    "valor_min": props.valor_min,
                    "valor_max": props.valor_max,
                    "media": round(props.media, 4),
                    "desviacion": round(props.desviacion, 4),
                    "mediana": props.mediana,
                    "escala_invertida": int(props.escala_invertida),
                    "spacing_y_mm": conj.spacing_mm if conj.spacing_mm else "",
                    "spacing_x_mm": conj.spacing_mm if conj.spacing_mm else "",
                    "spacing_z_mm": "",
                    "edad": edad,
                    "edad_meses": edad_meses,
                    "sexo": sexo,
                    "target": target,
                    "target_source": "sufijo_del_nombre_de_archivo",
                    "hallazgo_texto": hallazgo,
                    "mask_path": mascara,
                    "mask_source": conj.mask_source if mascara else "",
                    "quality_status": "revisar" if razones else "aceptada",
                    "quality_reason": ";".join(razones),
                    "split": "",
                }
            )

    return filas


def escribir(filas: list[dict[str, object]], destino: Path) -> None:
    """Escribe el manifiesto en CSV, ordenado por identificador de imagen."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    filas = sorted(filas, key=lambda f: str(f["image_id"]))
    with destino.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(COLUMNAS))
        escritor.writeheader()
        escritor.writerows(filas)
