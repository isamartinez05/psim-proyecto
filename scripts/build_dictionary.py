"""Genera el diccionario de variables y el contrato de entrada al modelo.

El diccionario se construye desde una especificacion declarada en este mismo
archivo y se contrasta contra las columnas reales de features.csv. Si una
columna existe en la tabla y no en la especificacion, o al reves, el script
falla: un diccionario que no describe la tabla que acompana no sirve.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TABLA = Path("data/processed/features.csv")
DICCIONARIO = Path("data/metadata/feature_dictionary.csv")
CONTRATO = Path("config/model_input.json")

DESCRIPTORES = (
    "compacidad",
    "relacion_aspecto",
    "fraccion_area",
    "simetria_izq_der",
    "energia_wavelet_d1",
)

DEFINICIONES = {
    "compacidad": (
        "4*pi*area/perimetro^2 de la region segmentada",
        "adimensional",
        "features.compacidad; perimetro por diferencia con la erosion",
        "0 a 1.3 sobre mascara discreta",
        "Un campo pulmonar con hallazgos extensos tiende a un contorno mas irregular",
    ),
    "relacion_aspecto": (
        "alto/ancho de la caja envolvente de la region",
        "adimensional",
        "features.relacion_aspecto",
        "0.3 a 3",
        "La proporcion del campo pulmonar varia con la edad y con la inspiracion",
    ),
    "fraccion_area": (
        "area pulmonar dividida por el area del campo irradiado",
        "adimensional",
        "features.fraccion_area; denominador = mascara de campo irradiado",
        "0 a 1",
        "El volumen pulmonar aireado se reduce con la consolidacion",
    ),
    "simetria_izq_der": (
        "|area_izq - area_der| / (area_izq + area_der)",
        "adimensional",
        "features.simetria_izq_der",
        "0 a 1",
        "La afectacion unilateral produce asimetria entre campos",
    ),
    "energia_wavelet_d1": (
        "energia media de los coeficientes de detalle de nivel 1 en la region",
        "adimensional",
        "features.energia_wavelet; wavelet db2, nivel 1",
        "0 a 1",
        "La textura del parenquima cambia con la presencia de infiltrados",
    ),
}

IDENTIDAD = {
    "sample_id": ("id", "str", "clave unica: image_id + config_id"),
    "image_id": ("id", "str", "identificador de la imagen fuente"),
    "subject_id": ("id", "str", "unidad independiente; vacio si se desconoce"),
    "config_id": ("control", "str", "estrategia de realce y umbralizacion"),
    "realce": ("control", "str", "componente de realce de la estrategia"),
    "umbral": ("control", "str", "componente de umbralizacion de la estrategia"),
    "split": ("control", "str", "particion asignada"),
    "dataset_id": ("control", "str", "conjunto de procedencia"),
    "source_sha256": ("control", "str", "hash del archivo original"),
    "lado_trabajo": ("control", "int", "lado mayor tras reducir la imagen"),
    "campo_px": ("control", "int", "area del campo irradiado en pixeles"),
    "mask_px": ("control", "int", "area de la mascara obtenida en pixeles"),
    "n_componentes": ("control", "int", "campos pulmonares detectados"),
    "umbral_usado": ("control", "float", "valor del umbral; vacio si es local"),
    "dice": ("control", "float", "solapamiento con la referencia"),
    "jaccard": ("control", "float", "solapamiento alternativo"),
    "target": ("objetivo", "int", "0 sin hallazgos, 1 con manifestaciones"),
    "edad": ("control", "int", "edad en anos declarada en la lectura"),
    "sexo": ("control", "str", "sexo declarado en la lectura"),
    "quality_status": ("control", "str", "aceptada o revisar"),
    "eligible_for_model": ("control", "int", "1 si aceptada y con dos campos"),
}


def construir(columnas: list[str]) -> pd.DataFrame:
    filas = []

    for col in columnas:
        if col in IDENTIDAD:
            rol, dtype, definicion = IDENTIDAD[col]
            filas.append(
                {
                    "name": col,
                    "role": rol,
                    "dtype": dtype,
                    "definition": definicion,
                    "unit": "",
                    "method": "manifiesto o segmentacion",
                    "parameters": "",
                    "source": "data/metadata/manifest.csv",
                    "roi_definition": "",
                    "valid_range": "",
                    "missing_policy": "campo vacio",
                    "available_at_prediction": "no"
                    if col in ("target", "dice", "jaccard")
                    else "si",
                    "biomedical_rationale": "",
                }
            )
            continue

        base = col.replace("ref_", "").replace("err_", "")
        if base not in DEFINICIONES:
            raise KeyError(f"Columna sin definicion: {col}")
        definicion, unidad, metodo, rango, razon = DEFINICIONES[base]

        if col.startswith("ref_"):
            rol, roi, disponible = "control", "mascara de referencia", "no"
            definicion = f"{definicion}, calculado sobre la mascara de referencia"
        elif col.startswith("err_"):
            rol, roi, disponible = "control", "ambas mascaras", "no"
            definicion = f"error relativo de {base} frente a su valor de referencia"
        else:
            rol, roi, disponible = "predictor", "mascara obtenida", "si"

        filas.append(
            {
                "name": col,
                "role": rol,
                "dtype": "float",
                "definition": definicion,
                "unit": unidad,
                "method": metodo,
                "parameters": "lado_trabajo=512; wavelet=db2; nivel=1",
                "source": "imagen reducida y mascara correspondiente",
                "roi_definition": roi,
                "valid_range": rango,
                "missing_policy": "campo vacio si la region es invalida",
                "available_at_prediction": disponible,
                "biomedical_rationale": razon,
            }
        )

    return pd.DataFrame(filas)


def main() -> None:
    tabla = pd.read_csv(TABLA, nrows=5)
    columnas = list(tabla.columns)

    dicc = construir(columnas)
    DICCIONARIO.parent.mkdir(parents=True, exist_ok=True)
    dicc.to_csv(DICCIONARIO, index=False)
    print(f"diccionario: {len(dicc)} filas en {DICCIONARIO}")

    predictores = [c for c in columnas if c in DESCRIPTORES]
    contrato = {
        "unidad_de_analisis": "imagen procesada bajo una estrategia",
        "clave": "sample_id",
        "columna_de_agrupacion": "image_id",
        "predictores": predictores,
        "columna_objetivo": None,
        "nota_objetivo": (
            "El proyecto es de segmentacion clasica y comparacion de estabilidad. "
            "La salida espacial es la mascara vinculada, no una etiqueta por fila. "
            "La columna target existe como variable de control para el analisis "
            "entre grupos del objetivo cinco, no como objetivo supervisado."
        ),
        "columnas_excluidas_de_predictor": (
            [c for c in columnas if c.startswith(("ref_", "err_"))]
            + ["dice", "jaccard", "target"]
        ),
        "razon_exclusion": (
            "Las columnas ref_ y err_ y las metricas de solapamiento derivan de la "
            "mascara de referencia, que no existe al procesar una imagen nueva."
        ),
        "estrategia_de_validacion": "agrupada por image_id; particion congelada en splits.csv",
        "metricas_previstas": ["dice", "jaccard", "error relativo por descriptor"],
        "linea_base": ["base_fijo", "base_otsu", "base_adap"],
    }
    CONTRATO.parent.mkdir(parents=True, exist_ok=True)
    CONTRATO.write_text(
        json.dumps(contrato, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"contrato: {len(predictores)} predictores en {CONTRATO}")

    print()
    print(dicc.groupby("role").size().to_string())


if __name__ == "__main__":
    main()
