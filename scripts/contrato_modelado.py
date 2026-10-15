"""Congela el contrato de modelado del lab005 y la tabla de decisiones del EDA.

Agrega la sección modelado_lab005 a config/model_input.json sin modificar el
contrato del lab004, y escribe docs/decisiones_modelado.csv.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

CONFIG = Path("config/model_input.json")
TRAIN = Path("data/processed/features.csv")

contrato = {
    "pregunta": (
        "¿Los descriptores de la segmentación automática base_adap permiten "
        "estimar la fracción pulmonar de la máscara de referencia en imágenes "
        "no usadas para ajustar el modelo?"
    ),
    "ruta": "continua",
    "ruta_binaria": (
        "No aplica: el proyecto no formula detección de TB y target está "
        "declarada como variable de control."
    ),
    "estrategia": "base_adap",
    "filtro": "eligible_for_model == 1",
    "grupo": "grupo_id",
    "agregacion": "media por grupo_id de predictores y objetivo",
    "objetivo": "ref_fraccion_area",
    "unidad_objetivo": "adimensional (fracción del campo irradiado)",
    "predictores": ["fraccion_area", "energia_wavelet_d1", "simetria_izq_der"],
    "excluidos": {
        "compacidad": "asociación nula con el objetivo en train",
        "relacion_aspecto": "signo de asociación inconsistente entre orígenes",
        "ref_*, err_*, dice, jaccard": "derivan de la máscara de referencia",
        "target, edad, sexo, dataset_id": "variables de control, no predictores",
    },
    "candidatos": ["grado_1", "grado_2"],
    "referencia_sencilla": "media de ref_fraccion_area en entrenamiento",
    "metrica_primaria": "RMSE",
    "metricas_secundarias": ["MAE", "R2"],
    "subgrupos_diagnostico": ["dataset_id"],
    "semilla": 2026,
    "umbral": None,
    "tablas": {
        "train": str(TRAIN.as_posix()),
        "validation": "data/processed/features_validation.csv",
        "test": "data/processed/features_test.csv",
    },
    "sha256_train": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
    "politica_prueba": (
        "features_test.csv se extrae solo después de congelar el protocolo y "
        "se evalúa una única vez."
    ),
}

config = json.loads(CONFIG.read_text(encoding="utf-8"))
config["modelado_lab005"] = contrato
CONFIG.write_text(
    json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

decisiones = [
    (
        (
            "fraccion_area tiene la mayor asociación con el objetivo "
            "(Spearman 0.70; 0.66 China, 0.63 Montgomery)"
        ),
        "Predictor principal",
        "RMSE en validación frente a la media de entrenamiento",
        "Comparte el denominador campo_px con el objetivo",
    ),
    (
        (
            "energia_wavelet_d1 se asocia con el objetivo (-0.51) con el mismo "
            "signo en ambos orígenes y redundancia moderada con fraccion_area (-0.41)"
        ),
        "Incluir como segundo predictor",
        "Coeficiente y comparación de grados 1 y 2 en validación",
        "Asociación observacional; su coeficiente no es un efecto causal",
    ),
    (
        "simetria_izq_der tiene asociación débil (-0.23) pero de signo consistente",
        "Incluir como tercer predictor",
        "Se reporta aunque no mejore el RMSE de validación",
        "Su aporte puede ser nulo",
    ),
    (
        (
            "compacidad no se asocia con el objetivo y relacion_aspecto invierte "
            "su signo entre orígenes"
        ),
        "Excluir ambos",
        "Ninguna adicional; la exclusión queda registrada en el contrato",
        "Una relación no lineal o condicional al centro no se explora",
    ),
    (
        "ref_fraccion_area es mayor en Montgomery (0.317) que en Shenzhen (0.260)",
        "No usar el origen como predictor; reportar errores por origen",
        "Residuos por dataset_id en validación y prueba",
        (
            "Montgomery aporta solo 20 imágenes en validación y en prueba; "
            "no se verifica transporte a otro centro"
        ),
    ),
    (
        "Las columnas ref_, err_, dice y jaccard derivan de la máscara manual",
        "Excluirlas como predictores",
        "Prueba automática que falla si aparecen entre los predictores",
        "El modelo solo usa información disponible sin máscara manual",
    ),
    (
        "Dos pacientes aportan varias imágenes (5 imágenes en 2 grupos)",
        "Agregar por grupo_id con la media",
        "Prueba que falla si un grupo se repite o cruza particiones",
        "Pérdida mínima de información (3 filas)",
    ),
    (
        (
            "Las segmentaciones con una sola componente no son elegibles "
            "(26 en train, 4 en validación)"
        ),
        "Aplicar la misma regla de elegibilidad en todas las particiones",
        "Conteos de elegibles por partición",
        "El modelo no aplica cuando la segmentación automática falla",
    ),
]
tabla = pd.DataFrame(
    decisiones, columns=["hallazgo_eda", "decision", "comprobacion", "limite"]
)
destino = Path("docs/decisiones_modelado.csv")
tabla.to_csv(destino, index=False, encoding="utf-8")
print(f"Contrato agregado a {CONFIG}")
print(f"Tabla de decisiones: {len(tabla)} filas en {destino}")
print(f"SHA-256 de features.csv: {contrato['sha256_train']}")
