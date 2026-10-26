"""Congela el protocolo del analisis entre grupos (OE5) antes de ver resultados.

Escribe config/analisis_oe5.json con las hipotesis, la prueba, el tamano de
efecto, la correccion por comparaciones multiples, el criterio de estabilidad
y los hashes de las tablas que se analizaran.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DESTINO = Path("config/analisis_oe5.json")
TABLAS = {
    "train": "data/processed/features.csv",
    "validation": "data/processed/features_validation.csv",
    "test": "data/processed/features_test.csv",
}

contrato = {
    "pregunta": (
        "Los descriptores morfologicos del campo pulmonar difieren entre "
        "radiografias con y sin hallazgos de tuberculosis, dentro de cada sitio?"
    ),
    "encuadre": "Caracterizacion morfometrica entre grupos; no es deteccion ni tamizaje.",
    "unidad": "paciente (grupo_id); las imagenes de un paciente se promedian",
    "poblacion": (
        "Pacientes con imagenes elegibles de base_adap en train, validation y test. "
        "OE5 no selecciona modelos, por lo que no requiere particion reservada."
    ),
    "estrategia": "base_adap",
    "filtro": "eligible_for_model == 1",
    "grupo": "target (0 = sin hallazgos, 1 = con hallazgos de TB)",
    "estratos": ["NLM-ChinaCXRSet", "NLM-MontgomeryCXRSet"],
    "descriptores": [
        "compacidad",
        "relacion_aspecto",
        "fraccion_area",
        "simetria_izq_der",
        "energia_wavelet_d1",
    ],
    "fuente_principal": "referencia (columnas ref_*)",
    "sensibilidad": [
        "automatica: descriptores de base_adap, misma poblacion",
        "estrategias: las nueve configuraciones, solo particion train",
    ],
    "prueba": "Mann-Whitney U bilateral",
    "efecto": "delta de Cliff (positivo = mayor en el grupo con hallazgos)",
    "intervalo": {
        "metodo": "bootstrap percentil por paciente",
        "replicas": 2000,
        "nivel": 0.95,
    },
    "correccion": "Holm sobre 10 pruebas (5 descriptores x 2 sitios) por fuente",
    "alfa": 0.05,
    "semilla": 2026,
    "criterio_estabilidad": (
        "Una conclusion es estable si referencia y automatica coinciden en la "
        "decision tras Holm y en el signo del delta de Cliff."
    ),
    "limitaciones": [
        "Montgomery aporta pocos pacientes por grupo",
        "La etiqueta proviene de lecturas clinicas, no de confirmacion diagnostica",
        "Edad y sexo pueden confundir la comparacion",
        "Las diferencias son asociaciones, no efectos causales",
    ],
    "sha256": {
        nombre: hashlib.sha256(Path(ruta).read_bytes()).hexdigest()
        for nombre, ruta in TABLAS.items()
    },
}

DESTINO.write_text(
    json.dumps(contrato, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(f"Contrato de OE5 escrito en {DESTINO}")
for nombre, h in contrato["sha256"].items():
    print(f"  {nombre}: {h[:16]}...")
