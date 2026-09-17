# Estabilidad de los descriptores morfológicos del campo pulmonar

Proyecto semestral de Procesamiento de Señales e Imágenes Médicas (PSIM - 2026II - 80).

**Equipo:** Juanita Trujillo Narváez (1000099467) y Karol Isabella Martínez Villarreal (1000099682)

**Docente:** Ing. Pablo Eduardo Caicedo-Rodríguez. Ph.D.

**Programa:** Ingeniería Biomédica — Escuela Colombiana de Ingeniería Julio Garavito

## Pregunta de investigación

¿En qué medida las estrategias de realce de intensidad, umbralización y
morfología modifican la estabilidad de los descriptores morfológicos del campo
pulmonar (compacidad, relación de aspecto, fracción de área, simetría
izquierda-derecha), respecto a los descriptores obtenidos sobre la máscara de
referencia, en radiografías de tórax de los conjuntos Montgomery y Shenzhen de
la National Library of Medicine?

## Conjunto de datos

Montgomery County CXR Set (138 imágenes, CR, 12 bits) y Shenzhen Hospital CXR
Set (662 imágenes, DR, 8 bits), publicados por la U.S. National Library of
Medicine. Total: 800 imágenes, 704 con máscara de referencia.

- **Unidad de análisis:** la imagen, asumida equivalente al sujeto.
- **Referencia:** Jaeger S et al. Quant Imaging Med Surg. 2014;4(6):475-477. PMCID: PMC4256233.
- **Ficha completa:** `docs/dataset_card.md`

## Estructura de las tres fases

| Fase | Semanas | Propósito |
|---|---|---|
| 1 | 1-5 | Formulación, adquisición y caracterización de los datos |
| 2 | 6-11 | Procesamiento, segmentación y extracción de características |
| 3 | 12-15 | Integración, validación y presentación en el Laboratorio 03 |

## Cómo reconstruir el entorno

Requisitos: Python 3.12 y uv (https://docs.astral.sh/uv/).

    git clone <url-del-repositorio>
    cd psim-proyecto
    uv sync --locked

El comando `uv sync --locked` instala exactamente las versiones fijadas en
`uv.lock`. No se usa `pip install` dentro del entorno del proyecto.

### Dependencias declaradas

numpy, scipy, matplotlib, pandas, scikit-image, pywavelets, statsmodels.

Dev: pytest, ruff.

**¿Por qué no está opencv-python?** La guía del curso lo incluye en el comando
de ejemplo, pero las 53 técnicas de la matriz de cobertura son cubribles con
scikit-image y scipy. No se agrega una dependencia sin uso concreto en el diseño
metodológico, conforme a la advertencia de la sección 14.1 de la guía.

## Cómo obtener los datos

Los datos originales no se redistribuyen en este repositorio. El procedimiento
completo de descarga, con URLs, versiones, estructura esperada y verificación,
está documentado en `data/raw/README.md`.

Resumen:

1. Descargar Montgomery y Shenzhen desde la NLM (enlaces en data/raw/README.md).
2. Descargar las máscaras de Shenzhen desde Mendeley (DOI 10.17632/8gf9vpkhgy.2).
3. Descomprimir en data/raw/ sin renombrar las carpetas.
4. Verificar la copia:

    uv run python scripts/check_dataset.py

El script comprueba conteos, tamaños y hashes SHA-256, y escribe el manifiesto
en results/fase1/manifiesto_datos.csv.

## Cómo reproducir el proyecto

    uv run python scripts/reproduce.py

Regenera todo lo que hay en data/processed/, results/ y figures/ a partir
de data/raw/, sin modificar los datos originales. También acepta --fase 1,
--fase 2 o --fase 3 para ejecutar una sola fase.

Actualmente el script es un esqueleto que documenta la estructura de cada fase.
Se irá llenando conforme avance el desarrollo.

## Política de datos

- data/raw/ contiene datos originales o instrucciones para obtenerlos. Nunca
  se sobrescriben.
- data/processed/ contiene datos derivados, regenerables con reproduce.py.
- results/ y figures/ contienen productos regenerables. Ninguna tabla ni
  figura del informe se edita a mano.
- El manifiesto (results/fase1/manifiesto_datos.csv) es el único producto
  derivado que se versiona, porque es la garantía de integridad.

## Estructura del repositorio

    psim-proyecto/
    ├── .python-version          Python 3.12
    ├── .gitignore
    ├── README.md                Este archivo
    ├── pyproject.toml           Dependencias declaradas
    ├── uv.lock                  Versiones fijadas
    ├── data/
    │   ├── raw/                 Datos originales (no versionados)
    │   │   └── README.md        Procedimiento de descarga y verificación
    │   └── processed/           Datos derivados (regenerables)
    ├── docs/
    │   ├── formulacion_proyecto.tex
    │   ├── sustentacion.tex
    │   ├── referencias.bib
    │   ├── dataset_card.md
    │   ├── datasets_candidatos.csv
    │   ├── matriz_tecnicas.csv
    │   ├── cronograma.csv
    │   └── Escuela_Rosario_logo.png
    ├── figures/
    │   ├── fase1/
    │   ├── fase2/
    │   └── fase3/
    ├── results/
    │   ├── fase1/
    │   ├── fase2/
    │   └── fase3/
    ├── scripts/
    │   ├── check_dataset.py     Verificación de integridad
    │   └── reproduce.py         Reconstrucción completa
    ├── src/
    │   └── psim/
    │       └── __init__.py
    └── tests/

## Verificación rápida

    uv sync --locked                              # entorno
    uv run ruff check src/ scripts/               # estilo
    uv run pytest                                 # pruebas
    uv run python scripts/check_dataset.py        # datos
    uv run python scripts/reproduce.py            # pipeline
