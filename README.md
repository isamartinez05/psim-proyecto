# Estabilidad de los descriptores morfologicos del campo pulmonar

Proyecto semestral de PSIM 2026II-80, Ingenieria Biomedica.
Juanita Trujillo Narvaez y Karol Isabella Martinez Villarreal.

## Que hace este proyecto

Compara nueve estrategias de segmentacion clasica del campo pulmonar en
radiografias de torax de dos sitios de adquisicion, y mide cuanto se desplaza
cada uno de cinco descriptores morfologicos respecto a mascaras trazadas por
personas.

El resultado principal es negativo y esta sustentado: ninguna estrategia de
realce mejora la linea base sin realce de forma consistente en los dos
origenes. La mejor combinacion es `base_adap`, con Dice mediano de 0.810 y
practicamente el mismo valor en ambos sitios.

## Reconstruccion

Desde una copia sin `.venv`, en Linux o PowerShell:


```
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/reproduce.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/reports reports/informe.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/slides slides/presentacion.tex
```

`scripts/reproduce.py` tarda unos 35 minutos la primera vez, de los cuales 26
corresponden a la extraccion de descriptores sobre 486 imagenes por nueve
estrategias. Con `--rapido` reutiliza `data/processed/features.csv` y baja a
unos 10 minutos. Con `--fase N` ejecuta solo una de las tres fases.

El guion crea las carpetas de salida que falten y verifica antes de empezar
que los datos originales esten en su sitio.

## Datos

Los datos no se versionan: sus condiciones de uso no permiten
redistribuirlos. `data/raw/README.md` documenta de donde obtenerlos y
`scripts/check_dataset.py` verifica que la copia local corresponde al conjunto
descrito, comparando conteos, tamanos y hashes SHA-256 contra
`results/fase1/manifiesto_datos.csv`.

Son 800 radiografias PA: 138 del Montgomery County Chest X-ray Set y 662 del
Shenzhen Hospital Chest X-ray Set, ambos de la U.S. National Library of
Medicine, mas las mascaras de Shenzhen publicadas por el Instituto
Politecnico de Kiev. En total unos 4.1 GB.

`data/raw/` es de solo lectura. Ningun guion del proyecto escribe en esa
carpeta; todo derivado va a `data/processed/`, `results/` o `figures/`.

## Estructura

```
config/            Contrato de entrada al modelado
data/metadata/         Manifiesto, particion y diccionario de variables
data/processed/       Tabla de caracteristicas, regenerable
data/raw/                   Datos originales, no versionados
figures/                     Figuras por fase, regenerables
reports/                     Fuente LaTeX del informe y tablas generadas
results/                     Tablas de resultados por fase
scripts/                     Guiones de construccion y verificacion
slides/                       Fuente Beamer de la sustentacion
src/psim/                   Paquete con las funciones del proyecto
tests/             Pruebas de estructura y contratos de datos
```

## Productos

| Archivo | Contenido |
|---|---|
| `data/metadata/manifest.csv` | 800 filas, una por imagen fuente |
| `data/metadata/splits.csv` | Particion 70/15/15 por unidad independiente |
| `data/metadata/feature_dictionary.csv` | 36 filas, una por columna de la tabla |
| `data/processed/features.csv` | 4374 filas: 486 imagenes por 9 estrategias |
| `config/model_input.json` | Predictores declarados y exclusiones razonadas |
| `results/fase3/auditoria_tabla.csv` | 11 comprobaciones sobre la tabla |
| `results/fase3/decisiones.csv` | 6 decisiones con evidencia y limitacion |
| `build/reports/informe.pdf` | Informe compilado |
| `build/slides/presentacion.pdf` | Sustentacion, 5 diapositivas |

## Modulos

| Modulo | Responsabilidad |
|---|---|
| `psim.io` | Lectura, reduccion a un canal y propiedades observables |
| `psim.manifest` | Inventario y lectura de las lecturas clinicas |
| `psim.splits` | Particion por unidad independiente |
| `psim.recorte` | Delimitacion del campo irradiado |
| `psim.mascaras` | Lectura unificada de las referencias de ambos conjuntos |
| `psim.segmentacion` | Las nueve estrategias y las metricas de solapamiento |
| `psim.features` | Los cinco descriptores y el error relativo |
| `psim.eda` | Galeria y exploracion de la cohorte de entrenamiento |

## Parametros fijados

| Parametro | Valor | Donde |
|---|---|---|
| Semilla de particion | 20262 | `psim.splits.SEMILLA` |
| Semilla de galeria | 20262 | `psim.eda.SEMILLA_GALERIA` |
| Lado de trabajo | 512 px | `psim.segmentacion.LADO_TRABAJO` |
| Umbral de fondo | 8 | `psim.recorte.UMBRAL_FONDO` |
| Umbral de mascara | 128 | `psim.mascaras.UMBRAL_MASCARA` |
| Wavelet | db2, nivel 1 | `psim.features.WAVELET` |
| Intervalos de histograma | 64 | `psim.eda.BINS` |

`results/fase1/entorno.json` registra las versiones de Python y de los
paquetes con que se produjo cada ejecucion.

## Nota sobre el banco comun

El protocolo del laboratorio contempla un banco de verificacion compartido
entre equipos. No se aplico por indicacion del docente, de modo que este
repositorio no incluye `data/common/` ni `results/common_features.csv`.

## Contribuciones

Juanita Trujillo mantuvo la infraestructura reproducible, el entorno y los
guiones de verificacion e inventario. Karol Martinez mantuvo el documento y la
coherencia entre resultados y narrativa. Cada bloque tecnico tuvo una
integrante que lo lidero y otra que lo reviso ejecutando el codigo.