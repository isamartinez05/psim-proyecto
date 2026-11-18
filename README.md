# Estabilidad de los descriptores morfologicos del campo pulmonar

Proyecto semestral de PSIM 2026II-80, Ingenieria Biomedica, Escuela Colombiana de
Ingenieria Julio Garavito. Juanita Trujillo Narvaez y Karol Isabella Martinez
Villarreal. Docente: Ing. Pablo Eduardo Caicedo-Rodriguez, Ph.D.

## Pregunta de investigacion

¿En que medida las estrategias de realce de intensidad, umbralizacion y morfologia
modifican la estabilidad de los descriptores morfologicos del campo pulmonar
(compacidad, relacion de aspecto, fraccion de area, simetria izquierda-derecha y
energia wavelet), respecto a los obtenidos sobre la mascara de referencia, en
radiografias de torax de dos sitios de adquisicion?

El proyecto caracteriza y compara; no detecta ni diagnostica tuberculosis.

## Datos

800 radiografias de torax posteroanteriores de la U.S. National Library of
Medicine: 138 del Montgomery County Chest X-ray Set (Estados Unidos) y 662 del
Shenzhen Hospital Chest X-ray Set (China), mas las mascaras pulmonares de Shenzhen
publicadas por el Instituto Politecnico de Kiev (DOI 10.17632/8gf9vpkhgy.2). En
total unos 4.1 GB; 704 imagenes tienen mascara de referencia.

Los datos no se versionan. `data/raw/README.md` indica como obtenerlos y
`scripts/check_dataset.py` verifica la copia local. `data/raw/` es de solo
lectura: todo derivado va a `data/processed/`, `results/` o `figures/`.

Particion congelada 70/15/15 por paciente, estratificada por origen y clase
(`data/metadata/splits.csv`). Dos pacientes con varias imagenes se agrupan para
que nunca crucen particiones.

## Objetivos y resultados principales

| Objetivo | Resultado principal |
|---|---|
| **OE1.** Caracterizar la digitalizacion | Montgomery es de 8 bits efectivos (no 12); ninguna imagen esta invertida; Montgomery conserva hasta 67 % de relleno negro, por lo que el campo irradiado se delimita como region. |
| **OE2.** Comparar realce y reduccion de ruido en dominio espacial y de frecuencia | De 9 estrategias evaluadas, solo el pasa-bajas gaussiano cumple el criterio de la formulacion, por un margen minimo. Los dos origenes tienen firmas espectrales separadas por completo (delta de Cliff = -1); la diferencia esta en las subbandas wavelet vertical y diagonal, no en la horizontal (anatomia). La especificacion de histograma iguala la intensidad, pero no elimina la diferencia de textura. |
| **OE3.** Comparar segmentaciones contra la referencia | `base_adap` (sin realce, umbral adaptativo) tiene el mayor Dice en ambos origenes, en entrenamiento (0.81) y en validacion (0.82 y 0.80). Ninguna estrategia de realce mejora la linea base de forma consistente entre sitios. |
| **OE4.** Cuantificar el desplazamiento de cada descriptor | Tres tipos de inestabilidad: sesgo sistematico (compacidad, unos 45 % menor), sesgo dependiente del sitio (relacion de aspecto) y error aleatorio (simetria). Estable en train, validacion y prueba. |
| **OE5.** Analizar diferencias entre grupos | Con la mascara de referencia, cuatro descriptores difieren entre radiografias con y sin hallazgos en Shenzhen; ninguno en Montgomery tras Holm. Segun la estrategia de segmentacion cambian entre 2 y 5 de 10 conclusiones. |

**Modelado supervisado (laboratorio de regresion).** Una regresion polinomica de
grado 2 sobre tres descriptores automaticos estima la fraccion pulmonar de la
mascara manual. En la prueba de 99 pacientes: RMSE 0.031 (IC 95 % bootstrap 0.027
a 0.036) frente a 0.055 de la media de entrenamiento, una reduccion del 43 %
(36 % a 51 %).

## Reconstruccion

Desde una copia sin `.venv`, en PowerShell o Linux:

```
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/reproduce.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/reports reports/informe.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/slides slides/presentacion.tex
```

`scripts/reproduce.py` encadena cinco fases:

| Fase | Contenido |
|---|---|
| 1 | Manifiesto, inventario de calidad y particion |
| 2 | Exploracion visual y numerica |
| 3 | Tabla de caracteristicas (486 imagenes x 9 estrategias), diccionario y decisiones |
| 4 | Modelado: seleccion en validacion, evaluacion unica en prueba y bootstrap |
| 5 | OE2 (cinco lotes), OE3 en validacion, OE4, OE5 y auditoria de la matriz de tecnicas |

La primera ejecucion tarda alrededor de una hora, sobre todo por la extraccion de
descriptores de la fase 3. Con `--rapido` reutiliza las tablas de descriptores. Con
`--fase N` ejecuta una sola fase.

## Protocolos congelados

Cada analisis inferencial o predictivo fijo sus decisiones antes de ver
resultados, y su historial queda en git:

| Archivo | Contenido |
|---|---|
| `config/model_input.json` | Contrato de entrada del analisis exploratorio |
| `config/modelado_lab005.json` | Objetivo, predictores, candidatos, metrica y hash de la tabla de entrenamiento |
| `results/modelado_proyecto/frozen_protocol.json` | Modelo elegido y coeficientes antes de abrir la prueba |
| `config/analisis_oe5.json` | Prueba, tamano de efecto, correccion, criterio de estabilidad y hashes |

## Estructura

```
config/            Contratos de entrada y protocolos
data/metadata/     Manifiesto, particion y diccionario de variables
data/processed/    Tablas de caracteristicas, regenerables
data/raw/          Datos originales, no versionados
docs/              Formulacion, matriz de tecnicas, cronograma y decisiones
figures/           Figuras por fase, regenerables
reports/           Fuentes LaTeX de los informes y tablas generadas
results/           Tablas de resultados por fase
scripts/           Guiones de construccion, analisis y verificacion
slides/            Fuentes Beamer de las sustentaciones
src/psim/          Paquete con las funciones del proyecto
tests/             Pruebas de estructura, contratos y funciones
```

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
| `psim.espacial` | Filtros, operadores de borde, realce y metricas de OE2 |
| `psim.frecuencia` | Espectro, perfil radial y filtros en frecuencia |
| `psim.wavelets` | DWT por niveles, energia por subbanda, reconstruccion y CWT |
| `psim.histograma` | Histograma, CDF, especificacion y ecualizacion |
| `psim.intensidad` | Transformaciones puntuales, diferencia y requantizacion |
| `psim.geometria` | Conectividad, transformada de distancia y morfologia |
| `psim.modeling` | Preparacion por paciente, pipeline de regresion y metricas |
| `psim.estadistica` | Mann-Whitney, delta de Cliff con bootstrap y Holm |

## Matriz de tecnicas

`docs/matriz_tecnicas.csv` conserva las 53 tecnicas comprometidas en la
formulacion, con su estado: 50 completadas y 3 que no aplican (correccion de escala
invertida, exploracion HSV y regresion logistica en el proyecto), justificadas en
`docs/matriz_tecnicas_implementacion.csv`, que indica el archivo y el producto
reales de cada tecnica. `scripts/auditar_matriz.py` lo verifica automaticamente.

## Pruebas

`uv run pytest` ejecuta las pruebas de estructura, de contratos de datos y de cada
modulo: particion por paciente, exclusion de columnas derivadas de la mascara
manual, imputacion ajustada solo en entrenamiento, metricas con valores conocidos y
reproducibilidad.

## Parametros fijados

| Parametro | Valor | Donde |
|---|---|---|
| Semilla de particion | 20262 | `psim.splits.SEMILLA` |
| Semilla de modelado, OE2 y OE5 | 2026 | contratos y guiones |
| Lado de trabajo | 512 px | `psim.segmentacion.LADO_TRABAJO` |
| Umbral de fondo | 8 | `psim.recorte.UMBRAL_FONDO` |
| Umbral de mascara | 128 | `psim.mascaras.UMBRAL_MASCARA` |
| Wavelet | db2 | `psim.features.WAVELET` |
| Replicas bootstrap | 2000 | OE5 y prueba del modelado |

`results/fase1/entorno.json` registra las versiones de Python y de los paquetes de
cada ejecucion.

## Nota sobre el banco comun

El protocolo del laboratorio contempla un banco de verificacion compartido entre
equipos. No se aplico por indicacion del docente.

## Contribuciones

Juanita Trujillo mantuvo la infraestructura reproducible, el entorno y los guiones
de verificacion, modelado y analisis. Karol Martinez mantuvo los documentos y la
coherencia entre resultados y narrativa. Cada bloque tecnico tuvo una integrante
que lo lidero y otra que lo reviso ejecutando el codigo.