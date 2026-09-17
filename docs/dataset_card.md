# Dataset card

Este proyecto utiliza dos conjuntos complementarios publicados por la U.S. National
Library of Medicine. Se documentan juntos porque constituyen la población de estudio
del proyecto, y por separado en cada campo donde difieren, ya que esa diferencia
entre sitios de adquisición es el objeto de la pregunta de investigación.

## Identificación

- **Nombre:** Montgomery County CXR Set y Shenzhen Hospital CXR Set (Tuberculosis Chest X-ray Datasets, NLM)
- **Fuente:** U.S. National Library of Medicine, Lister Hill National Center for Biomedical Communications. Montgomery: programa de control de tuberculosis del Departamento de Salud y Servicios Humanos del Condado de Montgomery, Maryland, EE. UU. Shenzhen: Shenzhen No. 3 People's Hospital, Guangdong, China.
- **URL permanente:**
  - Montgomery: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html
  - Shenzhen: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html
  - Máscaras de Shenzhen (fuente externa): https://www.kaggle.com/yoctoman/shcxr-lung-mask
  - Alternativa con DOI permanente para ambas máscaras: https://doi.org/10.17632/8gf9vpkhgy.2
- **Versión:** publicación original NLM 2014. Los archivos del servidor registran fecha de modificación 2024-08-28. Máscaras de Shenzhen publicadas en 2018 por el Instituto Politécnico de Kiev. El conjunto combinado de Mendeley corresponde a la versión 2 (3 de octubre de 2022).
- **Fecha de consulta:** 2026-09-16
- **Referencia/citación:**
  - Jaeger S, Candemir S, Antani S, Wáng YXJ, Lu PX, Thoma G. Two public chest X-ray datasets for computer-aided screening of pulmonary diseases. Quant Imaging Med Surg. 2014;4(6):475-477. doi:10.3978/j.issn.2223-4292.2014.11.20. PMID 25525580. PMCID PMC4256233.
  - Candemir S, Jaeger S, Palaniappan K, et al. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. IEEE Trans Med Imaging. 2014;33(2):577-590. doi:10.1109/TMI.2013.2290491. (Máscaras de Montgomery.)
  - Stirenko S, Kochura Y, Alienin O, et al. Chest X-ray analysis of tuberculosis by deep learning with segmentation and augmentation. 2018. (Máscaras de Shenzhen.)
- **Licencia o condiciones de uso:** Las imágenes de ambos conjuntos son de acceso público para investigación, con citación obligatoria de Jaeger et al. (2014) y con la condición de no redistribuir el conjunto fuera del grupo de investigación. Los datos fueron desidentificados por los proveedores y quedaron exentos de revisión IRB; su uso y publicación fueron exentos por la NIH Office of Human Research Protections Programs bajo el número 5357. Las máscaras de Shenzhen se distribuyen por separado bajo licencia CC BY-NC-SA 4.0, compatible con uso académico no comercial.

## Contenido

- **Modalidad:** radiografía de tórax, proyección posteroanterior (PA). Montgomery se adquirió mediante radiografía computarizada (CR); Shenzhen mediante radiografía digital directa (DR).
- **Población o procedencia:** Montgomery corresponde a personas tamizadas por el programa de control de tuberculosis del condado. Shenzhen corresponde a estudios adquiridos como parte de la atención de rutina del hospital. Ninguno de los dos conjuntos documenta criterios de inclusión, distribución etaria agregada ni composición demográfica más allá de los campos presentes en las lecturas clínicas individuales.
- **Número de sujetos:** 138 en Montgomery y 662 en Shenzhen, para un total de 800. La correspondencia uno a uno entre imagen y sujeto es la interpretación asumida, dado que ninguno de los conjuntos incluye un identificador de paciente que permita agrupar estudios. Debe verificarse en Fase 1.
- **Número de estudios/imágenes:** 800 imágenes en total (138 + 662). El conteo de archivos en el servidor de la NLM coincide exactamente con las cifras publicadas.
- **Dimensiones típicas:** Montgomery presenta 4020x4892 o 4892x4020 píxeles, con espaciado de 0.0875 mm en ambas direcciones. Shenzhen presenta resolución variable en torno a 3000x3000 píxeles y no documenta espaciado de píxel.
- **Profundidad de bits / rango de intensidad:** Montgomery en 12 bits en escala de grises. Shenzhen en 8 bits, codificado como PNG de tres canales RGB y con la escala de grises invertida respecto a la convención radiográfica habitual.
- **Formato de archivo:** PNG en ambos conjuntos. Las máscaras de Montgomery son PNG binarios separados por pulmón izquierdo y derecho, con el mismo nombre de archivo que la radiografía correspondiente.
- **Metadatos disponibles:** lecturas radiológicas en archivos de texto por imagen (`ClinicalReadings/`), que incluyen edad, sexo y descripción de la anormalidad pulmonar. Cada conjunto incluye además un archivo de región de interés de consenso (`montgomery_consensus_roi.csv` y `shenzhen_consensus_roi.csv`) y un documento descriptivo (`NLM-MontgomeryCXRSet-ReadMe.pdf`, `NLM-ChinaCXRSet-ReadMe.docx`). Shenzhen incluye adicionalmente anotaciones poligonales de hallazgos compatibles con tuberculosis para los 336 casos positivos, en carpetas `Annotations/` y `Annotations-2/`, que este proyecto no utiliza.

## Variable de interés

- **Unidad de análisis:** la imagen, asumida equivalente al sujeto. Los objetivos OE3 y OE4, que requieren comparación contra máscara de referencia, se restringen al subconjunto de 704 imágenes que disponen de ella (138 de Montgomery y 566 de Shenzhen), estratificado por origen. Los objetivos OE1 y OE2 operan sobre las 800.
- **Variable de salida, si existe:** el proyecto no predice una variable clínica. La salida son los descriptores morfológicos del campo pulmonar segmentado: compacidad, relación de aspecto, fracción de área, simetría izquierda-derecha y energía por subbanda wavelet. Todos adimensionales, porque Shenzhen no documenta espaciado de píxel y las medidas métricas no serían comparables entre orígenes.
- **Etiquetas o referencia:** la referencia del proyecto son las máscaras de campo pulmonar. Las de Montgomery fueron trazadas bajo supervisión de un radiólogo; las de Shenzhen por estudiantes y profesores del Departamento de Ingeniería Informática del Instituto Politécnico de Kiev. Existe además una etiqueta de tuberculosis codificada en el último carácter del nombre de archivo (`_0` normal, `_1` con manifestaciones), que se utiliza únicamente en OE5 como variable de estratificación para comparación de grupos, nunca como objetivo de detección.
- **Posibles factores de confusión:** tecnología del detector (CR frente a DR), profundidad de cuantización (12 frente a 8 bits), parámetros de exposición no documentados, presencia de hallazgos que opacifican el parénquima y alteran el contraste del borde pulmonar, y convención de trazado de la máscara de referencia, que difiere entre los dos orígenes.

## Calidad y limitaciones

- **Datos faltantes:** 96 de las 662 imágenes de Shenzhen carecen de máscara de referencia. Shenzhen no documenta espaciado de píxel para ninguna imagen. Ninguno de los conjuntos incluye parámetros de exposición ni identificador de paciente.
- **Desbalance, si aplica:** Montgomery contiene 80 estudios normales y 58 con manifestaciones de tuberculosis, un desbalance apreciable. Shenzhen contiene 326 normales y 336 positivos, prácticamente balanceado. El conjunto combinado queda en 406 normales y 394 positivos. El desbalance relevante para este proyecto no es el de clase sino el de origen: Shenzhen aporta el 83 por ciento de las imágenes, por lo que todo resultado agregado debe reportarse estratificado por sitio.
- **Variabilidad de adquisición:** es el objeto de estudio, no un defecto. Dos sitios, dos tecnologías de detector, dos profundidades de bits y dos convenciones de codificación de intensidad.
- **Artefactos conocidos:** las imágenes de Shenzhen presentan escala de grises invertida y codificación en tres canales RGB, ambas requieren corrección antes de cualquier procesamiento. Nueve archivos de Shenzhen pesan menos de 2 MB frente a un promedio de 5.43 MB para el conjunto, con un mínimo de 0.75 MB; esa dispersión sugiere resolución o codificación distinta en esos casos y debe verificarse en Fase 1. Montgomery es homogéneo, con tamaños entre 2.05 y 6.85 MB y sin valores atípicos.
- **Restricciones técnicas:** el volumen total de imágenes es de 4.08 GB (0.57 GB Montgomery, 3.51 GB Shenzhen), verificado en el índice del servidor de la NLM el 2026-09-16. El tamaño individual de las imágenes de Montgomery, de hasta 4892x4020 píxeles en 12 bits, implica cerca de 39 MB por imagen al cargarla como arreglo de enteros de 16 bits; el procesamiento debe hacerse imagen por imagen y no cargando el conjunto completo en memoria.

## Reproducibilidad

- **Método de descarga:** descarga directa desde los índices del servidor de la NLM enlazados arriba, sin registro ni solicitud de acceso. Las máscaras de Shenzhen se obtienen por separado desde Kaggle o desde el conjunto combinado de Mendeley con DOI permanente. El procedimiento exacto queda documentado en `data/raw/README.md`.
- **Identificador de versión:** la NLM no asigna número de versión a estos conjuntos. Como sustituto se registra la fecha de modificación de los archivos en el servidor (2024-08-28) junto con la fecha de consulta (2026-09-16). Para las máscaras existe el DOI versionado 10.17632/8gf9vpkhgy.2.
- **Hash o mecanismo de verificación, si está disponible:** la NLM no publica sumas de verificación. El equipo genera un manifiesto propio mediante `scripts/check_dataset.py`, que registra para cada archivo su nombre, tamaño en bytes y hash SHA-256, y verifica los conteos esperados (138 y 662 imágenes, 138 y 566 máscaras). Ese manifiesto se versiona en el repositorio y permite comprobar que la copia local corresponde al conjunto descrito en esta ficha.
- **Datos que se conservarán en data/raw/:** las imágenes originales, las máscaras de referencia y las lecturas clínicas, sin modificación alguna. Por volumen y por las condiciones de uso, estos archivos no se versionan en el repositorio; `data/raw/README.md` documenta de dónde obtenerlos, qué descargar y cómo verificar la copia contra el manifiesto.
- **Datos que serán regenerables en data/processed/:** todo producto derivado, incluyendo imágenes corregidas de inversión y canal, versiones normalizadas y redimensionadas, imágenes realzadas por cada estrategia, máscaras obtenidas por cada combinación de segmentación y morfología, y recortes de región de interés. Ninguno se versiona: todos se reconstruyen ejecutando `scripts/reproduce.py` a partir de `data/raw/`.
