# data/raw/ — Datos originales

Los datos originales **no se redistribuyen en este repositorio** y **no se
modifican nunca**. Este archivo documenta cómo obtenerlos y cómo verificar que la
copia local corresponde al conjunto descrito en `docs/dataset_card.md`.

Cualquier cambio de tamaño, intensidad, formato, región de interés, filtro o
máscara produce un dato derivado que se escribe en `data/processed/` o un
resultado regenerable en `results/` y `figures/`. Nada de lo que hay bajo
`data/raw/` se sobrescribe.

**Las carpetas conservan el nombre con el que vienen de su fuente.** No se
renombra nada. Descomprimir y dejar tal cual.

## Qué descargar

### 1. Montgomery County CXR Set

Origen: U.S. National Library of Medicine.

    https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html

Descargar `CXR_png/`, `ClinicalReadings/` y `ManualMask/`, más el archivo
`NLM-MontgomeryCXRSet-ReadMe.pdf`. La carpeta resultante se llama `MontgomerySet`.

### 2. Shenzhen Hospital CXR Set

Origen: U.S. National Library of Medicine.

    https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html

Descargar `CXR_png/` y `ClinicalReadings/`, más `NLM-ChinaCXRSet-ReadMe.docx`.

La carpeta se llama `ChinaSet_AllFiles`, no "Shenzhen". Es el nombre con el que la
NLM distribuye el conjunto y el que aparece en la literatura. **Se conserva tal
cual.**

Las carpetas `Annotations/` y `Annotations-2/` contienen anotaciones poligonales
de hallazgos compatibles con tuberculosis. **Este proyecto no las utiliza**: anotan
lesiones, no campo pulmonar. No descargarlas.

### 3. Máscaras de campo pulmonar de Shenzhen

No forman parte del paquete oficial de la NLM. Fueron preparadas por el Instituto
Politécnico de Kiev "Igor Sikorsky" y se distribuyen bajo licencia CC BY-NC-SA 4.0.

Fuente preferida, con DOI permanente y versión fija:

    https://doi.org/10.17632/8gf9vpkhgy.2

Fuente alternativa:

    https://www.kaggle.com/yoctoman/shcxr-lung-mask

Cubren 566 de las 662 imágenes de Shenzhen. Las 96 restantes no tienen máscara.
La carpeta se llama `shcxr-lung-mask` y contiene una subcarpeta `mask/`.

## Qué versión usar

Ninguno de los conjuntos de la NLM tiene número de versión. Como sustituto se
registran dos datos verificables:

| Elemento | Referencia de versión |
|---|---|
| Imágenes Montgomery y Shenzhen | Publicación NLM 2014; archivos con fecha de modificación 2024-08-28 en el servidor |
| Máscaras de Shenzhen | Mendeley Data, DOI 10.17632/8gf9vpkhgy, versión 2 (2022-10-03) |
| Fecha de consulta del equipo | 2026-09-16 |

## Estructura esperada

Después de descargar y descomprimir, sin renombrar nada:

    data/raw/
    ├── MontgomerySet/
    │   ├── CXR_png/                 138 archivos .png
    │   ├── ClinicalReadings/        138 archivos .txt
    │   ├── ManualMask/
    │   │   ├── leftMask/            138 archivos .png
    │   │   └── rightMask/           138 archivos .png
    │   └── NLM-MontgomeryCXRSet-ReadMe.pdf
    ├── ChinaSet_AllFiles/
    │   ├── CXR_png/                 662 archivos .png
    │   ├── ClinicalReadings/        662 archivos .txt
    │   └── NLM-ChinaCXRSet-ReadMe.docx
    └── shcxr-lung-mask/
        └── mask/                    566 archivos .png

Los nombres son sensibles a mayúsculas en Linux y macOS. Respetarlos exactamente
como aparecen arriba.

## Cómo verificar la copia

Ejecutar desde la raíz del proyecto:

    uv run python scripts/check_dataset.py

El script comprueba los conteos esperados, calcula el hash SHA-256 de cada archivo
y escribe un manifiesto en `results/fase1/manifiesto_datos.csv`. Ese manifiesto sí
se versiona en el repositorio: es lo que permite comprobar que dos copias del
conjunto son idénticas sin redistribuir las imágenes.

Cifras de referencia, verificadas en el índice del servidor de la NLM el
2026-09-16:

| Comprobación | Valor esperado |
|---|---|
| Imágenes Montgomery | 138 archivos, 614 034 765 bytes en total |
| Imágenes Shenzhen | 662 archivos, 3 772 099 214 bytes en total |
| Máscaras Montgomery | 138 izquierdas y 138 derechas |
| Máscaras Shenzhen | 566 archivos |
| Lecturas clínicas | 138 y 662 archivos de texto |

El total de las imágenes es de aproximadamente 4.08 GB.

Si algún conteo no coincide, la descarga está incompleta y debe repetirse antes de
continuar. Si los conteos coinciden pero los hashes difieren de los del manifiesto
versionado, la fuente cambió: documentar la discrepancia en el informe antes de
seguir, porque los resultados dejarían de ser comparables con los ya obtenidos.

## Nota sobre integridad

La NLM no publica sumas de verificación oficiales. El manifiesto que genera
`check_dataset.py` es la única garantía de integridad del proyecto, y por eso se
genera una sola vez, al inicio de la Fase 1, y no se regenera después salvo que se
documente la razón.

## Condiciones de uso

Ambos conjuntos fueron desidentificados por los proveedores y quedaron exentos de
revisión IRB; su uso y publicación fueron exentos por la NIH Office of Human
Research Protections Programs bajo el número 5357. La condición de uso incluye no
redistribuir los datos fuera del grupo de investigación, razón por la cual este
repositorio no los contiene.

Citación obligatoria en cualquier producto derivado:

> Jaeger S, Candemir S, Antani S, Wáng YXJ, Lu PX, Thoma G. Two public chest X-ray
> datasets for computer-aided screening of pulmonary diseases. Quant Imaging Med
> Surg. 2014;4(6):475-477. PMCID: PMC4256233.
