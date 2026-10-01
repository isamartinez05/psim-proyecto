"""Exploración visual y numérica de la cohorte de entrenamiento.

Todo lo que produce este módulo se calcula sobre la partición de
entrenamiento. Las de validación y prueba no se inspeccionan aquí: las
decisiones sobre parámetros y descriptores deben tomarse sin haber mirado los
datos contra los que después se reportará el resultado.

Dos advertencias que condicionan el diseño de las figuras:

- Los píxeles vecinos no son observaciones independientes. Un histograma que
  acumule todos los píxeles de todas las imágenes queda dominado por las
  imágenes de mayor tamaño y por el fondo, que ocupa la mayor parte del
  encuadre. Por eso cada histograma conjunto se acompaña de un resumen por
  imagen, que es la unidad de análisis declarada.

- Las imágenes de los dos conjuntos difieren en resolución hasta en un factor
  de cuatro. Una comparación de tamaños sin separar por origen mezcla dos
  efectos: la variabilidad propia del conjunto y la diferencia entre
  protocolos de adquisición.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psim import DATA_RAW, figura, resultado
from psim.io import leer_gris

# Semilla de la selección de la galería. Fija, para que la misma cohorte
# produzca siempre los mismos paneles.
SEMILLA_GALERIA = 20262

# Número de intervalos de los histogramas de intensidad. Se declara porque
# cambiarlo cambia cualquier descriptor derivado del histograma.
BINS = 64

# Rango de intensidad de los histogramas. Ambos conjuntos son de ocho bits
# efectivos, de modo que el rango es común y las distribuciones son
# directamente comparables.
RANGO = (0, 255)

AZUL = "#4A7FA5"
ROSA = "#C47F76"
GRIS = "#6E828C"


@dataclass(frozen=True)
class Panel:
    """Una entrada de la galería, con la razón por la que fue seleccionada."""

    image_id: str
    criterio: str
    nota: str


def cargar(manifiesto: Path, particion: Path, split: str = "train") -> pd.DataFrame:
    """Une manifiesto y partición, y devuelve una sola partición.

    La columna ``split`` del manifiesto se descarta: la partición es la
    autoridad sobre esa asignación.
    """
    m = pd.read_csv(manifiesto, keep_default_na=False)
    s = pd.read_csv(particion, keep_default_na=False)
    m = m.drop(columns=["split"], errors="ignore")
    d = m.merge(
        s[["image_id", "grupo_id", "estrato", "split"]],
        on="image_id",
        validate="1:1",
    )

    # La etiqueta se almacena como 0 y 1, que la lectura del CSV interpreta
    # como enteros. Compararla contra una cadena fallaría en silencio y
    # dejaría los filtros por clase vacíos, de modo que se normaliza aquí.
    d["target"] = d["target"].astype(str)

    return d[d["split"] == split].reset_index(drop=True)


def seleccionar_galeria(d: pd.DataFrame, semilla: int = SEMILLA_GALERIA) -> list[Panel]:
    """Selecciona los paneles de la galería según una regla registrada.

    La regla cubre las dos clases, los dos orígenes, los extremos de
    resolución, el grupo pediátrico y un caso que requiere revisión. Dentro de
    cada criterio la elección es aleatoria con semilla fija, de modo que no se
    escogen ejemplos de apariencia favorable.
    """
    rng = random.Random(semilla)
    elegidos: list[Panel] = []
    usados: set[str] = set()

    def tomar(sub: pd.DataFrame, n: int, criterio: str, nota: str) -> None:
        disponibles = [i for i in sub["image_id"] if i not in usados]
        for iid in rng.sample(disponibles, min(n, len(disponibles))):
            usados.add(iid)
            elegidos.append(Panel(iid, criterio, nota))

    es_mc = d["dataset_id"].str.endswith("MontgomeryCXRSet")

    tomar(d[es_mc & (d["target"] == "0")], 2, "clase_y_origen", "Montgomery sin hallazgos")
    tomar(d[es_mc & (d["target"] == "1")], 2, "clase_y_origen", "Montgomery con hallazgos")
    tomar(d[~es_mc & (d["target"] == "0")], 2, "clase_y_origen", "Shenzhen sin hallazgos")
    tomar(d[~es_mc & (d["target"] == "1")], 2, "clase_y_origen", "Shenzhen con hallazgos")

    # Extremos de resolución: la imagen con menos y con más píxeles.
    pixeles = d["height"] * d["width"]
    for idx, nota in [(pixeles.idxmin(), "menor resolucion"), (pixeles.idxmax(), "mayor resolucion")]:
        iid = d.loc[idx, "image_id"]
        if iid not in usados:
            usados.add(iid)
            elegidos.append(Panel(iid, "extremo_resolucion", nota))

    edad = pd.to_numeric(d["edad"], errors="coerce")
    tomar(d[edad < 2], 1, "pediatrico", "paciente menor de dos anos")

    tomar(d[d["quality_status"] == "revisar"], 1, "requiere_revision", "marcado en el inventario")

    return elegidos


def construir_galeria(d: pd.DataFrame, paneles: list[Panel], destino: Path) -> None:
    """Dibuja la galería con la misma escala de intensidad en todos los paneles.

    Las imágenes se muestran con ``vmin`` y ``vmax`` fijos en el rango de ocho
    bits. Usar una escala automática por panel haría que dos imágenes de
    contraste distinto se vieran iguales, que es precisamente lo que el
    proyecto quiere comparar.
    """
    n = len(paneles)
    cols = 4
    filas = (n + cols - 1) // cols
    fig, ejes = plt.subplots(filas, cols, figsize=(3.1 * cols, 3.5 * filas))
    ejes = np.atleast_1d(ejes).ravel()

    for eje, panel in zip(ejes, paneles, strict=False):
        fila = d[d["image_id"] == panel.image_id].iloc[0]
        img = leer_gris(DATA_RAW / fila["relative_path"])

        eje.imshow(img, cmap="gray", vmin=RANGO[0], vmax=RANGO[1])
        eje.set_xticks([])
        eje.set_yticks([])

        origen = "MC" if fila["dataset_id"].endswith("MontgomeryCXRSet") else "SH"
        clase = "TB+" if fila["target"] == "1" else "TB-"
        eje.set_title(
            f"{panel.image_id}\n{origen} · {clase} · {fila['height']}x{fila['width']}",
            fontsize=7.5,
        )
        eje.set_xlabel(panel.nota, fontsize=7, color=GRIS)

    for eje in ejes[n:]:
        eje.axis("off")

    fig.suptitle(
        "Galería de la cohorte de entrenamiento · escala común 0-255 · "
        "imagen completa sin recorte",
        fontsize=10,
    )
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=130, bbox_inches="tight")
    plt.close(fig)


def figura_dimensiones(d: pd.DataFrame, destino: Path) -> None:
    """Distribución de dimensiones, separada por origen."""
    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8))

    es_mc = d["dataset_id"].str.endswith("MontgomeryCXRSet")
    grupos = [("Montgomery", d[es_mc], AZUL), ("Shenzhen", d[~es_mc], ROSA)]

    for nombre, sub, color in grupos:
        ejes[0].scatter(sub["width"], sub["height"], s=9, alpha=0.45, color=color, label=nombre)
    ejes[0].set_xlabel("ancho (px)")
    ejes[0].set_ylabel("alto (px)")
    ejes[0].set_title(f"Dimensiones por origen (n={len(d)})", fontsize=10)
    ejes[0].legend(fontsize=8)

    for nombre, sub, color in grupos:
        mp = (sub["height"] * sub["width"]) / 1e6
        ejes[1].hist(mp, bins=30, alpha=0.6, color=color, label=nombre)
    ejes[1].set_xlabel("megapixeles por imagen")
    ejes[1].set_ylabel("imagenes")
    ejes[1].set_title("Tamano total", fontsize=10)
    ejes[1].legend(fontsize=8)

    tabla = pd.crosstab(
        np.where(es_mc, "Montgomery", "Shenzhen"),
        d["orientacion"],
    )
    tabla.plot(kind="bar", ax=ejes[2], color=[AZUL, ROSA], rot=0)
    ejes[2].set_ylabel("imagenes")
    ejes[2].set_xlabel("")
    ejes[2].set_title("Orientacion", fontsize=10)
    ejes[2].legend(fontsize=8)

    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=130, bbox_inches="tight")
    plt.close(fig)


def estadisticos_por_imagen(d: pd.DataFrame, muestra: int | None = None) -> pd.DataFrame:
    """Calcula mediana, IQR, media y desviación por imagen.

    Las estadísticas se calculan sobre la imagen completa en su escala de
    almacenamiento, sin normalizar. El rango intercuartílico no está en el
    manifiesto y se añade aquí porque resume la dispersión sin verse arrastrado
    por el fondo, a diferencia de la desviación.
    """
    filas = d if muestra is None else d.sample(muestra, random_state=SEMILLA_GALERIA)
    salida = []
    for _, fila in filas.iterrows():
        img = leer_gris(DATA_RAW / fila["relative_path"]).astype(np.float64)
        q25, q50, q75 = np.percentile(img, [25, 50, 75])
        salida.append({
            "image_id": fila["image_id"],
            "origen": "Montgomery" if fila["dataset_id"].endswith("MontgomeryCXRSet") else "Shenzhen",
            "target": fila["target"],
            "media": round(float(img.mean()), 3),
            "desviacion": round(float(img.std(ddof=0)), 3),
            "mediana": float(q50),
            "iqr": float(q75 - q25),
            "p25": float(q25),
            "p75": float(q75),
        })
    return pd.DataFrame(salida)


def figura_intensidad(
    d: pd.DataFrame,
    stats: pd.DataFrame,
    destino: Path,
    n_histogramas: int = 40,
) -> None:
    """Histogramas de intensidad y resúmenes por imagen.

    El panel izquierdo superpone el histograma de una muestra de imágenes,
    cada uno normalizado a su propio número de píxeles. Sin esa normalización
    las imágenes grandes dominarían el gráfico por tener más píxeles, no por
    tener una distribución distinta.
    """
    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8))
    bordes = np.linspace(RANGO[0], RANGO[1] + 1, BINS + 1)

    es_mc = d["dataset_id"].str.endswith("MontgomeryCXRSet")
    muestra = pd.concat([
        d[es_mc].sample(min(n_histogramas // 2, int(es_mc.sum())), random_state=SEMILLA_GALERIA),
        d[~es_mc].sample(min(n_histogramas // 2, int((~es_mc).sum())), random_state=SEMILLA_GALERIA),
    ])

    for _, fila in muestra.iterrows():
        img = leer_gris(DATA_RAW / fila["relative_path"])
        h, _ = np.histogram(img, bins=bordes)
        color = AZUL if fila["dataset_id"].endswith("MontgomeryCXRSet") else ROSA
        ejes[0].plot(bordes[:-1], h / h.sum(), color=color, alpha=0.3, linewidth=0.8)

    ejes[0].set_xlabel("intensidad almacenada (0-255)")
    ejes[0].set_ylabel("fraccion de pixeles")
    ejes[0].set_title(
        f"Histogramas por imagen (n={len(muestra)}, {BINS} intervalos)\n"
        "azul Montgomery · rosa Shenzhen",
        fontsize=9,
    )

    for nombre, color in [("Montgomery", AZUL), ("Shenzhen", ROSA)]:
        sub = stats[stats["origen"] == nombre]
        ejes[1].scatter(sub["mediana"], sub["iqr"], s=11, alpha=0.5, color=color, label=nombre)
    ejes[1].set_xlabel("mediana de intensidad")
    ejes[1].set_ylabel("rango intercuartilico")
    ejes[1].set_title(f"Resumen por imagen (n={len(stats)})", fontsize=10)
    ejes[1].legend(fontsize=8)

    datos = [stats[stats["origen"] == n]["mediana"].to_numpy() for n in ("Montgomery", "Shenzhen")]
    bp = ejes[2].boxplot(datos, tick_labels=["Montgomery", "Shenzhen"], patch_artist=True)
    for parche, color in zip(bp["boxes"], [AZUL, ROSA], strict=False):
        parche.set_facecolor(color)
        parche.set_alpha(0.6)
    ejes[2].set_ylabel("mediana de intensidad")
    ejes[2].set_title("Distribucion de la mediana", fontsize=10)

    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=130, bbox_inches="tight")
    plt.close(fig)


def resumen_calidad(d: pd.DataFrame) -> pd.DataFrame:
    """Tabla de casos aceptados y casos que requieren revisión, con su razón."""
    filas = []
    for estado, sub in d.groupby("quality_status"):
        if estado == "aceptada":
            filas.append({"estado": estado, "razon": "", "n": len(sub), "ejemplos": ""})
            continue
        for razon, s2 in sub.groupby("quality_reason"):
            filas.append({
                "estado": estado,
                "razon": razon,
                "n": len(s2),
                "ejemplos": ", ".join(sorted(s2["image_id"])[:3]),
            })
    return pd.DataFrame(filas).sort_values(["estado", "n"], ascending=[True, False])


def ejecutar(manifiesto: Path, particion: Path) -> dict[str, object]:
    """Produce todas las figuras y tablas de la exploración inicial."""
    d = cargar(manifiesto, particion, split="train")

    paneles = seleccionar_galeria(d)
    construir_galeria(d, paneles, figura(2, "galeria_entrenamiento.png"))

    figura_dimensiones(d, figura(2, "dimensiones.png"))

    stats = estadisticos_por_imagen(d)
    stats.to_csv(resultado(2, "estadisticos_por_imagen.csv"), index=False)
    figura_intensidad(d, stats, figura(2, "intensidad.png"))

    calidad = resumen_calidad(d)
    calidad.to_csv(resultado(2, "resumen_calidad.csv"), index=False)

    pd.DataFrame([{"image_id": p.image_id, "criterio": p.criterio, "nota": p.nota} for p in paneles]).to_csv(
        resultado(2, "galeria_seleccion.csv"), index=False
    )

    return {"n_train": len(d), "paneles": len(paneles), "stats": len(stats)}