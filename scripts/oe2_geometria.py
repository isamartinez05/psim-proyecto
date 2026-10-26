"""OE2, lote 5: geometria de regiones y morfologia.

Usa la misma muestra. Cuenta componentes con vecindad 4 y 8 sobre las mascaras
automaticas de base_adap, y genera las figuras de conectividad, transformada
de distancia, pasos morfologicos y top-hat / black-hat.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psim import DATA_RAW, RAIZ
from psim.espacial import a_flotante
from psim.geometria import (
    apertura,
    black_hat,
    borde,
    cierre,
    dilatacion,
    erosion,
    etiquetar,
    top_hat,
    transformada_distancia,
)
from psim.io import leer_gris
from psim.recorte import mascara_campo
from psim.segmentacion import reducir, segmentar

MUESTRA = RAIZ / "results" / "fase2" / "muestra_oe2.csv"
OUT = RAIZ / "results" / "fase2"
FIG1 = RAIZ / "figures" / "fase1"
FIG2 = RAIZ / "figures" / "fase2"
ORIGEN = {"NLM-ChinaCXRSet": "Shenzhen", "NLM-MontgomeryCXRSet": "Montgomery"}
ESTRATEGIA = "base_adap"
RADIO_MORFOLOGIA = 5
RADIO_SOMBRERO = 15


def preparar(fila: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Imagen en flotante y mascara automatica de la estrategia del proyecto."""
    g = reducir(leer_gris(DATA_RAW / fila["relative_path"]))
    campo = mascara_campo(g)
    return a_flotante(g), segmentar(g, ESTRATEGIA, campo).mascara


def conteo_componentes(m: pd.DataFrame) -> pd.DataFrame:
    """Componentes con vecindad 4 y 8 en la mascara de cada imagen."""
    filas = []
    for _, fila in m.iterrows():
        _, mascara = preparar(fila)
        filas.append(
            {
                "image_id": fila["image_id"],
                "origen": ORIGEN[fila["dataset_id"]],
                "componentes_4": etiquetar(mascara, 4)[1],
                "componentes_8": etiquetar(mascara, 8)[1],
            }
        )
    return pd.DataFrame(filas)


def guardar(fig: plt.Figure, ruta) -> None:
    """Guarda y cierra una figura."""
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figuras(m: pd.DataFrame) -> None:
    """Conectividad, distancia y borde, pasos morfologicos y sombreros."""
    ejemplos = {
        ORIGEN[o]: preparar(g.iloc[0]) for o, g in m.groupby("dataset_id", sort=True)
    }

    fig, ejes = plt.subplots(len(ejemplos), 2, figsize=(8, 8))
    for i, (origen, (_, mascara)) in enumerate(ejemplos.items()):
        for j, vecindad in enumerate((4, 8)):
            etiquetas, n = etiquetar(mascara, vecindad)
            ejes[i, j].imshow(np.ma.masked_equal(etiquetas, 0), cmap="tab20")
            ejes[i, j].set_title(
                f"{origen}: vecindad {vecindad}, {n} regiones", fontsize=9
            )
            ejes[i, j].axis("off")
    guardar(fig, FIG1 / "conectividad.png")

    fig, ejes = plt.subplots(len(ejemplos), 2, figsize=(8, 8))
    for i, (origen, (x, mascara)) in enumerate(ejemplos.items()):
        ejes[i, 0].imshow(transformada_distancia(mascara), cmap="viridis")
        ejes[i, 0].set_title(f"{origen}: transformada de distancia", fontsize=9)
        ejes[i, 1].imshow(x, cmap="gray", vmin=0, vmax=1)
        ejes[i, 1].contour(
            borde(mascara), levels=[0.5], colors="yellow", linewidths=0.8
        )
        ejes[i, 1].set_title(f"{origen}: borde de la mascara", fontsize=9)
        for eje in ejes[i]:
            eje.axis("off")
    guardar(fig, FIG1 / "distancia_borde.png")

    origen, (x, mascara) = next(iter(ejemplos.items()))
    pasos = {
        "mascara": mascara,
        f"erosion r={RADIO_MORFOLOGIA}": erosion(mascara, RADIO_MORFOLOGIA),
        f"dilatacion r={RADIO_MORFOLOGIA}": dilatacion(mascara, RADIO_MORFOLOGIA),
        f"apertura r={RADIO_MORFOLOGIA}": apertura(mascara, RADIO_MORFOLOGIA),
        f"cierre r={RADIO_MORFOLOGIA}": cierre(mascara, RADIO_MORFOLOGIA),
    }
    fig, ejes = plt.subplots(1, len(pasos), figsize=(17, 3.8))
    for eje, (titulo, imagen) in zip(ejes, pasos.items(), strict=True):
        eje.imshow(imagen, cmap="gray")
        eje.set_title(f"{origen}: {titulo}", fontsize=9)
        eje.axis("off")
    guardar(fig, FIG2 / "morfologia_pasos.png")

    fig, ejes = plt.subplots(len(ejemplos), 3, figsize=(12, 8))
    for i, (origen, (x, _)) in enumerate(ejemplos.items()):
        paneles = (
            ("original", x),
            (f"top-hat r={RADIO_SOMBRERO}", top_hat(x, RADIO_SOMBRERO)),
            (f"black-hat r={RADIO_SOMBRERO}", black_hat(x, RADIO_SOMBRERO)),
        )
        for j, (titulo, imagen) in enumerate(paneles):
            ejes[i, j].imshow(imagen, cmap="gray")
            ejes[i, j].set_title(f"{origen}: {titulo}", fontsize=9)
            ejes[i, j].axis("off")
    guardar(fig, FIG2 / "tophat_blackhat.png")


def main() -> None:
    """Conteo de componentes por vecindad y figuras de geometria y morfologia."""
    plt.switch_backend("Agg")
    if not MUESTRA.is_file():
        raise FileNotFoundError("Falta muestra_oe2.csv: ejecute antes oe2_espacial.py.")
    for carpeta in (OUT, FIG1, FIG2):
        carpeta.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MUESTRA)
    conteo = conteo_componentes(m)
    conteo.to_csv(OUT / "conectividad_componentes.csv", index=False)
    figuras(m)
    print("=== Componentes de la mascara automatica segun la vecindad ===")
    print(pd.crosstab(conteo["componentes_4"], conteo["componentes_8"]).to_string())
    distintos = conteo[conteo["componentes_4"] != conteo["componentes_8"]]
    print(
        f"\nImagenes donde la vecindad cambia el conteo: {len(distintos)} de {len(conteo)}"
    )
    elegible_4 = (conteo["componentes_4"] == 2).sum()
    elegible_8 = (conteo["componentes_8"] == 2).sum()
    print(
        f"Con exactamente 2 componentes: vecindad 4 = {elegible_4}, vecindad 8 = {elegible_8}"
    )


if __name__ == "__main__":
    main()
