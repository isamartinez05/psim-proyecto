"""OE2, lote 3: wavelets y multirresolucion.

Usa la misma muestra y el mismo recorte central de los lotes 1 y 2. Calcula la
energia relativa por subbanda de una DWT db2 de tres niveles, la compara entre
origenes, muestra la reconstruccion por niveles y el escalograma de una wavelet
continua sobre un perfil horizontal.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psim import DATA_RAW, RAIZ
from psim.espacial import a_flotante
from psim.estadistica import comparar_grupos
from psim.frecuencia import recorte_central
from psim.io import leer_gris
from psim.recorte import mascara_campo
from psim.segmentacion import reducir
from psim.wavelets import (
    energia_por_nivel,
    energia_subbandas,
    escalograma,
    nombres_subbandas,
    perfil_horizontal,
    reconstruir_sin_detalles,
)

MUESTRA = RAIZ / "results" / "fase2" / "muestra_oe2.csv"
OUT = RAIZ / "results" / "fase3"
FIG2 = RAIZ / "figures" / "fase2"
FIG3 = RAIZ / "figures" / "fase3"
ORIGEN = {"NLM-ChinaCXRSet": "Shenzhen", "NLM-MontgomeryCXRSet": "Montgomery"}
# Parametros fijados antes de ver resultados.
WAVELET = "db2"
NIVELES = 3
LADO_RECORTE = 192
WAVELET_CONTINUA = "mexh"
ESCALAS = np.arange(1, 65)
SEMILLA = 2026


def preparar(fila: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Imagen reducida en flotante y su campo irradiado."""
    g = reducir(leer_gris(DATA_RAW / fila["relative_path"]))
    return a_flotante(g), mascara_campo(g)


def energias(m: pd.DataFrame) -> pd.DataFrame:
    """Energia relativa por subbanda del recorte central de cada imagen."""
    filas = []
    for _, fila in m.iterrows():
        x, campo = preparar(fila)
        recorte = recorte_central(x, campo, LADO_RECORTE)
        filas.append(
            {
                "image_id": fila["image_id"],
                "origen": ORIGEN[fila["dataset_id"]],
                "target": fila["target"],
                **energia_subbandas(recorte, WAVELET, NIVELES),
            }
        )
    return pd.DataFrame(filas)


def comparar_origenes(e: pd.DataFrame) -> pd.DataFrame:
    """Delta de Cliff por subbanda, Montgomery frente a Shenzhen, con Holm."""
    datos = e.assign(
        es_montgomery=(e["origen"] == "Montgomery").astype(int), muestra="todas"
    )
    r = comparar_grupos(
        datos,
        nombres_subbandas(NIVELES),
        grupo="es_montgomery",
        estrato="muestra",
        semilla=SEMILLA,
    )
    return r.rename(columns={"mediana_con": "mediana_mc", "mediana_sin": "mediana_sh"})


def figura_energia(e: pd.DataFrame) -> None:
    """Energia relativa por escala (H + V + D) y por origen, en escala logaritmica."""
    por_nivel = pd.DataFrame(
        [
            {"origen": o, **energia_por_nivel(fila, NIVELES)}
            for o, fila in zip(
                e["origen"],
                e[nombres_subbandas(NIVELES)].to_dict("records"),
                strict=True,
            )
        ]
    )
    escalas = [c for c in por_nivel.columns if c != "origen"]
    medianas = por_nivel.groupby("origen")[escalas].median()
    x = np.arange(len(escalas))
    fig, eje = plt.subplots(figsize=(8, 4.5))
    for desplazamiento, origen in zip((-0.2, 0.2), medianas.index, strict=True):
        eje.bar(x + desplazamiento, medianas.loc[origen], width=0.4, label=origen)
    eje.set_yscale("log")
    eje.set_xticks(x, escalas)
    eje.set(
        ylabel="Energia relativa (mediana)",
        title=f"Energia por escala, DWT {WAVELET} de {NIVELES} niveles (recorte central)",
    )
    eje.legend()
    fig.tight_layout()
    fig.savefig(FIG3 / "energia_wavelets.png", dpi=150)
    plt.close(fig)


def figura_reconstruccion(m: pd.DataFrame) -> None:
    """Imagen reconstruida quitando detalles de escala en escala."""
    ejemplos = [g.iloc[0] for _, g in m.groupby("dataset_id", sort=True)]
    fig, ejes = plt.subplots(len(ejemplos), NIVELES + 1, figsize=(13, 7))
    for i, fila in enumerate(ejemplos):
        x, campo = preparar(fila)
        recorte = recorte_central(x, campo, LADO_RECORTE)
        for quitar in range(NIVELES + 1):
            imagen = reconstruir_sin_detalles(recorte, quitar, WAVELET, NIVELES)
            titulo = "original" if quitar == 0 else f"sin niveles 1 a {quitar}"
            ejes[i, quitar].imshow(imagen, cmap="gray", vmin=0, vmax=1)
            ejes[i, quitar].set_title(
                f"{ORIGEN[fila['dataset_id']]}: {titulo}", fontsize=9
            )
            ejes[i, quitar].axis("off")
    fig.tight_layout()
    fig.savefig(FIG3 / "reconstruccion_dwt.png", dpi=150)
    plt.close(fig)


def figura_cwt(m: pd.DataFrame) -> None:
    """Imagen con el perfil marcado y escalograma de la wavelet continua."""
    ejemplos = [g.iloc[0] for _, g in m.groupby("dataset_id", sort=True)]
    fig, ejes = plt.subplots(len(ejemplos), 3, figsize=(15, 7))
    for i, fila in enumerate(ejemplos):
        origen = ORIGEN[fila["dataset_id"]]
        x, campo = preparar(fila)
        y, perfil = perfil_horizontal(x, campo)
        ejes[i, 0].imshow(x, cmap="gray", vmin=0, vmax=1)
        ejes[i, 0].axhline(y, color="yellow", linewidth=1)
        ejes[i, 0].set_title(f"{origen}: perfil en la fila {y}", fontsize=9)
        ejes[i, 0].axis("off")
        ejes[i, 1].plot(perfil)
        ejes[i, 1].set(
            title="Intensidad del perfil", xlabel="Columna", ylabel="Intensidad"
        )
        ejes[i, 2].imshow(
            escalograma(perfil, ESCALAS, WAVELET_CONTINUA),
            aspect="auto",
            cmap="magma",
            extent=(0, len(perfil), ESCALAS[-1], ESCALAS[0]),
        )
        ejes[i, 2].set(
            title=f"Escalograma ({WAVELET_CONTINUA})", xlabel="Columna", ylabel="Escala"
        )
    fig.tight_layout()
    fig.savefig(FIG2 / "cwt_perfil.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Energia por subbanda, comparacion de origenes y figuras de wavelets."""
    plt.switch_backend("Agg")
    if not MUESTRA.is_file():
        raise FileNotFoundError("Falta muestra_oe2.csv: ejecute antes oe2_espacial.py.")
    for carpeta in (OUT, FIG2, FIG3):
        carpeta.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MUESTRA)
    e = energias(m)
    e.to_csv(OUT / "energia_subbandas.csv", index=False)
    comparacion = comparar_origenes(e)
    comparacion.to_csv(OUT / "energia_subbandas_por_origen.csv", index=False)
    figura_energia(e)
    figura_reconstruccion(m)
    figura_cwt(m)
    columnas = [
        "descriptor",
        "mediana_mc",
        "mediana_sh",
        "delta",
        "ic_inf",
        "ic_sup",
        "p_holm",
        "significativo",
    ]
    print("=== Energia relativa por subbanda: Montgomery frente a Shenzhen ===")
    print("(delta positivo = mas energia en Montgomery)")
    print(comparacion[columnas].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
