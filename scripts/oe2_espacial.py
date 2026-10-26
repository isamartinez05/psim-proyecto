"""OE2, lote 1: filtros espaciales, operadores de borde y metricas de realce.

Selecciona una muestra de entrenamiento estratificada por origen y clase,
aplica cada estrategia espacial y mide contraste RMS, ruido estimado y
conservacion de bordes dentro del campo irradiado. Aplica el criterio de la
formulacion con umbrales fijados antes de ver resultados.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psim import DATA_RAW, RAIZ
from psim.eda import cargar
from psim.espacial import (
    a_flotante,
    bordes,
    conservacion_bordes,
    contraste_rms,
    emboss,
    filtro_gaussiano,
    filtro_media,
    filtro_mediana,
    ruido_estimado,
    unsharp,
)
from psim.io import leer_gris
from psim.recorte import mascara_campo
from psim.segmentacion import reducir

SEMILLA = 2026
POR_ESTRATO = 10
OUT = RAIZ / "results" / "fase2"
FIG = RAIZ / "figures" / "fase2"
ORIGEN = {"NLM-ChinaCXRSet": "Shenzhen", "NLM-MontgomeryCXRSet": "Montgomery"}
# Criterio fijado antes de ver resultados.
TOLERANCIA_CONTRASTE = 0.05
TOLERANCIA_RUIDO = 0.05
MINIMO_BORDES = 0.90
ESTRATEGIAS = {
    "original": lambda x: x,
    "media_5": lambda x: filtro_media(x, 5),
    "mediana_5": lambda x: filtro_mediana(x, 5),
    "gaussiano_1.5": lambda x: filtro_gaussiano(x, 1.5),
    "unsharp_2_1": lambda x: unsharp(x, 2.0, 1.0),
}


def muestra() -> pd.DataFrame:
    """Muestra de train estratificada por origen y clase, con semilla fija."""
    d = cargar(RAIZ / "data/metadata/manifest.csv", RAIZ / "data/metadata/splits.csv")
    d = d[d["quality_status"] == "aceptada"]
    partes = [
        g.sample(n=POR_ESTRATO, random_state=SEMILLA)
        for _, g in d.groupby(["dataset_id", "target"], sort=True)
    ]
    m = pd.concat(partes, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    m[["image_id", "dataset_id", "target", "relative_path"]].to_csv(
        OUT / "muestra_oe2.csv", index=False
    )
    return m


def preparar(fila: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Imagen reducida en flotante y su campo irradiado."""
    g = reducir(leer_gris(DATA_RAW / fila["relative_path"]))
    return a_flotante(g), mascara_campo(g)


def medir(m: pd.DataFrame) -> pd.DataFrame:
    """Metricas de cada estrategia sobre cada imagen de la muestra."""
    filas = []
    for _, fila in m.iterrows():
        x, campo = preparar(fila)
        for nombre, funcion in ESTRATEGIAS.items():
            y = funcion(x)
            filas.append(
                {
                    "image_id": fila["image_id"],
                    "origen": ORIGEN[fila["dataset_id"]],
                    "target": fila["target"],
                    "estrategia": nombre,
                    "contraste_rms": contraste_rms(y, campo),
                    "ruido": ruido_estimado(y),
                    "conservacion_bordes": conservacion_bordes(x, y, campo),
                }
            )
    return pd.DataFrame(filas)


def decidir(metricas: pd.DataFrame) -> pd.DataFrame:
    """Aplica el criterio: mejora al menos una metrica sin degradar las otras."""
    mediana = metricas.groupby("estrategia")[
        ["contraste_rms", "ruido", "conservacion_bordes"]
    ].median()
    base = mediana.loc["original"]
    r = mediana.copy()
    r["cambio_contraste"] = r["contraste_rms"] / base["contraste_rms"] - 1
    r["cambio_ruido"] = r["ruido"] / base["ruido"] - 1
    r["mejora_contraste"] = r["cambio_contraste"] > TOLERANCIA_CONTRASTE
    r["mejora_ruido"] = r["cambio_ruido"] < -TOLERANCIA_RUIDO
    r["degrada"] = (
        (r["cambio_contraste"] < -TOLERANCIA_CONTRASTE)
        | (r["cambio_ruido"] > TOLERANCIA_RUIDO)
        | (r["conservacion_bordes"] < MINIMO_BORDES)
    )
    r["se_conserva"] = (r["mejora_contraste"] | r["mejora_ruido"]) & ~r["degrada"]
    r.loc["original", "se_conserva"] = False
    return r.round(4)


def ejemplos(m: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """La primera imagen de cada origen en la muestra."""
    return {
        ORIGEN[o]: preparar(g.iloc[0]) for o, g in m.groupby("dataset_id", sort=True)
    }


def figura_filtros(ej: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Original, media y gaussiano por origen."""
    columnas = ["original", "media_5", "gaussiano_1.5"]
    fig, ejes = plt.subplots(len(ej), len(columnas), figsize=(10, 7))
    for i, (origen, (x, _)) in enumerate(ej.items()):
        for j, nombre in enumerate(columnas):
            ejes[i, j].imshow(ESTRATEGIAS[nombre](x), cmap="gray", vmin=0, vmax=1)
            ejes[i, j].set_title(f"{origen}: {nombre}", fontsize=9)
            ejes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "filtros_comparacion.png", dpi=150)
    plt.close(fig)


def figura_mediana(ej: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Original y mediana, con un recorte ampliado del centro."""
    fig, ejes = plt.subplots(len(ej), 2, figsize=(7, 7))
    for i, (origen, (x, _)) in enumerate(ej.items()):
        c = x.shape[0] // 2
        recorte = slice(c - 64, c + 64), slice(c - 64, c + 64)
        for j, (nombre, y) in enumerate(
            (("original", x), ("mediana_5", filtro_mediana(x)))
        ):
            ejes[i, j].imshow(y[recorte], cmap="gray", vmin=0, vmax=1)
            ejes[i, j].set_title(f"{origen}: {nombre} (recorte)", fontsize=9)
            ejes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "median_comparison.png", dpi=150)
    plt.close(fig)


def figura_bordes(ej: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Los cuatro operadores de borde sobre una imagen de cada origen."""
    fig, ejes = plt.subplots(len(ej), 4, figsize=(13, 7))
    for i, (origen, (x, _)) in enumerate(ej.items()):
        for j, (nombre, magnitud) in enumerate(bordes(x).items()):
            ejes[i, j].imshow(magnitud, cmap="gray")
            ejes[i, j].set_title(f"{origen}: {nombre}", fontsize=9)
            ejes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "bordes_comparacion.png", dpi=150)
    plt.close(fig)


def figura_realce(ej: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Original frente a unsharp masking."""
    fig, ejes = plt.subplots(len(ej), 2, figsize=(7, 7))
    for i, (origen, (x, _)) in enumerate(ej.items()):
        for j, (nombre, y) in enumerate((("original", x), ("unsharp", unsharp(x)))):
            ejes[i, j].imshow(y, cmap="gray", vmin=0, vmax=1)
            ejes[i, j].set_title(f"{origen}: {nombre}", fontsize=9)
            ejes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "realce_comparacion.png", dpi=150)
    plt.close(fig)


def figura_emboss(ej: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Operador direccional de relieve por origen."""
    fig, ejes = plt.subplots(1, len(ej), figsize=(8, 4))
    for eje, (origen, (x, _)) in zip(ejes, ej.items(), strict=True):
        eje.imshow(emboss(x), cmap="gray")
        eje.set_title(f"{origen}: emboss", fontsize=9)
        eje.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "emboss.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Muestra, metricas, decision y figuras del lote espacial."""
    plt.switch_backend("Agg")
    FIG.mkdir(parents=True, exist_ok=True)
    m = muestra()
    print(f"Muestra: {len(m)} imagenes")
    print(pd.crosstab(m["dataset_id"], m["target"]).to_string())
    metricas = medir(m)
    metricas.to_csv(OUT / "metricas_realce.csv", index=False)
    decision = decidir(metricas)
    decision.to_csv(OUT / "decision_realce_espacial.csv")
    ej = ejemplos(m)
    figura_filtros(ej)
    figura_mediana(ej)
    figura_bordes(ej)
    figura_realce(ej)
    figura_emboss(ej)
    print("\n=== Mediana por estrategia y origen ===")
    print(
        metricas.groupby(["estrategia", "origen"])[
            ["contraste_rms", "ruido", "conservacion_bordes"]
        ]
        .median()
        .round(4)
        .to_string()
    )
    print("\n=== Decision con el criterio de la formulacion ===")
    print(
        decision[
            [
                "cambio_contraste",
                "cambio_ruido",
                "conservacion_bordes",
                "degrada",
                "se_conserva",
            ]
        ].to_string()
    )


if __name__ == "__main__":
    main()
