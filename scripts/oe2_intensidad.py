"""OE2, lote 4: histogramas e intensidad.

Usa la misma muestra de los lotes anteriores. Especifica el histograma de
Montgomery hacia Shenzhen dentro del campo y mide si eso acerca los origenes en
intensidad y en textura. Evalua la ecualizacion global y CLAHE con el criterio
de OE2 y genera las figuras de transformaciones de intensidad.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from oe2_espacial import decidir
from skimage.transform import resize

from psim import DATA_RAW, RAIZ
from psim.espacial import a_flotante, conservacion_bordes, contraste_rms, ruido_estimado
from psim.estadistica import delta_cliff, intervalo_delta
from psim.frecuencia import con_ventana, fraccion_alta, recorte_central
from psim.histograma import (
    cdf,
    clahe,
    distancia,
    ecualizar_global,
    especificar,
    histograma,
)
from psim.intensidad import (
    diferencia,
    gamma,
    logaritmica,
    normalizar_rango,
    requantizar,
)
from psim.io import leer_gris
from psim.recorte import mascara_campo
from psim.segmentacion import reducir
from psim.wavelets import energia_subbandas

MUESTRA = RAIZ / "results" / "fase2" / "muestra_oe2.csv"
OUT = RAIZ / "results" / "fase2"
FIG1 = RAIZ / "figures" / "fase1"
FIG2 = RAIZ / "figures" / "fase2"
ORIGEN = {"NLM-ChinaCXRSet": "Shenzhen", "NLM-MontgomeryCXRSet": "Montgomery"}
# Parametros fijados antes de ver resultados.
SEMILLA = 2026
PIXELES_REFERENCIA = 200_000
LADO_RECORTE = 192
MEDIDAS = ["mediana_intensidad", "fraccion_alta", "D1", "V1"]


def cargar(m: pd.DataFrame) -> list[dict]:
    """Imagen reducida, campo y origen de cada imagen de la muestra."""
    datos = []
    for _, fila in m.iterrows():
        g = reducir(leer_gris(DATA_RAW / fila["relative_path"]))
        datos.append(
            {
                "image_id": fila["image_id"],
                "origen": ORIGEN[fila["dataset_id"]],
                "x": a_flotante(g),
                "campo": mascara_campo(g),
            }
        )
    return datos


def referencia_shenzhen(datos: list[dict]) -> np.ndarray:
    """Pixeles del campo de las imagenes de Shenzhen, submuestreados con semilla."""
    rng = np.random.default_rng(SEMILLA)
    pixeles = np.concatenate(
        [d["x"][d["campo"]] for d in datos if d["origen"] == "Shenzhen"]
    )
    return rng.choice(
        pixeles, size=min(PIXELES_REFERENCIA, pixeles.size), replace=False
    )


def medir(x: np.ndarray, campo: np.ndarray, referencia: np.ndarray) -> dict[str, float]:
    """Intensidad, alta frecuencia y energia de detalle fino de una imagen."""
    recorte = recorte_central(x, campo, LADO_RECORTE)
    energias = energia_subbandas(recorte)
    return {
        "mediana_intensidad": float(np.median(x[campo])),
        "fraccion_alta": fraccion_alta(con_ventana(recorte)),
        "D1": energias["D1"],
        "V1": energias["V1"],
        "distancia_a_shenzhen": distancia(x[campo], referencia),
    }


def especificacion(datos: list[dict], referencia: np.ndarray) -> pd.DataFrame:
    """Medidas originales de todas y de Montgomery especificado hacia Shenzhen."""
    filas = []
    for d in datos:
        filas.append(
            {
                "image_id": d["image_id"],
                "origen": d["origen"],
                "version": "original",
                **medir(d["x"], d["campo"], referencia),
            }
        )
        if d["origen"] == "Montgomery":
            ajustada = especificar(d["x"], d["campo"], referencia)
            filas.append(
                {
                    "image_id": d["image_id"],
                    "origen": d["origen"],
                    "version": "especificada",
                    **medir(ajustada, d["campo"], referencia),
                }
            )
    return pd.DataFrame(filas)


def comparar(tabla: pd.DataFrame) -> pd.DataFrame:
    """Delta de Cliff de Montgomery frente a Shenzhen, antes y despues."""
    sh = tabla[(tabla["origen"] == "Shenzhen") & (tabla["version"] == "original")]
    filas = []
    for version in ("original", "especificada"):
        mc = tabla[(tabla["origen"] == "Montgomery") & (tabla["version"] == version)]
        for medida in [*MEDIDAS, "distancia_a_shenzhen"]:
            inferior, superior = intervalo_delta(
                mc[medida], sh[medida], semilla=SEMILLA
            )
            filas.append(
                {
                    "medida": medida,
                    "montgomery": version,
                    "mediana_mc": float(mc[medida].median()),
                    "mediana_sh": float(sh[medida].median()),
                    "delta_mc_vs_sh": delta_cliff(mc[medida], sh[medida]),
                    "ic_inf": inferior,
                    "ic_sup": superior,
                }
            )
    return pd.DataFrame(filas)


def realce(datos: list[dict]) -> pd.DataFrame:
    """Metricas de OE2 para la ecualizacion global y CLAHE."""
    estrategias = {
        "original": lambda x: x,
        "ecualizacion_global": ecualizar_global,
        "clahe_0.01": clahe,
    }
    filas = []
    for d in datos:
        for nombre, funcion in estrategias.items():
            y = funcion(d["x"])
            filas.append(
                {
                    "image_id": d["image_id"],
                    "origen": d["origen"],
                    "estrategia": nombre,
                    "contraste_rms": contraste_rms(y, d["campo"]),
                    "ruido": ruido_estimado(y),
                    "conservacion_bordes": conservacion_bordes(d["x"], y, d["campo"]),
                }
            )
    return pd.DataFrame(filas)


def guardar(fig: plt.Figure, ruta) -> None:
    """Guarda y cierra una figura."""
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def panel(eje: plt.Axes, imagen: np.ndarray, titulo: str, fijo: bool = True) -> None:
    """Muestra una imagen en gris sin ejes."""
    opciones = {"vmin": 0, "vmax": 1} if fijo else {}
    eje.imshow(imagen, cmap="gray", **opciones)
    eje.set_title(titulo, fontsize=9)
    eje.axis("off")


def figuras(datos: list[dict], referencia: np.ndarray) -> None:
    """Histogramas, especificacion, ecualizacion y transformaciones de intensidad."""
    ejemplo = {d["origen"]: d for d in reversed(datos)}
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    for origen in ORIGEN.values():
        pixeles = np.concatenate(
            [d["x"][d["campo"]] for d in datos if d["origen"] == origen]
        )
        ejes[0].plot(*histograma(pixeles), label=origen)
        ejes[1].plot(*cdf(pixeles), label=origen)
    ejes[0].set(
        title="Histograma del campo (muestra)", xlabel="Intensidad", ylabel="Densidad"
    )
    ejes[1].set(
        title="Funcion de distribucion acumulada", xlabel="Intensidad", ylabel="F(x)"
    )
    ejes[0].legend()
    guardar(fig, FIG1 / "histogramas.png")

    mc = ejemplo["Montgomery"]
    ajustada = especificar(mc["x"], mc["campo"], referencia)
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.5))
    panel(ejes[0], mc["x"], "Montgomery original")
    panel(ejes[1], ajustada, "Montgomery especificada hacia Shenzhen")
    for valores, etiqueta in (
        (mc["x"][mc["campo"]], "Montgomery original"),
        (ajustada[mc["campo"]], "Montgomery especificada"),
        (referencia, "Referencia Shenzhen"),
    ):
        ejes[2].plot(*histograma(valores), label=etiqueta)
    ejes[2].set(title="Histogramas del campo", xlabel="Intensidad")
    ejes[2].legend(fontsize=8)
    guardar(fig, FIG2 / "histogram_matching.png")

    fig, ejes = plt.subplots(2, 3, figsize=(11, 7))
    for i, (origen, d) in enumerate(sorted(ejemplo.items())):
        panel(ejes[i, 0], d["x"], f"{origen}: original")
        panel(ejes[i, 1], ecualizar_global(d["x"]), f"{origen}: ecualizacion global")
        panel(ejes[i, 2], clahe(d["x"]), f"{origen}: CLAHE")
    guardar(fig, FIG2 / "ecualizacion_comparacion.png")

    d = ejemplo["Shenzhen"]
    transformadas = {
        "original": d["x"],
        "rango 1-99 %": normalizar_rango(d["x"], d["campo"]),
        "gamma 0.5": gamma(d["x"], 0.5),
        "gamma 2.0": gamma(d["x"], 2.0),
        "logaritmica": logaritmica(d["x"]),
    }
    fig, ejes = plt.subplots(1, len(transformadas), figsize=(17, 3.8))
    for eje, (titulo, imagen) in zip(ejes, transformadas.items(), strict=True):
        panel(eje, imagen, f"Shenzhen: {titulo}")
    guardar(fig, FIG2 / "transformaciones_intensidad.png")

    mapas = {
        "|ecualizacion - original|": diferencia(ecualizar_global(d["x"]), d["x"]),
        "|CLAHE - original|": diferencia(clahe(d["x"]), d["x"]),
        "|gamma 0.5 - original|": diferencia(gamma(d["x"], 0.5), d["x"]),
    }
    fig, ejes = plt.subplots(1, len(mapas), figsize=(13, 4.2))
    for eje, (titulo, imagen) in zip(ejes, mapas.items(), strict=True):
        panel(eje, imagen, f"Shenzhen: {titulo}", fijo=False)
    guardar(fig, FIG2 / "mapas_diferencia.png")

    fig, ejes = plt.subplots(1, 4, figsize=(15, 4))
    for eje, bits in zip(ejes, (8, 4, 2, 1), strict=True):
        panel(eje, requantizar(d["x"], bits), f"{bits} bits ({2**bits} niveles)")
    guardar(fig, FIG1 / "requantizacion.png")

    fig, ejes = plt.subplots(1, 3, figsize=(12, 4.2))
    panel(ejes[0], d["x"], "512 px")
    for eje, suavizar in zip(ejes[1:], (False, True), strict=True):
        pequena = resize(
            d["x"], (64, 64), anti_aliasing=suavizar, order=0 if not suavizar else 1
        )
        nombre = "64 px con antialias" if suavizar else "64 px sin antialias (aliasing)"
        panel(eje, pequena, nombre)
    guardar(fig, FIG1 / "submuestreo.png")


def main() -> None:
    """Especificacion entre origenes, ecualizacion y figuras de intensidad."""
    plt.switch_backend("Agg")
    if not MUESTRA.is_file():
        raise FileNotFoundError("Falta muestra_oe2.csv: ejecute antes oe2_espacial.py.")
    for carpeta in (OUT, FIG1, FIG2):
        carpeta.mkdir(parents=True, exist_ok=True)
    datos = cargar(pd.read_csv(MUESTRA))
    referencia = referencia_shenzhen(datos)
    tabla = especificacion(datos, referencia)
    tabla.to_csv(OUT / "especificacion_histograma.csv", index=False)
    comparacion = comparar(tabla)
    comparacion.to_csv(OUT / "especificacion_por_origen.csv", index=False)
    metricas = realce(datos)
    metricas.to_csv(OUT / "metricas_realce_histograma.csv", index=False)
    decision = decidir(metricas)
    decision.to_csv(OUT / "decision_realce_histograma.csv")
    figuras(datos, referencia)
    print("=== Montgomery frente a Shenzhen, antes y despues de especificar ===")
    print("(delta negativo = menor en Montgomery)")
    print(comparacion.round(5).to_string(index=False))
    print("\n=== Ecualizacion con el criterio de OE2 ===")
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
