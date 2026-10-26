"""OE2, lote 2: dominio de frecuencia.

Usa la misma muestra del lote 1. Compara el espectro de cada origen sobre un
recorte central del campo con ventana de Hann, evalua dos pasa-bajas con las
metricas y el criterio de OE2, y muestra pasa-altas y pasa-banda.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psim import DATA_RAW, RAIZ
from psim.espacial import a_flotante, conservacion_bordes, contraste_rms, ruido_estimado
from psim.estadistica import delta_cliff, intervalo_delta
from psim.frecuencia import (
    con_ventana,
    espectro,
    filtrar,
    fraccion_alta,
    pasaaltas_butterworth,
    pasabajas_butterworth,
    pasabajas_gaussiano,
    pasabanda_butterworth,
    perfil_radial,
    recorte_central,
)
from psim.io import leer_gris
from psim.recorte import mascara_campo
from psim.segmentacion import reducir

OUT = RAIZ / "results" / "fase2"
FIG = RAIZ / "figures" / "fase2"
MUESTRA = OUT / "muestra_oe2.csv"
ORIGEN = {"NLM-ChinaCXRSet": "Shenzhen", "NLM-MontgomeryCXRSet": "Montgomery"}
SEMILLA = 2026
LADO_RECORTE = 192
CORTE_ALTA = 0.25
# Parametros y criterio fijados antes de ver resultados (iguales al lote 1).
CORTE_PASABAJAS = 0.15
ORDEN = 2
CORTE_PASAALTAS = 0.05
BANDA = (0.03, 0.15)
TOLERANCIA_CONTRASTE = 0.05
TOLERANCIA_RUIDO = 0.05
MINIMO_BORDES = 0.90


def preparar(fila: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Imagen reducida en flotante y su campo irradiado."""
    g = reducir(leer_gris(DATA_RAW / fila["relative_path"]))
    return a_flotante(g), mascara_campo(g)


def estrategias(forma: tuple[int, int]) -> dict[str, np.ndarray | None]:
    """Funciones de transferencia de los pasa-bajas evaluados."""
    return {
        "original": None,
        "pb_gaussiano_0.15": pasabajas_gaussiano(forma, CORTE_PASABAJAS),
        "pb_butterworth_0.15": pasabajas_butterworth(forma, CORTE_PASABAJAS, ORDEN),
    }


def analizar(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Metricas de los pasa-bajas, fraccion de alta frecuencia y perfiles por origen."""
    metricas, espectral = [], []
    perfiles: dict[str, list[np.ndarray]] = {o: [] for o in ORIGEN.values()}
    espectros2d: dict[str, list[np.ndarray]] = {o: [] for o in ORIGEN.values()}
    for _, fila in m.iterrows():
        origen = ORIGEN[fila["dataset_id"]]
        x, campo = preparar(fila)
        for nombre, h in estrategias(x.shape).items():
            y = x if h is None else np.clip(filtrar(x, h), 0.0, 1.0)
            metricas.append(
                {
                    "image_id": fila["image_id"],
                    "origen": origen,
                    "estrategia": nombre,
                    "contraste_rms": contraste_rms(y, campo),
                    "ruido": ruido_estimado(y),
                    "conservacion_bordes": conservacion_bordes(x, y, campo),
                }
            )
        recorte = con_ventana(recorte_central(x, campo, LADO_RECORTE))
        frecuencias, potencia = perfil_radial(recorte)
        perfiles[origen].append(potencia / potencia.sum())
        espectros2d[origen].append(espectro(recorte))
        espectral.append(
            {
                "image_id": fila["image_id"],
                "origen": origen,
                "target": fila["target"],
                "fraccion_alta": fraccion_alta(recorte, CORTE_ALTA),
                "fraccion_campo_en_recorte": float(
                    recorte_central(campo, campo, LADO_RECORTE).mean()
                ),
            }
        )
    extra = {"frecuencias": frecuencias, "perfiles": perfiles, "espectros": espectros2d}
    return pd.DataFrame(metricas), pd.DataFrame(espectral), extra


def decidir(metricas: pd.DataFrame) -> pd.DataFrame:
    """Mismo criterio del lote 1: mejora una metrica sin degradar las otras."""
    mediana = metricas.groupby("estrategia")[
        ["contraste_rms", "ruido", "conservacion_bordes"]
    ].median()
    base = mediana.loc["original"]
    r = mediana.copy()
    r["cambio_contraste"] = r["contraste_rms"] / base["contraste_rms"] - 1
    r["cambio_ruido"] = r["ruido"] / base["ruido"] - 1
    mejora = (r["cambio_contraste"] > TOLERANCIA_CONTRASTE) | (
        r["cambio_ruido"] < -TOLERANCIA_RUIDO
    )
    r["degrada"] = (
        (r["cambio_contraste"] < -TOLERANCIA_CONTRASTE)
        | (r["cambio_ruido"] > TOLERANCIA_RUIDO)
        | (r["conservacion_bordes"] < MINIMO_BORDES)
    )
    r["se_conserva"] = mejora & ~r["degrada"]
    r.loc["original", "se_conserva"] = False
    return r.round(4)


def comparar_origenes(espectral: pd.DataFrame) -> dict[str, float]:
    """Delta de Cliff de la fraccion de alta frecuencia, Montgomery frente a Shenzhen."""
    mc = espectral.loc[espectral["origen"] == "Montgomery", "fraccion_alta"]
    sh = espectral.loc[espectral["origen"] == "Shenzhen", "fraccion_alta"]
    inferior, superior = intervalo_delta(mc, sh, semilla=SEMILLA)
    return {
        "mediana_montgomery": float(mc.median()),
        "mediana_shenzhen": float(sh.median()),
        "delta_mc_vs_sh": delta_cliff(mc, sh),
        "ic_inf": inferior,
        "ic_sup": superior,
    }


def figura_espectros(extra: dict) -> None:
    """Espectro 2D medio de cada origen y perfil radial normalizado."""
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
    for eje, origen in zip(ejes[:2], ORIGEN.values(), strict=True):
        eje.imshow(np.mean(extra["espectros"][origen], axis=0), cmap="magma")
        eje.set_title(f"{origen}: log |FFT| medio (recorte central)", fontsize=9)
        eje.axis("off")
    for origen in ORIGEN.values():
        perfiles = np.array(extra["perfiles"][origen])
        ejes[2].semilogy(extra["frecuencias"], perfiles.mean(axis=0), label=origen)
    ejes[2].axvline(CORTE_ALTA, color="gray", linestyle="--", linewidth=1)
    ejes[2].set(
        xlabel="Frecuencia (ciclos/pixel)",
        ylabel="Potencia normalizada",
        title="Perfil radial de potencia",
    )
    ejes[2].legend()
    fig.tight_layout()
    fig.savefig(FIG / "espectros.png", dpi=150)
    plt.close(fig)


def figura_filtros(m: pd.DataFrame) -> None:
    """Pasa-bajas, pasa-altas y pasa-banda sobre una imagen de cada origen."""
    ejemplos = [g.iloc[0] for _, g in m.groupby("dataset_id", sort=True)]
    filas_pb, filas_pa, filas_bp = [], [], []
    for fila in ejemplos:
        x, _ = preparar(fila)
        origen = ORIGEN[fila["dataset_id"]]
        filas_pb.append(
            (
                origen,
                [
                    ("original", x),
                    (
                        "pb gaussiano",
                        filtrar(x, pasabajas_gaussiano(x.shape, CORTE_PASABAJAS)),
                    ),
                    (
                        "pb Butterworth",
                        filtrar(
                            x, pasabajas_butterworth(x.shape, CORTE_PASABAJAS, ORDEN)
                        ),
                    ),
                ],
            )
        )
        filas_pa.append(
            (
                origen,
                [
                    (
                        "pasa-altas",
                        filtrar(x, pasaaltas_butterworth(x.shape, CORTE_PASAALTAS)),
                    )
                ],
            )
        )
        filas_bp.append(
            (
                origen,
                [("pasa-banda", filtrar(x, pasabanda_butterworth(x.shape, *BANDA)))],
            )
        )
    for nombre_archivo, filas, gris_fijo in (
        ("pasabajas_comparacion.png", filas_pb, True),
        ("pasaaltas.png", filas_pa, False),
        ("pasabanda.png", filas_bp, False),
    ):
        columnas = len(filas[0][1])
        fig, ejes = plt.subplots(
            len(filas), columnas, figsize=(4 * columnas, 7), squeeze=False
        )
        for i, (origen, paneles) in enumerate(filas):
            for j, (titulo, imagen) in enumerate(paneles):
                opciones = {"vmin": 0, "vmax": 1} if gris_fijo else {}
                ejes[i, j].imshow(imagen, cmap="gray", **opciones)
                ejes[i, j].set_title(f"{origen}: {titulo}", fontsize=9)
                ejes[i, j].axis("off")
        fig.tight_layout()
        fig.savefig(FIG / nombre_archivo, dpi=150)
        plt.close(fig)


def main() -> None:
    """Lote de frecuencia: metricas, decision, comparacion de origenes y figuras."""
    plt.switch_backend("Agg")
    if not MUESTRA.is_file():
        raise FileNotFoundError("Falta muestra_oe2.csv: ejecute antes oe2_espacial.py.")
    FIG.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MUESTRA)
    metricas, espectral, extra = analizar(m)
    metricas.to_csv(OUT / "metricas_frecuencia.csv", index=False)
    espectral.to_csv(OUT / "espectro_por_imagen.csv", index=False)
    decision = decidir(metricas)
    decision.to_csv(OUT / "decision_realce_frecuencia.csv")
    comparacion = comparar_origenes(espectral)
    pd.DataFrame([comparacion]).to_csv(OUT / "espectro_por_origen.csv", index=False)
    figura_espectros(extra)
    figura_filtros(m)

    print("=== Decision con el criterio de OE2 ===")
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
    print("\n=== Fraccion de potencia sobre 0.25 ciclos/pixel, por origen ===")
    print(espectral.groupby("origen")["fraccion_alta"].describe().round(5).to_string())
    print("\nMontgomery frente a Shenzhen:")
    for clave, valor in comparacion.items():
        print(f"  {clave}: {valor:.4f}")
    print(
        "\nFraccion minima del recorte dentro del campo:",
        f"{espectral['fraccion_campo_en_recorte'].min():.3f}",
    )


if __name__ == "__main__":
    main()
