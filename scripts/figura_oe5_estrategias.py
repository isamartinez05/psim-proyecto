"""Mapa de calor del delta de Cliff por estrategia, descriptor y sitio (OE5).

Lee results/fase3/oe5_estrategias_train.csv; no recalcula nada. Cada celda es
el delta de Cliff entre grupos en train. El asterisco marca significancia tras
Holm y el recuadro, las celdas que no coinciden con la referencia en decision
o en signo.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from psim import RAIZ

ENTRADA = RAIZ / "results" / "fase3" / "oe5_estrategias_train.csv"
FIG = RAIZ / "figures" / "fase3"
ORIGEN = {"NLM-ChinaCXRSet": "Shenzhen", "NLM-MontgomeryCXRSet": "Montgomery"}
DESCRIPTORES = [
    "compacidad",
    "relacion_aspecto",
    "fraccion_area",
    "simetria_izq_der",
    "energia_wavelet_d1",
]
LIMITE = 0.6


def matrices(r: pd.DataFrame, sitio: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delta y significancia por estrategia y descriptor, con la referencia primero."""
    s = r[r["estrato"] == sitio]
    orden = ["referencia"] + sorted(
        e for e in s["estrategia"].unique() if e != "referencia"
    )
    delta = s.pivot(index="estrategia", columns="descriptor", values="delta")
    sig = s.pivot(index="estrategia", columns="descriptor", values="significativo")
    return delta.loc[orden, DESCRIPTORES], sig.loc[orden, DESCRIPTORES].astype(bool)


def discrepancias(delta: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """Celdas que difieren de la referencia en decision o en signo."""
    distinta_decision = sig.ne(sig.loc["referencia"], axis="columns")
    distinto_signo = np.sign(delta).ne(np.sign(delta.loc["referencia"]), axis="columns")
    resultado = distinta_decision | distinto_signo
    resultado.loc["referencia"] = False
    return resultado


def dibujar_panel(eje: plt.Axes, delta: pd.DataFrame, sig: pd.DataFrame, titulo: str):
    """Un panel del mapa de calor con valores, asteriscos y recuadros."""
    difiere = discrepancias(delta, sig)
    imagen = eje.imshow(
        delta.to_numpy(), cmap="RdBu_r", vmin=-LIMITE, vmax=LIMITE, aspect="auto"
    )
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            valor = delta.iat[i, j]
            texto = f"{valor:+.2f}" + ("*" if sig.iat[i, j] else "")
            color = "white" if abs(valor) > LIMITE * 0.6 else "black"
            eje.text(j, i, texto, ha="center", va="center", fontsize=8, color=color)
            if difiere.iat[i, j]:
                eje.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2,
                    )
                )
    eje.axhline(0.5, color="black", linewidth=1.5)
    eje.set_xticks(range(len(DESCRIPTORES)), DESCRIPTORES, rotation=30, ha="right")
    eje.set_yticks(range(len(delta.index)), delta.index)
    for etiqueta in eje.get_yticklabels():
        if etiqueta.get_text() == "base_adap":
            etiqueta.set_fontweight("bold")
    eje.set_title(titulo)
    return imagen


def main() -> None:
    """Genera la figura y cuenta las discrepancias por estrategia."""
    plt.switch_backend("Agg")
    FIG.mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(ENTRADA)
    fig, ejes = plt.subplots(1, 2, figsize=(13, 6), sharey=True, layout="constrained")
    conteo = {}
    for eje, (sitio, nombre) in zip(ejes, ORIGEN.items(), strict=True):
        delta, sig = matrices(r, sitio)
        imagen = dibujar_panel(eje, delta, sig, nombre)
        conteo[nombre] = discrepancias(delta, sig).sum(axis=1)
    fig.colorbar(
        imagen,
        ax=ejes,
        shrink=0.8,
        label="Delta de Cliff (positivo = mayor con hallazgos)",
    )
    fig.suptitle(
        "Diferencias entre grupos segun la estrategia de segmentacion (train)\n"
        "* significativo tras Holm; recuadro: no coincide con la referencia"
    )
    fig.savefig(FIG / "oe5_estrategias_train.png", dpi=160)
    plt.close(fig)
    tabla = pd.DataFrame(conteo).drop(index="referencia")
    tabla["total"] = tabla.sum(axis=1)
    print("Celdas que no coinciden con la referencia (de 5 por sitio):")
    print(tabla.sort_values("total").to_string())


if __name__ == "__main__":
    main()
