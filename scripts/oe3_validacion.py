"""OE3 en validacion: las nueve estrategias contra la mascara de referencia.

La formulacion pide Dice y Jaccard por imagen sobre validacion, estratificados
por origen. En el lab004 la estrategia se eligio en train; aqui se verifica, sin
tocar la prueba, si la eleccion se sostiene en validacion. No modifica
features_validation.csv, cuyo hash esta congelado en los contratos.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

from psim import DATA_RAW, RAIZ
from psim.eda import cargar
from psim.estadistica import holm
from psim.features import DESCRIPTORES, error_relativo, extraer
from psim.io import leer_gris
from psim.mascaras import leer_referencia
from psim.recorte import mascara_campo
from psim.segmentacion import CONFIGURACIONES, dice, jaccard, reducir, segmentar

OUT = RAIZ / "results" / "fase2"
FIG = RAIZ / "figures" / "fase2"
ORIGEN = {"NLM-ChinaCXRSet": "Shenzhen", "NLM-MontgomeryCXRSet": "Montgomery"}
ELEGIDA = "base_adap"
ALFA = 0.05


def extraer_validacion() -> pd.DataFrame:
    """Dice, Jaccard y error de cada descriptor por imagen y estrategia."""
    d = cargar(
        RAIZ / "data/metadata/manifest.csv",
        RAIZ / "data/metadata/splits.csv",
        split="validation",
    )
    con = d[d["mask_path"] != ""].reset_index(drop=True)
    print(
        f"Validacion: {len(con)} imagenes con mascara x {len(CONFIGURACIONES)} estrategias"
    )
    filas = []
    t0 = time.time()
    for i, (_, r) in enumerate(con.iterrows(), 1):
        img = leer_gris(DATA_RAW / r["relative_path"])
        ref = leer_referencia(r.to_dict(), img.shape)
        g = reducir(img)
        campo = mascara_campo(g)
        ref_total = reducir(ref.total)
        d_ref = extraer(
            g, ref_total, campo, reducir(ref.izquierdo), reducir(ref.derecho)
        )
        for cfg in CONFIGURACIONES:
            s = segmentar(g, cfg, campo)
            d_obt = extraer(g, s.mascara, campo)
            fila = {
                "image_id": r["image_id"],
                "origen": ORIGEN[r["dataset_id"]],
                "estrategia": cfg,
                "n_componentes": s.n_componentes,
                "dice": dice(s.mascara, ref_total),
                "jaccard": jaccard(s.mascara, ref_total),
            }
            for k in DESCRIPTORES:
                fila[f"err_{k}"] = error_relativo(d_obt[k], d_ref[k])
            filas.append(fila)
        if i % 25 == 0 or i == len(con):
            print(f"  {i:>4}/{len(con)}  {(time.time() - t0) / 60:5.1f} min")
    return pd.DataFrame(filas)


def comparar_con_train(v: pd.DataFrame) -> pd.DataFrame:
    """Dice mediano y ranking por estrategia y origen, en train y en validacion."""
    t = pd.read_csv(RAIZ / "data" / "processed" / "features.csv")
    t = t[t["split"] == "train"].assign(origen=lambda x: x["dataset_id"].map(ORIGEN))
    train = t.groupby(["config_id", "origen"])["dice"].median().rename("dice_train")
    val = v.groupby(["estrategia", "origen"])["dice"].median().rename("dice_val")
    train.index = train.index.set_names(["estrategia", "origen"])
    r = pd.concat([train, val], axis=1).reset_index()
    r["rango_train"] = (
        r.groupby("origen")["dice_train"].rank(ascending=False).astype(int)
    )
    r["rango_val"] = r.groupby("origen")["dice_val"].rank(ascending=False).astype(int)
    return r.sort_values(["origen", "rango_val"]).round(4)


def pruebas_pareadas(v: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon pareado del Dice: la elegida frente a cada otra estrategia."""
    ancho = v.pivot_table(
        index=["image_id", "origen"], columns="estrategia", values="dice"
    )
    filas = []
    for origen, g in ancho.groupby(level="origen"):
        for cfg in CONFIGURACIONES:
            if cfg == ELEGIDA:
                continue
            diferencia = g[ELEGIDA] - g[cfg]
            prueba = wilcoxon(g[ELEGIDA], g[cfg], alternative="two-sided")
            filas.append(
                {
                    "origen": origen,
                    "comparada": cfg,
                    "n": len(g),
                    "dif_mediana": float(diferencia.median()),
                    "pct_elegida_mejor": round(100 * float((diferencia > 0).mean()), 1),
                    "p": float(prueba.pvalue),
                }
            )
    r = pd.DataFrame(filas)
    r["p_holm"] = holm(r["p"])
    r["significativo"] = r["p_holm"] < ALFA
    return r.round(4)


def graficar(v: pd.DataFrame) -> None:
    """Dice por estrategia en validacion, un panel por origen."""
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for eje, origen in zip(ejes, ["Shenzhen", "Montgomery"], strict=True):
        s = v[v["origen"] == origen]
        orden = (
            s.groupby("estrategia")["dice"].median().sort_values(ascending=False).index
        )
        cajas = eje.boxplot(
            [s.loc[s["estrategia"] == c, "dice"] for c in orden], showfliers=False
        )
        for caja, nombre in zip(cajas["boxes"], orden, strict=True):
            if nombre == ELEGIDA:
                caja.set_color("tab:red")
                caja.set_linewidth(2)
        eje.set_xticks(range(1, len(orden) + 1), orden, rotation=40, ha="right")
        n = s["image_id"].nunique()
        eje.set_title(f"{origen} (n = {n} imagenes)")
    ejes[0].set_ylabel("Dice frente a la mascara de referencia")
    fig.suptitle(
        "OE3 en validacion: Dice por estrategia (rojo = base_adap, elegida en train)"
    )
    fig.tight_layout()
    fig.savefig(FIG / "oe3_dice_validacion.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Extraccion, resumen, comparacion con train, pruebas pareadas y figura."""
    plt.switch_backend("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    v = extraer_validacion()
    v.to_csv(OUT / "oe3_validacion_por_imagen.csv", index=False)
    errores = [f"err_{k}" for k in DESCRIPTORES]
    resumen = (
        v.assign(dos_componentes=v["n_componentes"] == 2)
        .groupby(["estrategia", "origen"])
        .agg(
            n=("dice", "size"),
            dice_mediana=("dice", "median"),
            jaccard_mediana=("jaccard", "median"),
            pct_dos_componentes=("dos_componentes", "mean"),
            **{f"{e}_mediana": (e, "median") for e in errores},
        )
        .round(4)
        .reset_index()
    )
    resumen.to_csv(OUT / "oe3_validacion_resumen.csv", index=False)
    ranking = comparar_con_train(v)
    ranking.to_csv(OUT / "oe3_ranking_train_validacion.csv", index=False)
    pareadas = pruebas_pareadas(v)
    pareadas.to_csv(OUT / "oe3_pruebas_pareadas.csv", index=False)
    graficar(v)
    print("\n=== Dice mediano y ranking: train frente a validacion ===")
    print(ranking.to_string(index=False))
    print(f"\n=== {ELEGIDA} frente a cada estrategia (Wilcoxon pareado, Holm) ===")
    print("(dif_mediana > 0: la elegida tiene mayor Dice)")
    print(pareadas.to_string(index=False))


if __name__ == "__main__":
    main()
