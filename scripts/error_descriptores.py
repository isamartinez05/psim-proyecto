"""OE4: desplazamiento de cada descriptor respecto a la referencia, por particion.

Usa la estrategia base_adap y las imagenes elegibles de las tres particiones.
El error relativo absoluto ya esta en las tablas (columnas err_*), calculado con
precision completa antes de redondear; aqui se le agrega el signo y se resume
por descriptor, particion y origen con estadisticos robustos.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psim import RAIZ

DESCRIPTORES = [
    "compacidad",
    "relacion_aspecto",
    "fraccion_area",
    "simetria_izq_der",
    "energia_wavelet_d1",
]
TABLAS = {
    "train": "features.csv",
    "validation": "features_validation.csv",
    "test": "features_test.csv",
}
ORIGEN = {"NLM-ChinaCXRSet": "Shenzhen", "NLM-MontgomeryCXRSet": "Montgomery"}
OUT = RAIZ / "results" / "fase3"
FIG = RAIZ / "figures" / "fase3"


def cargar() -> pd.DataFrame:
    """Une las tres particiones con base_adap y solo imagenes elegibles."""
    partes = []
    for split, archivo in TABLAS.items():
        t = pd.read_csv(RAIZ / "data" / "processed" / archivo)
        t = t[
            (t["config_id"] == "base_adap")
            & (t["split"] == split)
            & (t["eligible_for_model"] == 1)
        ]
        partes.append(t)
    return pd.concat(partes, ignore_index=True)


def a_largo(t: pd.DataFrame) -> pd.DataFrame:
    """Una fila por imagen y descriptor, con error firmado, absoluto y diferencia."""
    partes = []
    for d in DESCRIPTORES:
        parte = t[["image_id", "split", "dataset_id"]].copy()
        parte["descriptor"] = d
        parte["obtenido"] = t[d].to_numpy()
        parte["referencia"] = t[f"ref_{d}"].to_numpy()
        parte["err_tabla"] = t[f"err_{d}"].to_numpy()
        partes.append(parte)
    largo = pd.concat(partes, ignore_index=True)
    largo["diferencia"] = largo["obtenido"] - largo["referencia"]
    # Magnitud con precision completa (calculada antes de redondear la tabla);
    # el signo se toma de la diferencia entre los valores guardados.
    recalculado = (largo["diferencia"] / largo["referencia"].abs()).replace(
        [np.inf, -np.inf], np.nan
    )
    largo["error_relativo"] = np.sign(largo["diferencia"]) * largo["err_tabla"]
    largo["error_abs"] = largo["err_tabla"]
    largo["origen"] = largo["dataset_id"].map(ORIGEN)
    dif = (recalculado.abs() - largo["err_tabla"]).abs()
    afectadas = dif > 1e-3
    print(
        "Filas donde el redondeo a 6 decimales altera el error recalculado:",
        int(afectadas.sum()),
    )
    print(largo.loc[afectadas, "descriptor"].value_counts().to_string())
    print(
        "Referencia maxima entre esas filas:",
        f"{largo.loc[afectadas, 'referencia'].abs().max():.6f}",
    )
    return largo.dropna(subset=["error_relativo"])


def resumir(largo: pd.DataFrame) -> pd.DataFrame:
    """Sesgo (firmado), magnitud (absoluta) y diferencia en unidades del descriptor."""
    return (
        largo.groupby(["descriptor", "split", "origen"])
        .agg(
            n=("error_relativo", "size"),
            sesgo_mediana=("error_relativo", "median"),
            abs_mediana=("error_abs", "median"),
            abs_p25=("error_abs", lambda s: s.quantile(0.25)),
            abs_p75=("error_abs", lambda s: s.quantile(0.75)),
            dif_mediana=("diferencia", "median"),
            dif_abs_mediana=("diferencia", lambda s: s.abs().median()),
            pct_sobreestima=("diferencia", lambda s: round(100 * (s > 0).mean(), 1)),
        )
        .round(4)
        .reset_index()
    )


def graficar(largo: pd.DataFrame) -> None:
    """Caja del error relativo firmado por descriptor, particion y origen."""
    fig, ejes = plt.subplots(1, len(DESCRIPTORES), figsize=(18, 4))
    grupos = [(s, o) for s in TABLAS for o in ORIGEN.values()]
    for eje, d in zip(ejes, DESCRIPTORES, strict=True):
        datos = [
            largo.query("descriptor == @d and split == @s and origen == @o")[
                "error_relativo"
            ]
            for s, o in grupos
        ]
        eje.boxplot(datos, showfliers=False)
        eje.axhline(0, color="black", linestyle="--", linewidth=1)
        eje.set_xticks(
            range(1, len(grupos) + 1),
            [f"{s[:3]}\n{o[:3]}" for s, o in grupos],
            fontsize=7,
        )
        eje.set_title(d, fontsize=9)
    ejes[0].set_ylabel("Error relativo firmado")
    fig.suptitle(
        "Desplazamiento de cada descriptor frente a la referencia (base_adap); "
        "positivo = la segmentacion automatica sobreestima"
    )
    fig.tight_layout()
    fig.savefig(FIG / "error_descriptores_particiones.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Genera la tabla y la figura del desplazamiento por particion."""
    plt.switch_backend("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    largo = a_largo(cargar())
    resumen = resumir(largo)
    resumen.to_csv(OUT / "error_descriptores_particiones.csv", index=False)
    graficar(largo)
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
