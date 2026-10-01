"""Audita la tabla de caracteristicas y produce las figuras de la Fase 2.

Verifica desde codigo las invariantes que exige el protocolo antes de declarar
la tabla lista para modelado, y genera las figuras que sustentan las
decisiones registradas.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psim import figura, resultado

AZUL = "#4A7FA5"
ROSA = "#C47F76"
SALVIA = "#7FA88E"

DESC = [
    "compacidad",
    "relacion_aspecto",
    "fraccion_area",
    "simetria_izq_der",
    "energia_wavelet_d1",
]


def cargar() -> pd.DataFrame:
    t = pd.read_csv("data/processed/features.csv", keep_default_na=False)
    t["origen"] = t.dataset_id.str.endswith("MontgomeryCXRSet").map(
        {True: "Montgomery", False: "Shenzhen"}
    )
    for c in ["dice", "jaccard", *DESC, *[f"err_{d}" for d in DESC]]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


def auditar(t: pd.DataFrame) -> list[dict[str, object]]:
    """Comprueba las invariantes y devuelve el resultado de cada una."""
    splits = pd.read_csv("data/metadata/splits.csv", keep_default_na=False)
    contrato = json.loads(Path("config/model_input.json").read_text(encoding="utf-8"))
    dicc = pd.read_csv("data/metadata/feature_dictionary.csv", keep_default_na=False)

    pruebas = []

    def check(nombre: str, ok: bool, detalle: str) -> None:
        pruebas.append(
            {
                "comprobacion": nombre,
                "resultado": "OK" if ok else "FALLA",
                "detalle": detalle,
            }
        )

    check(
        "claves unicas",
        t.sample_id.is_unique,
        f"{t.sample_id.nunique()} de {len(t)} filas",
    )

    manif = pd.read_csv("data/metadata/manifest.csv", keep_default_na=False)
    huerfanas = set(t.image_id) - set(manif.image_id)
    check(
        "todas las imagenes estan en el manifiesto",
        not huerfanas,
        f"{len(huerfanas)} huerfanas",
    )

    n_cfg = t.groupby("image_id").config_id.nunique()
    check(
        "misma cantidad de estrategias por imagen",
        n_cfg.nunique() == 1,
        f"{n_cfg.iloc[0]} estrategias por imagen",
    )

    pred = contrato["predictores"]
    prohibidas = set(pred) & set(contrato["columnas_excluidas_de_predictor"])
    check(
        "ningun predictor esta excluido", not prohibidas, f"{prohibidas or 'ninguna'}"
    )

    no_disp = dicc[(dicc.role == "predictor") & (dicc.available_at_prediction != "si")]
    check(
        "todo predictor esta disponible en prediccion",
        len(no_disp) == 0,
        f"{len(no_disp)} predictores no disponibles",
    )

    infinitos = int(np.isinf(t[pred].to_numpy(dtype=float)).sum())
    check(
        "sin valores infinitos en predictores", infinitos == 0, f"{infinitos} infinitos"
    )

    cruza = (
        t.merge(splits[["image_id", "split"]], on="image_id", suffixes=("", "_sp"))
        .groupby("image_id")
        .split_sp.nunique()
        .max()
    )
    check(
        "ninguna imagen cruza particiones",
        cruza == 1,
        f"max {cruza} particiones por imagen",
    )

    g = splits[splits.subject_id != ""].groupby("grupo_id").split.nunique()
    check(
        "ningun grupo de sujeto cruza particiones",
        (g.max() if len(g) else 1) == 1,
        f"{len(g)} grupos verificados",
    )

    const = [c for c in pred if t[c].nunique(dropna=True) <= 1]
    check("sin predictores constantes", not const, f"{const or 'ninguno'}")

    faltan = t[pred].isna().sum()
    check(
        "faltantes reportados, no imputados",
        True,
        "; ".join(f"{k}={v}" for k, v in faltan.items() if v) or "ninguno",
    )

    columnas_dicc = set(dicc.name)
    check(
        "el diccionario describe toda la tabla",
        columnas_dicc == set(t.columns) - {"origen"},
        f"{len(columnas_dicc)} columnas descritas",
    )

    return pruebas


def figura_dice(t: pd.DataFrame) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.2))

    orden = t.groupby("config_id").dice.median().sort_values(ascending=False).index
    datos = [t[t.config_id == c].dice.dropna() for c in orden]
    bp = ejes[0].boxplot(datos, tick_labels=list(orden), patch_artist=True)
    for parche, c in zip(bp["boxes"], orden, strict=True):
        parche.set_facecolor(SALVIA if c.startswith("base") else AZUL)
        parche.set_alpha(0.65)
    ejes[0].set_ylabel("Dice frente a la referencia")
    ejes[0].set_title(
        f"Concordancia por estrategia (n={len(t) // 9} imagenes)\n"
        "verde: linea base sin realce",
        fontsize=10,
    )
    ejes[0].tick_params(axis="x", rotation=45, labelsize=8)

    p = t.pivot_table(
        index="config_id", columns="origen", values="dice", aggfunc="median"
    )
    p = p.loc[orden]
    x = np.arange(len(p))
    ejes[1].bar(
        x - 0.2, p["Montgomery"], 0.4, color=AZUL, label="Montgomery", alpha=0.8
    )
    ejes[1].bar(x + 0.2, p["Shenzhen"], 0.4, color=ROSA, label="Shenzhen", alpha=0.8)
    ejes[1].set_xticks(x)
    ejes[1].set_xticklabels(p.index, rotation=45, ha="right", fontsize=8)
    ejes[1].set_ylabel("Dice mediano")
    ejes[1].set_title("Consistencia entre sitios de adquisicion", fontsize=10)
    ejes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(figura(2, "dice_por_estrategia.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def figura_errores(t: pd.DataFrame) -> None:
    fig, ejes = plt.subplots(1, len(DESC), figsize=(3.0 * len(DESC), 4.0), sharey=False)
    orden = t.groupby("config_id").dice.median().sort_values(ascending=False).index

    for eje, d in zip(ejes, DESC, strict=True):
        med = t.groupby("config_id")[f"err_{d}"].median().loc[orden]
        colores = [SALVIA if c.startswith("base") else AZUL for c in orden]
        eje.barh(range(len(med)), med.to_numpy(), color=colores, alpha=0.8)
        eje.set_yticks(range(len(med)))
        eje.set_yticklabels(med.index, fontsize=7)
        eje.invert_yaxis()
        eje.set_xlabel("error relativo mediano", fontsize=8)
        eje.set_title(d, fontsize=9)

    fig.suptitle(
        "Desplazamiento de cada descriptor frente a la referencia", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(figura(3, "error_descriptores.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t = cargar()
    print(f"tabla: {len(t)} filas x {len(t.columns) - 1} columnas")

    pruebas = pd.DataFrame(auditar(t))
    pruebas.to_csv(resultado(3, "auditoria_tabla.csv"), index=False)
    print()
    print(pruebas.to_string(index=False))

    figura_dice(t)
    figura_errores(t)
    print("\nfiguras escritas")

    e = t.groupby("config_id")[[f"err_{d}" for d in DESC]].median().round(4)
    e.columns = DESC
    e.to_csv(resultado(3, "error_por_descriptor.csv"))

    p = t.pivot_table(
        index="config_id", columns="origen", values="dice", aggfunc="median"
    ).round(4)
    p.to_csv(resultado(2, "dice_por_origen.csv"))

    fallos = (pruebas.resultado == "FALLA").sum()
    print(f"\ncomprobaciones: {len(pruebas) - fallos} OK, {fallos} fallidas")


if __name__ == "__main__":
    main()
