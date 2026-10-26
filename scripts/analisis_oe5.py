"""OE5: diferencias entre grupos con y sin hallazgos, y su estabilidad.

Aplica el protocolo congelado en config/analisis_oe5.json. Compara los
descriptores por paciente entre grupos, dentro de cada sitio, con los valores de
la mascara de referencia (analisis principal) y con los de la segmentacion
automatica (sensibilidad), y repite la comparacion con las nueve estrategias en
la particion de entrenamiento, siempre sobre los mismos pacientes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psim import RAIZ
from psim.estadistica import comparar_grupos

CONTRATO = RAIZ / "config" / "analisis_oe5.json"
OUT = RAIZ / "results" / "fase3"
FIG = RAIZ / "figures" / "fase3"
TABLAS = {
    "train": "data/processed/features.csv",
    "validation": "data/processed/features_validation.csv",
    "test": "data/processed/features_test.csv",
}
ORIGEN = {"NLM-ChinaCXRSet": "Shenzhen", "NLM-MontgomeryCXRSet": "Montgomery"}


def sha256(ruta: Path) -> str:
    """Hash SHA-256 de un archivo."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def cargar(contrato: dict[str, Any]) -> pd.DataFrame:
    """Une las tres tablas, verifica sus hashes y agrega el grupo de cada imagen."""
    partes = []
    for split, relativa in TABLAS.items():
        ruta = RAIZ / relativa
        if sha256(ruta) != contrato["sha256"][split]:
            raise ValueError(f"{ruta.name} cambio despues de congelar el protocolo.")
        t = pd.read_csv(ruta)
        partes.append(t[t["split"] == split])
    t = pd.concat(partes, ignore_index=True)
    splits = pd.read_csv(RAIZ / "data" / "metadata" / "splits.csv")
    return t.merge(
        splits[["image_id", "grupo_id"]],
        on="image_id",
        how="left",
        validate="many_to_one",
    )


def por_paciente(t: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Una fila por paciente: promedia sus imagenes y conserva grupo y sitio."""
    if t["grupo_id"].isna().any():
        raise ValueError("Hay imagenes sin grupo.")
    mixtos = t.groupby("grupo_id")["target"].nunique()
    if (mixtos > 1).any():
        raise ValueError(
            f"Pacientes con etiquetas mixtas: {list(mixtos[mixtos > 1].index)}"
        )
    return (
        t.groupby("grupo_id")
        .agg(
            dataset_id=("dataset_id", "first"),
            target=("target", "first"),
            edad=("edad", "mean"),
            sexo=("sexo", "first"),
            n_imagenes=("image_id", "size"),
            **{c: (c, "mean") for c in columnas},
        )
        .reset_index()
    )


def principal_y_automatica(
    pacientes: pd.DataFrame, descriptores: list[str], contrato: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara grupos con la referencia y con la segmentacion automatica."""
    opciones = {
        "replicas": contrato["intervalo"]["replicas"],
        "semilla": contrato["semilla"],
        "alfa": contrato["alfa"],
    }
    base = ["grupo_id", "dataset_id", "target"]
    referencia = pacientes[base + [f"ref_{d}" for d in descriptores]].rename(
        columns={f"ref_{d}": d for d in descriptores}
    )
    automatica = pacientes[base + descriptores]
    r_ref = comparar_grupos(referencia, descriptores, **opciones)
    r_auto = comparar_grupos(automatica, descriptores, **opciones)
    return r_ref.assign(fuente="referencia"), r_auto.assign(fuente="automatica")


def estabilidad(r_ref: pd.DataFrame, r_auto: pd.DataFrame) -> pd.DataFrame:
    """Aplica el criterio del contrato: misma decision tras Holm y mismo signo."""
    columnas = [
        "estrato",
        "descriptor",
        "delta",
        "ic_inf",
        "ic_sup",
        "p_holm",
        "significativo",
    ]
    e = r_ref[columnas].merge(
        r_auto[columnas], on=["estrato", "descriptor"], suffixes=("_ref", "_auto")
    )
    e["misma_decision"] = e["significativo_ref"] == e["significativo_auto"]
    e["mismo_signo"] = np.sign(e["delta_ref"]) == np.sign(e["delta_auto"])
    e["estable"] = e["misma_decision"] & e["mismo_signo"]
    return e


def estrategias_train(
    t: pd.DataFrame,
    poblacion: set[str],
    descriptores: list[str],
    contrato: dict[str, Any],
) -> pd.DataFrame:
    """Repite la comparacion con cada estrategia, en train y con los mismos pacientes."""
    opciones = {
        "replicas": contrato["intervalo"]["replicas"],
        "semilla": contrato["semilla"],
        "alfa": contrato["alfa"],
    }
    train = t[(t["split"] == "train") & t["grupo_id"].isin(poblacion)]
    resultados = []
    base_adap = train[train["config_id"] == "base_adap"]
    ref_cols = [f"ref_{d}" for d in descriptores]
    referencia = por_paciente(base_adap, ref_cols).rename(
        columns={f"ref_{d}": d for d in descriptores}
    )
    resultados.append(
        comparar_grupos(referencia, descriptores, **opciones).assign(
            estrategia="referencia"
        )
    )
    for cfg in sorted(train["config_id"].unique()):
        pacientes = por_paciente(train[train["config_id"] == cfg], descriptores)
        resultados.append(
            comparar_grupos(pacientes, descriptores, **opciones).assign(estrategia=cfg)
        )
    return pd.concat(resultados, ignore_index=True)


def acuerdo_estrategias(r: pd.DataFrame) -> pd.DataFrame:
    """Cuantas estrategias coinciden con la referencia en decision y signo."""
    ref = r[r["estrategia"] == "referencia"].set_index(["estrato", "descriptor"])
    otras = r[r["estrategia"] != "referencia"].copy()
    clave = list(zip(otras["estrato"], otras["descriptor"], strict=True))
    otras["coincide"] = [
        (fila.significativo == ref.loc[k, "significativo"])
        and (np.sign(fila.delta) == np.sign(ref.loc[k, "delta"]))
        for fila, k in zip(otras.itertuples(), clave, strict=True)
    ]
    resumen = (
        otras.groupby(["estrato", "descriptor"])
        .agg(
            estrategias=("estrategia", "size"),
            coinciden=("coincide", "sum"),
            delta_min=("delta", "min"),
            delta_max=("delta", "max"),
        )
        .reset_index()
    )
    resumen["delta_referencia"] = [
        ref.loc[(e, d), "delta"]
        for e, d in zip(resumen["estrato"], resumen["descriptor"], strict=True)
    ]
    return resumen.round(3)


def covariables(pacientes: pd.DataFrame) -> pd.DataFrame:
    """Edad y sexo por sitio y grupo, para discutir la confusion."""
    return (
        pacientes.groupby(["dataset_id", "target"])
        .agg(
            n=("grupo_id", "size"),
            edad_mediana=("edad", "median"),
            pct_masculino=(
                "sexo",
                lambda s: round(
                    100 * s.astype(str).str.lower().str.startswith("m").mean(), 1
                ),
            ),
        )
        .reset_index()
    )


def graficar(
    r_ref: pd.DataFrame, r_auto: pd.DataFrame, descriptores: list[str]
) -> None:
    """Delta de Cliff con intervalo bootstrap, referencia frente a automatica."""
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    series = ((r_ref, "Referencia", "o", -0.15), (r_auto, "Automatica", "s", 0.15))
    for eje, (sitio, nombre) in zip(ejes, ORIGEN.items(), strict=True):
        for r, etiqueta, marcador, desplazamiento in series:
            s = r[r["estrato"] == sitio].set_index("descriptor").loc[descriptores]
            yy = np.arange(len(descriptores)) + desplazamiento
            errores = [s["delta"] - s["ic_inf"], s["ic_sup"] - s["delta"]]
            eje.errorbar(
                s["delta"], yy, xerr=errores, fmt=marcador, capsize=3, label=etiqueta
            )
        n = r_ref[r_ref["estrato"] == sitio].iloc[0]
        eje.axvline(0, color="black", linestyle="--", linewidth=1)
        eje.set(
            title=f"{nombre} (con hallazgos n={n['n_con']}, sin n={n['n_sin']})",
            xlabel="Delta de Cliff (positivo = mayor con hallazgos)",
            xlim=(-1, 1),
        )
    ejes[0].set_yticks(range(len(descriptores)), descriptores)
    ejes[0].legend(fontsize=8)
    fig.suptitle("Diferencias entre grupos por paciente, IC bootstrap 95 %")
    fig.tight_layout()
    fig.savefig(FIG / "oe5_delta_cliff.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Ejecuta el analisis principal, la estabilidad y la sensibilidad."""
    plt.switch_backend("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    contrato = json.loads(CONTRATO.read_text(encoding="utf-8"))
    descriptores = contrato["descriptores"]
    t = cargar(contrato)

    elegibles = t[(t["config_id"] == "base_adap") & (t["eligible_for_model"] == 1)]
    pacientes = por_paciente(
        elegibles, [f"ref_{d}" for d in descriptores] + descriptores
    )
    print(f"Pacientes: {len(pacientes)}")
    print(pd.crosstab(pacientes["dataset_id"], pacientes["target"]).to_string())

    r_ref, r_auto = principal_y_automatica(pacientes, descriptores, contrato)
    pd.concat([r_ref, r_auto]).to_csv(OUT / "oe5_comparacion.csv", index=False)
    e = estabilidad(r_ref, r_auto)
    e.to_csv(OUT / "oe5_estabilidad.csv", index=False)
    cov = covariables(pacientes)
    cov.to_csv(OUT / "oe5_covariables.csv", index=False)
    graficar(r_ref, r_auto, descriptores)

    poblacion = set(elegibles.loc[elegibles["split"] == "train", "grupo_id"])
    r_est = estrategias_train(t, poblacion, descriptores, contrato)
    r_est.to_csv(OUT / "oe5_estrategias_train.csv", index=False)
    acuerdo = acuerdo_estrategias(r_est)
    acuerdo.to_csv(OUT / "oe5_acuerdo_estrategias.csv", index=False)

    vista = [
        "estrato",
        "descriptor",
        "mediana_con",
        "mediana_sin",
        "delta",
        "ic_inf",
        "ic_sup",
        "p_holm",
        "significativo",
    ]
    print("\n=== Principal: mascara de referencia ===")
    print(r_ref[vista].round(4).to_string(index=False))
    print("\n=== Sensibilidad: segmentacion automatica ===")
    print(r_auto[vista].round(4).to_string(index=False))
    print("\n=== Estabilidad (criterio del contrato) ===")
    print(
        e[
            [
                "estrato",
                "descriptor",
                "delta_ref",
                "delta_auto",
                "significativo_ref",
                "significativo_auto",
                "estable",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\n=== Covariables por sitio y grupo ===")
    print(cov.round(1).to_string(index=False))
    print("\n=== Acuerdo de las 9 estrategias con la referencia (train) ===")
    print(acuerdo.to_string(index=False))


if __name__ == "__main__":
    main()
