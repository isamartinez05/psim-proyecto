"""Genera las tablas y cifras del informe final integrado desde los resultados.

Ningun numero del informe final se transcribe a mano: este guion lee los CSV de
todas las fases y escribe reports/tablas/final_*.tex y final_cifras.tex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from build_tablas_tex import SALIDA, tabla

from psim import RAIZ

R2 = RAIZ / "results" / "fase2"
R3 = RAIZ / "results" / "fase3"
RM = RAIZ / "results" / "modelado_proyecto"
META = RAIZ / "data" / "metadata"
ORIGEN = {"NLM-ChinaCXRSet": "Shenzhen", "NLM-MontgomeryCXRSet": "Montgomery"}
SITIOS = ["Shenzhen", "Montgomery"]
Cifras = dict[str, str]


def si_no(serie: pd.Series) -> pd.Series:
    """Convierte valores logicos a si / no."""
    return serie.map(lambda v: "si" if str(v) == "True" else "no")


def datos(c: Cifras) -> None:
    """Conjuntos de datos y particion."""
    m = pd.read_csv(META / "manifest.csv", keep_default_na=False)
    m["origen"] = m["dataset_id"].map(ORIGEN)
    for columna in ("target", "height", "width", "bits_efectivos", "channels", "edad"):
        m[columna] = pd.to_numeric(m[columna], errors="coerce")
    filas = []
    for origen in SITIOS:
        g = m[m["origen"] == origen]
        filas.append(
            {
                "origen": origen,
                "imagenes": len(g),
                "con_mascara": int((g["mask_path"] != "").sum()),
                "con_hallazgos": int((g["target"] == 1).sum()),
                "alto": int(g["height"].median()),
                "ancho": int(g["width"].median()),
                "bits": int(g["bits_efectivos"].mode().iloc[0]),
                "canales": int(g["channels"].mode().iloc[0]),
                "edad": int(g["edad"].median()),
            }
        )
    tabla(
        pd.DataFrame(filas),
        "final_datos",
        "Conjuntos de datos (medianas de alto, ancho y edad; moda de bits y canales)",
        "lrrrrrrrr",
        [
            "Origen",
            "Imagenes",
            "Con mascara",
            "Con hallazgos",
            "Alto px",
            "Ancho px",
            "Bits",
            "Canales",
            "Edad",
        ],
    )
    c["FinNImagenes"] = str(len(m))
    c["FinNConMascara"] = str(int((m["mask_path"] != "").sum()))
    for f in filas:
        sufijo = "SH" if f["origen"] == "Shenzhen" else "MC"
        c[f"FinN{sufijo}"] = str(f["imagenes"])
        c[f"FinBits{sufijo}"] = str(f["bits"])
    invertidas = pd.to_numeric(m["escala_invertida"], errors="coerce").fillna(0)
    c["FinInvertidas"] = str(int(invertidas.sum()))

    s = pd.read_csv(META / "splits.csv")
    d = s.merge(m[["image_id", "origen", "target"]], on="image_id")
    particion = []
    for split, nombre in (
        ("train", "Entrenamiento"),
        ("validation", "Validacion"),
        ("test", "Prueba"),
    ):
        g = d[d["split"] == split]
        particion.append(
            {
                "particion": nombre,
                "Shenzhen": int((g["origen"] == "Shenzhen").sum()),
                "Montgomery": int((g["origen"] == "Montgomery").sum()),
                "total": len(g),
                "con_hallazgos": int((g["target"] == 1).sum()),
            }
        )
    tabla(
        pd.DataFrame(particion),
        "final_particion",
        "Particion por paciente, estratificada por origen y clase",
        "lrrrr",
        ["Particion", "Shenzhen", "Montgomery", "Total", "Con hallazgos"],
    )


def oe3(c: Cifras) -> None:
    """Segmentacion: ranking en train y validacion y pruebas pareadas."""
    r = pd.read_csv(R2 / "oe3_ranking_train_validacion.csv")
    ancho = r.pivot(
        index="estrategia",
        columns="origen",
        values=["dice_train", "dice_val", "rango_val"],
    )
    ancho.columns = [f"{a}_{b}" for a, b in ancho.columns]
    for sitio in SITIOS:
        ancho[f"rango_val_{sitio}"] = ancho[f"rango_val_{sitio}"].astype(int)
    ancho = ancho.sort_values("rango_val_Shenzhen").reset_index()
    columnas = ["estrategia"] + [
        f"{a}_{s}" for s in SITIOS for a in ("dice_train", "dice_val", "rango_val")
    ]
    tabla(
        ancho[columnas],
        "final_oe3_ranking",
        "Dice mediano y puesto en validacion de las nueve estrategias",
        "lrrrrrr",
        [
            "Estrategia",
            "SH train",
            "SH valid.",
            "SH puesto",
            "MC train",
            "MC valid.",
            "MC puesto",
        ],
    )
    base = r[r["estrategia"] == "base_adap"].set_index("origen")
    for sitio, sufijo in (("Shenzhen", "SH"), ("Montgomery", "MC")):
        c[f"FinDiceTrain{sufijo}"] = f"{base.loc[sitio, 'dice_train']:.3f}"
        c[f"FinDiceVal{sufijo}"] = f"{base.loc[sitio, 'dice_val']:.3f}"

    p = pd.read_csv(R2 / "oe3_pruebas_pareadas.csv")
    c["FinParesSig"] = str(int((p["significativo"].astype(str) == "True").sum()))
    c["FinParesTotal"] = str(len(p))
    p["significativo"] = si_no(p["significativo"])
    tabla(
        p[
            [
                "origen",
                "comparada",
                "dif_mediana",
                "pct_elegida_mejor",
                "p_holm",
                "significativo",
            ]
        ],
        "final_oe3_pareadas",
        "base\\_adap frente a cada estrategia en validacion (Wilcoxon pareado con Holm)",
        "llrrrl",
        ["Origen", "Comparada", "Dif. Dice", "% mejor", "p Holm", "Signif."],
    )


def oe2(c: Cifras) -> None:
    """Realce, frecuencia, wavelets, especificacion y conectividad."""
    filas = []
    for archivo, lote in (
        ("decision_realce_espacial.csv", "espacial"),
        ("decision_realce_frecuencia.csv", "frecuencia"),
        ("decision_realce_histograma.csv", "histograma"),
    ):
        d = pd.read_csv(R2 / archivo, index_col=0).drop(index="original")
        for estrategia, f in d.iterrows():
            filas.append(
                {
                    "lote": lote,
                    "estrategia": estrategia,
                    "contraste": float(f["cambio_contraste"]),
                    "ruido": float(f["cambio_ruido"]),
                    "bordes": float(f["conservacion_bordes"]),
                    "se_conserva": "si" if str(f["se_conserva"]) == "True" else "no",
                }
            )
    decision = pd.DataFrame(filas)
    tabla(
        decision,
        "final_oe2_decision",
        "Estrategias de realce y filtrado frente al criterio de la formulacion",
        "llrrrl",
        [
            "Lote",
            "Estrategia",
            "Cambio contraste",
            "Cambio ruido",
            "Bordes",
            "Se conserva",
        ],
    )
    c["FinRealceTotal"] = str(len(decision))
    c["FinRealceConservadas"] = str(int((decision["se_conserva"] == "si").sum()))

    e = pd.read_csv(R2 / "espectro_por_origen.csv").iloc[0]
    c["FinAltaMC"] = f"{e['mediana_montgomery']:.5f}"
    c["FinAltaSH"] = f"{e['mediana_shenzhen']:.5f}"
    c["FinDeltaEspectro"] = f"{e['delta_mc_vs_sh']:.2f}"

    w = pd.read_csv(R3 / "energia_subbandas_por_origen.csv")
    w["significativo"] = si_no(w["significativo"])
    tabla(
        w[
            [
                "descriptor",
                "mediana_mc",
                "mediana_sh",
                "delta",
                "ic_inf",
                "ic_sup",
                "p_holm",
                "significativo",
            ]
        ],
        "final_oe2_wavelets",
        "Energia relativa por subbanda (DWT db2, 3 niveles); delta positivo = mayor en Montgomery",
        "lrrrrrrl",
        [
            "Subbanda",
            "Mediana MC",
            "Mediana SH",
            "Delta",
            "IC inf",
            "IC sup",
            "p Holm",
            "Signif.",
        ],
    )

    s = pd.read_csv(R2 / "especificacion_por_origen.csv")
    s = s[s["medida"] != "distancia_a_shenzhen"]
    antes_despues = s.pivot(
        index="medida", columns="montgomery", values="delta_mc_vs_sh"
    )
    antes_despues = antes_despues[["original", "especificada"]].reset_index()
    tabla(
        antes_despues,
        "final_oe2_especificacion",
        "Delta de Cliff Montgomery frente a Shenzhen antes y despues de especificar el histograma",
        "lrr",
        ["Medida", "Original", "Especificada"],
    )
    indice = antes_despues.set_index("medida")
    c["FinIntensidadAntes"] = f"{indice.loc['mediana_intensidad', 'original']:.2f}"
    c["FinIntensidadDespues"] = (
        f"{indice.loc['mediana_intensidad', 'especificada']:.2f}"
    )
    c["FinAltaDespues"] = f"{indice.loc['fraccion_alta', 'especificada']:.2f}"

    cc = pd.read_csv(R2 / "conectividad_componentes.csv")
    c["FinConectN"] = str(len(cc))
    c["FinConectIguales"] = str(int((cc["componentes_4"] == cc["componentes_8"]).sum()))


def oe4(c: Cifras) -> None:
    """Desplazamiento de cada descriptor en la particion de prueba."""
    e = pd.read_csv(R3 / "error_descriptores_particiones.csv")
    prueba = e[e["split"] == "test"].sort_values(["descriptor", "origen"])
    tabla(
        prueba[
            [
                "descriptor",
                "origen",
                "n",
                "sesgo_mediana",
                "abs_mediana",
                "dif_abs_mediana",
                "pct_sobreestima",
            ]
        ],
        "final_oe4",
        "Error de cada descriptor automatico frente a la referencia (particion de prueba)",
        "llrrrrr",
        [
            "Descriptor",
            "Origen",
            "n",
            "Sesgo rel.",
            "Error abs. rel.",
            "Dif. abs.",
            "% sobreestima",
        ],
    )
    compacidad = e.loc[e["descriptor"] == "compacidad", "sesgo_mediana"]
    c["FinSesgoCompacidad"] = f"{-100 * compacidad.median():.0f}"


def oe5(c: Cifras) -> None:
    """Diferencias entre grupos, estabilidad, covariables y estrategias."""
    comparacion = pd.read_csv(R3 / "oe5_comparacion.csv")
    ref = comparacion[comparacion["fuente"] == "referencia"].copy()
    ref["sitio"] = ref["estrato"].map(ORIGEN)
    c["FinOECincoSigSH"] = str(
        int(
            (
                (ref["sitio"] == "Shenzhen")
                & (ref["significativo"].astype(str) == "True")
            ).sum()
        )
    )
    ref["significativo"] = si_no(ref["significativo"])
    tabla(
        ref[
            [
                "sitio",
                "descriptor",
                "mediana_con",
                "mediana_sin",
                "delta",
                "ic_inf",
                "ic_sup",
                "p_holm",
                "significativo",
            ]
        ],
        "final_oe5_referencia",
        "Diferencias entre grupos con la mascara de referencia, por paciente (delta positivo = mayor con hallazgos)",
        "llrrrrrrl",
        [
            "Origen",
            "Descriptor",
            "Med. con",
            "Med. sin",
            "Delta",
            "IC inf",
            "IC sup",
            "p Holm",
            "Signif.",
        ],
    )

    est = pd.read_csv(R3 / "oe5_estabilidad.csv")
    est["sitio"] = est["estrato"].map(ORIGEN)
    c["FinEstables"] = str(int((est["estable"].astype(str) == "True").sum()))
    c["FinEstabilidadTotal"] = str(len(est))
    for columna in ("significativo_ref", "significativo_auto", "estable"):
        est[columna] = si_no(est[columna])
    tabla(
        est[
            [
                "sitio",
                "descriptor",
                "delta_ref",
                "delta_auto",
                "significativo_ref",
                "significativo_auto",
                "estable",
            ]
        ],
        "final_oe5_estabilidad",
        "Estabilidad de las conclusiones: mascara de referencia frente a segmentacion automatica",
        "llrrlll",
        [
            "Origen",
            "Descriptor",
            "Delta ref.",
            "Delta auto.",
            "Signif. ref.",
            "Signif. auto.",
            "Estable",
        ],
    )

    cov = pd.read_csv(R3 / "oe5_covariables.csv")
    cov["sitio"] = cov["dataset_id"].map(ORIGEN)
    cov["grupo"] = cov["target"].map({0: "sin hallazgos", 1: "con hallazgos"})
    tabla(
        cov[["sitio", "grupo", "n", "edad_mediana", "pct_masculino"]],
        "final_oe5_covariables",
        "Edad y sexo por origen y grupo",
        "llrrr",
        ["Origen", "Grupo", "Pacientes", "Edad mediana", "% masculino"],
    )
    pacientes = pd.read_csv(R3 / "oe5_pacientes.csv")
    c["FinPacientesOECinco"] = str(len(pacientes))

    r = pd.read_csv(R3 / "oe5_estrategias_train.csv")
    conteo: dict[str, dict[str, int]] = {}
    for estrato, g in r.groupby("estrato"):
        referencia = g[g["estrategia"] == "referencia"].set_index("descriptor")
        for estrategia, h in g[g["estrategia"] != "referencia"].groupby("estrategia"):
            h = h.set_index("descriptor").loc[referencia.index]
            difiere = (h["significativo"] != referencia["significativo"]) | (
                np.sign(h["delta"]) != np.sign(referencia["delta"])
            )
            conteo.setdefault(estrategia, {})[ORIGEN[estrato]] = int(difiere.sum())
    t = pd.DataFrame(conteo).T[SITIOS]
    t["total"] = t.sum(axis=1)
    t = t.sort_values("total").reset_index(names="estrategia")
    tabla(
        t,
        "final_oe5_estrategias",
        "Conclusiones entre grupos que no coinciden con la referencia, por estrategia (de 5 por origen)",
        "lrrr",
        ["Estrategia", "Shenzhen", "Montgomery", "Total"],
    )
    c["FinDiscMin"] = str(int(t["total"].min()))
    c["FinDiscMax"] = str(int(t["total"].max()))
    c["FinDiscBaseAdap"] = str(int(t.set_index("estrategia").loc["base_adap", "total"]))


def modelado(c: Cifras) -> None:
    """Intervalo bootstrap del error en prueba."""
    b = pd.read_csv(RM / "test_bootstrap.csv")
    tabla(
        b,
        "final_bootstrap",
        "Error en prueba del modelo congelado con intervalo bootstrap por pacientes al 95 por ciento",
        "lrlrrr",
        ["Grupo", "Pacientes", "Medida", "Estimacion", "IC inf", "IC sup"],
    )
    todos = b[b["grupo"] == "Todos"].set_index("medida")
    c["FinRMSE"] = f"{todos.loc['rmse_modelo', 'estimacion']:.3f}"
    c["FinRMSEInf"] = f"{todos.loc['rmse_modelo', 'ic_inf']:.3f}"
    c["FinRMSESup"] = f"{todos.loc['rmse_modelo', 'ic_sup']:.3f}"
    c["FinRMSEMedia"] = f"{todos.loc['rmse_media_train', 'estimacion']:.3f}"
    for clave, columna in (
        ("FinReduccion", "estimacion"),
        ("FinReduccionInf", "ic_inf"),
        ("FinReduccionSup", "ic_sup"),
    ):
        c[clave] = f"{100 * todos.loc['reduccion_relativa', columna]:.0f}"


def matriz(c: Cifras) -> None:
    """Estado de la matriz de tecnicas por bloque."""
    a = pd.read_csv(R3 / "auditoria_matriz.csv", keep_default_na=False)
    resumen = pd.crosstab(
        a["bloque"], a["estado_real"], margins=True, margins_name="Total"
    )
    tabla(
        resumen.reset_index(),
        "final_matriz",
        "Estado de las tecnicas de la matriz por bloque",
        "lrrr",
        ["Bloque", *[str(col) for col in resumen.columns]],
    )
    c["FinTecnicasTotal"] = str(len(a))
    c["FinTecnicasCompletas"] = str(int((a["estado_real"] == "completo").sum()))
    c["FinTecnicasNoAplica"] = str(int((a["estado_real"] == "no_aplica").sum()))


def main() -> None:
    """Genera todas las tablas y cifras del informe final."""
    cifras: Cifras = {}
    print("Tablas del informe final:")
    for bloque in (datos, oe3, oe2, oe4, oe5, modelado, matriz):
        bloque(cifras)
    lineas = [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in cifras.items()]
    (SALIDA / "final_cifras.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"  final_cifras.tex  ({len(cifras)} macros)")


if __name__ == "__main__":
    main()
