"""Genera las tablas .tex y las cifras del lab005 desde los resultados del modelado.

Como en build_tablas_tex.py, ningun numero del informe se transcribe a mano:
el documento incluye estas tablas y macros con \\input.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from build_tablas_tex import SALIDA, escapar, tabla

from psim import RAIZ

PROY = RAIZ / "results" / "modelado_proyecto"
DIDA = RAIZ / "results" / "modelado_didactico"
PROC = RAIZ / "data" / "processed"
SH, MC = "NLM-ChinaCXRSet", "NLM-MontgomeryCXRSet"
ORIGENES = {"todos": "Todos", SH: "Shenzhen", MC: "Montgomery"}
MODELOS = {
    "grado_1": "Grado 1",
    "grado_2": "Grado 2",
    "referencia_media": "Media de train",
    "referencia_mediana": "Mediana de train",
    "degree_1": "Grado 1",
    "degree_2": "Grado 2",
    "baseline": "Referencia constante",
    "C_0.1": "Logistica C=0.1",
    "C_1.0": "Logistica C=1.0",
}
CABECERA_METRICAS = ["Modelo", "Origen", "n", "MAE", "RMSE", "R2"]


def metricas_proyecto(nombre_csv: str) -> pd.DataFrame:
    """Metricas del proyecto por modelo y origen, con nombres legibles."""
    df = pd.read_csv(PROY / nombre_csv)
    df["modelo"] = df["modelo"].map(MODELOS)
    df["subgrupo"] = df["subgrupo"].map(ORIGENES)
    return df[["modelo", "subgrupo", "n", "MAE", "RMSE", "R2"]]


def didactico() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une validacion y prueba de la practica comun, separadas por ruta."""
    partes = []
    for archivo, poblacion in (
        ("validation_metrics.csv", "Validacion"),
        ("test_metrics.csv", "Prueba"),
    ):
        df = pd.read_csv(DIDA / archivo)
        df["model"] = df["model"].map(MODELOS)
        partes.append(df.assign(poblacion=poblacion))
    df = pd.concat(partes, ignore_index=True)
    reg = df[df["route"] == "continuous"][["poblacion", "model", "MAE", "RMSE", "R2"]]
    binaria = df[df["route"] == "binary"][
        ["poblacion", "model", "ROC_AUC", "AP", "Brier", "sensitivity", "specificity"]
    ]
    return reg, binaria


def valor(df: pd.DataFrame, modelo: str, subgrupo: str, columna: str) -> float:
    """Un valor de una tabla de metricas del proyecto."""
    fila = df[(df["modelo"] == modelo) & (df["subgrupo"] == subgrupo)]
    return float(fila[columna].iloc[0])


def valor_didactico(df: pd.DataFrame, modelo: str, columna: str) -> float:
    """Un valor de una tabla de metricas de la practica comun."""
    return float(df.loc[df["model"] == modelo, columna].iloc[0])


def escribir_cifras(proto: dict[str, Any], evaluacion: dict[str, Any]) -> None:
    """Escribe las macros numericas que usa el texto del informe."""
    eleg = proto["modelo_elegido"]
    val = pd.read_csv(PROY / "validation_metrics.csv")
    test = pd.read_csv(PROY / "test_metrics.csv")
    pred = pd.read_csv(PROY / "test_predictions.csv")
    asoc = pd.read_csv(PROY / "asociacion_con_objetivo_train.csv", index_col=0)
    dproto = json.loads((DIDA / "frozen_protocol.json").read_text(encoding="utf-8"))
    dva = pd.read_csv(DIDA / "validation_metrics.csv")
    dte = pd.read_csv(DIDA / "test_metrics.csv")

    train = pd.read_csv(PROC / "features.csv")
    train = train[(train["config_id"] == "base_adap") & (train["split"] == "train")]
    fval = pd.read_csv(PROC / "features_validation.csv")
    ftest = pd.read_csv(PROC / "features_test.csv")
    media_origen = (
        train[train["eligible_for_model"] == 1]
        .groupby("dataset_id")["ref_fraccion_area"]
        .mean()
    )
    residuo = pred.groupby("dataset_id")["residuo"].mean()
    n_origen = pred["dataset_id"].value_counts()
    rmse_te = valor(test, eleg, "todos", "RMSE")
    rmse_base = valor(test, "referencia_media", "todos", "RMSE")
    diseno = proto["diseno_train"][eleg]
    dreg, dbin = dproto["continuous_model"], dproto["binary_model"]

    cifras = {
        "ModElegidoNombre": MODELOS[eleg],
        "ModGrado": str(proto["grado"]),
        "ModMascTrain": str(len(train)),
        "ModMascVal": str(len(fval)),
        "ModMascTest": str(len(ftest)),
        "ModElegTrain": str(int(train["eligible_for_model"].sum())),
        "ModElegVal": str(int(fval["eligible_for_model"].sum())),
        "ModElegTest": str(int(ftest["eligible_for_model"].sum())),
        "ModNTrain": str(proto["n_train_grupos"]),
        "ModNVal": str(proto["n_validacion_grupos"]),
        "ModNTest": str(evaluacion["n_test_grupos"]),
        "ModNTestSH": str(int(n_origen[SH])),
        "ModNTestMC": str(int(n_origen[MC])),
        "ModRMSEValUno": f"{valor(val, 'grado_1', 'todos', 'RMSE'):.4f}",
        "ModRMSEValDos": f"{valor(val, 'grado_2', 'todos', 'RMSE'):.4f}",
        "ModRMSEValBase": f"{valor(val, 'referencia_media', 'todos', 'RMSE'):.4f}",
        "ModRVal": f"{valor(val, eleg, 'todos', 'R2'):.2f}",
        "ModRMSETest": f"{rmse_te:.4f}",
        "ModRMSETestBase": f"{rmse_base:.4f}",
        "ModRTest": f"{valor(test, eleg, 'todos', 'R2'):.2f}",
        "ModMAETest": f"{valor(test, eleg, 'todos', 'MAE'):.4f}",
        "ModMejoraTest": f"{100 * (1 - rmse_te / rmse_base):.0f}",
        "ModRMSETestSH": f"{valor(test, eleg, SH, 'RMSE'):.4f}",
        "ModRMSETestMC": f"{valor(test, eleg, MC, 'RMSE'):.4f}",
        "ModResTestSH": f"{residuo[SH]:+.4f}",
        "ModResTestMC": f"{residuo[MC]:+.4f}",
        "ModRefMediaSH": f"{media_origen[SH]:.3f}",
        "ModRefMediaMC": f"{media_origen[MC]:.3f}",
        "ModRangoDiseno": f"{diseno['rango']}/{diseno['columnas']}",
        "ModCondicion": f"{diseno['condicion']:.1f}",
        "ModRhoFA": f"{asoc.loc['fraccion_area', 'spearman']:.2f}",
        "ModRhoEW": f"{asoc.loc['energia_wavelet_d1', 'spearman']:.2f}",
        "ModRhoSI": f"{asoc.loc['simetria_izq_der', 'spearman']:.2f}",
        "DidModeloReg": MODELOS[dreg],
        "DidModeloBin": escapar(MODELOS[dbin]),
        "DidRMSEValReg": f"{valor_didactico(dva, dreg, 'RMSE'):.3f}",
        "DidRMSETestReg": f"{valor_didactico(dte, dreg, 'RMSE'):.3f}",
        "DidRMSETestBase": f"{dte[dte['route'] == 'continuous'].iloc[-1]['RMSE']:.3f}",
        "DidAUCVal": f"{valor_didactico(dva, dbin, 'ROC_AUC'):.2f}",
        "DidAUCTest": f"{valor_didactico(dte, dbin, 'ROC_AUC'):.2f}",
        "DidAPTest": f"{valor_didactico(dte, dbin, 'AP'):.2f}",
        "DidPrevTest": f"{dte[dte['route'] == 'binary'].iloc[-1]['AP']:.2f}",
        "DidSensTest": f"{valor_didactico(dte, dbin, 'sensitivity'):.2f}",
        "DidEspTest": f"{valor_didactico(dte, dbin, 'specificity'):.2f}",
    }
    lineas = [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in cifras.items()]
    (SALIDA / "lab005_cifras.tex").write_text(
        "\n".join(lineas) + "\n", encoding="utf-8"
    )
    print(f"  lab005_cifras.tex  ({len(cifras)} macros)")


def main() -> None:
    """Genera todas las tablas y cifras del informe del lab005."""
    proto = json.loads((PROY / "frozen_protocol.json").read_text(encoding="utf-8"))
    evaluacion = json.loads((PROY / "test_evaluation.json").read_text(encoding="utf-8"))
    print("Tablas del lab005:")
    tabla(
        metricas_proyecto("validation_metrics.csv"),
        "lab005_validacion",
        "Metricas en validacion por modelo y origen (fraccion adimensional)",
        "llrrrr",
        CABECERA_METRICAS,
    )
    tabla(
        metricas_proyecto("test_metrics.csv"),
        "lab005_prueba",
        "Metricas en la prueba, evaluada una sola vez con el modelo congelado",
        "llrrrr",
        CABECERA_METRICAS,
    )
    tabla(
        pd.DataFrame(proto["coeficientes"]),
        "lab005_coeficientes",
        "Coeficientes del modelo elegido, ajustado en train "
        "(predictores estandarizados con media y DE de train)",
        "lr",
        ["Termino", "Coeficiente"],
    )
    tabla(
        pd.read_csv(RAIZ / "docs" / "decisiones_modelado.csv"),
        "lab005_decisiones",
        "Del analisis exploratorio a las decisiones de modelado",
        r"p{3.6cm}p{2.8cm}p{3.4cm}p{3.4cm}",
        ["Hallazgo del EDA", "Decision", "Comprobacion", "Limite"],
    )
    reg, binaria = didactico()
    tabla(
        reg,
        "lab005_didactico_reg",
        "Practica comun, ruta continua (datos sinteticos, unidades arbitrarias)",
        "llrrr",
        ["Poblacion", "Modelo", "MAE", "RMSE", "R2"],
    )
    tabla(
        binaria,
        "lab005_didactico_bin",
        "Practica comun, ruta binaria (datos sinteticos, umbral 0.5)",
        "llrrrrr",
        ["Poblacion", "Modelo", "ROC-AUC", "AP", "Brier", "Sens.", "Espec."],
    )
    escribir_cifras(proto, evaluacion)


if __name__ == "__main__":
    main()
