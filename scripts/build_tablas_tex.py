"""Genera las tablas .tex que consume el informe, desde los CSV de resultados.

Ningun numero del informe se transcribe a mano: todas las tablas se producen
aqui y el documento las incluye con \\input. Si un resultado cambia, basta con
volver a ejecutar este guion.
"""

from __future__ import annotations

import pandas as pd

from psim import RAIZ

SALIDA = RAIZ / "reports" / "tablas"
DESC = [
    "compacidad",
    "relacion_aspecto",
    "fraccion_area",
    "simetria_izq_der",
    "energia_wavelet_d1",
]
NOMBRES = {
    "compacidad": "Compacidad",
    "relacion_aspecto": "Relacion de aspecto",
    "fraccion_area": "Fraccion de area",
    "simetria_izq_der": "Simetria izq-der",
    "energia_wavelet_d1": "Energia wavelet",
}


def escapar(texto: str) -> str:
    """Protege los caracteres que LaTeX interpreta."""
    for a, b in [
        ("\\", r"\textbackslash "),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde "),
        ("^", r"\textasciicircum "),
    ]:
        texto = texto.replace(a, b)
    return texto


def tabla(
    df: pd.DataFrame,
    nombre: str,
    caption: str,
    formato: str,
    cabeceras: list[str] | None = None,
) -> None:
    """Escribe un entorno tabular listo para incluir con \\input."""
    cabeceras = cabeceras or list(df.columns)
    lineas = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabular}}{{{formato}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{escapar(c)}}}" for c in cabeceras) + r" \\",
        r"\midrule",
    ]
    for _, fila in df.iterrows():
        celdas = []
        for v in fila:
            if isinstance(v, float):
                celdas.append(f"{v:.4f}" if pd.notna(v) else "---")
            else:
                celdas.append(escapar(str(v)))
        lineas.append(" & ".join(celdas) + r" \\")
    lineas += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / f"{nombre}.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"  {nombre}.tex  ({len(df)} filas)")


def main() -> None:
    print("tablas generadas:")

    # Inventario por origen
    m = pd.read_csv(RAIZ / "data/metadata/manifest.csv", keep_default_na=False)
    m["origen"] = m.dataset_id.str.endswith("MontgomeryCXRSet").map(
        {True: "Montgomery", False: "Shenzhen"}
    )
    inv = pd.DataFrame(
        {
            "Origen": ["Montgomery", "Shenzhen", "Total"],
            "Imagenes": [
                int((m.origen == "Montgomery").sum()),
                int((m.origen == "Shenzhen").sum()),
                len(m),
            ],
            "Con mascara": [
                int(((m.origen == "Montgomery") & (m.mask_path != "")).sum()),
                int(((m.origen == "Shenzhen") & (m.mask_path != "")).sum()),
                int((m.mask_path != "").sum()),
            ],
            "TB positivo": [
                int(((m.origen == "Montgomery") & (m.target == 1)).sum()),
                int(((m.origen == "Shenzhen") & (m.target == 1)).sum()),
                int((m.target == 1).sum()),
            ],
            "A revisar": [
                int(
                    ((m.origen == "Montgomery") & (m.quality_status == "revisar")).sum()
                ),
                int(((m.origen == "Shenzhen") & (m.quality_status == "revisar")).sum()),
                int((m.quality_status == "revisar").sum()),
            ],
        }
    )
    tabla(inv, "inventario", "Inventario de la cohorte por origen.", "lrrrr")

    # Particion
    s = pd.read_csv(RAIZ / "data/metadata/splits.csv", keep_default_na=False)
    par = s.groupby(["estrato", "split"]).size().unstack(fill_value=0)
    par = par[["train", "validation", "test"]].reset_index()
    par.columns = ["Estrato", "Entrenamiento", "Validacion", "Prueba"]
    tabla(
        par,
        "particion",
        "Particion por estrato. MC es Montgomery, SH Shenzhen; el sufijo es la clase.",
        "lrrr",
    )

    # Dice por estrategia y origen
    d = pd.read_csv(RAIZ / "results/fase2/dice_por_origen.csv")
    d.columns = ["Estrategia", "Montgomery", "Shenzhen"]
    glob = pd.read_csv(RAIZ / "results/fase2/dice_por_estrategia.csv")
    d = d.merge(glob[["config_id", "dice"]], left_on="Estrategia", right_on="config_id")
    d = d[["Estrategia", "Montgomery", "Shenzhen", "dice"]]
    d.columns = ["Estrategia", "Montgomery", "Shenzhen", "Global"]
    d = d.sort_values("Global", ascending=False)
    tabla(
        d,
        "dice",
        "Coeficiente Dice mediano frente a la referencia, por estrategia y origen.",
        "lrrr",
    )

    # Error por descriptor
    e = pd.read_csv(RAIZ / "results/fase3/error_por_descriptor.csv")
    e = e.rename(columns={"config_id": "Estrategia", **NOMBRES})
    tabla(
        e,
        "errores",
        "Error relativo mediano de cada descriptor frente a su valor de referencia.",
        "l" + "r" * len(DESC),
    )

    # Auditoria
    a = pd.read_csv(RAIZ / "results/fase3/auditoria_tabla.csv", keep_default_na=False)
    a.columns = ["Comprobacion", "Resultado", "Detalle"]
    tabla(
        a,
        "auditoria",
        "Comprobaciones sobre la tabla de caracteristicas.",
        "p{6.0cm}lp{5.0cm}",
    )

    # Decisiones, en formato de descripcion por ser texto largo
    dec = pd.read_csv(RAIZ / "results/fase3/decisiones.csv", keep_default_na=False)
    lineas = []
    for _, r in dec.iterrows():
        lineas += [
            rf"\paragraph{{{escapar(r['ambito'].capitalize())}}}",
            rf"\textbf{{Hallazgo.}} {escapar(r['hallazgo'])}",
            "",
            rf"\textbf{{Evidencia.}} {escapar(r['evidencia'])}",
            "",
            rf"\textbf{{Decision.}} {escapar(r['decision'])}",
            "",
            rf"\textbf{{Efecto esperado.}} {escapar(r['efecto_esperado'])}",
            "",
            rf"\textbf{{Limitacion.}} {escapar(r['limitacion'])}",
            "",
        ]
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "decisiones.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"  decisiones.tex  ({len(dec)} decisiones)")

    # Cifras sueltas que el texto cita
    t = pd.read_csv(RAIZ / "data/processed/features.csv", keep_default_na=False)
    cifras = {
        "NFilasTabla": len(t),
        "NColumnasTabla": len(t.columns),
        "NImagenesTotal": len(m),
        "NImagenesTrain": int((s.split == "train").sum()),
        "NConMascara": int((m.mask_path != "").sum()),
        "NEstrategias": int(t.config_id.nunique()),
        "MejorEstrategia": escapar(str(d.iloc[0]["Estrategia"])),
        "MejorDice": f"{d.iloc[0]['Global']:.3f}",
        "MejorDiceMC": f"{d.iloc[0]['Montgomery']:.3f}",
        "MejorDiceSH": f"{d.iloc[0]['Shenzhen']:.3f}",
    }
    macros = [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in cifras.items()]
    (SALIDA / "cifras.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"  cifras.tex  ({len(cifras)} macros)")


if __name__ == "__main__":
    main()
