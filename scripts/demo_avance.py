"""Demostracion en vivo del avance: resultados clave y figuras, paso a paso.

Uso:
    uv run python scripts/demo_avance.py

En cada paso imprime los numeros clave leidos de los resultados, abre la figura
y espera Enter. El paso de las estrategias se regenera en vivo. Al final corre
las pruebas y la auditoria de la matriz de tecnicas.
"""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from psim import RAIZ

R2 = RAIZ / "results" / "fase2"
R3 = RAIZ / "results" / "fase3"
ORIGEN = {"NLM-ChinaCXRSet": "Shenzhen", "NLM-MontgomeryCXRSet": "Montgomery"}
FALTA = [
    "Informe final integrado en LaTeX (fases 1 a 3, modelado y OE2 a OE5)",
    "Presentacion final en Beamer",
    "README del repositorio actualizado",
    "Reproduccion completa desde cero en otro computador",
]
pd.set_option("display.width", 160)


def titulo(texto: str) -> None:
    """Encabezado de un paso."""
    print("\n" + "=" * 78)
    print(f"  {texto}")
    print("=" * 78)


def abrir(relativa: str) -> None:
    """Abre una figura con el visor del sistema."""
    ruta = RAIZ / relativa
    if not ruta.is_file():
        print(f"\n  (falta la figura {relativa})")
        return
    print(f"\n  Abriendo {relativa}")
    if sys.platform == "win32":
        os.startfile(ruta)
    else:
        webbrowser.open(ruta.as_uri())


def correr(*argumentos: str) -> None:
    """Ejecuta un comando de Python del proyecto mostrando su salida."""
    print(f"\n  $ python {' '.join(argumentos)}\n")
    subprocess.run([sys.executable, *argumentos], cwd=RAIZ, check=False)


def leer(ruta: Path, **opciones: object) -> pd.DataFrame | None:
    """Lee un resultado o avisa si falta."""
    if not ruta.is_file():
        print(f"  (falta {ruta.relative_to(RAIZ)}; ejecute reproduce.py --fase 5)")
        return None
    return pd.read_csv(ruta, **opciones)


def pausa() -> None:
    """Espera Enter para seguir, o q para salir."""
    respuesta = input("\n  [Enter] siguiente paso    [q] salir: ").strip().lower()
    if respuesta == "q":
        raise SystemExit(0)


def paso_espectro() -> None:
    """Firma espectral de cada hospital."""
    titulo("1. Cada hospital tiene una firma de frecuencia propia (OE2, frecuencia)")
    d = leer(R2 / "espectro_por_origen.csv")
    if d is not None:
        f = d.iloc[0]
        print("  Fraccion de potencia sobre 0.25 ciclos/pixel (mediana):")
        print(f"    Montgomery: {f['mediana_montgomery']:.5f}")
        print(f"    Shenzhen:   {f['mediana_shenzhen']:.5f}")
        print(
            f"  Delta de Cliff Montgomery vs Shenzhen: {f['delta_mc_vs_sh']:+.2f} "
            f"[{f['ic_inf']:+.2f}, {f['ic_sup']:+.2f}]"
        )
    abrir("figures/fase2/espectros.png")


def paso_wavelets() -> None:
    """Energia por subbanda y orientacion."""
    titulo(
        "2. La firma del sitio esta en la adquisicion, no en la anatomia (OE2, wavelets)"
    )
    d = leer(R3 / "energia_subbandas_por_origen.csv")
    if d is not None:
        columnas = [
            "descriptor",
            "mediana_mc",
            "mediana_sh",
            "delta",
            "p_holm",
            "significativo",
        ]
        print(d[columnas].round(4).to_string(index=False))
        print("\n  H = bordes horizontales (costillas, diafragma): anatomia")
        print("  V y D = textura y adquisicion: ahi se separan los hospitales")
    abrir("figures/fase3/energia_wavelets.png")


def paso_realce() -> None:
    """Especificacion de histograma y decision de las estrategias de realce."""
    titulo(
        "3. Normalizar la intensidad no elimina el efecto de sitio (OE2, intensidad)"
    )
    d = leer(R2 / "especificacion_por_origen.csv")
    if d is not None:
        tabla = d.pivot(index="medida", columns="montgomery", values="delta_mc_vs_sh")
        tabla = tabla.drop(index="distancia_a_shenzhen")[["original", "especificada"]]
        print(
            "  Delta de Cliff Montgomery vs Shenzhen, antes y despues de especificar:"
        )
        print(tabla.round(3).to_string())
    print("\n  Estrategias de realce que cumplen el criterio de la formulacion:")
    total, conservadas = 0, []
    for archivo in (
        "decision_realce_espacial.csv",
        "decision_realce_frecuencia.csv",
        "decision_realce_histograma.csv",
    ):
        decision = leer(R2 / archivo, index_col=0)
        if decision is None:
            continue
        decision = decision.drop(index="original")
        total += len(decision)
        conservadas += list(decision.index[decision["se_conserva"]])
    print(f"    {len(conservadas)} de {total}: {', '.join(conservadas) or 'ninguna'}")
    abrir("figures/fase2/histogram_matching.png")


def paso_oe4() -> None:
    """Desplazamiento de cada descriptor."""
    titulo("4. OE4: tres tipos de inestabilidad de los descriptores")
    d = leer(R3 / "error_descriptores_particiones.csv")
    if d is not None:
        prueba = d[d["split"] == "test"]
        resumen = prueba.pivot(
            index="descriptor",
            columns="origen",
            values=["sesgo_mediana", "pct_sobreestima"],
        )
        print(
            "  Particion de prueba (sesgo relativo mediano y % de imagenes sobreestimadas):"
        )
        print(resumen.round(3).to_string())
        print(
            "\n  Compacidad: sesgo sistematico | Relacion de aspecto: sesgo por sitio"
        )
        print("  Simetria: error aleatorio (centrado en cero, del tamano de la medida)")
    abrir("figures/fase3/error_descriptores_particiones.png")


def paso_oe5() -> None:
    """Diferencias entre grupos y estabilidad."""
    titulo("5. OE5: diferencias entre grupos (protocolo congelado antes de mirar)")
    d = leer(R3 / "oe5_comparacion.csv")
    if d is not None:
        ref = d[d["fuente"] == "referencia"].copy()
        ref["sitio"] = ref["estrato"].map(ORIGEN)
        columnas = [
            "sitio",
            "descriptor",
            "delta",
            "ic_inf",
            "ic_sup",
            "p_holm",
            "significativo",
        ]
        print("  Mascara de referencia, por paciente:")
        print(ref[columnas].round(3).to_string(index=False))
    e = leer(R3 / "oe5_estabilidad.csv")
    if e is not None:
        estables = int(e["estable"].sum())
        print(
            f"\n  Conclusiones que se mantienen con la segmentacion automatica: {estables} de {len(e)}"
        )
    abrir("figures/fase3/oe5_delta_cliff.png")


def paso_estrategias() -> None:
    """Conclusiones por estrategia, regenerado en vivo."""
    titulo("6. La segmentacion cambia las conclusiones (se genera en vivo)")
    correr("scripts/figura_oe5_estrategias.py")
    abrir("figures/fase3/oe5_estrategias_train.png")


def paso_cierre() -> None:
    """Pruebas y auditoria en vivo, y lo que falta."""
    titulo("Reproducibilidad, en vivo: pruebas y auditoria de la matriz de tecnicas")
    correr("-m", "pytest", "-q")
    correr("scripts/auditar_matriz.py")
    titulo("Lo que falta (semana 15)")
    for i, pendiente in enumerate(FALTA, 1):
        print(f"  {i}. {pendiente}")
    print()


PASOS: list[Callable[[], None]] = [
    paso_espectro,
    paso_wavelets,
    paso_realce,
    paso_oe4,
    paso_oe5,
    paso_estrategias,
    paso_cierre,
]


def main() -> None:
    """Recorre los pasos de la demostracion."""
    titulo("Estabilidad de los descriptores morfologicos del campo pulmonar - avance")
    print("  Nuevo desde la ultima entrega: OE2, OE4 y OE5 completos;")
    print(
        "  50 de 53 tecnicas de la matriz implementadas y 3 justificadas como no aplicables."
    )
    pausa()
    for i, paso in enumerate(PASOS):
        paso()
        if i < len(PASOS) - 1:
            pausa()


if __name__ == "__main__":
    main()
