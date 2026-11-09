"""Audita la matriz de tecnicas contra la implementacion real del proyecto.

Para cada tecnica de docs/matriz_tecnicas.csv conserva lo prometido en la
formulacion y registra donde quedo implementada, que producto la evidencia y
por que cambio, si cambio. Sin argumentos solo audita; con --actualizar agrega
las columnas archivo_real, producto_real y nota, y actualiza el estado.
"""

from __future__ import annotations

import argparse

import pandas as pd

from psim import RAIZ

MATRIZ = RAIZ / "docs" / "matriz_tecnicas.csv"
SALIDA = RAIZ / "results" / "fase3" / "auditoria_matriz.csv"
MANIFIESTO = "results/fase1/manifiesto_datos.csv"
NO_APLICA = "no_aplica"

# tecnica: (archivo real, producto real, nota). Una nota que empieza con
# NO_APLICA marca una tecnica que se verifico y no corresponde aplicar.
MAPEO: dict[str, tuple[str, str, str]] = {
    "Inspeccion de formato y metadatos": ("src/psim/io.py", MANIFIESTO, ""),
    "Verificacion de resolucion espacial y muestreo": (
        "src/psim/io.py",
        MANIFIESTO,
        "",
    ),
    "Analisis de cuantizacion y resolucion de intensidad": (
        "src/psim/io.py",
        MANIFIESTO,
        "bits efectivos por imagen en el manifiesto",
    ),
    "Submuestreo controlado": (
        "src/psim/segmentacion.py",
        "figures/fase1/submuestreo.png",
        "reduccion a 512 px; aliasing ilustrado en oe2_intensidad.py",
    ),
    "Requantizacion controlada": (
        "src/psim/intensidad.py",
        "figures/fase1/requantizacion.png",
        "",
    ),
    "Correccion de escala invertida Shenzhen": (
        "src/psim/io.py",
        MANIFIESTO,
        f"{NO_APLICA}: se verifico que ninguna imagen esta invertida",
    ),
    "Vecindad adyacencia y conectividad": (
        "src/psim/geometria.py",
        "figures/fase1/conectividad.png",
        "",
    ),
    "Etiquetado de regiones conexas": (
        "src/psim/geometria.py",
        "results/fase2/conectividad_componentes.csv",
        "conteo con vecindad 4 y 8 en la muestra de OE2",
    ),
    "Transformada de distancia y extraccion de borde": (
        "src/psim/geometria.py",
        "figures/fase1/distancia_borde.png",
        "",
    ),
    "Verificacion de canales RGB y conversion a gris": (
        "src/psim/io.py",
        MANIFIESTO,
        "canales_son_identicos y a_gris",
    ),
    "Exploracion del espacio HSV": (
        "src/psim/io.py",
        MANIFIESTO,
        f"{NO_APLICA}: radiografias en gris con canales identicos; HSV no aporta",
    ),
    "Operaciones aritmeticas entre imagenes": (
        "src/psim/intensidad.py",
        "figures/fase2/mapas_diferencia.png",
        "",
    ),
    "Normalizacion de rango dinamico": (
        "src/psim/intensidad.py",
        "figures/fase2/transformaciones_intensidad.png",
        "",
    ),
    "Transformaciones puntuales de intensidad": (
        "src/psim/intensidad.py",
        "figures/fase2/transformaciones_intensidad.png",
        "gamma y logaritmica",
    ),
    "Filtro de media": (
        "src/psim/espacial.py",
        "figures/fase2/filtros_comparacion.png",
        "",
    ),
    "Filtro de mediana": (
        "src/psim/espacial.py",
        "figures/fase2/median_comparison.png",
        "",
    ),
    "Filtro gaussiano": (
        "src/psim/espacial.py",
        "figures/fase2/filtros_comparacion.png",
        "",
    ),
    "Operador Sobel": (
        "src/psim/espacial.py",
        "figures/fase2/bordes_comparacion.png",
        "",
    ),
    "Operador Prewitt": (
        "src/psim/espacial.py",
        "figures/fase2/bordes_comparacion.png",
        "",
    ),
    "Operador Roberts": (
        "src/psim/espacial.py",
        "figures/fase2/bordes_comparacion.png",
        "",
    ),
    "Operador Laplaciano": (
        "src/psim/espacial.py",
        "figures/fase2/bordes_comparacion.png",
        "",
    ),
    "Realce por unsharp masking": (
        "src/psim/espacial.py",
        "figures/fase2/realce_comparacion.png",
        "",
    ),
    "Operador direccional emboss": (
        "src/psim/espacial.py",
        "figures/fase2/emboss.png",
        "",
    ),
    "Histograma de intensidad y funcion de distribucion acumulada": (
        "src/psim/histograma.py",
        "figures/fase1/histogramas.png",
        "",
    ),
    "Comparacion de distribuciones entre origenes": (
        "src/psim/histograma.py",
        "figures/fase1/histogramas.png",
        "histograma y CDF por origen en la misma figura",
    ),
    "Ecualizacion global de histograma": (
        "src/psim/histograma.py",
        "figures/fase2/ecualizacion_comparacion.png",
        "tambien como estrategia eq_* en segmentacion.py",
    ),
    "Ecualizacion adaptativa limitada por contraste": (
        "src/psim/histograma.py",
        "figures/fase2/ecualizacion_comparacion.png",
        "tambien como estrategia clahe_* en segmentacion.py",
    ),
    "Especificacion de histograma entre origenes": (
        "src/psim/histograma.py",
        "figures/fase2/histogram_matching.png",
        "",
    ),
    "Umbral global fijo": (
        "src/psim/segmentacion.py",
        "figures/fase2/dice_por_estrategia.png",
        "",
    ),
    "Umbral de Otsu": (
        "src/psim/segmentacion.py",
        "figures/fase2/dice_por_estrategia.png",
        "",
    ),
    "Umbral adaptativo local": (
        "src/psim/segmentacion.py",
        "figures/fase2/dice_por_estrategia.png",
        "",
    ),
    "Erosion": ("src/psim/geometria.py", "figures/fase2/morfologia_pasos.png", ""),
    "Dilatacion": ("src/psim/geometria.py", "figures/fase2/morfologia_pasos.png", ""),
    "Apertura morfologica": (
        "src/psim/geometria.py",
        "figures/fase2/morfologia_pasos.png",
        "en el flujo del proyecto: refinar en segmentacion.py",
    ),
    "Cierre morfologico": (
        "src/psim/geometria.py",
        "figures/fase2/morfologia_pasos.png",
        "en el flujo del proyecto: refinar en segmentacion.py",
    ),
    "Transformacion top-hat": (
        "src/psim/geometria.py",
        "figures/fase2/tophat_blackhat.png",
        "",
    ),
    "Transformacion black-hat": (
        "src/psim/geometria.py",
        "figures/fase2/tophat_blackhat.png",
        "",
    ),
    "Relleno de huecos y seleccion de componentes mayores": (
        "src/psim/segmentacion.py",
        "results/fase2/conectividad_componentes.csv",
        (
            "seleccion de componentes en refinar; los huecos del campo no se "
            "rellenan por diseno, para no incorporar las axilas"
        ),
    ),
    "Transformada de Fourier 2D y espectro de magnitud": (
        "src/psim/frecuencia.py",
        "figures/fase2/espectros.png",
        "",
    ),
    "Filtro pasa-bajas gaussiano": (
        "src/psim/frecuencia.py",
        "figures/fase2/pasabajas_comparacion.png",
        "",
    ),
    "Filtro pasa-bajas Butterworth": (
        "src/psim/frecuencia.py",
        "figures/fase2/pasabajas_comparacion.png",
        "",
    ),
    "Filtro pasa-altas": ("src/psim/frecuencia.py", "figures/fase2/pasaaltas.png", ""),
    "Filtro pasa-banda": ("src/psim/frecuencia.py", "figures/fase2/pasabanda.png", ""),
    "Transformada wavelet continua sobre perfil 1D": (
        "src/psim/wavelets.py",
        "figures/fase2/cwt_perfil.png",
        "",
    ),
    "Descomposicion wavelet discreta 2D": (
        "src/psim/wavelets.py",
        "results/fase3/energia_subbandas.csv",
        "tres niveles db2; en el proyecto, nivel 1 en features.py",
    ),
    "Reconstruccion inversa por niveles": (
        "src/psim/wavelets.py",
        "figures/fase3/reconstruccion_dwt.png",
        "",
    ),
    "Comparacion de energia por subbanda entre origenes": (
        "src/psim/wavelets.py",
        "figures/fase3/energia_wavelets.png",
        "",
    ),
    "Estadistica descriptiva de intensidad por region": (
        "src/psim/eda.py",
        "results/fase2/especificacion_histograma.csv",
        "mediana de intensidad dentro del campo; estadisticos globales en el manifiesto",
    ),
    "Extraccion de descriptores morfologicos": (
        "src/psim/features.py",
        "data/processed/features.csv",
        "",
    ),
    "Estimacion de la distribucion de los descriptores": (
        "scripts/analisis_oe5.py",
        "figures/fase3/distribuciones.png",
        "",
    ),
    "Prueba de hipotesis entre grupos": (
        "src/psim/estadistica.py",
        "results/fase3/oe5_comparacion.csv",
        "Mann-Whitney, delta de Cliff con bootstrap y Holm",
    ),
    "Regresion logistica sobre descriptores explicitos": (
        "scripts/modelado_didactico.py",
        "results/modelado_didactico/test_metrics.csv",
        (
            f"{NO_APLICA}: la ruta binaria no aplica al proyecto; se desarrollo "
            "en la practica comun"
        ),
    ),
}
ESTADOS = {
    "completo": "completado",
    "parcial": "parcial",
    "falta": "pendiente",
    NO_APLICA: NO_APLICA,
}


def existe(ruta: str) -> bool:
    """True si la ruta declarada existe en el proyecto."""
    ruta = str(ruta).strip()
    return bool(ruta) and (RAIZ / ruta).exists()


def auditar(m: pd.DataFrame) -> pd.DataFrame:
    """Agrega la implementacion real y su estado a cada tecnica."""
    m = m.copy()
    reales = [
        MAPEO.get(t, (a, p, ""))
        for t, a, p in zip(
            m["tecnica"], m["archivo_codigo"], m["figura_o_tabla"], strict=True
        )
    ]
    m["archivo_real"] = [r[0] for r in reales]
    m["producto_real"] = [r[1] for r in reales]
    m["nota"] = [r[2] for r in reales]
    codigo = m["archivo_real"].map(existe)
    producto = m["producto_real"].map(existe)
    m["estado_real"] = "falta"
    m.loc[codigo | producto, "estado_real"] = "parcial"
    m.loc[codigo & producto, "estado_real"] = "completo"
    m.loc[m["nota"].str.startswith(NO_APLICA) & codigo, "estado_real"] = NO_APLICA
    return m


def main() -> None:
    """Audita la matriz y, con --actualizar, escribe el estado real en docs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actualizar", action="store_true")
    args = parser.parse_args()

    original = pd.read_csv(MATRIZ, keep_default_na=False)
    sobrantes = set(MAPEO) - set(original["tecnica"])
    if sobrantes:
        print("AVISO: tecnicas del mapeo que no estan en la matriz:")
        for t in sorted(sobrantes):
            print(f"  {t}")
    m = auditar(original)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(SALIDA, index=False)

    print("=== Resumen por bloque ===")
    print(pd.crosstab(m["bloque"], m["estado_real"], margins=True).to_string())
    pendientes = m[~m["estado_real"].isin(["completo", NO_APLICA])]
    print(f"\n=== Tecnicas no completas: {len(pendientes)} ===")
    for fila in pendientes.itertuples():
        print(
            f"  {fila.estado_real:<8} | {fila.archivo_real:<28} | "
            f"{fila.producto_real:<45} | {fila.tecnica}"
        )
    print("\n=== No aplica ===")
    for fila in m[m["estado_real"] == NO_APLICA].itertuples():
        print(f"  {fila.tecnica}: {fila.nota}")

    if args.actualizar:
        salida = original.copy()
        salida["archivo_real"] = m["archivo_real"]
        salida["producto_real"] = m["producto_real"]
        salida["nota"] = m["nota"]
        salida["estado"] = m["estado_real"].map(ESTADOS)
        salida.to_csv(MATRIZ, index=False, encoding="utf-8")
        print(f"\nMatriz actualizada en {MATRIZ.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
