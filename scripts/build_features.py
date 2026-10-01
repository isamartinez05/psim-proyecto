"""Extrae la tabla de caracteristicas sobre la cohorte de entrenamiento.

Una fila por combinacion de imagen y estrategia. Con 486 imagenes que tienen
mascara de referencia y nueve estrategias, la tabla tiene 4374 filas.

Las etiquetas no entran en el calculo de ningun descriptor: se incorporan al
final mediante una union por image_id, y la tabla las marca como columna de
control, no como predictor.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from psim import DATA_RAW, resultado
from psim.eda import cargar
from psim.features import DESCRIPTORES, error_relativo, extraer
from psim.io import leer_gris
from psim.mascaras import leer_referencia
from psim.recorte import mascara_campo
from psim.segmentacion import CONFIGURACIONES, dice, jaccard, reducir, segmentar


def main() -> None:
    d = cargar("data/metadata/manifest.csv", "data/metadata/splits.csv")
    con = d[d["mask_path"] != ""].reset_index(drop=True)
    print(f"entrenamiento {len(d)}, con mascara {len(con)}")
    print(f"esperadas {len(con) * len(CONFIGURACIONES)} filas")

    filas: list[dict[str, object]] = []
    t0 = time.time()

    for i, (_, r) in enumerate(con.iterrows(), 1):
        img = leer_gris(DATA_RAW / r["relative_path"])
        ref = leer_referencia(r.to_dict(), img.shape)

        g = reducir(img)
        campo = mascara_campo(g)
        ref_total = reducir(ref.total)
        ref_izq = reducir(ref.izquierdo)
        ref_der = reducir(ref.derecho)

        d_ref = extraer(g, ref_total, campo, ref_izq, ref_der)

        for cfg in CONFIGURACIONES:
            s = segmentar(g, cfg, campo)
            d_obt = extraer(g, s.mascara, campo)

            fila: dict[str, object] = {
                "sample_id": f"{r['image_id']}__{cfg}",
                "image_id": r["image_id"],
                "subject_id": r["subject_id"],
                "config_id": cfg,
                "realce": cfg.split("_")[0],
                "umbral": cfg.split("_")[1],
                "split": r["split"],
                "dataset_id": r["dataset_id"],
                "source_sha256": r["sha256"],
                "lado_trabajo": max(g.shape),
                "campo_px": int(campo.sum()),
                "mask_px": int(s.mascara.sum()),
                "n_componentes": s.n_componentes,
                "umbral_usado": round(float(s.umbral_usado), 6)
                if np.isfinite(s.umbral_usado)
                else "",
                "dice": round(dice(s.mascara, ref_total), 6),
                "jaccard": round(jaccard(s.mascara, ref_total), 6),
            }

            for k in DESCRIPTORES:
                fila[k] = round(d_obt[k], 6) if np.isfinite(d_obt[k]) else ""
                fila[f"ref_{k}"] = round(d_ref[k], 6) if np.isfinite(d_ref[k]) else ""
                e = error_relativo(d_obt[k], d_ref[k])
                fila[f"err_{k}"] = round(e, 6) if np.isfinite(e) else ""

            fila["target"] = r["target"]
            fila["edad"] = r["edad"]
            fila["sexo"] = r["sexo"]
            fila["quality_status"] = r["quality_status"]
            fila["eligible_for_model"] = int(
                r["quality_status"] == "aceptada" and s.n_componentes == 2
            )
            filas.append(fila)

        if i % 25 == 0 or i == len(con):
            t = time.time() - t0
            print(
                f"  {i:>4}/{len(con)}  {t / 60:5.1f} min  "
                f"restan {(len(con) - i) * t / i / 60:5.1f} min"
            )

    tabla = pd.DataFrame(filas).sort_values("sample_id")
    destino = Path("data/processed/features.csv")
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino, index=False)
    print(f"\nescritas {len(tabla)} filas x {len(tabla.columns)} columnas en {destino}")

    resumen = (
        tabla.groupby("config_id")[["dice", "jaccard"]]
        .median()
        .round(4)
        .sort_values("dice", ascending=False)
    )
    resumen.to_csv(resultado(2, "dice_por_estrategia.csv"))
    print()
    print(resumen.to_string())


if __name__ == "__main__":
    main()
