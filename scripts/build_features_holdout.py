"""Extrae descriptores de validación o prueba con el pipeline congelado del lab004.

Reutiliza las funciones de build_features.py, restringidas a la estrategia
elegida en entrenamiento (base_adap) y a las imágenes con máscara de
referencia, la misma regla de cohorte de entrenamiento. No modifica
features.csv. La prueba se extrae solo después de congelar el protocolo.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from psim import DATA_RAW
from psim.eda import cargar
from psim.features import DESCRIPTORES, error_relativo, extraer
from psim.io import leer_gris
from psim.mascaras import leer_referencia
from psim.recorte import mascara_campo
from psim.segmentacion import CONFIGURACIONES, dice, jaccard, reducir, segmentar

ESTRATEGIA = "base_adap"


def main() -> None:
    """Genera data/processed/features_<split>.csv para una partición reservada."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["validation", "test"])
    args = parser.parse_args()
    if ESTRATEGIA not in CONFIGURACIONES:
        raise ValueError(f"{ESTRATEGIA} no está en CONFIGURACIONES")

    d = cargar(
        Path("data/metadata/manifest.csv"),
        Path("data/metadata/splits.csv"),
        split=args.split,
    )
    con = d[d["mask_path"] != ""].reset_index(drop=True)
    faltan = [p for p in con["relative_path"] if not (DATA_RAW / p).is_file()]
    if faltan:
        raise FileNotFoundError(f"{len(faltan)} imágenes ausentes, p. ej. {faltan[0]}")
    print(f"{args.split}: {len(d)} imágenes, {len(con)} con máscara")

    filas: list[dict[str, object]] = []
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

        s = segmentar(g, ESTRATEGIA, campo)
        d_obt = extraer(g, s.mascara, campo)
        fila: dict[str, object] = {
            "sample_id": f"{r['image_id']}__{ESTRATEGIA}",
            "image_id": r["image_id"],
            "subject_id": r["subject_id"],
            "config_id": ESTRATEGIA,
            "realce": ESTRATEGIA.split("_")[0],
            "umbral": ESTRATEGIA.split("_")[1],
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
            print(f"  {i:>4}/{len(con)}  {(time.time() - t0) / 60:5.1f} min")

    tabla = pd.DataFrame(filas).sort_values("sample_id")
    destino = Path(f"data/processed/features_{args.split}.csv")
    tabla.to_csv(destino, index=False)
    # Solo conteos: no se resumen métricas ni descriptores aquí.
    print(
        f"escritas {len(tabla)} filas en {destino}; "
        f"elegibles {int(tabla['eligible_for_model'].sum())}"
    )


if __name__ == "__main__":
    main()
