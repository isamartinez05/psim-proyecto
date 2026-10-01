import csv
from pathlib import Path

from psim.splits import SEMILLA, construir, escribir, verificar

with open("data/metadata/manifest.csv", encoding="utf-8", newline="") as f:
    filas = list(csv.DictReader(f))

p = construir(filas)
problemas = verificar(p)
if problemas:
    for x in problemas:
        print("PROBLEMA:", x)
    raise SystemExit(1)

escribir(p, Path("data/metadata/splits.csv"))
print(f"particion escrita: {len(p)} filas, semilla {SEMILLA}")
