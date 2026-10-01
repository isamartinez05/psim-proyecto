from pathlib import Path

from psim.manifest import construir, escribir

filas = construir(calcular_hash=False)
print(f"filas: {len(filas)}")
escribir(filas, Path("data/metadata/manifest.csv"))
print("escrito en data/metadata/manifest.csv")
