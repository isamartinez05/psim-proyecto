from pathlib import Path

from psim.eda import ejecutar

r = ejecutar(Path("data/metadata/manifest.csv"), Path("data/metadata/splits.csv"))
print(f"entrenamiento: {r['n_train']} imagenes")
print(f"galeria: {r['paneles']} paneles")
print(f"estadisticos: {r['stats']} filas")
