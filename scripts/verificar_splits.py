import pandas as pd

m = pd.read_csv("data/metadata/manifest.csv", keep_default_na=False)
s = pd.read_csv("data/metadata/splits.csv", keep_default_na=False)

# El manifiesto trae una columna split vacia; la particion es la autoridad.
m = m.drop(columns=["split"])
d = m.merge(
    s[["image_id", "grupo_id", "estrato", "split"]], on="image_id", validate="1:1"
)
print(f"union: {len(d)} filas  (manifiesto {len(m)}, particion {len(s)})")
print()

print("--- proporciones ---")
v = d["split"].value_counts()
for k in ["train", "validation", "test"]:
    print(f"  {k:<11} {v[k]:>4}  ({v[k] / len(d) * 100:.1f} %)")
print()

print("--- por estrato ---")
print(
    pd.crosstab(d["estrato"], d["split"])[["train", "validation", "test"]].to_string()
)
print()

print("--- edad, que no se estratifico ---")
e = pd.to_numeric(d["edad"], errors="coerce")
print(
    d.assign(e=e)
    .groupby("split")["e"]
    .describe()[["count", "mean", "50%", "min", "max"]]
    .round(1)
    .to_string()
)
print()
d2 = d.assign(
    e=e, rango=pd.cut(e, bins=[0, 18, 200], right=False, labels=["<18a", ">=18a"])
)
print("porcentaje por particion:")
print(
    pd.crosstab(d2["rango"], d2["split"], normalize="columns")
    .mul(100)
    .round(1)
    .to_string()
)
print()

print("--- mascara disponible ---")
print(pd.crosstab(d["mask_path"] != "", d["split"]).to_string())
print()

print("--- grupos de sujeto ---")
for g, sub in d[d["subject_id"] != ""].groupby("subject_id"):
    print(f"  {g}: split={set(sub['split'])}  n={len(sub)}")
