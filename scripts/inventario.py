import pandas as pd

d = pd.read_csv("data/metadata/manifest.csv", keep_default_na=False)
print(f"filas: {len(d)}   image_id unicos: {d.image_id.nunique()}")
print()

print("--- por conjunto ---")
print(d.groupby("dataset_id").size().to_string())
print()

print("--- etiqueta ---")
print(pd.crosstab(d.dataset_id, d.target).to_string())
print()

print("--- calidad ---")
print(d.quality_status.value_counts().to_string())
print()
print("razones:")
for r, n in d[d.quality_reason != ""].quality_reason.value_counts().items():
    print(f"  {n:>4}  {r}")
print()

print("--- mascara de referencia ---")
con = (d.mask_path != "").groupby(d.dataset_id).sum()
tot = d.groupby("dataset_id").size()
for k in tot.index:
    print(f"  {k}: {con[k]} de {tot[k]}")
print()

print("--- orientacion y canales ---")
print(pd.crosstab(d.dataset_id, d.orientacion).to_string())
print()
print(pd.crosstab(d.dataset_id, [d.dtype, d.bits_efectivos, d.channels]).to_string())
print()

print("--- sujetos agrupados ---")
g = d[d.subject_id != ""]
print(f"  imagenes con subject_id: {len(g)}  sujetos: {g.subject_id.nunique()}")
for s, sub in g.groupby("subject_id"):
    print(f"  {s}: {list(sub.image_id)}")
print()

print("--- demografia ---")
print(
    f"  edad faltante: {(d.edad == '').sum()}   sexo faltante: {(d.sexo == '').sum()}"
)
print("  sexo:", d.sexo.value_counts().to_dict())
e = pd.to_numeric(d.edad, errors="coerce")
print(f"  edad: min={e.min():.0f} max={e.max():.0f} mediana={e.median():.0f}")
