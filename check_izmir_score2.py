"""Izmir score2 cikisini dogrula."""
import pandas as pd
import numpy as np

f = r"C:\TUBITAK_Proje\results\scored2\izmir_earthquake_score2.csv.gz"
d = pd.read_csv(f)
print(f"Satir={len(d):,}  kolonlar={list(d.columns)}")
print(f"NaN: {d.isna().any().any()}")
print()
emo = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
print("Duygu ortalamalari:")
for c in emo:
    print(f"  {c:10s}: {d[c].mean():.4f}")
print(f"  val_sav   : {d['val_sav'].mean():+.4f}")
am = d[emo].values.argmax(1)
print("\nArgmax dagilimi:")
for j, c in enumerate(emo):
    print(f"  {c:10s}: {(am == j).mean()*100:5.1f}%")
print(f"\nsavasy: poz %{(d['val_sav']>0).mean()*100:.1f}, "
      f"neg %{(d['val_sav']<0).mean()*100:.1f}")
print(f"Datetime aralik: {d['Datetime'].min()} -> {d['Datetime'].max()}")
