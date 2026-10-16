"""km2023 score2 cikislarini dogrula: satir sayisi, kolonlar, duygu dagilimi."""
import glob
import numpy as np
import pandas as pd

files = sorted(glob.glob(r"C:\TUBITAK_Proje\results\scored2\deprem*_score2.csv.gz"))
print("Dosyalar:")
tot = 0
frames = []
for f in files:
    d = pd.read_csv(f)
    tot += len(d)
    frames.append(d)
    print(f"  {f.split(chr(92))[-1]:32s} satir={len(d):>7,} kolon={list(d.columns)}")
df = pd.concat(frames, ignore_index=True)
print(f"\nTOPLAM: {len(df):,} satir (beklenen 942,180)")
print(f"NaN var mi: {df.isna().any().any()}")

print("\n=== DUYGU ORTALAMALARI (tweet bazli) ===")
for c in ["anger", "disgust", "fear", "joy", "sadness", "surprise"]:
    print(f"  {c:10s}: ort={df[c].mean():.4f}")
print(f"  {'val_sav':10s}: ort={df['val_sav'].mean():+.4f}")

print("\n=== ARGMAX DAGILIMI (duygu) ===")
emocols = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
am = df[emocols].values.argmax(1)
for j, c in enumerate(emocols):
    print(f"  {c:10s}: {(am==j).mean()*100:5.1f}%")

print("\n=== savasy dagilimi ===")
print(f"  pozitif: {(df['val_sav']>0).mean()*100:.1f}%  negatif: {(df['val_sav']<0).mean()*100:.1f}%")
