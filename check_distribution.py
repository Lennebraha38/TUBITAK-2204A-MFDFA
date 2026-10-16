"""Skorlanmis verilerin dagilimini kontrol et (model kalibrasyonu)."""
import pandas as pd
import numpy as np

files = [f"results/scored/depremGun{i}tr_scored.csv.gz" for i in range(1, 6)]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"Toplam: {len(df):,}")
print(f"p_pos ort: {df['p_pos'].mean():.4f}")
print(f"p_neu ort: {df['p_neu'].mean():.4f}")
print(f"p_neg ort: {df['p_neg'].mean():.4f}")

# Argmax sinif dagilimi
am = df[["p_pos", "p_neu", "p_neg"]].values.argmax(axis=1)  # 0=pos,1=neu,2=neg
tot = len(am)
print(f"\nArgmax dagilimi:")
print(f"  POZITIF : {(am==0).sum():>8,} ({(am==0).mean()*100:.1f}%)")
print(f"  NOTR    : {(am==1).sum():>8,} ({(am==1).mean()*100:.1f}%)")
print(f"  NEGATIF : {(am==2).sum():>8,} ({(am==2).mean()*100:.1f}%)")

print(f"\nValans dagilimi:")
print(f"  ort={df['valence'].mean():+.4f} std={df['valence'].std():.4f}")
for lo, hi in [(-1.0,-0.5),(-0.5,-0.1),(-0.1,0.1),(0.1,0.5),(0.5,1.0)]:
    m = (df['valence']>=lo)&(df['valence']<hi)
    print(f"  [{lo:+.1f},{hi:+.1f}) : {m.sum():>8,} ({m.mean()*100:.1f}%)")

# Notr'e yakinsa valans kucuk olmali; kontrol: notr orani yuksekken valans?
print(f"\nNotr olasiligi > 0.5 olan tweet orani: {(df['p_neu']>0.5).mean()*100:.1f}%")
