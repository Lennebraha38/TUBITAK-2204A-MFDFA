"""
Skorlanmis tweetlerden (score2 cikisi) zaman serisi uret - TUM sinyal kolonlari icin.

Kullanim: python build_series2.py <glob> <tarih_kolonu> <cikti_onek>
Ornek   : python build_series2.py "results\scored2\deprem*_score2.csv.gz" Date km2023

Kol: anger, disgust, fear, joy, sadness, surprise, val_sav, p_sav_pos, p_sav_neg
Her biri icin 1-dk ve 5-dk ortalamalari alinir.
"""
import sys
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pattern = sys.argv[1]
datecol = sys.argv[2]
prefix = sys.argv[3]

# Sinyal kolonlari (sayisal, ortalamasi alinacak)
SIGNALS = ["anger", "disgust", "fear", "joy", "sadness", "surprise",
           "val_sav", "p_sav_pos", "p_sav_neg"]

files = sorted(glob.glob(pattern))
print(f"Dosyalar ({len(files)}):", flush=True)
for f in files:
    print("  ", f, flush=True)

frames = [pd.read_csv(f) for f in files]
df = pd.concat(frames, ignore_index=True)
df[datecol] = pd.to_datetime(df[datecol], utc=True)
df = df.set_index(datecol).sort_index()

# mevcut sinyal kolonlarini bul
cols = [c for c in SIGNALS if c in df.columns]
missing = [c for c in SIGNALS if c not in df.columns]
if missing:
    print(f"UYARI: eksik kolonlar: {missing}", flush=True)

print(f"\nToplam tweet: {len(df):,}", flush=True)
print(f"Aralik: {df.index.min()} -> {df.index.max()}", flush=True)

# Turetilmis bilesik sinyaller
df["neg_emo"] = df[["sadness", "fear", "anger", "disgust"]].sum(axis=1)
df["distress"] = df[["sadness", "fear"]].sum(axis=1)
df["valence_emo"] = df["joy"] - df["neg_emo"]
cols_all = cols + ["neg_emo", "distress", "valence_emo"]

outdir = r"C:\TUBITAK_Proje\results"
PRIMARY = "distress"  # ana sinyal (uzuntu+korku)

for rule, tag in [("1min", "1min"), ("5min", "5min")]:
    agg = df[cols_all].resample(rule).mean()
    cnt = df[cols_all[0]].resample(rule).size()
    out = agg.copy()
    out["count"] = cnt
    n_nan = int(out[PRIMARY].isna().sum())
    print(f"[{tag}] kutu={len(out)} bos={n_nan} "
          f"tweet/kutu med={int(cnt.median())} min={int(cnt[cnt > 0].min())}",
          flush=True)
    out[cols_all] = out[cols_all].interpolate(method="linear",
                                              limit_direction="both")
    out["count"] = out["count"].fillna(0)
    p = os.path.join(outdir, f"{prefix}_series_{tag}.csv")
    out.to_csv(p)
    print(f"  -> {p}", flush=True)

# Grafik (1-dk)
s1 = pd.read_csv(os.path.join(outdir, f"{prefix}_series_1min.csv"), index_col=0,
                 parse_dates=True)
fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
for col, c in [("sadness", "navy"), ("fear", "darkorange"),
               ("anger", "crimson"), ("joy", "green")]:
    if col in s1.columns:
        ax[0].plot(s1.index, s1[col], lw=0.6, color=c, label=col)
ax[0].set_ylabel("Duygu olasiligi")
ax[0].legend(loc="upper right", ncol=4, fontsize=8)
ax[0].set_title(f"Duygu Zaman Serileri (1-dk) - {prefix}")
ax[0].grid(alpha=0.3)
ax[1].plot(s1.index, s1[PRIMARY], lw=0.6, color="purple")
ax[1].set_ylabel(f"{PRIMARY}\n(uzuntu+korku)")
ax[1].grid(alpha=0.3)
ax[2].plot(s1.index, s1["count"], lw=0.5, color="darkred")
ax[2].set_ylabel("Tweet / dakika")
ax[2].set_xlabel("Zaman (UTC)")
ax[2].grid(alpha=0.3)
plt.tight_layout()
pp = os.path.join(outdir, f"{prefix}_series_plot.png")
plt.savefig(pp, dpi=140)
print(f"Grafik: {pp}", flush=True)
print(f"\n1-dk N={len(s1)} | {PRIMARY} ort={s1[PRIMARY].mean():+.4f} "
      f"std={s1[PRIMARY].std():.4f}", flush=True)
if "val_sav" in s1.columns:
    print(f"val_sav ort={s1['val_sav'].mean():+.4f} std={s1['val_sav'].std():.4f}",
          flush=True)
