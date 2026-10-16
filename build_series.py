"""
Skorlanmis tweetlerden zaman serisi uret (genel - her olay icin).
Kullanim: python build_series.py <glob> <tarih_kolonu> <cikti_onek>
Ornek   : python build_series.py "results\scored\deprem*_scored.csv.gz" Date km2023
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

files = sorted(glob.glob(pattern))
print(f"Dosyalar ({len(files)}):", flush=True)
for f in files:
    print("  ", f, flush=True)

frames = [pd.read_csv(f) for f in files]
df = pd.concat(frames, ignore_index=True)
df[datecol] = pd.to_datetime(df[datecol], utc=True)
df = df.set_index(datecol).sort_index()
print(f"\nToplam tweet: {len(df):,}", flush=True)
print(f"Aralik: {df.index.min()} -> {df.index.max()}", flush=True)

outdir = r"C:\TUBITAK_Proje\results"
summary = {}
for rule, tag in [("1min", "1min"), ("5min", "5min")]:
    agg = df["valence"].resample(rule).mean()
    cnt = df["valence"].resample(rule).size()
    out = pd.DataFrame({"valence": agg, "count": cnt})
    n_nan = int(out["valence"].isna().sum())
    print(f"[{tag}] kutu={len(out)} bos={n_nan} "
          f"tweet/kutu med={int(cnt.median())} min={cnt[cnt>0].min()}", flush=True)
    out["valence"] = out["valence"].interpolate(method="linear",
                                                limit_direction="both")
    # count serisini de doldur (log hacim icin onemli)
    out["count"] = out["count"].fillna(0)
    p = os.path.join(outdir, f"{prefix}_series_{tag}.csv")
    out.to_csv(p)
    print(f"  -> {p}", flush=True)
    if tag == "1min":
        summary = out

fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
ax[0].plot(summary.index, summary["valence"], lw=0.5, color="navy")
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_ylabel("Valans (pos-neg)")
ax[0].set_title(f"Duygu Zaman Serisi (1-dk) - {prefix}")
ax[0].grid(alpha=0.3)
ax[1].plot(summary.index, summary["count"], lw=0.5, color="darkred")
ax[1].set_ylabel("Tweet / dakika")
ax[1].set_xlabel("Zaman (UTC)")
ax[1].grid(alpha=0.3)
plt.tight_layout()
pp = os.path.join(outdir, f"{prefix}_series_plot.png")
plt.savefig(pp, dpi=140)
print(f"Grafik: {pp}", flush=True)
print(f"\n1-dk N={len(summary)} | valans ort={summary['valence'].mean():+.4f} "
      f"std={summary['valence'].std():.4f}", flush=True)
