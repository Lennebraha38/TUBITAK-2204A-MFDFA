"""
Iki olayi (ve tum sinyalleri) karsilastir: MF-DFA sonuc tablosu + figur.

Sonuclari results\*_*_mfdfa.csv dosyalarindan okur.
Cikti: results\comparison_all.csv, results\comparison_summary.txt,
       results\comparison_plot.png
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = r"C:\TUBITAK_Proje\results"

files = sorted(glob.glob(os.path.join(RES, "*_*_mfdfa.csv")))
files = [f for f in files if "comparison" not in os.path.basename(f)]
print(f"Bulunan MF-DFA sonuc dosyasi: {len(files)}", flush=True)
for f in files:
    print("  ", os.path.basename(f), flush=True)

if not files:
    raise SystemExit("Hic sonuc dosyasi yok.")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset=["event", "signal", "window"], keep="last")
df.to_csv(os.path.join(RES, "comparison_all.csv"), index=False)

lines = []
def w(s=""):
    lines.append(s)
    print(s, flush=True)


w("=" * 90)
w("MF-DFA OZET: Delta_alpha ve IAAFT z-skoru (event x sinyal x pencere)")
w("=" * 90)

for event in sorted(df["event"].unique()):
    sub = df[df["event"] == event]
    w("")
    w(f"### OLAY: {event}")
    piv = sub.pivot_table(index="signal", columns="window",
                          values="delta_alpha", aggfunc="mean")
    w("Delta_alpha:")
    w(piv.round(4).to_string())
    w("")
    pivz = sub.pivot_table(index="signal", columns="window",
                           values="iaaft_z", aggfunc="mean")
    w("IAAFT z (buyuk = IAAFT null'dan anlamli sapiyor = dogrusal-olmayan coklu fraktal):")
    w(pivz.round(2).to_string())

# Ana sinyal: distress
w("")
w("=" * 90)
w("ANA SINYAL (distress = uzuntu+korku) - pencere karsilastirmasi")
w("=" * 90)
prim = df[df["signal"] == "distress"]
w(prim[["event", "window", "N", "delta_alpha", "delta_h",
        "iaaft_z", "iaaft_p", "shuffle_z", "shuffle_p"]].to_string(index=False))

# Figur: sinyal bazinda Delta_alpha (TUM penceresi), iki olay
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
tum = df[df["window"] == "TUM"]
events = sorted(tum["event"].unique())
signals = sorted(tum["signal"].unique())
xpos = np.arange(len(signals))
wid = 0.8 / max(1, len(events))
for k, ev in enumerate(events):
    vals = [tum[(tum["event"] == ev) & (tum["signal"] == s)]["delta_alpha"].mean()
            for s in signals]
    zs = [tum[(tum["event"] == ev) & (tum["signal"] == s)]["iaaft_z"].mean()
          for s in signals]
    ax[0].bar(xpos + k * wid, vals, wid, label=ev)
    ax[1].bar(xpos + k * wid, zs, wid, label=ev)
ax[0].set_xticks(xpos + wid * (len(events) - 1) / 2)
ax[0].set_xticklabels(signals, rotation=45, ha="right")
ax[0].set_ylabel(r"$\Delta\alpha$ (TUM)")
ax[0].set_title("Coklu fraktallik genisligi - tum seri")
ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")
ax[1].set_xticks(xpos + wid * (len(events) - 1) / 2)
ax[1].set_xticklabels(signals, rotation=45, ha="right")
ax[1].set_ylabel("IAAFT z")
ax[1].set_title("IAAFT null'a gore z-skoru")
ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
pp = os.path.join(RES, "comparison_plot.png")
plt.savefig(pp, dpi=140)
w("")
w(f"Figur: {pp}")

txt = os.path.join(RES, "comparison_summary.txt")
open(txt, "w", encoding="utf-8").write("\n".join(lines))
w(f"Ozet: {txt}")
