"""
Iki deprem olayini karsilastir (MF-DFA sonuclari uzerinden).
Girdi : results/km2023_mfdfa.csv, results/izmir2020_mfdfa.csv
Cikti : results/event_comparison.csv + results/event_comparison.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = r"C:\TUBITAK_Proje\results"
files = [("Kahramanmaras 2023", os.path.join(R, "km2023_mfdfa.csv")),
         ("Izmir 2020", os.path.join(R, "izmir2020_mfdfa.csv"))]

frames = []
for name, f in files:
    if os.path.exists(f):
        d = pd.read_csv(f)
        d["event"] = name
        frames.append(d)
    else:
        print(f"UYARI: yok -> {f}")

if not frames:
    print("Hicbir MF-DFA sonucu bulunamadi. Once analyze_series.py calistir.")
    raise SystemExit(1)

allr = pd.concat(frames, ignore_index=True)
allr.to_csv(os.path.join(R, "event_comparison.csv"), index=False)

print("=" * 78)
print(allr[["event", "window", "N", "width", "hspread",
            "iaaft_z", "iaaft_p", "shuffle_z", "shuffle_p"]].to_string(index=False))
print("=" * 78)

# Gorsel: her olay icin Delta_alpha (pencere bazli)
windows = [w for w in allr["window"].unique() if w.startswith(("TUM", "AKUT", "NORM"))]
wlabel = {"TUM": "Tüm", "AKUT_24s": "Akut (24s)", "NORM_son48s": "Normalleşme (48s)"}
wlabel = {k: v for k, v in wlabel.items()}

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(windows))
wcol = 0.35
for i, (name, _) in enumerate(files):
    sub = allr[allr["event"] == name].set_index("window")
    widths = [sub.loc[w, "width"] if w in sub.index else np.nan for w in windows]
    ax[0].bar(x + (i - 0.5) * wcol, widths, wcol, label=name)
    zs = [sub.loc[w, "iaaft_z"] if w in sub.index else np.nan for w in windows]
    ax[1].bar(x + (i - 0.5) * wcol, zs, wcol, label=name)

ax[0].set_xticks(x)
ax[0].set_xticklabels([wlabel.get(w, w) for w in windows])
ax[0].set_ylabel(r"$\Delta\alpha$ (çoklu fraktallik genişliği)")
ax[0].set_title(r"Çoklu Fraktallik Genişliği: $\Delta\alpha$")
ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")

ax[1].axhline(2, color="r", ls="--", label="z=2 (anlamlılık)")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xticks(x)
ax[1].set_xticklabels([wlabel.get(w, w) for w in windows])
ax[1].set_ylabel("IAAFT surrogate z-skoru")
ax[1].set_title("Surrogate testi: gerçek > rastgele? (z)")
ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
pp = os.path.join(R, "event_comparison.png")
plt.savefig(pp, dpi=140)
print(f"\nKaydedildi: results/event_comparison.csv, {pp}")
